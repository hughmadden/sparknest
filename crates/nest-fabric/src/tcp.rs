//! TCP lanes: the bulk data path for hosts off the RDMA fabric (ADR-048).
//!
//! Off-grid hosts (ADR-047) reach the cluster over 10GbE. One TCP stream
//! does not fill such a link on every host (WSL2 needs four to eight,
//! benchmarks/M10-LAN-TCP.md), so each peer pair keeps several lanes open
//! and spreads requests over them. A lane is a plain TCP connection,
//! authenticated with the cluster secret, that carries requests both ways:
//! the off-grid host dials every lane (it may sit behind NAT) and either end
//! asks the other for byte ranges on it.
//!
//! ```text
//! request   kind=1 | id u64 | file u64 | generation u64 | offset u64 | len u32
//! answer    kind=2 | id u64 | len u32 | <len bytes of the file>
//! failure   kind=3 | id u64 | len u32 | <postcard NestError>
//! ```
//!
//! Each lane has a reader and a writer thread on a blocking socket. The
//! answer's header goes out with MSG_MORE and its body with `sendfile`
//! straight from the object (no copy through user space on the serving
//! host); TCP_NODELAY keeps requests and the last segment of an answer from
//! waiting. Objects are opened through the same `ReadSource` as RDMA reads,
//! so serving checks and fencing are shared: the object's guard is held
//! until its bytes are on the wire.

use crate::{ReadSource, ServeFile};
use nest_types::{FileId, Generation, NestError, NodeId};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr};
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak, mpsc};
use std::time::Duration;
use tokio::sync::oneshot;

const REQ: u8 = 1;
const ANSWER: u8 = 2;
const FAILURE: u8 = 3;
/// Bytes after the kind byte.
const REQ_LEN: usize = 8 + 8 + 8 + 8 + 4;
const ANSWER_LEN: usize = 8 + 4;
/// Longest failure message accepted.
const MAX_FAILURE: usize = 64 << 10;

#[derive(Clone, Debug)]
pub struct LanesConfig {
    /// Lanes kept open to each peer this host dials.
    pub per_peer: usize,
    /// Largest single read.
    pub chunk: usize,
    /// Bytes of reads this host has outstanding at once, over all peers.
    pub inflight_bytes: u64,
    pub connect_timeout: Duration,
    /// How long a read may wait for its answer.
    pub read_timeout: Duration,
}

impl Default for LanesConfig {
    fn default() -> Self {
        LanesConfig {
            per_peer: 4,
            chunk: 4 << 20,
            inflight_bytes: 64 << 20,
            connect_timeout: Duration::from_secs(2),
            read_timeout: Duration::from_secs(30),
        }
    }
}

/// Counters for tests and the status page.
#[derive(Default)]
pub struct LaneStats {
    /// Bytes this host read from others.
    pub read_bytes: AtomicU64,
    pub reads: AtomicU64,
    /// Bytes this host sent to others.
    pub served_bytes: AtomicU64,
    pub served: AtomicU64,
}

/// Unit the in-flight budget counts in.
const BUDGET_UNIT: usize = 64 << 10;

pub struct Lanes {
    me: NodeId,
    cluster: String,
    secret: Vec<u8>,
    cfg: LanesConfig,
    peers: Mutex<HashMap<NodeId, Vec<Arc<Lane>>>>,
    /// Peers this host dials, and where.
    dial: Mutex<HashMap<NodeId, SocketAddr>>,
    source: Mutex<Option<Weak<dyn ReadSource>>>,
    budget: Arc<tokio::sync::Semaphore>,
    rt: tokio::runtime::Handle,
    stop: tokio::sync::watch::Sender<bool>,
    local_addr: Option<SocketAddr>,
    pub stats: LaneStats,
}

type Waiter = oneshot::Sender<Result<Vec<u8>, NestError>>;

struct Lane {
    peer: NodeId,
    out: mpsc::Sender<Out>,
    pending: Mutex<HashMap<u64, Waiter>>,
    next_id: AtomicU64,
    /// Our requests waiting for an answer on this lane.
    outstanding: AtomicUsize,
    dead: AtomicBool,
    sock: std::net::TcpStream,
}

enum Out {
    Req {
        id: u64,
        file: FileId,
        generation: Generation,
        offset: u64,
        len: u32,
    },
    Answer {
        id: u64,
        r: Result<(ServeFile, u64, u32), NestError>,
    },
}

impl Lanes {
    /// Start the lanes service; with `listen`, accept lanes there (a host
    /// others dial).
    pub async fn start(
        me: NodeId,
        cluster: String,
        secret: Vec<u8>,
        cfg: LanesConfig,
        listen: Option<SocketAddr>,
    ) -> std::io::Result<Arc<Lanes>> {
        let listener = match listen {
            Some(a) => Some(tokio::net::TcpListener::bind(a).await?),
            None => None,
        };
        let units = (cfg.inflight_bytes as usize / BUDGET_UNIT).max(cfg.chunk / BUDGET_UNIT);
        let (stop, _) = tokio::sync::watch::channel(false);
        let lanes = Arc::new(Lanes {
            me,
            cluster,
            secret,
            local_addr: listener.as_ref().and_then(|l| l.local_addr().ok()),
            cfg,
            peers: Mutex::new(HashMap::new()),
            dial: Mutex::new(HashMap::new()),
            source: Mutex::new(None),
            budget: Arc::new(tokio::sync::Semaphore::new(units)),
            rt: tokio::runtime::Handle::current(),
            stop,
            stats: LaneStats::default(),
        });
        if let Some(l) = listener {
            let weak = Arc::downgrade(&lanes);
            let mut stop = lanes.stop.subscribe();
            tokio::spawn(async move {
                loop {
                    let accepted = tokio::select! {
                        a = l.accept() => a,
                        _ = stop.changed() => return,
                    };
                    let Ok((s, _)) = accepted else { continue };
                    let Some(lanes) = weak.upgrade() else { return };
                    tokio::spawn(async move {
                        if let Err(e) = lanes.accept(s).await {
                            tracing::debug!(error = %e, "rejected TCP lane");
                        }
                    });
                }
            });
        }
        lanes.keep_dialled();
        Ok(lanes)
    }

    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.local_addr
    }

    pub fn chunk(&self) -> usize {
        self.cfg.chunk
    }

    pub fn set_source(&self, source: Weak<dyn ReadSource>) {
        *self.source.lock() = Some(source);
    }

    /// Keep `per_peer` lanes open to `peer` at `addr` (an off-grid host
    /// dials every host it knows; the others only accept).
    pub fn dial(&self, peer: NodeId, addr: SocketAddr) {
        self.dial.lock().insert(peer, addr);
    }

    /// Live lanes to `peer`.
    pub fn connected(&self, peer: NodeId) -> usize {
        self.peers.lock().get(&peer).map_or(0, |v| {
            v.iter().filter(|l| !l.dead.load(Ordering::SeqCst)).count()
        })
    }

    /// Read `len` bytes (≤ chunk) of exactly `generation` of `file` from
    /// `peer`; fewer only at the end of the file.
    pub async fn read(
        &self,
        peer: NodeId,
        file: FileId,
        generation: Generation,
        offset: u64,
        len: usize,
    ) -> Result<Vec<u8>, NestError> {
        assert!(len <= self.cfg.chunk);
        let lane = self
            .pick(peer)
            .ok_or_else(|| NestError::Unavailable(format!("no TCP lane to {peer}")))?;
        let _budget = self
            .budget
            .acquire_many(len.div_ceil(BUDGET_UNIT).max(1) as u32)
            .await
            .map_err(|_| NestError::Unavailable("lanes stopped".into()))?;
        let id = lane.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        lane.pending.lock().insert(id, tx);
        lane.outstanding.fetch_add(1, Ordering::Relaxed);
        let sent = lane.out.send(Out::Req {
            id,
            file,
            generation,
            offset,
            len: len as u32,
        });
        let r = match sent {
            Err(_) => Err(NestError::Unavailable(format!("TCP lane to {peer} closed"))),
            Ok(()) => match tokio::time::timeout(self.cfg.read_timeout, rx).await {
                Ok(Ok(r)) => r,
                Ok(Err(_)) => Err(NestError::Unavailable(format!("TCP lane to {peer} lost"))),
                Err(_) => Err(NestError::Unavailable(format!(
                    "TCP read from {peer} timed out"
                ))),
            },
        };
        lane.pending.lock().remove(&id);
        lane.outstanding.fetch_sub(1, Ordering::Relaxed);
        if let Ok(b) = &r {
            self.stats.reads.fetch_add(1, Ordering::Relaxed);
            self.stats
                .read_bytes
                .fetch_add(b.len() as u64, Ordering::Relaxed);
        }
        r
    }

    /// The live lane to `peer` with the fewest of our reads waiting.
    fn pick(&self, peer: NodeId) -> Option<Arc<Lane>> {
        self.peers
            .lock()
            .get(&peer)?
            .iter()
            .filter(|l| !l.dead.load(Ordering::SeqCst))
            .min_by_key(|l| l.outstanding.load(Ordering::Relaxed))
            .cloned()
    }

    /// Close every lane; outstanding reads fail.
    pub fn shutdown(&self) {
        let _ = self.stop.send(true);
        self.dial.lock().clear();
        for (_, v) in self.peers.lock().drain() {
            for l in v {
                l.kill();
            }
        }
    }

    async fn accept(self: Arc<Self>, mut s: tokio::net::TcpStream) -> std::io::Result<()> {
        s.set_nodelay(true)?;
        let peer = tokio::time::timeout(
            self.cfg.connect_timeout,
            nest_rpc::auth::server(&mut s, &self.secret, &self.cluster, self.me),
        )
        .await
        .map_err(|_| std::io::Error::other("handshake timeout"))??;
        self.run(peer, s)
    }

    async fn connect(self: Arc<Self>, peer: NodeId, addr: SocketAddr) -> std::io::Result<()> {
        let mut s = tokio::time::timeout(
            self.cfg.connect_timeout,
            tokio::net::TcpStream::connect(addr),
        )
        .await
        .map_err(|_| std::io::Error::other("connect timeout"))??;
        s.set_nodelay(true)?;
        let who = tokio::time::timeout(
            self.cfg.connect_timeout,
            nest_rpc::auth::client(&mut s, &self.secret, &self.cluster, self.me),
        )
        .await
        .map_err(|_| std::io::Error::other("handshake timeout"))??;
        if who != peer {
            return Err(std::io::Error::other(format!(
                "{addr} is node {who}, expected {peer}"
            )));
        }
        self.run(peer, s)
    }

    /// Hand an authenticated stream to its two threads and register it.
    fn run(self: &Arc<Self>, peer: NodeId, s: tokio::net::TcpStream) -> std::io::Result<()> {
        let s = s.into_std()?;
        s.set_nonblocking(false)?;
        let (out, rx) = mpsc::channel();
        let lane = Arc::new(Lane {
            peer,
            out,
            pending: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            outstanding: AtomicUsize::new(0),
            dead: AtomicBool::new(false),
            sock: s.try_clone()?,
        });
        let (r, w) = (s.try_clone()?, s);
        let (l1, me1) = (lane.clone(), Arc::downgrade(self));
        std::thread::Builder::new()
            .name(format!("lane-rx-{peer}"))
            .spawn(move || reader(l1, me1, r))?;
        let (l2, me2) = (lane.clone(), Arc::downgrade(self));
        std::thread::Builder::new()
            .name(format!("lane-tx-{peer}"))
            .spawn(move || writer(l2, me2, w, rx))?;
        let mut peers = self.peers.lock();
        let v = peers.entry(peer).or_default();
        v.retain(|l| !l.dead.load(Ordering::SeqCst));
        v.push(lane);
        Ok(())
    }

    /// Top up dialled peers to `per_peer` live lanes every half second.
    fn keep_dialled(self: &Arc<Self>) {
        let weak = Arc::downgrade(self);
        let mut stop = self.stop.subscribe();
        tokio::spawn(async move {
            let connecting: Arc<Mutex<HashMap<NodeId, usize>>> = Arc::default();
            loop {
                let Some(me) = weak.upgrade() else { return };
                let dial: Vec<(NodeId, SocketAddr)> =
                    me.dial.lock().iter().map(|(p, a)| (*p, *a)).collect();
                for (peer, addr) in dial {
                    let busy = connecting.lock().get(&peer).copied().unwrap_or(0);
                    let missing = me.cfg.per_peer.saturating_sub(me.connected(peer) + busy);
                    for _ in 0..missing {
                        *connecting.lock().entry(peer).or_default() += 1;
                        let (me, connecting) = (me.clone(), connecting.clone());
                        tokio::spawn(async move {
                            if let Err(e) = me.connect(peer, addr).await {
                                tracing::debug!(%peer, error = %e, "TCP lane connect failed");
                            }
                            *connecting.lock().entry(peer).or_default() -= 1;
                        });
                    }
                }
                drop(me);
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_millis(500)) => {}
                    _ = stop.changed() => return,
                }
            }
        });
    }
}

impl Lane {
    /// Close the lane: its threads exit and its waiting reads fail.
    fn kill(&self) {
        if !self.dead.swap(true, Ordering::SeqCst) {
            let _ = self.sock.shutdown(Shutdown::Both);
        }
        for (_, tx) in self.pending.lock().drain() {
            let _ = tx.send(Err(NestError::Unavailable(format!(
                "TCP lane to {} lost",
                self.peer
            ))));
        }
    }
}

/// Reads requests (answered through the writer) and answers (handed to
/// their waiting reads).
fn reader(lane: Arc<Lane>, lanes: Weak<Lanes>, mut s: std::net::TcpStream) {
    let r = (|| -> std::io::Result<()> {
        loop {
            let mut kind = [0u8; 1];
            s.read_exact(&mut kind)?;
            match kind[0] {
                REQ => {
                    let mut b = [0u8; REQ_LEN];
                    s.read_exact(&mut b)?;
                    let id = u64::from_le_bytes(b[0..8].try_into().unwrap());
                    let file = FileId(u64::from_le_bytes(b[8..16].try_into().unwrap()));
                    let generation = Generation(u64::from_le_bytes(b[16..24].try_into().unwrap()));
                    let offset = u64::from_le_bytes(b[24..32].try_into().unwrap());
                    let len = u32::from_le_bytes(b[32..36].try_into().unwrap());
                    let Some(l) = lanes.upgrade() else {
                        return Ok(());
                    };
                    let source = l.source.lock().as_ref().and_then(|w| w.upgrade());
                    let out = lane.out.clone();
                    let answer = move |r: Result<ServeFile, NestError>| {
                        let _ = out.send(Out::Answer {
                            id,
                            r: r.map(|f| (f, offset, len)),
                        });
                    };
                    match source {
                        None => answer(Err(NestError::Unavailable("not serving yet".into()))),
                        Some(src) => match src.open_now(file, generation) {
                            Some(r) => answer(r),
                            None => {
                                let open = src.open(file, generation);
                                l.rt.spawn(async move { answer(open.await) });
                            }
                        },
                    }
                }
                ANSWER | FAILURE => {
                    let mut b = [0u8; ANSWER_LEN];
                    s.read_exact(&mut b)?;
                    let id = u64::from_le_bytes(b[0..8].try_into().unwrap());
                    let len = u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize;
                    if kind[0] == FAILURE && len > MAX_FAILURE {
                        return Err(std::io::Error::other("oversized failure"));
                    }
                    let mut body = vec![0u8; len];
                    s.read_exact(&mut body)?;
                    let r = if kind[0] == ANSWER {
                        Ok(body)
                    } else {
                        Err(nest_rpc::decode::<NestError>(&body)
                            .unwrap_or_else(|e| NestError::Io(e.to_string())))
                    };
                    // A read that gave up meanwhile has no waiter: dropped.
                    if let Some(tx) = lane.pending.lock().remove(&id) {
                        let _ = tx.send(r);
                    }
                }
                k => return Err(std::io::Error::other(format!("bad lane frame kind {k}"))),
            }
        }
    })();
    if let Err(e) = r
        && !lane.dead.load(Ordering::SeqCst)
    {
        tracing::debug!(peer = %lane.peer, error = %e, "TCP lane closed");
    }
    lane.kill();
}

/// Sends our requests and our answers, in the order they become ready.
fn writer(
    lane: Arc<Lane>,
    lanes: Weak<Lanes>,
    mut s: std::net::TcpStream,
    rx: mpsc::Receiver<Out>,
) {
    let r = (|| -> std::io::Result<()> {
        while let Ok(o) = rx.recv() {
            if lane.dead.load(Ordering::SeqCst) {
                return Ok(());
            }
            match o {
                Out::Req {
                    id,
                    file,
                    generation,
                    offset,
                    len,
                } => {
                    let mut b = [0u8; 1 + REQ_LEN];
                    b[0] = REQ;
                    b[1..9].copy_from_slice(&id.to_le_bytes());
                    b[9..17].copy_from_slice(&file.0.to_le_bytes());
                    b[17..25].copy_from_slice(&generation.0.to_le_bytes());
                    b[25..33].copy_from_slice(&offset.to_le_bytes());
                    b[33..37].copy_from_slice(&len.to_le_bytes());
                    s.write_all(&b)?;
                }
                Out::Answer {
                    id,
                    r: Ok((f, offset, len)),
                } => {
                    let size = f.file.metadata()?.len();
                    let n = (len as u64).min(size.saturating_sub(offset)) as u32;
                    send_more(&s, &answer_header(ANSWER, id, n))?;
                    send_file(&s, &f.file, offset, n as usize)?;
                    drop(f); // the object's guard: its bytes are on the wire
                    if let Some(l) = lanes.upgrade() {
                        l.stats.served.fetch_add(1, Ordering::Relaxed);
                        l.stats.served_bytes.fetch_add(n as u64, Ordering::Relaxed);
                    }
                }
                Out::Answer { id, r: Err(e) } => {
                    let body = nest_rpc::encode(&e).unwrap_or_default();
                    s.write_all(&answer_header(FAILURE, id, body.len() as u32))?;
                    s.write_all(&body)?;
                }
            }
        }
        Ok(())
    })();
    if let Err(e) = r
        && !lane.dead.load(Ordering::SeqCst)
    {
        tracing::debug!(peer = %lane.peer, error = %e, "TCP lane write failed");
    }
    lane.kill();
}

fn answer_header(kind: u8, id: u64, len: u32) -> [u8; 1 + ANSWER_LEN] {
    let mut b = [0u8; 1 + ANSWER_LEN];
    b[0] = kind;
    b[1..9].copy_from_slice(&id.to_le_bytes());
    b[9..13].copy_from_slice(&len.to_le_bytes());
    b
}

/// Send `buf` with MSG_MORE: it leaves with the first segment of the body.
fn send_more(s: &std::net::TcpStream, mut buf: &[u8]) -> std::io::Result<()> {
    while !buf.is_empty() {
        let n = nix::sys::socket::send(
            s.as_raw_fd(),
            buf,
            // nix has no name for MSG_MORE.
            nix::sys::socket::MsgFlags::from_bits_retain(libc::MSG_MORE | libc::MSG_NOSIGNAL),
        )
        .map_err(std::io::Error::from)?;
        buf = &buf[n..];
    }
    Ok(())
}

/// Send `len` bytes of `file` from `offset` with sendfile(2).
fn send_file(
    s: &std::net::TcpStream,
    file: &std::fs::File,
    offset: u64,
    len: usize,
) -> std::io::Result<()> {
    let mut off = offset as libc::off_t;
    let mut left = len;
    while left > 0 {
        let n = match nix::sys::sendfile::sendfile(s, file, Some(&mut off), left) {
            Ok(n) => n,
            Err(nix::errno::Errno::EINTR) => continue,
            Err(e) => return Err(e.into()),
        };
        if n == 0 {
            // The object shrank under us (never for a STABLE generation):
            // the stream can no longer be framed.
            return Err(std::io::Error::other("object ended before its answer"));
        }
        left -= n;
    }
    Ok(())
}
