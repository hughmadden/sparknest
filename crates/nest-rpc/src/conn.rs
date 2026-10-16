use crate::{Filter, RpcError, auth};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use futures::future::BoxFuture;
use nest_types::NodeId;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufWriter};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot, watch};

const MAX_FRAME: usize = 256 << 20;
const KIND_REQ: u8 = 0;
const KIND_OK: u8 = 1;
const KIND_ERR: u8 = 2;

#[derive(Clone, Debug)]
pub struct RpcConfig {
    pub node: NodeId,
    pub cluster: String,
    pub secret: Vec<u8>,
    pub listen: SocketAddr,
    pub connect_timeout: Duration,
}

/// A service implementation. Errors are returned to the caller as
/// [`RpcError::Remote`].
pub trait Handler: Send + Sync + 'static {
    fn call(&self, peer: NodeId, body: Bytes) -> BoxFuture<'static, Result<Bytes, String>>;
}

impl<F> Handler for F
where
    F: Fn(NodeId, Bytes) -> BoxFuture<'static, Result<Bytes, String>> + Send + Sync + 'static,
{
    fn call(&self, peer: NodeId, body: Bytes) -> BoxFuture<'static, Result<Bytes, String>> {
        self(peer, body)
    }
}

/// Handle to this node's RPC endpoint: server for incoming requests and
/// client pool for outgoing ones. Cheap to clone.
#[derive(Clone)]
pub struct Rpc {
    inner: Arc<Inner>,
}

struct Inner {
    cfg: RpcConfig,
    local_addr: SocketAddr,
    handlers: parking_lot::RwLock<HashMap<u8, Arc<dyn Handler>>>,
    addrs: parking_lot::RwLock<HashMap<NodeId, SocketAddr>>,
    pinned: parking_lot::RwLock<HashMap<NodeId, SocketAddr>>,
    conns: Mutex<HashMap<NodeId, ConnSlot>>,
    filter: Filter,
    shutdown: watch::Sender<bool>,
}

struct Frame {
    id: u64,
    kind: u8,
    service: u8,
    body: Bytes,
}

async fn write_frame(w: &mut BufWriter<OwnedWriteHalf>, f: &Frame) -> std::io::Result<()> {
    let mut hdr = BytesMut::with_capacity(14);
    hdr.put_u32_le((10 + f.body.len()) as u32);
    hdr.put_u64_le(f.id);
    hdr.put_u8(f.kind);
    hdr.put_u8(f.service);
    w.write_all(&hdr).await?;
    w.write_all(&f.body).await
}

async fn read_frame(r: &mut OwnedReadHalf) -> std::io::Result<Frame> {
    let len = r.read_u32_le().await? as usize;
    if !(10..=MAX_FRAME).contains(&len) {
        return Err(std::io::Error::other(format!("bad frame length {len}")));
    }
    let mut buf = BytesMut::zeroed(len);
    r.read_exact(&mut buf).await?;
    let mut hdr = &buf[..10];
    let id = hdr.get_u64_le();
    let kind = hdr.get_u8();
    let service = hdr.get_u8();
    let body = buf.freeze().slice(10..);
    Ok(Frame {
        id,
        kind,
        service,
        body,
    })
}

/// Writer task: drains frames and flushes when the queue is momentarily
/// empty, so bursts coalesce into few syscalls.
async fn writer(mut w: BufWriter<OwnedWriteHalf>, mut rx: mpsc::Receiver<Frame>) {
    while let Some(f) = rx.recv().await {
        if write_frame(&mut w, &f).await.is_err() {
            return;
        }
        while let Ok(f) = rx.try_recv() {
            if write_frame(&mut w, &f).await.is_err() {
                return;
            }
        }
        if w.flush().await.is_err() {
            return;
        }
    }
}

/// Per-peer slot; the async mutex serializes (re)connection attempts.
type ConnSlot = Arc<tokio::sync::Mutex<Option<Arc<Conn>>>>;

type Pending = Mutex<HashMap<u64, oneshot::Sender<Result<Bytes, RpcError>>>>;

/// An authenticated connection to one peer. Either end may send requests
/// on it (ADR-047): each numbers its own, and the frame kind tells a
/// request from the other end apart from an answer to one of ours. A node
/// dials the peers it has an address for; a peer it cannot reach (behind
/// NAT) dials in, and its connection is used for both directions.
struct Conn {
    peer: NodeId,
    tx: mpsc::Sender<Frame>,
    pending: Arc<Pending>,
    next_id: AtomicU64,
    dead: Arc<AtomicBool>,
}

impl Conn {
    async fn open(
        inner: &Arc<Inner>,
        peer: NodeId,
        addr: SocketAddr,
    ) -> Result<Arc<Conn>, RpcError> {
        let unreachable = |e: String| RpcError::Unreachable(peer, e);
        let mut s = tokio::time::timeout(inner.cfg.connect_timeout, TcpStream::connect(addr))
            .await
            .map_err(|_| unreachable("connect timeout".into()))?
            .map_err(|e| unreachable(e.to_string()))?;
        s.set_nodelay(true).ok();
        crate::sockopt::set_rto_min(&s, crate::sockopt::RTO_MIN);
        let who = tokio::time::timeout(
            inner.cfg.connect_timeout,
            auth::client(
                &mut s,
                &inner.cfg.secret,
                &inner.cfg.cluster,
                inner.cfg.node,
            ),
        )
        .await
        .map_err(|_| unreachable("handshake timeout".into()))?
        .map_err(|e| unreachable(e.to_string()))?;
        if who != peer {
            return Err(unreachable(format!(
                "{addr} is node {who}, expected {peer}"
            )));
        }
        Ok(Conn::start(inner, peer, s))
    }

    /// Run an authenticated stream: answer the peer's requests and route
    /// answers to ours.
    fn start(inner: &Arc<Inner>, peer: NodeId, s: TcpStream) -> Arc<Conn> {
        let (mut r, w) = s.into_split();
        let (tx, rx) = mpsc::channel(1024);
        tokio::spawn(writer(BufWriter::with_capacity(256 << 10, w), rx));
        let conn = Arc::new(Conn {
            peer,
            tx: tx.clone(),
            pending: Arc::default(),
            next_id: AtomicU64::new(1),
            dead: Arc::new(AtomicBool::new(false)),
        });
        let (pending, dead) = (conn.pending.clone(), conn.dead.clone());
        let inner = inner.clone();
        let mut stop = inner.shutdown.subscribe();
        tokio::spawn(async move {
            loop {
                let f = tokio::select! {
                    f = read_frame(&mut r) => f,
                    _ = stop.changed() => break,
                };
                let Ok(f) = f else { break };
                match f.kind {
                    KIND_REQ => answer(&inner, peer, &tx, f),
                    KIND_OK | KIND_ERR => {
                        let res = if f.kind == KIND_OK {
                            Ok(f.body)
                        } else {
                            Err(match f.service {
                                0 => {
                                    RpcError::Remote(String::from_utf8_lossy(&f.body).into_owned())
                                }
                                s => RpcError::NoService(s),
                            })
                        };
                        if let Some(tx) = pending.lock().remove(&f.id) {
                            let _ = tx.send(res);
                        }
                    }
                    _ => break,
                }
            }
            dead.store(true, Ordering::SeqCst);
            for (_, tx) in pending.lock().drain() {
                let _ = tx.send(Err(RpcError::Protocol(format!(
                    "connection to {peer} lost"
                ))));
            }
        });
        conn
    }

    fn alive(&self) -> bool {
        !self.dead.load(Ordering::SeqCst)
    }

    async fn call(&self, service: u8, body: Bytes) -> Result<Bytes, RpcError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().insert(id, tx);
        if !self.alive()
            || self
                .tx
                .send(Frame {
                    id,
                    kind: KIND_REQ,
                    service,
                    body,
                })
                .await
                .is_err()
        {
            self.pending.lock().remove(&id);
            return Err(RpcError::Unreachable(self.peer, "connection closed".into()));
        }
        rx.await
            .unwrap_or_else(|_| Err(RpcError::Protocol("connection dropped".into())))
    }
}

/// Run a request from `peer` and send the answer back on its connection.
fn answer(inner: &Arc<Inner>, peer: NodeId, tx: &mpsc::Sender<Frame>, f: Frame) {
    if inner.filter.is_blocked(peer) {
        return; // partitioned: the request is lost
    }
    let handler = inner.handlers.read().get(&f.service).cloned();
    let (tx, inner) = (tx.clone(), inner.clone());
    tokio::spawn(async move {
        let (kind, service, body) = match handler {
            None => (KIND_ERR, f.service, Bytes::new()),
            Some(h) => match h.call(peer, f.body).await {
                Ok(b) => (KIND_OK, 0, b),
                Err(e) => (KIND_ERR, 0, Bytes::from(e.into_bytes())),
            },
        };
        if inner.filter.is_blocked(peer) {
            return;
        }
        let _ = tx
            .send(Frame {
                id: f.id,
                kind,
                service,
                body,
            })
            .await;
    });
}

impl Rpc {
    /// Bind the listener and start serving. Handlers may be registered
    /// afterwards; requests for unregistered services get `NoService`.
    pub async fn bind(cfg: RpcConfig) -> std::io::Result<Rpc> {
        let listener = TcpListener::bind(cfg.listen).await?;
        let local_addr = listener.local_addr()?;
        let (shutdown, _) = watch::channel(false);
        let inner = Arc::new(Inner {
            cfg,
            local_addr,
            handlers: Default::default(),
            addrs: Default::default(),
            pinned: Default::default(),
            conns: Default::default(),
            filter: Filter::default(),
            shutdown,
        });
        let weak = Arc::downgrade(&inner);
        let mut stop = inner.shutdown.subscribe();
        tokio::spawn(async move {
            loop {
                let accepted = tokio::select! {
                    a = listener.accept() => a,
                    _ = stop.changed() => return,
                };
                let Ok((s, _)) = accepted else { continue };
                let Some(inner) = weak.upgrade() else { return };
                tokio::spawn(serve(inner, s));
            }
        });
        let rpc = Rpc { inner };
        rpc.keep_pinned_connected();
        Ok(rpc)
    }

    pub fn node(&self) -> NodeId {
        self.inner.cfg.node
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.inner.local_addr
    }

    pub fn filter(&self) -> &Filter {
        &self.inner.filter
    }

    pub fn register(&self, service: u8, handler: Arc<dyn Handler>) {
        self.inner.handlers.write().insert(service, handler);
    }

    /// Learn a peer's address (from membership). Ignored for a peer whose
    /// address was pinned.
    pub fn set_peer(&self, node: NodeId, addr: SocketAddr) {
        if self.inner.pinned.read().contains_key(&node) {
            return;
        }
        let old = self.inner.addrs.write().insert(node, addr);
        if old.is_some_and(|o| o != addr) {
            self.inner.conns.lock().remove(&node);
        }
    }

    /// Fix a peer's address (from this host's own configuration): nothing
    /// learned later replaces it, and the connection to it is kept open so
    /// the peer can send its requests on it (an off-grid host pins the
    /// fabric hosts' LAN addresses, ADR-047).
    pub fn pin_peer(&self, node: NodeId, addr: SocketAddr) {
        self.inner.pinned.write().insert(node, addr);
        let old = self.inner.addrs.write().insert(node, addr);
        if old.is_some_and(|o| o != addr) {
            self.inner.conns.lock().remove(&node);
        }
    }

    /// Stop dialling `node` (it went off-grid and dials in, ADR-047). A
    /// pinned address stays; an open connection stays until it drops.
    pub fn forget_peer(&self, node: NodeId) {
        if self.inner.pinned.read().contains_key(&node) {
            return;
        }
        self.inner.addrs.write().remove(&node);
    }

    pub fn peer_addr(&self, node: NodeId) -> Option<SocketAddr> {
        self.inner.addrs.read().get(&node).copied()
    }

    /// Whether a live connection to `peer` is open (either direction).
    pub fn connected(&self, peer: NodeId) -> bool {
        let slot = self.inner.conns.lock().get(&peer).cloned();
        slot.and_then(|s| {
            s.try_lock()
                .ok()
                .and_then(|c| c.as_ref().map(|c| c.alive()))
        })
        .unwrap_or(false)
    }

    /// Send one request and wait for its response.
    pub async fn call(
        &self,
        peer: NodeId,
        service: u8,
        body: Bytes,
        timeout: Duration,
    ) -> Result<Bytes, RpcError> {
        if self.inner.filter.is_blocked(peer) {
            return Err(RpcError::Unreachable(peer, "blocked by filter".into()));
        }
        if peer == self.inner.cfg.node {
            // Local calls skip the network but keep the same semantics.
            let h = self.inner.handlers.read().get(&service).cloned();
            let h = h.ok_or(RpcError::NoService(service))?;
            return tokio::time::timeout(timeout, h.call(peer, body))
                .await
                .map_err(|_| RpcError::Timeout(peer))?
                .map_err(RpcError::Remote);
        }
        tokio::time::timeout(timeout, async {
            let conn = self.conn(peer).await?;
            conn.call(service, body).await
        })
        .await
        .map_err(|_| RpcError::Timeout(peer))?
    }

    async fn conn(&self, peer: NodeId) -> Result<Arc<Conn>, RpcError> {
        let slot = self.inner.conns.lock().entry(peer).or_default().clone();
        let mut slot = slot.lock().await;
        if let Some(c) = slot.as_ref()
            && c.alive()
        {
            return Ok(c.clone());
        }
        let addr = self.peer_addr(peer).ok_or_else(|| {
            RpcError::Unreachable(peer, "no address, and it has not connected here".into())
        })?;
        let c = Conn::open(&self.inner, peer, addr).await?;
        *slot = Some(c.clone());
        Ok(c)
    }

    /// Reconnect to pinned peers as soon as their connection drops, so
    /// they can always reach this host.
    fn keep_pinned_connected(&self) {
        let weak = Arc::downgrade(&self.inner);
        let mut stop = self.inner.shutdown.subscribe();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_millis(500)) => {}
                    _ = stop.changed() => return,
                }
                let Some(inner) = weak.upgrade() else { return };
                let rpc = Rpc { inner };
                let peers: Vec<NodeId> = rpc.inner.pinned.read().keys().copied().collect();
                for p in peers {
                    if !rpc.inner.filter.is_blocked(p) && !rpc.connected(p) {
                        let r = rpc.clone();
                        tokio::spawn(async move {
                            let _ = r.conn(p).await;
                        });
                    }
                }
            }
        });
    }

    /// Stop serving and close connections. Outstanding calls fail.
    pub fn shutdown(&self) {
        let _ = self.inner.shutdown.send(true);
        self.inner.conns.lock().clear();
    }
}

async fn serve(inner: Arc<Inner>, mut s: TcpStream) {
    s.set_nodelay(true).ok();
    crate::sockopt::set_rto_min(&s, crate::sockopt::RTO_MIN);
    let hs = tokio::time::timeout(
        inner.cfg.connect_timeout,
        auth::server(
            &mut s,
            &inner.cfg.secret,
            &inner.cfg.cluster,
            inner.cfg.node,
        ),
    )
    .await;
    let peer = match hs {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "rejected rpc connection");
            return;
        }
        Err(_) => return,
    };
    let conn = Conn::start(&inner, peer, s);
    // A peer we cannot dial (no address: it is behind NAT) is reached over
    // the connection it opened; the newest one wins. Peers we can dial keep
    // their own outgoing connection for our requests.
    if !inner.addrs.read().contains_key(&peer) {
        let slot = inner.conns.lock().entry(peer).or_default().clone();
        *slot.lock().await = Some(conn);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::FutureExt;

    async fn node(id: u64, secret: &[u8]) -> Rpc {
        let rpc = Rpc::bind(RpcConfig {
            node: NodeId(id),
            cluster: "test".into(),
            secret: secret.to_vec(),
            listen: "127.0.0.1:0".parse().unwrap(),
            connect_timeout: Duration::from_secs(2),
        })
        .await
        .unwrap();
        let me = id;
        rpc.register(
            7,
            Arc::new(move |peer: NodeId, body: Bytes| {
                async move {
                    if body.as_ref() == b"fail" {
                        return Err("nope".to_string());
                    }
                    let mut v = format!("{me}<-{peer}:").into_bytes();
                    v.extend_from_slice(&body);
                    Ok(Bytes::from(v))
                }
                .boxed()
            }),
        );
        rpc
    }

    #[tokio::test]
    async fn roundtrip_errors_and_filter() {
        let a = node(1, b"s").await;
        let b = node(2, b"s").await;
        a.set_peer(NodeId(2), b.local_addr());
        let t = Duration::from_secs(2);
        let r = a
            .call(NodeId(2), 7, Bytes::from_static(b"hi"), t)
            .await
            .unwrap();
        assert_eq!(&r[..], b"2<-1:hi");
        // Many concurrent calls over one connection.
        let futs: Vec<_> = (0..200)
            .map(|i| a.call(NodeId(2), 7, Bytes::from(format!("{i}")), t))
            .collect();
        for (i, r) in futures::future::join_all(futs)
            .await
            .into_iter()
            .enumerate()
        {
            assert_eq!(r.unwrap(), Bytes::from(format!("2<-1:{i}")));
        }
        assert_eq!(
            a.call(NodeId(2), 7, Bytes::from_static(b"fail"), t).await,
            Err(RpcError::Remote("nope".into()))
        );
        assert_eq!(
            a.call(NodeId(2), 9, Bytes::new(), t).await,
            Err(RpcError::NoService(9))
        );
        // Local call path.
        assert_eq!(
            &a.call(NodeId(1), 7, Bytes::from_static(b"x"), t)
                .await
                .unwrap()[..],
            b"1<-1:x"
        );
        // Partition.
        a.filter().block(NodeId(2));
        assert!(matches!(
            a.call(NodeId(2), 7, Bytes::new(), t).await,
            Err(RpcError::Unreachable(..))
        ));
        a.filter().unblock(NodeId(2));
        b.filter().block(NodeId(1));
        assert_eq!(
            a.call(NodeId(2), 7, Bytes::new(), Duration::from_millis(200))
                .await,
            Err(RpcError::Timeout(NodeId(2)))
        );
    }

    #[tokio::test]
    async fn wrong_secret_is_rejected() {
        let a = node(1, b"secret-a").await;
        let b = node(2, b"secret-b").await;
        a.set_peer(NodeId(2), b.local_addr());
        let r = a
            .call(NodeId(2), 7, Bytes::new(), Duration::from_secs(2))
            .await;
        assert!(matches!(r, Err(RpcError::Unreachable(..))), "{r:?}");
    }

    #[tokio::test]
    async fn reconnects_after_peer_restart() {
        let a = node(1, b"s").await;
        let b = node(2, b"s").await;
        a.set_peer(NodeId(2), b.local_addr());
        let t = Duration::from_secs(2);
        a.call(NodeId(2), 7, Bytes::new(), t).await.unwrap();
        let addr = b.local_addr();
        b.shutdown();
        drop(b);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(a.call(NodeId(2), 7, Bytes::new(), t).await.is_err());
        let b2 = Rpc::bind(RpcConfig {
            node: NodeId(2),
            cluster: "test".into(),
            secret: b"s".to_vec(),
            listen: addr,
            connect_timeout: t,
        })
        .await
        .unwrap();
        b2.register(
            7,
            Arc::new(|_p: NodeId, _b: Bytes| async { Ok(Bytes::from_static(b"back")) }.boxed()),
        );
        assert_eq!(
            &a.call(NodeId(2), 7, Bytes::new(), t).await.unwrap()[..],
            b"back"
        );
    }

    /// ADR-047: a host behind NAT dials out; the host it dialled has no
    /// address for it and sends its own requests back over that connection.
    #[tokio::test]
    async fn a_peer_behind_nat_is_reached_over_the_connection_it_opened() {
        let core = node(1, b"s").await;
        let offgrid = node(2, b"s").await;
        let t = Duration::from_secs(2);
        // No address for node 2 on the core side, and none learned later.
        assert!(matches!(
            core.call(NodeId(2), 7, Bytes::new(), t).await,
            Err(RpcError::Unreachable(..))
        ));
        offgrid.pin_peer(NodeId(1), core.local_addr());
        // Membership later names node 1 by a fabric address: ignored.
        offgrid.set_peer(NodeId(1), "10.255.255.1:9".parse().unwrap());
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while !core.connected(NodeId(2)) {
            assert!(
                std::time::Instant::now() < deadline,
                "node 2 never dialled in"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let r = core.call(NodeId(2), 7, Bytes::from_static(b"hi"), t).await;
        assert_eq!(&r.unwrap()[..], b"2<-1:hi");
        let r = offgrid
            .call(NodeId(1), 7, Bytes::from_static(b"yo"), t)
            .await;
        assert_eq!(&r.unwrap()[..], b"1<-2:yo");
        // Both directions at once, many in flight.
        let ups: Vec<_> = (0..100)
            .map(|i| core.call(NodeId(2), 7, Bytes::from(format!("{i}")), t))
            .collect();
        let downs: Vec<_> = (0..100)
            .map(|i| offgrid.call(NodeId(1), 7, Bytes::from(format!("{i}")), t))
            .collect();
        let (ups, downs) = futures::join!(
            futures::future::join_all(ups),
            futures::future::join_all(downs)
        );
        for (i, (u, d)) in ups.into_iter().zip(downs).enumerate() {
            assert_eq!(u.unwrap(), Bytes::from(format!("2<-1:{i}")));
            assert_eq!(d.unwrap(), Bytes::from(format!("1<-2:{i}")));
        }

        // The core host restarts: the off-grid host dials in again and the
        // core reaches it as before.
        let addr = core.local_addr();
        core.shutdown();
        drop(core);
        tokio::time::sleep(Duration::from_millis(100)).await;
        let core = Rpc::bind(RpcConfig {
            node: NodeId(1),
            cluster: "test".into(),
            secret: b"s".to_vec(),
            listen: addr,
            connect_timeout: t,
        })
        .await
        .unwrap();
        core.register(
            7,
            Arc::new(|p: NodeId, b: Bytes| {
                async move {
                    let mut v = format!("1'<-{p}:").into_bytes();
                    v.extend_from_slice(&b);
                    Ok(Bytes::from(v))
                }
                .boxed()
            }),
        );
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while !core.connected(NodeId(2)) {
            assert!(
                std::time::Instant::now() < deadline,
                "node 2 did not reconnect"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let r = core
            .call(NodeId(2), 7, Bytes::from_static(b"again"), t)
            .await;
        assert_eq!(&r.unwrap()[..], b"2<-1:again");
        let r = offgrid
            .call(NodeId(1), 7, Bytes::from_static(b"x"), t)
            .await;
        assert_eq!(&r.unwrap()[..], b"1'<-2:x");
    }
}
