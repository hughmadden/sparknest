//! The standby path (ADR-057): a second control connection per fabric
//! peer over the 10G LAN, idle until the fabric connection fails.
//!
//! **Who dials.** Of each pair the lower node id dials the standby and
//! redials it when it drops; the higher one uses the connection it
//! accepts. Each pair therefore has exactly one standby connection, and
//! either end sends its requests on it when failed over. The dialler's
//! first request on it is a standby hello (`SVC_STANDBY`): the accepting
//! end learns from it that this connection is the dialler's standby, not
//! its primary. The auth handshake is unchanged, so 0.4.4 peers keep
//! working on the primary connection: they answer the hello `NoService`
//! and the dialler closes that connection and tries again only after a
//! minute. Telling the paths apart by the local listener would not work:
//! fabric hosts listen on `0.0.0.0`, so one listener accepts both.
//!
//! **When requests move to the standby.** Only when the primary fails,
//! and only while the standby answers a probe:
//! - the primary cannot be dialled (refused, reset, unreachable, connect
//!   timeout), or
//! - a request on the primary has been waiting `silence` (1 s) and nothing
//!   at all has arrived from the peer on that connection since it was sent,
//!   not even the answer to a probe sent halfway through. The probe is
//!   answered by the peer's connection reader, never by a handler, so a
//!   slow handler (its answer late but its connection alive) never counts:
//!   slowness alone never switches.
//!
//! The primary is then closed: requests in flight on it fail `Unreachable`
//! and are not resent here (openraft resends its own; ADR-056, no hedging).
//!
//! **Coming back.** While failed over the primary is redialled every
//! `redial_every` (2 s) and probed every `ping_every`; after it has been
//! connected and answering for `return_after` (10 s) without a break,
//! requests return to it. Requests already in flight on the standby finish
//! there. If the standby itself drops meanwhile, requests return to the
//! primary at once.
//!
//! **The LAN preferred** (`Rpc::prefer_standby`, the cluster setting
//! `control-on-lan`, ADR-068). Requests move to the
//! standby once it has answered for `return_after`, the primary left open
//! as the fallback; there they are watched for silence as the primary is
//! (a request waiting `silence` with nothing arriving), and moved back to
//! the primary, the LAN connection closed and dialled again. The primary
//! failing changes nothing then.

use crate::RpcError;
use crate::conn::{Conn, Rpc, SVC_PING};
use bytes::Bytes;
use nest_types::NodeId;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

/// When a peer's requests move to its standby connection and back. The
/// defaults are ADR-057's; tests shorten them.
#[derive(Clone, Debug)]
pub struct StandbyConfig {
    /// How long a request may wait on the primary with nothing at all
    /// arriving there before the primary counts as gone.
    pub silence: Duration,
    /// How often each end probes the standby (and, while failed over, the
    /// primary).
    pub ping_every: Duration,
    /// A path counts as answering while its last probe answer is this
    /// recent.
    pub ping_fresh: Duration,
    /// While failed over, how often the primary is dialled again.
    pub redial_every: Duration,
    /// How long the primary must stay connected and answering before
    /// requests return to it.
    pub return_after: Duration,
}

impl Default for StandbyConfig {
    fn default() -> Self {
        StandbyConfig {
            silence: Duration::from_secs(1),
            ping_every: Duration::from_millis(500),
            ping_fresh: Duration::from_millis(1500),
            redial_every: Duration::from_secs(2),
            return_after: Duration::from_secs(10),
        }
    }
}

/// Why requests went on the standby when the LAN is preferred.
const PREFERRED: &str = "the LAN is preferred and answering";

/// One peer's paths, for diagnostics (`nest net`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathState {
    /// Requests to the peer use the standby connection.
    pub on_standby: bool,
    /// When requests last moved (or when the standby was configured).
    pub since: SystemTime,
    /// Moves to the standby, and back, since this process started.
    pub failovers: u64,
    pub returns: u64,
    /// The standby connection is open and answered its last probe.
    pub standby_up: bool,
    /// Round trip of the last standby probe.
    pub standby_rtt: Option<Duration>,
    /// The standby address, when this host was given one.
    pub standby_addr: Option<SocketAddr>,
    /// Why requests last moved.
    pub reason: Option<String>,
}

#[derive(Default)]
pub(crate) struct Paths {
    peers: Mutex<HashMap<NodeId, Peer>>,
    /// Requests go on the standby (LAN) while it answers, the primary
    /// (fabric) becoming the fallback (ADR-068): the control connection
    /// then never queues behind the fabric's bulk traffic.
    pub(crate) prefer_standby: std::sync::atomic::AtomicBool,
}

struct Peer {
    /// Where to dial (only the lower id of a pair dials).
    addr: Option<SocketAddr>,
    conn: Option<Arc<Conn>>,
    /// Standby probes: last answer, its round trip, one in flight.
    answered: Option<Instant>,
    rtt: Option<Duration>,
    probing: bool,
    /// Standby dialling: in progress, and not again before.
    dialling: bool,
    next_dial: Instant,
    dial_wait: Duration,
    on_standby: bool,
    /// A failover is being decided (its standby probe is out).
    switching: bool,
    since: SystemTime,
    failovers: u64,
    returns: u64,
    reason: Option<String>,
    /// While failed over: the primary's redial and probes.
    redialled: Option<Instant>,
    primary_answered: Option<Instant>,
    primary_probing: bool,
    primary_ok_since: Option<Instant>,
    /// Preferred standby: since when it has answered without a break.
    standby_ok_since: Option<Instant>,
}

impl Peer {
    fn new(addr: Option<SocketAddr>) -> Peer {
        Peer {
            addr,
            conn: None,
            answered: None,
            rtt: None,
            probing: false,
            dialling: false,
            next_dial: Instant::now(),
            dial_wait: Duration::ZERO,
            on_standby: false,
            switching: false,
            since: SystemTime::now(),
            failovers: 0,
            returns: 0,
            reason: None,
            redialled: None,
            primary_answered: None,
            primary_probing: false,
            primary_ok_since: None,
            standby_ok_since: None,
        }
    }

    fn live(&self) -> Option<Arc<Conn>> {
        self.conn.clone().filter(|c| c.alive())
    }

    fn answering(&self, fresh: Duration) -> bool {
        self.live().is_some() && self.answered.is_some_and(|t| t.elapsed() < fresh)
    }

    fn return_to_primary(&mut self, peer: NodeId, why: &str) {
        self.on_standby = false;
        self.standby_ok_since = None;
        self.since = SystemTime::now();
        self.returns += 1;
        self.reason = Some(why.to_string());
        self.redialled = None;
        self.primary_answered = None;
        self.primary_ok_since = None;
        tracing::warn!(%peer, reason = why, "control path on the fabric");
    }
}

impl Paths {
    pub(crate) fn set_standby(&self, peer: NodeId, addr: SocketAddr) {
        let mut g = self.peers.lock();
        let p = g.entry(peer).or_insert_with(|| Peer::new(Some(addr)));
        if p.addr != Some(addr) {
            p.addr = Some(addr);
            p.next_dial = Instant::now();
            p.dial_wait = Duration::ZERO;
        }
    }

    /// The peer dialled its standby to us: our standby to it.
    pub(crate) fn accepted_standby(&self, peer: NodeId, conn: Arc<Conn>) {
        let mut g = self.peers.lock();
        let p = g.entry(peer).or_insert_with(|| Peer::new(None));
        p.conn = Some(conn);
        p.answered = Some(Instant::now());
        tracing::info!(%peer, "standby control connection accepted");
    }

    /// Whether `peer` has a standby configured or accepted.
    pub(crate) fn has(&self, peer: NodeId) -> bool {
        self.peers.lock().contains_key(&peer)
    }

    pub(crate) fn standby_conn(&self, peer: NodeId) -> Option<Arc<Conn>> {
        self.peers.lock().get(&peer)?.live()
    }

    /// The standby connection while `peer` is failed over to it. A standby
    /// that dropped meanwhile sends requests back to the primary.
    pub(crate) fn active_standby(&self, peer: NodeId) -> Option<Arc<Conn>> {
        let mut g = self.peers.lock();
        let p = g.get_mut(&peer)?;
        if !p.on_standby {
            return None;
        }
        match p.live() {
            Some(c) => Some(c),
            None => {
                p.return_to_primary(peer, "standby connection lost");
                None
            }
        }
    }

    /// Move `peer`'s requests to its standby, if it answers a probe now;
    /// `primary` (if any) is closed, failing what is in flight on it.
    /// Returns the standby connection when requests now go there.
    pub(crate) async fn fail_over(
        &self,
        rpc: &Rpc,
        peer: NodeId,
        why: &str,
        primary: Option<Arc<Conn>>,
    ) -> Option<Arc<Conn>> {
        let conn = {
            let mut g = self.peers.lock();
            let p = g.get_mut(&peer)?;
            if p.on_standby {
                return p.live();
            }
            if p.switching {
                return None;
            }
            let c = p.live()?;
            p.switching = true;
            c
        };
        // Alive and answering now: a host that is restarting closes both
        // connections, and its standby must not take requests it cannot.
        let budget = rpc.inner.standby.silence / 2;
        let ok = answered(tokio::time::timeout(budget, conn.call(SVC_PING, Bytes::new())).await)
            && conn.alive();
        let mut g = self.peers.lock();
        let p = g.get_mut(&peer)?;
        p.switching = false;
        if !ok {
            return None;
        }
        if !p.on_standby {
            p.on_standby = true;
            p.since = SystemTime::now();
            p.failovers += 1;
            p.reason = Some(why.to_string());
            p.answered = Some(Instant::now());
            p.redialled = Some(Instant::now());
            p.primary_answered = None;
            p.primary_ok_since = None;
            tracing::warn!(%peer, reason = why, "control path failed over to the standby");
        }
        drop(g);
        if let Some(c) = primary {
            c.close("primary path failed; switched to the standby");
        }
        Some(conn)
    }

    pub(crate) fn state(&self, peer: NodeId, fresh: Duration) -> Option<PathState> {
        let g = self.peers.lock();
        g.get(&peer).map(|p| snapshot(p, fresh))
    }

    pub(crate) fn all(&self, fresh: Duration) -> Vec<(NodeId, PathState)> {
        let g = self.peers.lock();
        let mut v: Vec<_> = g.iter().map(|(n, p)| (*n, snapshot(p, fresh))).collect();
        v.sort_by_key(|(n, _)| *n);
        v
    }

    pub(crate) fn clear(&self) {
        for (_, p) in self.peers.lock().drain() {
            if let Some(c) = p.conn {
                c.close("shutting down");
            }
        }
    }
}

fn snapshot(p: &Peer, fresh: Duration) -> PathState {
    PathState {
        on_standby: p.on_standby,
        since: p.since,
        failovers: p.failovers,
        returns: p.returns,
        standby_up: p.answering(fresh),
        standby_rtt: p.rtt,
        standby_addr: p.addr,
        reason: p.reason.clone(),
    }
}

/// A probe's outcome: any answer at all, `NoService` included (a peer that
/// predates probes still answers from its connection's reader).
fn answered(r: Result<Result<Bytes, RpcError>, tokio::time::error::Elapsed>) -> bool {
    matches!(r, Ok(Ok(_)) | Ok(Err(RpcError::NoService(_))))
}

/// Keep the standby connections up and probed, watch the primaries for
/// silence, and bring failed-over peers back (see the module docs).
pub(crate) fn watch(rpc: &Rpc) {
    let weak = Arc::downgrade(&rpc.inner);
    let mut stop = rpc.inner.shutdown.subscribe();
    let tick = (rpc.inner.standby.silence / 10)
        .clamp(Duration::from_millis(5), Duration::from_millis(100));
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = tokio::time::sleep(tick) => {}
                _ = stop.changed() => return,
            }
            let Some(inner) = weak.upgrade() else { return };
            let rpc = Rpc { inner };
            let peers: Vec<NodeId> = rpc.inner.paths.peers.lock().keys().copied().collect();
            for peer in peers {
                step(&rpc, peer);
            }
        }
    });
}

fn step(rpc: &Rpc, peer: NodeId) {
    let cfg = &rpc.inner.standby;
    let me = rpc.node();
    let now = Instant::now();
    let mut primary = rpc.primary_now(peer);
    if rpc.inner.filter.is_cut(peer)
        && let Some(c) = primary.take()
    {
        c.close("primary path cut by filter");
    }
    let prefer = rpc
        .inner
        .paths
        .prefer_standby
        .load(std::sync::atomic::Ordering::Relaxed);
    let mut g = rpc.inner.paths.peers.lock();
    let Some(p) = g.get_mut(&peer) else { return };

    // The standby connection: dialled by the lower id, probed by both.
    match p.live() {
        None => {
            p.answered = None;
            if me < peer
                && let Some(addr) = p.addr
                && !p.dialling
                && now >= p.next_dial
            {
                p.dialling = true;
                let rpc = rpc.clone();
                tokio::spawn(async move { dial(rpc, peer, addr).await });
            }
        }
        Some(c) => {
            let due = p.answered.is_none_or(|t| t.elapsed() >= cfg.ping_every);
            if due && !p.probing {
                p.probing = true;
                let (rpc, fresh) = (rpc.clone(), cfg.ping_fresh);
                tokio::spawn(async move {
                    let t = Instant::now();
                    let ok =
                        answered(tokio::time::timeout(fresh, c.call(SVC_PING, Bytes::new())).await);
                    let mut g = rpc.inner.paths.peers.lock();
                    if let Some(p) = g.get_mut(&peer) {
                        p.probing = false;
                        if ok {
                            p.answered = Some(Instant::now());
                            p.rtt = Some(t.elapsed());
                            p.standby_ok_since.get_or_insert_with(Instant::now);
                        } else {
                            p.standby_ok_since = None;
                        }
                    }
                });
            }
        }
    }

    if p.live().is_none() {
        p.standby_ok_since = None;
    }
    if !p.on_standby {
        // The standby preferred: move onto it once it has answered for
        // `return_after` (the primary stays open, the fallback).
        if prefer
            && !p.switching
            && p.answering(cfg.ping_fresh)
            && p.standby_ok_since
                .is_some_and(|t| t.elapsed() >= cfg.return_after)
        {
            p.on_standby = true;
            p.since = SystemTime::now();
            p.reason = Some(PREFERRED.into());
            p.redialled = Some(now);
            p.primary_answered = None;
            p.primary_ok_since = None;
            tracing::info!(%peer, "control path on the LAN (preferred)");
            return;
        }
        // A request waiting with the peer silent: probe halfway through,
        // and fail over once the whole budget has passed unanswered.
        let Some(c) = primary else { return };
        let Some((quiet, probed)) = c.silence() else {
            return;
        };
        if quiet >= cfg.silence / 2 && !probed {
            c.probe();
        } else if quiet >= cfg.silence && probed && !p.switching && p.answering(cfg.ping_fresh) {
            drop(g);
            let rpc = rpc.clone();
            tokio::spawn(async move {
                let why = format!("no answer on the fabric for {} ms", quiet.as_millis());
                rpc.inner.paths.fail_over(&rpc, peer, &why, Some(c)).await;
            });
        }
        return;
    }

    // Failed over: redial the primary, probe it, and return once it has
    // been healthy long enough.
    let Some(lan) = p.live() else {
        p.return_to_primary(peer, "standby connection lost");
        return;
    };
    if !prefer && p.reason.as_deref() == Some(PREFERRED) {
        p.return_to_primary(peer, "the LAN is no longer preferred");
        return;
    }
    if prefer {
        // On the preferred LAN: a request waiting there with nothing at all
        // arriving falls back to the fabric (as the fabric falls over to
        // the LAN), the stuck connection closed; the LAN is taken again
        // once it has answered for `return_after`.
        if let Some((quiet, probed)) = lan.silence() {
            if quiet >= cfg.silence / 2 && !probed {
                lan.probe();
            } else if quiet >= cfg.silence && probed {
                let why = format!("no answer on the LAN for {} ms", quiet.as_millis());
                p.return_to_primary(peer, &why);
                drop(g);
                lan.close("LAN path silent; requests back on the fabric");
                return;
            }
        }
        // Keep the fabric connected for a fallback.
        if primary.is_none() && p.redialled.is_none_or(|t| t.elapsed() >= cfg.redial_every) {
            p.redialled = Some(now);
            let rpc = rpc.clone();
            tokio::spawn(async move {
                if let Err(e) = rpc.conn(peer).await {
                    tracing::debug!(%peer, error = %e, "fabric unreachable (LAN in use)");
                }
            });
        }
        return;
    }
    match primary {
        None => {
            p.primary_ok_since = None;
            p.primary_answered = None;
            if p.redialled.is_none_or(|t| t.elapsed() >= cfg.redial_every) {
                p.redialled = Some(now);
                let rpc = rpc.clone();
                tokio::spawn(async move {
                    if let Err(e) = rpc.conn(peer).await {
                        tracing::debug!(%peer, error = %e, "fabric still unreachable");
                    }
                });
            }
        }
        Some(c) => {
            let healthy = p
                .primary_answered
                .is_some_and(|t| t.elapsed() < cfg.ping_fresh);
            if healthy {
                let since = *p.primary_ok_since.get_or_insert(now);
                if since.elapsed() >= cfg.return_after {
                    let why = format!(
                        "fabric connected and answering for {} s",
                        cfg.return_after.as_secs_f64()
                    );
                    p.return_to_primary(peer, &why);
                    return;
                }
            } else if p.primary_answered.is_some() {
                // It stopped answering: start over.
                p.primary_ok_since = None;
            }
            let due = p
                .primary_answered
                .is_none_or(|t| t.elapsed() >= cfg.ping_every);
            if due && !p.primary_probing {
                p.primary_probing = true;
                let (rpc, fresh) = (rpc.clone(), cfg.ping_fresh);
                tokio::spawn(async move {
                    let ok =
                        answered(tokio::time::timeout(fresh, c.call(SVC_PING, Bytes::new())).await);
                    let mut g = rpc.inner.paths.peers.lock();
                    if let Some(p) = g.get_mut(&peer) {
                        p.primary_probing = false;
                        if ok {
                            p.primary_answered = Some(Instant::now());
                        } else {
                            p.primary_answered = None;
                            p.primary_ok_since = None;
                        }
                    }
                });
            }
        }
    }
}

/// Dial `peer`'s standby, backing off while it fails (a minute for a peer
/// that predates standbys).
async fn dial(rpc: Rpc, peer: NodeId, addr: SocketAddr) {
    let r = rpc.open_standby(peer, addr).await;
    let mut g = rpc.inner.paths.peers.lock();
    let Some(p) = g.get_mut(&peer) else { return };
    p.dialling = false;
    match r {
        Ok(c) => {
            p.conn = Some(c);
            p.answered = Some(Instant::now());
            p.dial_wait = Duration::ZERO;
            tracing::info!(%peer, %addr, "standby control connection up");
        }
        Err(e) => {
            let old =
                matches!(&e, RpcError::Unreachable(_, why) if why.contains("no standby support"));
            p.dial_wait = if old {
                Duration::from_secs(60)
            } else {
                (p.dial_wait * 2)
                    .max(rpc.inner.standby.ping_every)
                    .min(Duration::from_secs(30))
            };
            p.next_dial = Instant::now() + p.dial_wait;
            tracing::debug!(%peer, %addr, error = %e, retry_in = ?p.dial_wait, "standby dial failed");
        }
    }
}
