//! Moving leadership on purpose (ADR-057): a stopping leader hands off to
//! its most caught-up voter instead of leaving the others to time out and
//! elect, and a leader hands off to the cluster's preferred leader once
//! that host has been reachable and caught up for a while. Every leader
//! change is logged, and the Raft network's round trips are exposed per
//! peer.

use super::MetaNode;
use crate::network::{self, Features};
use crate::stats::RaftStats;
use nest_types::{NodeId, PeerRaftStats, Setting};
use openraft::Instant as _;
use openraft::ServerState;
use openraft::async_runtime::WatchReceiver;
use parking_lot::{Mutex, RwLock};
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

/// Timing of leadership handoffs. Tests shorten them.
#[derive(Clone, Debug)]
pub struct HandoffConfig {
    /// The preferred leader must have been reachable and caught up this
    /// long without a break before the leader hands over to it.
    pub settle: Duration,
    /// No handoff to the preferred leader within this long of this host's
    /// last handoff, or of the preferred host last taking over, so a
    /// flapping host never bounces leadership.
    pub cooldown: Duration,
    /// Caught up: at most this many entries behind the leader's log.
    pub max_lag: u64,
    /// Reachable: acknowledged the leader within this long.
    pub max_ack_age: Duration,
    /// How often the leader looks at the preferred leader.
    pub check_every: Duration,
    /// How long a stopping leader waits for its successor to take over.
    /// Shutdown never waits more than this plus half a second.
    pub shutdown_wait: Duration,
    /// How long to wait for a peer to say whether it can take over.
    pub probe_timeout: Duration,
}

impl Default for HandoffConfig {
    fn default() -> Self {
        HandoffConfig {
            settle: Duration::from_secs(120),
            cooldown: Duration::from_secs(600),
            max_lag: 256,
            max_ack_age: Duration::from_secs(1),
            check_every: Duration::from_secs(1),
            shutdown_wait: Duration::from_secs(2),
            probe_timeout: Duration::from_millis(300),
        }
    }
}

/// Whether a peer's link is healthy enough for it to lead (the daemon
/// plugs in the fabric path state; default: every peer is).
pub type PeerHealth = Arc<dyn Fn(NodeId) -> bool + Send + Sync>;

pub(super) struct State {
    pub(super) stats: Arc<RaftStats>,
    pub(super) legacy: Arc<AtomicBool>,
    healthy: RwLock<PeerHealth>,
    /// When each host was last seen to take over as leader (not counting
    /// the leader this host found when it started).
    became_leader: Mutex<HashMap<u64, Instant>>,
    /// When this host last handed leadership to the preferred leader.
    last_handoff: Mutex<Option<Instant>>,
    stopping: AtomicBool,
}

impl State {
    pub(super) fn new() -> State {
        State {
            stats: Arc::new(RaftStats::default()),
            legacy: Arc::new(AtomicBool::new(false)),
            healthy: RwLock::new(Arc::new(|_| true)),
            became_leader: Mutex::new(HashMap::new()),
            last_handoff: Mutex::new(None),
            stopping: AtomicBool::new(false),
        }
    }

    fn healthy(&self, peer: u64) -> bool {
        let f = self.healthy.read().clone();
        f(NodeId(peer))
    }
}

/// Log every leader change and remember when each host took over.
pub(super) fn watch_leader(node: &Arc<MetaNode>) {
    let mut m = node.raft.metrics();
    let lead = node.lead.clone();
    tokio::spawn(async move {
        // The leader and term last seen.
        let mut known: Option<(u64, u64)> = None;
        loop {
            let (leader, term) = {
                let b = m.borrow_and_update();
                (b.current_leader, b.current_term)
            };
            if let Some(l) = leader
                && known.is_none_or(|(k, t)| k != l || t != term)
            {
                match known {
                    None => tracing::info!(leader = l, term, "metadata leader is {l}"),
                    Some((old, _)) => {
                        lead.became_leader.lock().insert(l, Instant::now());
                        tracing::warn!(
                            from = old,
                            to = l,
                            term,
                            "metadata leader changed: {old} -> {l} (term {term})"
                        );
                    }
                }
                known = Some((l, term));
            }
            if m.changed().await.is_err() {
                return;
            }
        }
    });
}

/// On the leader: hand over to the preferred leader once it has settled.
pub(super) fn watch_preferred(node: &Arc<MetaNode>) {
    let weak: Weak<MetaNode> = Arc::downgrade(node);
    let cfg = node.cfg.handoff.clone();
    tokio::spawn(async move {
        // The preferred host and since when it has looked ready.
        let mut ready_since: Option<(u64, Instant)> = None;
        let mut refused: Option<(u64, Instant)> = None;
        loop {
            tokio::time::sleep(cfg.check_every).await;
            let Some(n) = weak.upgrade() else { return };
            if n.lead.stopping.load(Ordering::Relaxed) {
                return;
            }
            if n.metrics().state != ServerState::Leader {
                ready_since = None;
                continue;
            }
            let Some(p) = n.preferred_leader().await else {
                ready_since = None;
                continue;
            };
            if !n.ready_to_lead(p) {
                if ready_since.is_some() {
                    tracing::debug!(preferred = p, "preferred leader not ready; settle restarts");
                }
                ready_since = None;
                continue;
            }
            let since = match ready_since {
                Some((q, t)) if q == p => t,
                _ => {
                    let now = Instant::now();
                    ready_since = Some((p, now));
                    now
                }
            };
            if since.elapsed() < cfg.settle || !n.cooled_down(p) {
                continue;
            }
            if refused.is_some_and(|(q, t)| q == p && t.elapsed() < cfg.cooldown) {
                continue;
            }
            if !n.can_take_over(p).await {
                tracing::info!(
                    preferred = p,
                    "preferred leader {p} cannot take leadership over (an older release?); \
                     staying leader"
                );
                refused = Some((p, Instant::now()));
                continue;
            }
            *n.lead.last_handoff.lock() = Some(Instant::now());
            n.transfer_to(p, "preferred leader", cfg.shutdown_wait)
                .await;
            ready_since = None;
        }
    });
}

impl MetaNode {
    /// Plug in which peers are healthy enough to lead (ADR-057: the
    /// preferred leader is skipped while its fabric link is degraded).
    pub fn set_peer_health(&self, f: PeerHealth) {
        *self.lead.healthy.write() = f;
    }

    /// Answer the Raft requests 0.4.4 does not know (leadership transfer,
    /// the feature probe) as 0.4.4 does: with a decode error. For tests of
    /// mixed-version clusters.
    #[doc(hidden)]
    pub fn emulate_0_4_4_raft(&self, on: bool) {
        self.lead.legacy.store(on, Ordering::Relaxed);
    }

    /// Every Raft peer with the round trips this host's Raft network
    /// measured to it; replication progress too when this host leads.
    pub fn raft_peers(&self) -> Vec<PeerRaftStats> {
        let m = self.metrics();
        let me = self.cfg.node.0;
        let mem = m.membership_config.membership();
        let voters: BTreeSet<u64> = mem.voter_ids().collect();
        let mut ids: BTreeSet<u64> = mem.nodes().map(|(id, _)| *id).collect();
        ids.extend(self.lead.stats.peers());
        ids.remove(&me);
        let last = m.last_log_index.unwrap_or(0);
        ids.into_iter()
            .map(|p| {
                let [heartbeat, append, vote, snapshot, transfer] =
                    self.lead.stats.peer(p).unwrap_or_default();
                let matched = m
                    .replication
                    .as_ref()
                    .and_then(|r| r.get(&p))
                    .map(|l| l.as_ref().map(|l| l.index()).unwrap_or(0));
                let last_ack_ms = m
                    .heartbeat
                    .as_ref()
                    .and_then(|h| h.get(&p).cloned().flatten())
                    .map(|t| t.into_inner().elapsed().as_millis() as u64);
                PeerRaftStats {
                    peer: NodeId(p),
                    voter: voters.contains(&p),
                    heartbeat,
                    append,
                    vote,
                    snapshot,
                    transfer,
                    matched,
                    lag: matched.map(|x| last.saturating_sub(x)),
                    last_ack_ms,
                }
            })
            .collect()
    }

    /// The `preferred-leader` setting from the local replica; `None` when
    /// unset.
    async fn preferred_leader(&self) -> Option<u64> {
        let path = self.meta_path.clone();
        let rows = tokio::task::spawn_blocking(move || {
            nest_meta::open_read(&path).and_then(|c| nest_meta::query::settings(&c))
        })
        .await
        .ok()?
        .ok()?;
        rows.into_iter()
            .find(|(k, _)| k == Setting::PreferredLeader.name())
            .map(|(_, v)| v)
            .filter(|v| *v != 0)
    }

    /// Whether `p` may take over from this leader now: another voter,
    /// healthy, acknowledging us and caught up.
    fn ready_to_lead(&self, p: u64) -> bool {
        let cfg = &self.cfg.handoff;
        let m = self.metrics();
        if m.state != ServerState::Leader || p == m.id {
            return false;
        }
        if !m.membership_config.membership().voter_ids().any(|v| v == p) {
            return false;
        }
        if !self.lead.healthy(p) {
            return false;
        }
        let acked = m
            .heartbeat
            .as_ref()
            .and_then(|h| h.get(&p).cloned().flatten())
            .is_some_and(|t| t.into_inner().elapsed() <= cfg.max_ack_age);
        let matched = m
            .replication
            .as_ref()
            .and_then(|r| r.get(&p).cloned().flatten())
            .map(|l| l.index());
        let last = m.last_log_index.unwrap_or(0);
        acked && matched.is_some_and(|x| x + cfg.max_lag >= last)
    }

    fn cooled_down(&self, p: u64) -> bool {
        let cool = self.cfg.handoff.cooldown;
        let ours = self
            .lead
            .last_handoff
            .lock()
            .is_none_or(|t| t.elapsed() >= cool);
        let theirs = self
            .lead
            .became_leader
            .lock()
            .get(&p)
            .is_none_or(|t| t.elapsed() >= cool);
        ours && theirs
    }

    async fn can_take_over(&self, p: u64) -> bool {
        network::features(
            &self.rpc,
            self.incarnation.clone(),
            NodeId(p),
            self.cfg.handoff.probe_timeout,
        )
        .await
        .is_some_and(|f: Features| f.transfer_leader)
    }

    /// Hand leadership to `to` and wait up to `wait` for another host to
    /// lead. Returns whether one does.
    async fn transfer_to(&self, to: u64, why: &str, wait: Duration) -> bool {
        let me = self.cfg.node.0;
        let term = self.metrics().current_term;
        let t = Instant::now();
        if let Err(e) = self.raft.trigger().transfer_leader(to).await {
            tracing::warn!(to, error = %e, "could not hand leadership over ({why})");
            return false;
        }
        let r = self
            .raft
            .wait(Some(wait))
            .metrics(
                |m| m.current_leader.is_some_and(|l| l != me),
                "leadership handoff",
            )
            .await;
        match r {
            Ok(m) => {
                tracing::info!(
                    to,
                    leader = ?m.current_leader,
                    from_term = term,
                    term = m.current_term,
                    took_ms = t.elapsed().as_millis() as u64,
                    "handed metadata leadership to {} ({why})",
                    m.current_leader.unwrap_or(to)
                );
                true
            }
            Err(_) => {
                tracing::warn!(
                    to,
                    waited_ms = wait.as_millis() as u64,
                    "{to} did not take metadata leadership over ({why}); the others elect a leader"
                );
                false
            }
        }
    }

    /// A stopping leader hands off to the most caught-up voter that can
    /// take over, so the others do not wait out an election timeout.
    pub(super) async fn hand_off_on_shutdown(&self) {
        let m = self.metrics();
        let me = self.cfg.node.0;
        if m.state != ServerState::Leader {
            return;
        }
        let voters: Vec<u64> = m
            .membership_config
            .membership()
            .voter_ids()
            .filter(|v| *v != me)
            .collect();
        if voters.is_empty() {
            return;
        }
        let probes = voters.iter().map(|&v| async move {
            let ok = self.can_take_over(v).await;
            (v, ok)
        });
        let (able, unable): (Vec<_>, Vec<_>) = futures::future::join_all(probes)
            .await
            .into_iter()
            .partition(|(_, ok)| *ok);
        if !unable.is_empty() {
            let skipped: Vec<u64> = unable.into_iter().map(|(v, _)| v).collect();
            tracing::info!(
                ?skipped,
                "not handing leadership to {skipped:?}: no answer, or an older release"
            );
        }
        let preferred = self.preferred_leader().await;
        let m = self.metrics();
        let last = m.last_log_index.unwrap_or(0);
        let matched = |v: u64| {
            m.replication
                .as_ref()
                .and_then(|r| r.get(&v).cloned().flatten())
                .map(|l| l.index())
        };
        // Healthy first, then fully caught up, then the preferred leader,
        // then the furthest along.
        let best = able.into_iter().map(|(v, _)| v).max_by_key(|&v| {
            let x = matched(v);
            (
                self.lead.healthy(v),
                x.is_some_and(|x| x >= last),
                preferred == Some(v),
                x,
            )
        });
        let Some(to) = best else {
            tracing::info!(
                "stopping as metadata leader with no voter able to take over; the others elect one"
            );
            return;
        };
        self.transfer_to(to, "this host is stopping", self.cfg.handoff.shutdown_wait)
            .await;
    }

    pub async fn shutdown(&self) {
        self.lead.stopping.store(true, Ordering::Relaxed);
        let budget = self.cfg.handoff.shutdown_wait + Duration::from_millis(500);
        if tokio::time::timeout(budget, self.hand_off_on_shutdown())
            .await
            .is_err()
        {
            tracing::info!("leadership handoff took too long; stopping anyway");
        }
        let _ = self.raft.shutdown().await;
    }
}
