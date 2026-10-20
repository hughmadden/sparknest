//! Push loads (ADR-063), the sparknestd side.
//!
//! Three roles, any of which a host may play in one group:
//! - **hub**: the coordinator's host. Takes the plan on the engine's
//!   socket, resolves and checks it, gives each file to one copy, tells
//!   every rank host which sources to expect and every source its files,
//!   then reports progress and the end.
//! - **source**: a host whose store holds assigned files. Reads each file
//!   once, in offset order, packs every run into the destination rank's
//!   open slot, and sends a slot whenever it fills and a credit allows.
//! - **rank host**: where a rank's engine joined with its ring. Hands out a
//!   credit per free ring slot, round robin over the sources owing it;
//!   writes each arriving slot into the ring slot its credit named, checks
//!   it, and tells the engine.
//!
//! Slots travel between hosts as `PushMsg::Slot` over the control RPC (a
//! direct call on the same host). RDMA writes straight into the ring
//! replace that where both ends have a fabric link.

use crate::vfs::Vfs;
use futures::FutureExt;
use futures::future::BoxFuture;
use nest_push::client::Conn;
use nest_push::plan::{Entry, Pattern, Plan, RunWalk};
use nest_push::proto::{Event, Forecast, Missing, RankSpec, Request, StreamForecast};
use nest_push::slot;
use nest_types::{FileId, Generation, NestError, NestResult, NodeId};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Bytes read from a file at once.
const READ_CHUNK: u64 = 16 << 20;
/// Gaps between a file's runs shorter than this are read through.
const GAP: u64 = 1 << 20;
/// A group, or a source waiting for a credit, with no progress this long
/// fails.
const STALL: Duration = Duration::from_secs(30);
/// How long `Start` waits for every rank to join.
const JOIN_WAIT: Duration = Duration::from_secs(60);
/// RPC deadline between hosts.
const RPC: Duration = Duration::from_secs(30);
/// Rank hosts report delivered bytes to the hub this often (bytes).
const REPORT_EVERY: u64 = 256 << 20;

/// Between sparknestds, on `service::PUSH`. Append-only.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) enum PushMsg {
    /// Hub → rank host: rank `rank` of `group` will join here.
    Expect {
        group: u64,
        token: u64,
        hub: NodeId,
        rank: u32,
        arena: u64,
    },
    /// Rank host → hub: the rank joined; its ring's slots are this big.
    Joined {
        group: u64,
        rank: u32,
        slot_bytes: u32,
    },
    /// Hub → rank host: the sources that will send this rank slots.
    Sources {
        group: u64,
        rank: u32,
        sources: Vec<NodeId>,
    },
    /// Hub → source: stream these files.
    Stream {
        group: u64,
        hub: NodeId,
        patterns: Vec<Pattern>,
        files: Vec<SourceFile>,
        /// Per rank: its host and slot size.
        ranks: Vec<(NodeId, u32)>,
    },
    /// Rank host → source: `n` more slots of `rank` may be sent.
    Credit { group: u64, rank: u32, n: u32 },
    /// Source → rank host: a packed slot (`slot_bytes` long) for `rank`.
    Slot {
        group: u64,
        rank: u32,
        source: NodeId,
        #[serde(with = "serde_bytes")]
        bytes: Vec<u8>,
    },
    /// Source → hub: finished. `slots` sent per rank; files it could not
    /// read, by plan index.
    SourceDone {
        group: u64,
        source: NodeId,
        slots: Vec<u64>,
        failed: Vec<(u32, String)>,
    },
    /// Rank host → hub: payload bytes delivered so far (progress), and
    /// whether that rank has everything.
    Delivered {
        group: u64,
        rank: u32,
        bytes: u64,
        complete: bool,
    },
    /// Hub → rank host: everything for `rank` is `slots` slots in all.
    RankDone { group: u64, rank: u32, slots: u64 },
    /// Any → any: the group ended (cancelled or failed).
    End { group: u64, reason: String },
}

/// A file a source streams, with the entries that read it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct SourceFile {
    pub index: u32,
    pub file: FileId,
    pub generation: Generation,
    pub entries: Vec<Entry>,
}

fn invalid(m: impl Into<String>) -> NestError {
    NestError::Invalid(m.into())
}

// ------------------------------------------------------------------ state

/// What an engine's socket is told; its writer thread sends these in order.
type EngineTx = tokio::sync::mpsc::UnboundedSender<Event>;

/// A group as the hub holds it.
struct HubGroup {
    token: u64,
    ranks: Vec<RankSpec>,
    rank_hosts: Vec<NodeId>,
    files: Vec<String>,
    patterns: Vec<Pattern>,
    entries: Vec<Entry>,
    engine: EngineTx,
    /// Rank → its ring's slot size, once joined.
    joined: HashMap<u32, u32>,
    joined_notify: Arc<tokio::sync::Notify>,
    /// Sources still streaming, and every source used.
    streaming: Vec<NodeId>,
    sources: Vec<NodeId>,
    /// Slots per rank, summed over the sources that finished.
    slots: Vec<u64>,
    failed: Vec<(u32, String)>,
    total: u64,
    delivered: Vec<u64>,
    complete: Vec<bool>,
    progress_at: tokio::time::Instant,
    started: bool,
}

/// A rank this host receives for.
struct RankSlot {
    hub: NodeId,
    token: u64,
    arena: u64,
    ring: Option<Arc<Ring>>,
    engine: Option<EngineTx>,
    filled: u64,
    expected: Option<u64>,
    bytes: u64,
    reported: u64,
    /// Ring slots no credit names.
    free: Vec<u32>,
    /// Per source, the ring slots its granted credits fill, in order.
    granted: HashMap<NodeId, VecDeque<u32>>,
    /// Sources owing this rank slots, granted to round robin.
    sources: Vec<NodeId>,
    next: usize,
    done_sent: bool,
}

/// A source stream's credits per rank, and its wake-up.
struct SourceState {
    credits: Vec<u32>,
    notify: Arc<tokio::sync::Notify>,
    cancelled: bool,
}

/// The engine's ring, mapped into this process: `slots` of `slot_bytes`.
struct Ring {
    map: nest_fabric::verbs::SharedMap,
    slots: u32,
    slot_bytes: u32,
}

impl Ring {
    fn map(fd: std::os::fd::OwnedFd, slots: u32, slot_bytes: u32) -> NestResult<Ring> {
        let len = (slots as usize)
            .checked_mul(slot_bytes as usize)
            .filter(|l| *l > 0 && slot_bytes as usize > slot::FOOTER + slot::FRAG)
            .ok_or_else(|| invalid("bad ring geometry"))?;
        let map = nest_fabric::verbs::SharedMap::map(fd, len)
            .map_err(|e| invalid(format!("mapping the ring: {e}")))?;
        Ok(Ring {
            map,
            slots,
            slot_bytes,
        })
    }

    /// Copy a received slot into ring slot `i`, which this host owns: a
    /// credit named it and it is not handed to the engine until checked.
    fn write_slot(&self, i: u32, bytes: &[u8]) -> NestResult<()> {
        if i >= self.slots || bytes.len() != self.slot_bytes as usize {
            return Err(invalid("slot does not fit the ring"));
        }
        self.map
            .write_owned(i as usize * self.slot_bytes as usize, bytes);
        Ok(())
    }

    /// Ring slot `i`, owned as for `write_slot`.
    fn slot(&self, i: u32) -> &[u8] {
        self.map.read_owned(
            i as usize * self.slot_bytes as usize,
            self.slot_bytes as usize,
        )
    }
}

#[derive(Default)]
struct State {
    hub: HashMap<u64, HubGroup>,
    ranks: HashMap<(u64, u32), RankSlot>,
    sources: HashMap<u64, SourceState>,
    /// Credits that came before their group's `Stream` (rank hosts start
    /// granting as soon as they learn their sources): (group, rank) → n.
    early: HashMap<(u64, u32), u32>,
}

/// The push service of one host.
pub struct Push {
    vfs: std::sync::Weak<Vfs>,
    state: Mutex<State>,
    next_group: AtomicU64,
}

/// A file resolved at start: its current sealed generation, pinned for the
/// group (a later write makes its source fail it as stale).
struct Resolved {
    file: FileId,
    generation: Generation,
    size: u64,
    /// Hosts with a live copy, this host first if it holds one.
    holders: Vec<NodeId>,
}

/// Give each file to one holder, balancing bytes per holder (largest files
/// first; each to the holder with the fewest bytes so far, this host's own
/// copy winning ties).
fn assign(files: &[Resolved], me: NodeId) -> Vec<NodeId> {
    let mut order: Vec<usize> = (0..files.len()).collect();
    order.sort_by_key(|i| std::cmp::Reverse(files[*i].size));
    let mut load: HashMap<NodeId, u64> = HashMap::new();
    let mut out = vec![me; files.len()];
    for i in order {
        let f = &files[i];
        let pick = *f
            .holders
            .iter()
            .min_by_key(|h| (load.get(h).copied().unwrap_or(0), **h != me))
            .expect("resolved files have holders");
        *load.entry(pick).or_default() += f.size;
        out[i] = pick;
    }
    out
}

impl Push {
    pub fn new(vfs: &Arc<Vfs>) -> Arc<Push> {
        let p = Arc::new(Push {
            vfs: Arc::downgrade(vfs),
            state: Mutex::new(State::default()),
            next_group: AtomicU64::new(rand::random::<u64>() >> 16 << 16),
        });
        vfs.rpc().register(
            nest_rpc::service::PUSH,
            Arc::new(PushService(Arc::downgrade(&p))),
        );
        let w = Arc::downgrade(&p);
        if let Ok(rt) = tokio::runtime::Handle::try_current() {
            rt.spawn(async move {
                loop {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    let Some(p) = w.upgrade() else { return };
                    p.sweep();
                }
            });
        }
        p
    }

    fn vfs(&self) -> NestResult<Arc<Vfs>> {
        self.vfs
            .upgrade()
            .ok_or_else(|| NestError::Unavailable("shutting down".into()))
    }

    fn me(&self) -> NestResult<NodeId> {
        Ok(self.vfs()?.me())
    }

    /// Groups this host takes part in, for status: (group, role).
    pub fn groups(&self) -> Vec<(u64, &'static str)> {
        let st = self.state.lock();
        let mut v: Vec<(u64, &'static str)> = st.hub.keys().map(|g| (*g, "hub")).collect();
        v.extend(st.sources.keys().map(|g| (*g, "source")));
        v.extend(st.ranks.keys().map(|(g, _)| (*g, "rank")));
        v.sort_unstable();
        v.dedup();
        v
    }

    // ----------------------------------------------------------- transport

    /// Deliver `m` to `to`: a direct call when it is this host.
    fn send(self: &Arc<Self>, to: NodeId, m: PushMsg) -> BoxFuture<'static, NestResult<()>> {
        let me = self.clone();
        async move {
            let here = me.me()?;
            if to == here {
                return me.on_msg(here, m).await;
            }
            let v = me.vfs()?;
            let r: Result<Result<(), String>, _> =
                nest_rpc::typed::call(v.rpc(), to, nest_rpc::service::PUSH, &m, RPC).await;
            match r {
                Ok(Ok(())) => Ok(()),
                Ok(Err(e)) => Err(NestError::Unavailable(e)),
                Err(e) => Err(NestError::Unavailable(e.to_string())),
            }
        }
        .boxed()
    }

    /// A message from another host, or this one.
    fn on_msg(self: &Arc<Self>, _from: NodeId, m: PushMsg) -> BoxFuture<'static, NestResult<()>> {
        let me = self.clone();
        async move {
            match m {
                PushMsg::Expect {
                    group,
                    token,
                    hub,
                    rank,
                    arena,
                } => me.expect(group, token, hub, rank, arena),
                PushMsg::Joined {
                    group,
                    rank,
                    slot_bytes,
                } => me.joined(group, rank, slot_bytes),
                PushMsg::Sources {
                    group,
                    rank,
                    sources,
                } => {
                    me.rank_sources(group, rank, sources);
                    Ok(())
                }
                PushMsg::Stream {
                    group,
                    hub,
                    patterns,
                    files,
                    ranks,
                } => {
                    me.stream(group, hub, patterns, files, ranks);
                    Ok(())
                }
                PushMsg::Credit { group, rank, n } => {
                    me.credit(group, rank, n);
                    Ok(())
                }
                PushMsg::Slot {
                    group,
                    rank,
                    source,
                    bytes,
                } => me.slot(group, rank, source, &bytes),
                PushMsg::SourceDone {
                    group,
                    source,
                    slots,
                    failed,
                } => {
                    me.source_done(group, source, slots, failed).await;
                    Ok(())
                }
                PushMsg::Delivered {
                    group,
                    rank,
                    bytes,
                    complete,
                } => {
                    me.delivered(group, rank, bytes, complete);
                    Ok(())
                }
                PushMsg::RankDone { group, rank, slots } => {
                    me.rank_done(group, rank, slots);
                    Ok(())
                }
                PushMsg::End { group, reason } => {
                    me.end_here(group, &reason);
                    Ok(())
                }
            }
        }
        .boxed()
    }

    // ---------------------------------------------------------------- hub

    fn create(
        &self,
        ranks: Vec<RankSpec>,
        files: Vec<String>,
        engine: EngineTx,
    ) -> NestResult<(u64, u64)> {
        if ranks.is_empty() {
            return Err(invalid("no ranks"));
        }
        if ranks.iter().enumerate().any(|(i, r)| r.rank as usize != i) {
            return Err(invalid("ranks must be numbered 0.. in order"));
        }
        let rank_hosts = {
            let c = self.vfs()?.cluster()?;
            ranks
                .iter()
                .map(|r| {
                    c["hosts"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .find(|h| h["name"] == r.host.as_str())
                        .and_then(|h| h["node"].as_u64())
                        .map(NodeId)
                        .ok_or_else(|| invalid(format!("no host named {}", r.host)))
                })
                .collect::<NestResult<Vec<_>>>()?
        };
        let group = self.next_group.fetch_add(1, Ordering::Relaxed);
        let token = rand::random::<u64>();
        let n = ranks.len();
        self.state.lock().hub.insert(
            group,
            HubGroup {
                token,
                ranks,
                rank_hosts,
                files,
                patterns: Vec::new(),
                entries: Vec::new(),
                engine,
                joined: HashMap::new(),
                joined_notify: Arc::new(tokio::sync::Notify::new()),
                streaming: Vec::new(),
                sources: Vec::new(),
                slots: vec![0; n],
                failed: Vec::new(),
                total: 0,
                delivered: vec![0; n],
                complete: vec![false; n],
                progress_at: tokio::time::Instant::now(),
                started: false,
            },
        );
        Ok((group, token))
    }

    fn add_plan(&self, group: u64, patterns: Vec<Pattern>, entries: Vec<Entry>) -> NestResult<()> {
        let mut st = self.state.lock();
        let g = st
            .hub
            .get_mut(&group)
            .ok_or_else(|| invalid("no such group"))?;
        if g.started {
            return Err(invalid("already started"));
        }
        g.patterns.extend(patterns);
        g.entries.extend(entries);
        Ok(())
    }

    fn resolve(&self, files: &[String]) -> NestResult<Vec<Resolved>> {
        let v = self.vfs()?;
        files
            .iter()
            .map(|p| {
                let id = v
                    .resolve_path(p)?
                    .ok_or_else(|| invalid(format!("{p}: not found")))?;
                let (a, _) = v.copies(id)?;
                if a.kind != nest_types::FileKind::Regular {
                    return Err(invalid(format!("{p}: not a regular file")));
                }
                if !a.sealed || a.gen_state != nest_types::GenState::Stable {
                    return Err(invalid(format!("{p}: not sealed")));
                }
                let mut holders = v.read_order(id)?;
                if v.holds_live(&a) {
                    holders.insert(0, v.me());
                }
                if holders.is_empty() {
                    return Err(invalid(format!("{p}: no live copy")));
                }
                Ok(Resolved {
                    file: id,
                    generation: a.generation,
                    size: a.size,
                    holders,
                })
            })
            .collect()
    }

    /// `Start`: check the plan, tell rank hosts what to expect, wait for
    /// every rank to join, then start the sources.
    async fn start(self: &Arc<Self>, group: u64) -> NestResult<Forecast> {
        let me = self.me()?;
        let (plan, ranks, rank_hosts, token, notify) = {
            let mut st = self.state.lock();
            let g = st
                .hub
                .get_mut(&group)
                .ok_or_else(|| invalid("no such group"))?;
            if g.started {
                return Err(invalid("already started"));
            }
            g.started = true;
            let plan = Plan {
                files: g.files.clone(),
                patterns: std::mem::take(&mut g.patterns),
                entries: std::mem::take(&mut g.entries),
            };
            (
                plan,
                g.ranks.clone(),
                g.rank_hosts.clone(),
                g.token,
                g.joined_notify.clone(),
            )
        };
        let resolved = {
            let me2 = self.clone();
            let files = plan.files.clone();
            tokio::task::spawn_blocking(move || me2.resolve(&files))
                .await
                .map_err(|e| NestError::Io(e.to_string()))??
        };
        let sizes: Vec<u64> = resolved.iter().map(|r| r.size).collect();
        let arenas: Vec<u64> = ranks.iter().map(|r| r.arena_bytes).collect();
        let totals = plan
            .check(&sizes, &arenas)
            .map_err(|e| invalid(format!("plan: {e}")))?;
        let holder = assign(&resolved, me);

        for (r, host) in rank_hosts.iter().enumerate() {
            self.send(
                *host,
                PushMsg::Expect {
                    group,
                    token,
                    hub: me,
                    rank: r as u32,
                    arena: ranks[r].arena_bytes,
                },
            )
            .await?;
        }
        let deadline = tokio::time::Instant::now() + JOIN_WAIT;
        let slot_bytes = loop {
            let n = notify.notified();
            {
                let st = self.state.lock();
                let g = st.hub.get(&group).ok_or_else(|| invalid("group ended"))?;
                if g.joined.len() == ranks.len() {
                    break (0..ranks.len() as u32)
                        .map(|r| g.joined[&r])
                        .collect::<Vec<u32>>();
                }
            }
            if tokio::time::timeout_at(deadline, n).await.is_err() {
                return Err(NestError::Unavailable(
                    "not every rank joined in time".into(),
                ));
            }
        };

        // Each source's files, with the entries that read them.
        let mut per_file: Vec<Vec<Entry>> = vec![Vec::new(); plan.files.len()];
        for e in plan.entries {
            per_file[e.file as usize].push(e);
        }
        let mut by_source: BTreeMap<NodeId, Vec<SourceFile>> = BTreeMap::new();
        for (i, entries) in per_file.into_iter().enumerate() {
            if entries.is_empty() {
                continue;
            }
            let r = &resolved[i];
            by_source.entry(holder[i]).or_default().push(SourceFile {
                index: i as u32,
                file: r.file,
                generation: r.generation,
                entries,
            });
        }
        // Which sources owe each rank anything.
        let mut owing: Vec<Vec<NodeId>> = vec![Vec::new(); ranks.len()];
        for (src, files) in &by_source {
            let mut rs: Vec<u32> = files
                .iter()
                .flat_map(|f| f.entries.iter().map(|e| e.rank))
                .collect();
            rs.sort_unstable();
            rs.dedup();
            for r in rs {
                owing[r as usize].push(*src);
            }
        }
        let forecast = Forecast {
            streams: by_source
                .iter()
                .map(|(n, fs)| StreamForecast {
                    store: self.name_of(*n),
                    bytes: fs.iter().map(|f| totals.file_bytes[f.index as usize]).sum(),
                    files: fs.len() as u32,
                    transport: if *n == me { "local" } else { "rpc" }.into(),
                    bytes_per_s: 0,
                })
                .collect(),
            rank_bytes: totals.rank_bytes.clone(),
            millis: 0,
        };
        {
            let mut st = self.state.lock();
            let g = st
                .hub
                .get_mut(&group)
                .ok_or_else(|| invalid("group ended"))?;
            g.total = totals.rank_bytes.iter().sum();
            g.streaming = by_source.keys().copied().collect();
            g.sources = g.streaming.clone();
            g.progress_at = tokio::time::Instant::now();
        }
        for (r, host) in rank_hosts.iter().enumerate() {
            self.send(
                *host,
                PushMsg::Sources {
                    group,
                    rank: r as u32,
                    sources: owing[r].clone(),
                },
            )
            .await?;
        }
        let rank_info: Vec<(NodeId, u32)> = rank_hosts.iter().copied().zip(slot_bytes).collect();
        for (src, files) in by_source {
            self.send(
                src,
                PushMsg::Stream {
                    group,
                    hub: me,
                    patterns: plan.patterns.clone(),
                    files,
                    ranks: rank_info.clone(),
                },
            )
            .await?;
        }
        // A plan with nothing to read is done at once.
        if self
            .state
            .lock()
            .hub
            .get(&group)
            .is_some_and(|g| g.streaming.is_empty())
        {
            self.finish(group).await;
        }
        Ok(forecast)
    }

    fn name_of(&self, n: NodeId) -> String {
        self.vfs()
            .and_then(|v| v.cluster())
            .ok()
            .and_then(|c| {
                c["hosts"]
                    .as_array()?
                    .iter()
                    .find(|h| h["node"].as_u64() == Some(n.0))
                    .and_then(|h| h["name"].as_str().map(str::to_string))
            })
            .unwrap_or_else(|| format!("node{}", n.0))
    }

    fn joined(&self, group: u64, rank: u32, slot_bytes: u32) -> NestResult<()> {
        let mut st = self.state.lock();
        let g = st
            .hub
            .get_mut(&group)
            .ok_or_else(|| invalid("no such group"))?;
        g.joined.insert(rank, slot_bytes);
        g.joined_notify.notify_waiters();
        Ok(())
    }

    /// A rank host's report: progress to the coordinator, and `Done` once
    /// every rank has everything.
    fn delivered(&self, group: u64, rank: u32, bytes: u64, complete: bool) {
        let mut st = self.state.lock();
        let Some(g) = st.hub.get_mut(&group) else {
            return;
        };
        if let Some(d) = g.delivered.get_mut(rank as usize) {
            *d = (*d).max(bytes);
        }
        if complete && let Some(c) = g.complete.get_mut(rank as usize) {
            *c = true;
        }
        g.progress_at = tokio::time::Instant::now();
        let _ = g.engine.send(Event::Progress {
            done: g.delivered.iter().sum(),
            total: g.total,
        });
        if g.complete.iter().all(|c| *c) {
            let _ = g.engine.send(Event::Done);
            st.hub.remove(&group);
        }
    }

    async fn source_done(
        self: &Arc<Self>,
        group: u64,
        source: NodeId,
        slots: Vec<u64>,
        failed: Vec<(u32, String)>,
    ) {
        let all_done = {
            let mut st = self.state.lock();
            let Some(g) = st.hub.get_mut(&group) else {
                return;
            };
            g.streaming.retain(|s| *s != source);
            for (i, n) in slots.iter().enumerate() {
                if let Some(t) = g.slots.get_mut(i) {
                    *t += n;
                }
            }
            g.failed.extend(failed);
            g.progress_at = tokio::time::Instant::now();
            g.streaming.is_empty()
        };
        if all_done {
            self.finish(group).await;
        }
    }

    /// Every source reported: fail the group if any file failed (phase 1:
    /// the engine reloads the whole group), else tell each rank host how
    /// many slots its rank gets in all.
    async fn finish(self: &Arc<Self>, group: u64) {
        let (failed, files, slots, hosts) = {
            let st = self.state.lock();
            let Some(g) = st.hub.get(&group) else {
                return;
            };
            (
                g.failed.clone(),
                g.files.clone(),
                g.slots.clone(),
                g.rank_hosts.clone(),
            )
        };
        if let Some((_, why)) = failed.first() {
            let n = hosts.len() as u32;
            let missing = failed
                .iter()
                .filter_map(|(i, _)| files.get(*i as usize))
                .map(|f| Missing {
                    file: f.clone(),
                    ranks: (0..n).collect(),
                })
                .collect();
            self.fail(group, why.clone(), missing).await;
            return;
        }
        for (r, host) in hosts.iter().enumerate() {
            if let Err(e) = self
                .send(
                    *host,
                    PushMsg::RankDone {
                        group,
                        rank: r as u32,
                        slots: slots[r],
                    },
                )
                .await
            {
                self.fail(group, format!("telling rank {r}'s host: {e}"), Vec::new())
                    .await;
                return;
            }
        }
    }

    /// End the group everywhere with `reason`.
    async fn fail(self: &Arc<Self>, group: u64, reason: String, missing: Vec<Missing>) {
        let hosts: Vec<NodeId> = {
            let mut st = self.state.lock();
            let Some(g) = st.hub.remove(&group) else {
                return;
            };
            let _ = g.engine.send(Event::Failed {
                reason: reason.clone(),
                missing,
            });
            let mut h = g.rank_hosts.clone();
            h.extend(g.sources.iter().copied());
            h.sort_unstable();
            h.dedup();
            h
        };
        for h in hosts {
            let _ = self
                .send(
                    h,
                    PushMsg::End {
                        group,
                        reason: reason.clone(),
                    },
                )
                .await;
        }
    }

    /// Fail groups that made no progress for `STALL`.
    fn sweep(self: &Arc<Self>) {
        let stale: Vec<u64> = {
            let st = self.state.lock();
            st.hub
                .iter()
                .filter(|(_, g)| g.started && g.progress_at.elapsed() > STALL)
                .map(|(id, _)| *id)
                .collect()
        };
        for g in stale {
            let me = self.clone();
            tokio::spawn(async move {
                me.fail(
                    g,
                    format!("no progress for {} s", STALL.as_secs()),
                    Vec::new(),
                )
                .await
            });
        }
    }

    // ---------------------------------------------------------- rank host

    fn expect(&self, group: u64, token: u64, hub: NodeId, rank: u32, arena: u64) -> NestResult<()> {
        self.state.lock().ranks.insert(
            (group, rank),
            RankSlot {
                hub,
                token,
                arena,
                ring: None,
                engine: None,
                filled: 0,
                expected: None,
                bytes: 0,
                reported: 0,
                free: Vec::new(),
                granted: HashMap::new(),
                sources: Vec::new(),
                next: 0,
                done_sent: false,
            },
        );
        Ok(())
    }

    /// `Join` from a rank's engine on this host.
    async fn join(
        self: &Arc<Self>,
        group: u64,
        token: u64,
        rank: u32,
        ring: Ring,
        engine: EngineTx,
    ) -> NestResult<()> {
        let (hub, slot_bytes) = {
            let mut st = self.state.lock();
            let r = st
                .ranks
                .get_mut(&(group, rank))
                .ok_or_else(|| invalid("this host does not expect that rank"))?;
            if r.token != token {
                return Err(invalid("bad token"));
            }
            if r.ring.is_some() {
                return Err(invalid("rank already joined"));
            }
            r.free = (0..ring.slots).rev().collect();
            let b = ring.slot_bytes;
            r.ring = Some(Arc::new(ring));
            r.engine = Some(engine);
            (r.hub, b)
        };
        self.send(
            hub,
            PushMsg::Joined {
                group,
                rank,
                slot_bytes,
            },
        )
        .await
    }

    fn rank_sources(self: &Arc<Self>, group: u64, rank: u32, sources: Vec<NodeId>) {
        if let Some(r) = self.state.lock().ranks.get_mut(&(group, rank)) {
            r.sources = sources;
        }
        self.grant(group, rank);
    }

    /// Hand free ring slots to sources as credits, round robin.
    fn grant(self: &Arc<Self>, group: u64, rank: u32) {
        let mut out: Vec<(NodeId, u32)> = Vec::new();
        {
            let mut st = self.state.lock();
            let Some(r) = st.ranks.get_mut(&(group, rank)) else {
                return;
            };
            if r.sources.is_empty() {
                return;
            }
            while let Some(slot) = r.free.pop() {
                let s = r.sources[r.next % r.sources.len()];
                r.next = r.next.wrapping_add(1);
                r.granted.entry(s).or_default().push_back(slot);
                match out.iter_mut().find(|(n, _)| *n == s) {
                    Some((_, k)) => *k += 1,
                    None => out.push((s, 1)),
                }
            }
        }
        for (src, n) in out {
            let me = self.clone();
            tokio::spawn(async move {
                if let Err(e) = me.send(src, PushMsg::Credit { group, rank, n }).await {
                    me.end_rank(group, rank, format!("credit to a source: {e}"));
                }
            });
        }
    }

    /// A slot from `source` for `rank`: into the ring slot its oldest
    /// credit named, checked, then to the engine.
    fn slot(
        self: &Arc<Self>,
        group: u64,
        rank: u32,
        source: NodeId,
        bytes: &[u8],
    ) -> NestResult<()> {
        let (hub, delivered, complete, report) = {
            let mut st = self.state.lock();
            let r = st
                .ranks
                .get_mut(&(group, rank))
                .ok_or_else(|| invalid("no such rank here"))?;
            let ring = r.ring.clone().ok_or_else(|| invalid("rank not joined"))?;
            let i = r
                .granted
                .get_mut(&source)
                .and_then(|q| q.pop_front())
                .ok_or_else(|| invalid("a slot without a credit"))?;
            ring.write_slot(i, bytes)?;
            let p = match slot::parse(ring.slot(i), r.arena) {
                Ok(p) => p,
                Err(e) => {
                    r.free.push(i);
                    return Err(invalid(format!("bad slot: {e}")));
                }
            };
            r.filled += 1;
            r.bytes += p.payload as u64;
            if let Some(e) = &r.engine {
                let _ = e.send(Event::Filled {
                    slot: i,
                    len: ring.slot_bytes,
                });
            }
            let complete = r.expected == Some(r.filled) && !r.done_sent;
            if complete {
                r.done_sent = true;
                if let Some(e) = &r.engine {
                    let _ = e.send(Event::Done);
                }
            }
            let report = complete || r.bytes - r.reported >= REPORT_EVERY;
            if report {
                r.reported = r.bytes;
            }
            (r.hub, r.bytes, complete, report)
        };
        if report {
            self.report(group, rank, hub, delivered, complete);
        }
        Ok(())
    }

    fn report(self: &Arc<Self>, group: u64, rank: u32, hub: NodeId, bytes: u64, complete: bool) {
        let me = self.clone();
        tokio::spawn(async move {
            let _ = me
                .send(
                    hub,
                    PushMsg::Delivered {
                        group,
                        rank,
                        bytes,
                        complete,
                    },
                )
                .await;
        });
    }

    /// The engine released ring slot `slot`: free for a new credit.
    fn release(self: &Arc<Self>, group: u64, rank: u32, slot: u32) {
        {
            let mut st = self.state.lock();
            let Some(r) = st.ranks.get_mut(&(group, rank)) else {
                return;
            };
            if r.ring.as_ref().is_some_and(|ring| slot < ring.slots) && !r.free.contains(&slot) {
                r.free.push(slot);
            }
        }
        self.grant(group, rank);
    }

    fn rank_done(self: &Arc<Self>, group: u64, rank: u32, slots: u64) {
        let report = {
            let mut st = self.state.lock();
            let Some(r) = st.ranks.get_mut(&(group, rank)) else {
                return;
            };
            r.expected = Some(slots);
            // Sources are done: no more credits needed.
            r.sources.clear();
            if r.filled == slots && !r.done_sent {
                r.done_sent = true;
                if let Some(e) = &r.engine {
                    let _ = e.send(Event::Done);
                }
                Some((r.hub, r.bytes))
            } else {
                None
            }
        };
        if let Some((hub, bytes)) = report {
            self.report(group, rank, hub, bytes, true);
        }
    }

    fn end_rank(&self, group: u64, rank: u32, reason: String) {
        if let Some(r) = self.state.lock().ranks.remove(&(group, rank))
            && let Some(e) = r.engine
            && !r.done_sent
        {
            let _ = e.send(Event::Failed {
                reason,
                missing: Vec::new(),
            });
        }
    }

    /// `End` for `group`: drop this host's part of it.
    fn end_here(&self, group: u64, reason: &str) {
        let mut st = self.state.lock();
        let ranks: Vec<(u64, u32)> = st
            .ranks
            .keys()
            .filter(|(g, _)| *g == group)
            .copied()
            .collect();
        for k in ranks {
            if let Some(r) = st.ranks.remove(&k)
                && let Some(e) = r.engine
                && !r.done_sent
            {
                let _ = e.send(Event::Failed {
                    reason: reason.to_string(),
                    missing: Vec::new(),
                });
            }
        }
        if let Some(s) = st.sources.get_mut(&group) {
            s.cancelled = true;
            s.notify.notify_waiters();
        }
        st.early.retain(|(g, _), _| *g != group);
        if let Some(g) = st.hub.remove(&group) {
            let _ = g.engine.send(Event::Failed {
                reason: reason.to_string(),
                missing: Vec::new(),
            });
        }
    }

    // -------------------------------------------------------------- source

    fn stream(
        self: &Arc<Self>,
        group: u64,
        hub: NodeId,
        patterns: Vec<Pattern>,
        files: Vec<SourceFile>,
        ranks: Vec<(NodeId, u32)>,
    ) {
        let notify = Arc::new(tokio::sync::Notify::new());
        {
            let mut st = self.state.lock();
            let mut credits = vec![0; ranks.len()];
            for (r, c) in credits.iter_mut().enumerate() {
                *c = st.early.remove(&(group, r as u32)).unwrap_or(0);
            }
            st.sources.insert(
                group,
                SourceState {
                    credits,
                    notify: notify.clone(),
                    cancelled: false,
                },
            );
        }
        let me = self.clone();
        tokio::spawn(async move {
            let src = me.me().unwrap_or(NodeId(0));
            let mut s = Stream {
                push: me.clone(),
                group,
                me: src,
                ranks,
                notify,
                open: Vec::new(),
                sent: Vec::new(),
            };
            let failed = s.run(&patterns, files).await;
            let slots = std::mem::take(&mut s.sent);
            me.state.lock().sources.remove(&group);
            let _ = me
                .send(
                    hub,
                    PushMsg::SourceDone {
                        group,
                        source: src,
                        slots,
                        failed,
                    },
                )
                .await;
        });
    }

    fn credit(&self, group: u64, rank: u32, n: u32) {
        let mut st = self.state.lock();
        match st.sources.get_mut(&group) {
            Some(s) => {
                if let Some(c) = s.credits.get_mut(rank as usize) {
                    *c += n;
                    s.notify.notify_waiters();
                }
            }
            None => *st.early.entry((group, rank)).or_default() += n,
        }
    }

    /// Wait for a credit for `rank` and take it; false if cancelled or
    /// none came within `STALL`.
    async fn take_credit(&self, group: u64, rank: u32, notify: &tokio::sync::Notify) -> bool {
        loop {
            let n = notify.notified();
            {
                let mut st = self.state.lock();
                let Some(s) = st.sources.get_mut(&group) else {
                    return false;
                };
                if s.cancelled {
                    return false;
                }
                if s.credits[rank as usize] > 0 {
                    s.credits[rank as usize] -= 1;
                    return true;
                }
            }
            if tokio::time::timeout(STALL, n).await.is_err() {
                return false;
            }
        }
    }

    // ----------------------------------------------------- engine sockets

    /// Serve one engine connection until it closes (its own threads: the
    /// socket is blocking).
    pub fn serve_engine(self: &Arc<Self>, conn: Conn) {
        let me = self.clone();
        let rt = tokio::runtime::Handle::current();
        std::thread::Builder::new()
            .name("push-engine".into())
            .spawn(move || me.engine_loop(conn, rt))
            .expect("thread");
    }

    fn engine_loop(self: Arc<Self>, conn: Conn, rt: tokio::runtime::Handle) {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
        let conn = Arc::new(conn);
        let out = conn.clone();
        let writer = std::thread::Builder::new()
            .name("push-events".into())
            .spawn(move || {
                while let Some(e) = rx.blocking_recv() {
                    if out.send_event(&e).is_err() {
                        return;
                    }
                }
            })
            .expect("thread");
        let mut owns_group: Option<u64> = None;
        let mut owns_rank: Option<(u64, u32)> = None;
        while let Ok(Some((req, fd))) = conn.next_request() {
            let fresh = owns_group.is_none() && owns_rank.is_none();
            let ev = match req {
                Request::Create {
                    version: _,
                    ranks,
                    files,
                } if fresh => match self.create(ranks, files, tx.clone()) {
                    Ok((group, token)) => {
                        owns_group = Some(group);
                        Some(Event::Created { group, token })
                    }
                    Err(e) => Some(Event::Refused(e.to_string())),
                },
                Request::Patterns(p) => owns_group
                    .ok_or_else(|| invalid("no group"))
                    .and_then(|g| self.add_plan(g, p, Vec::new()))
                    .err()
                    .map(|e| Event::Refused(e.to_string())),
                Request::Entries(e) => owns_group
                    .ok_or_else(|| invalid("no group"))
                    .and_then(|g| self.add_plan(g, Vec::new(), e))
                    .err()
                    .map(|e| Event::Refused(e.to_string())),
                Request::Start => match owns_group {
                    None => Some(Event::Refused("no group".into())),
                    Some(g) => {
                        let (me, tx2) = (self.clone(), tx.clone());
                        rt.spawn(async move {
                            match me.start(g).await {
                                Ok(f) => {
                                    let _ = tx2.send(Event::Planned(f));
                                }
                                Err(e) => {
                                    let _ = tx2.send(Event::Refused(e.to_string()));
                                    me.fail(g, e.to_string(), Vec::new()).await;
                                }
                            }
                        });
                        None
                    }
                },
                Request::Join {
                    version: _,
                    group,
                    token,
                    rank,
                    slots,
                    slot_bytes,
                } if fresh => {
                    let joined = fd
                        .ok_or_else(|| invalid("Join needs the ring's fd"))
                        .and_then(|fd| Ring::map(fd, slots, slot_bytes))
                        .and_then(|ring| {
                            rt.block_on(self.join(group, token, rank, ring, tx.clone()))
                        });
                    match joined {
                        Ok(()) => {
                            owns_rank = Some((group, rank));
                            Some(Event::Joined)
                        }
                        Err(e) => Some(Event::Refused(e.to_string())),
                    }
                }
                Request::Release { slot } => {
                    if let Some((g, r)) = owns_rank {
                        let me = self.clone();
                        rt.spawn(async move { me.release(g, r, slot) });
                    }
                    None
                }
                _ => Some(Event::Refused("unexpected request".into())),
            };
            if let Some(e) = ev {
                let _ = tx.send(e);
            }
        }
        // The engine closed: a coordinator cancels its group, a rank that
        // had not finished fails it.
        if let Some(g) = owns_group {
            let me = self.clone();
            rt.spawn(async move {
                me.fail(g, "the coordinator closed its socket".into(), Vec::new())
                    .await
            });
        }
        if let Some((g, r)) = owns_rank {
            let unfinished = {
                let mut st = self.state.lock();
                st.ranks
                    .remove(&(g, r))
                    .and_then(|rs| (!rs.done_sent).then_some(rs.hub))
            };
            if let Some(hub) = unfinished {
                let me = self.clone();
                rt.spawn(async move {
                    let _ = me
                        .send(
                            hub,
                            PushMsg::End {
                                group: g,
                                reason: format!("rank {r} closed its socket"),
                            },
                        )
                        .await;
                });
            }
        }
        drop(tx);
        let _ = writer.join();
    }
}

/// One source's pass over its files for a group.
struct Stream {
    push: Arc<Push>,
    group: u64,
    me: NodeId,
    /// Per rank: its host and slot size.
    ranks: Vec<(NodeId, u32)>,
    notify: Arc<tokio::sync::Notify>,
    open: Vec<Option<slot::Packer>>,
    /// Slots sent per rank.
    sent: Vec<u64>,
}

impl Stream {
    /// Stream every file; the files that failed, by plan index.
    async fn run(&mut self, patterns: &[Pattern], files: Vec<SourceFile>) -> Vec<(u32, String)> {
        let n = self.ranks.len();
        self.open = (0..n).map(|_| None).collect();
        self.sent = vec![0; n];
        let mut failed = Vec::new();
        let vfs = match self.push.vfs() {
            Ok(v) => v,
            Err(e) => return files.iter().map(|f| (f.index, e.to_string())).collect(),
        };
        for f in files {
            if let Err(e) = self.file(&vfs, patterns, &f).await {
                failed.push((f.index, e));
                if self.cancelled() {
                    return failed;
                }
            }
        }
        for r in 0..n as u32 {
            if let Err(e) = self.flush(r).await {
                failed.push((u32::MAX, e));
                break;
            }
        }
        failed
    }

    fn cancelled(&self) -> bool {
        let st = self.push.state.lock();
        st.sources.get(&self.group).is_none_or(|s| s.cancelled)
    }

    async fn file(
        &mut self,
        vfs: &Arc<Vfs>,
        patterns: &[Pattern],
        f: &SourceFile,
    ) -> Result<(), String> {
        let obj = {
            let (v, file, generation) = (vfs.clone(), f.file, f.generation);
            tokio::task::spawn_blocking(move || v.push_open(file, generation))
                .await
                .map_err(|e| e.to_string())?
                .map_err(|e| e.to_string())?
        };
        // The file's ranges, merged across small gaps: one pass in order.
        let mut spans: Vec<(u64, u64)> = f
            .entries
            .iter()
            .filter_map(|e| {
                let x = patterns.get(e.pattern as usize)?.check().ok()?;
                (x.bytes > 0).then(|| (e.src_base, e.src_base + x.src_len))
            })
            .collect();
        spans.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::new();
        for (a, b) in spans {
            match merged.last_mut() {
                Some(last) if a <= last.1.saturating_add(GAP) => last.1 = last.1.max(b),
                _ => merged.push((a, b)),
            }
        }
        for (a, b) in merged {
            let mut at = a;
            while at < b {
                let end = (at + READ_CHUNK).min(b);
                let buf = {
                    let (o, len) = (obj.clone(), (end - at) as usize);
                    tokio::task::spawn_blocking(move || {
                        use std::os::unix::fs::FileExt;
                        let mut b = vec![0u8; len];
                        o.read_exact_at(&mut b, at).map(|_| b)
                    })
                    .await
                    .map_err(|e| e.to_string())?
                    .map_err(|e| format!("reading: {e}"))?
                };
                for e in f.entries.iter() {
                    let p = &patterns[e.pattern as usize];
                    for (s, d, n) in RunWalk::new(p, e, at, end) {
                        let off = (s - at) as usize;
                        self.put(e.rank, d, &buf[off..off + n as usize]).await?;
                    }
                }
                at = end;
            }
        }
        Ok(())
    }

    /// Pack `data` (for arena offset `dst` of rank `r`), sending slots as
    /// they fill.
    async fn put(&mut self, r: u32, mut dst: u64, mut data: &[u8]) -> Result<(), String> {
        let ri = r as usize;
        while !data.is_empty() {
            let len = self.ranks[ri].1 as usize;
            let s = self.open[ri].get_or_insert_with(|| slot::Packer::new(vec![0u8; len]));
            let took = s.push(dst, data);
            if took == 0 {
                self.flush(r).await?;
                continue;
            }
            data = &data[took..];
            dst += took as u64;
        }
        Ok(())
    }

    /// Send rank `r`'s open slot once a credit allows.
    async fn flush(&mut self, r: u32) -> Result<(), String> {
        let ri = r as usize;
        let Some(s) = self.open[ri].take() else {
            return Ok(());
        };
        let (bytes, _, _) = s.finish();
        if !self.push.take_credit(self.group, r, &self.notify).await {
            return Err("cancelled, or no credit came".into());
        }
        self.push
            .send(
                self.ranks[ri].0,
                PushMsg::Slot {
                    group: self.group,
                    rank: r,
                    source: self.me,
                    bytes,
                },
            )
            .await
            .map_err(|e| format!("sending a slot: {e}"))?;
        self.sent[ri] += 1;
        Ok(())
    }
}

/// The `service::PUSH` handler.
struct PushService(std::sync::Weak<Push>);

impl nest_rpc::Handler for PushService {
    fn call(
        &self,
        peer: NodeId,
        body: bytes::Bytes,
    ) -> BoxFuture<'static, Result<bytes::Bytes, String>> {
        let p = self.0.clone();
        async move {
            let m: PushMsg = nest_rpc::decode(&body).map_err(|e| e.to_string())?;
            let Some(p) = p.upgrade() else {
                return Err("shutting down".to_string());
            };
            let r: Result<(), String> = p.on_msg(peer, m).await.map_err(|e| e.to_string());
            nest_rpc::encode(&r)
                .map(bytes::Bytes::from)
                .map_err(|e| e.to_string())
        }
        .boxed()
    }
}

/// Listen for engines on `path` (`@name`: abstract) until the process
/// ends. Only this process's uid and root may connect (ADR-063 §8).
pub fn listen(push: Arc<Push>, path: &str) -> std::io::Result<()> {
    let l = nest_push::client::Listener::bind(path)?;
    let uid = nix::unistd::getuid().as_raw();
    let rt = tokio::runtime::Handle::current();
    std::thread::Builder::new()
        .name("push-listen".into())
        .spawn(move || {
            let _rt = rt.enter();
            loop {
                let conn = match l.accept() {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::error!(error = %e, "push socket accept failed");
                        return;
                    }
                };
                match conn.peer() {
                    Ok((_, u, _)) if u == uid || u == 0 => push.serve_engine(conn),
                    Ok((pid, u, _)) => tracing::warn!(
                        pid,
                        uid = u,
                        "push socket: refused a connection from another user"
                    ),
                    Err(_) => {}
                }
            }
        })?;
    Ok(())
}
