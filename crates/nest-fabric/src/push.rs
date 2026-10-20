//! Push loads over RDMA (ADR-063 §5, §7): a source's NIC writes a whole
//! slot straight into a rank's ring, which the rank's host registered.
//!
//! ```text
//! rank host                                   source
//!   register_push_ring(ring)  -> rkey per device, address
//!   (credit: slot i, addr + i·slot_bytes, rkeys, token)  ->
//!                                             copy the slot image into a
//!                                             registered staging slot
//!                 <- RDMA WRITE_WITH_IMM(staging -> ring slot), imm = PUSH|token
//!   PushSink::landed(token, len)              completion: staging slot free
//! ```
//!
//! The write rides an existing link's lane (any lane of the link: the
//! ring is registered on every device). Its immediate has `IMM_PUSH` set,
//! which read replies never do (their immediate is a landing slot id), so
//! the completion thread routes it here. The sink learns which ring slot
//! landed from the 31-bit token the rank host chose for that credit.

use crate::engine::{Fabric, wr};
use crate::pool::{Pool, Slot};
use crate::verbs::{Access, Region, SharedMap};
use nest_types::{NestError, NestResult, NodeId};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};
use std::time::Duration;
use tokio::sync::oneshot;

/// Immediate bit marking a push write (read replies carry a slot id).
pub(crate) const IMM_PUSH: u32 = 1 << 31;
/// Work request kind of a push write.
pub(crate) const WR_PUSH: u64 = 5;
/// A push write whose completion has not come after this fails, and its
/// lane is retired.
const PUSH_TIMEOUT: Duration = Duration::from_secs(30);

/// Told when a push write lands in a registered ring (on the completion
/// thread: return at once).
pub trait PushSink: Send + Sync + 'static {
    fn landed(&self, token: u32, len: u32);
}

/// Where a source writes one ring slot.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PushTarget {
    pub addr: u64,
    /// The ring's rkey on each of the rank host's rails (the source writes
    /// on whichever lane it has, so it needs the key of the device at that
    /// lane's far end: the rail in its lane's subnet).
    pub rkeys: Vec<(std::net::Ipv4Addr, u8, u32)>,
    /// Told to the rank host with the write (31 bits).
    pub token: u32,
}

/// An engine's ring, registered on every device of this host.
pub struct PushRing {
    regions: Vec<(crate::Rail, Region)>,
    base: u64,
}

impl PushRing {
    /// The target for the `len` bytes at `offset` of the ring.
    pub fn target(&self, offset: u64, token: u32) -> PushTarget {
        PushTarget {
            addr: self.base + offset,
            rkeys: self
                .regions
                .iter()
                .map(|(rail, r)| (rail.addr, rail.prefix, r.rkey()))
                .collect(),
            token: token & !IMM_PUSH,
        }
    }
}

/// A source's registered staging slots, one pool per device.
pub struct PushPool {
    pools: Vec<(u32, Arc<Pool>)>,
    slot_bytes: usize,
}

impl PushPool {
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
}

/// A push write in flight: its staging slot (held until the NIC is done
/// with it), its length, and the writer waiting for the completion.
type InFlight = (Slot, u32, oneshot::Sender<bool>);

/// Push writes in flight, and where landed writes are reported.
#[derive(Default)]
pub(crate) struct PushState {
    in_flight: Mutex<HashMap<u64, InFlight>>,
    sink: Mutex<Option<Weak<dyn PushSink>>>,
    next: std::sync::atomic::AtomicU32,
}

impl Fabric {
    /// Registered staging for a push source: `slots` of `slot_bytes` on
    /// every device. Freed when dropped (after the group's writes finish).
    pub fn push_pool(&self, slots: u32, slot_bytes: usize) -> std::io::Result<PushPool> {
        let pools = self
            .devices
            .iter()
            .map(|d| {
                Ok((
                    d.id,
                    Pool::with_access(&d.ctx, slots, slot_bytes, Access::Local, 0)?,
                ))
            })
            .collect::<std::io::Result<Vec<_>>>()?;
        Ok(PushPool { pools, slot_bytes })
    }

    /// Register an engine's ring on every device so peers can write into
    /// it. Deregistered when the returned ring drops.
    pub fn register_push_ring(&self, map: Arc<SharedMap>) -> std::io::Result<PushRing> {
        let regions = self
            .rails()
            .iter()
            .map(|rail| {
                let dev = self
                    .devices
                    .iter()
                    .find(|d| d.ctx.name == rail.ibdev)
                    .expect("rail device opened");
                Ok((
                    rail.clone(),
                    Region::register_shared(&dev.ctx, map.clone(), Access::RemoteWrite)?,
                ))
            })
            .collect::<std::io::Result<Vec<_>>>()?;
        Ok(PushRing {
            base: map.addr(),
            regions,
        })
    }

    pub fn set_push_sink(&self, sink: Weak<dyn PushSink>) {
        *self.push.sink.lock() = Some(sink);
    }

    /// Write `data` (at most the pool's slot size) into `target` on
    /// `peer`, through a lane of the link to it. Returns once the NIC has
    /// finished (the bytes are in the peer's ring and its host was told).
    pub async fn push_write(
        &self,
        peer: NodeId,
        pool: &PushPool,
        data: &[u8],
        target: &PushTarget,
    ) -> NestResult<()> {
        if data.len() > pool.slot_bytes {
            return Err(NestError::Invalid(
                "slot larger than the push pool's".into(),
            ));
        }
        let link = self.link(peer).await?;
        let lane = link.lanes
            [link.next.fetch_add(1, Ordering::Relaxed) as usize % link.lanes.len()]
        .clone();
        // The rank host's device at the far end of this lane shares its
        // subnet with ours.
        let rkey = target
            .rkeys
            .iter()
            .find(|(addr, prefix, _)| {
                lane.local.same_subnet(&crate::Rail {
                    addr: *addr,
                    prefix: *prefix,
                    ..lane.local.clone()
                })
            })
            .map(|(_, _, k)| *k)
            .ok_or_else(|| {
                NestError::Invalid("no ring registration in this lane's subnet".into())
            })?;
        let pool_of = pool
            .pools
            .iter()
            .find(|(d, _)| *d == lane.dev.id)
            .map(|(_, p)| p.clone())
            .ok_or_else(|| NestError::Unavailable("no push staging on that device".into()))?;
        let mut slot = pool_of.acquire().await;
        slot.as_mut_slice(data.len()).copy_from_slice(data);
        if lane.dead.load(Ordering::SeqCst) {
            return Err(NestError::Unavailable("RDMA link failed".into()));
        }
        let n = self.push.next.fetch_add(1, Ordering::Relaxed);
        let id = wr(WR_PUSH, lane.id, n);
        let (tx, rx) = oneshot::channel();
        let (ptr, lkey) = (slot.ptr(), slot.lkey());
        self.push
            .in_flight
            .lock()
            .insert(id, (slot, data.len() as u32, tx));
        // SAFETY: the staging slot stays in `in_flight` until this write's
        // completion (success or flush); the remote range is a ring slot
        // the rank host registered and granted to us for this write.
        let posted = unsafe {
            lane.qp.post_write_imm(
                id,
                ptr,
                data.len() as u32,
                lkey,
                target.addr,
                rkey,
                IMM_PUSH | target.token,
            )
        };
        if let Err(e) = posted {
            drop(self.push.in_flight.lock().remove(&id));
            self.fail_lane(&lane);
            return Err(NestError::Unavailable(e.to_string()));
        }
        match tokio::time::timeout(PUSH_TIMEOUT, rx).await {
            Ok(Ok(true)) => {
                self.stats.push_writes.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .push_bytes
                    .fetch_add(data.len() as u64, Ordering::Relaxed);
                Ok(())
            }
            Ok(_) => Err(NestError::Unavailable("push write failed".into())),
            Err(_) => {
                // The NIC may still read the slot: never reuse it.
                if let Some((slot, _, _)) = self.push.in_flight.lock().remove(&id) {
                    std::mem::forget(slot);
                }
                self.fail_lane(&lane);
                Err(NestError::Unavailable("push write timed out".into()))
            }
        }
    }

    /// A push write's completion (on the completion thread): its bytes.
    pub(crate) fn push_written(&self, wr_id: u64, ok: bool) -> Option<u32> {
        let (slot, n, tx) = self.push.in_flight.lock().remove(&wr_id)?;
        drop(slot);
        let _ = tx.send(ok);
        ok.then_some(n)
    }

    /// A push write landed in one of our registered rings.
    pub(crate) fn push_landed(&self, token: u32, len: u32) {
        let sink = self.push.sink.lock().as_ref().and_then(|w| w.upgrade());
        if let Some(s) = sink {
            s.landed(token, len);
        }
    }
}
