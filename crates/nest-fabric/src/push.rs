//! Push loads over RDMA (ADR-063 §5, §7): a source's NIC writes a whole
//! slot straight into a rank's ring, which the rank's host registered.
//!
//! ```text
//! rank host                                   source
//!   register_push_ring(ring)  -> rkey per device, address
//!   (credit: slot i, addr + i·slot_bytes, rkeys, token)  ->
//!                                             pack the slot image in a
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
use crate::verbs::{Access, Region, SharedMap};
use nest_types::{NestError, NodeId};
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

/// A source's registered staging: one mapping of `slots × slot_bytes`,
/// registered on every device, so a slot is packed once and written
/// through whichever device the lane uses.
pub struct PushPool(Arc<Staging>);

struct Staging {
    /// Per device: its registration of `map`. Dropped before `map`.
    regions: Vec<(u32, Region)>,
    map: Arc<SharedMap>,
    slot_bytes: usize,
    free: Mutex<Vec<u32>>,
    returned: tokio::sync::Notify,
}

impl PushPool {
    pub fn slot_bytes(&self) -> usize {
        self.0.slot_bytes
    }

    /// A free slot, viewed as its first `len` bytes (at most `slot_bytes`;
    /// the rank's slot size). Waits while every slot is in use.
    pub async fn acquire(&self, len: usize) -> PushSlot {
        assert!(len <= self.0.slot_bytes, "slot larger than the push pool's");
        loop {
            let notified = self.0.returned.notified();
            if let Some(index) = self.0.free.lock().pop() {
                return PushSlot {
                    s: self.0.clone(),
                    index,
                    len,
                };
            }
            notified.await;
        }
    }
}

/// One staging slot, owned: the packer fills it, `push_write` hands it to
/// the NIC, and it returns to its pool once the write completes.
pub struct PushSlot {
    s: Arc<Staging>,
    index: u32,
    len: usize,
}

impl PushSlot {
    fn offset(&self) -> usize {
        self.index as usize * self.s.slot_bytes
    }
}

impl AsRef<[u8]> for PushSlot {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: the slot is ours alone (the NIC reads it only while a
        // write owns it in `in_flight`); len is within the slot.
        unsafe { self.s.map.slice(self.offset(), self.len) }
    }
}

impl AsMut<[u8]> for PushSlot {
    fn as_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` excludes other views.
        unsafe { self.s.map.slice_mut(self.offset(), self.len) }
    }
}

impl Drop for PushSlot {
    fn drop(&mut self) {
        self.s.free.lock().push(self.index);
        self.s.returned.notify_one();
    }
}

/// A push write in flight: its staging slot (held until the NIC is done
/// with it), its length, and the writer waiting for the completion (sent
/// the slot back if the write failed).
type InFlight = (PushSlot, u32, oneshot::Sender<Result<(), PushSlot>>);

/// Push writes in flight, and where landed writes are reported.
#[derive(Default)]
pub(crate) struct PushState {
    in_flight: Mutex<HashMap<u64, InFlight>>,
    sink: Mutex<Option<Weak<dyn PushSink>>>,
    next: std::sync::atomic::AtomicU32,
}

impl Fabric {
    /// Registered staging for a push source: `slots` of `slot_bytes`,
    /// registered on every device. Freed when dropped (after the group's
    /// writes finish).
    pub fn push_pool(&self, slots: u32, slot_bytes: usize) -> std::io::Result<PushPool> {
        let map = Arc::new(SharedMap::anonymous(slots as usize * slot_bytes)?);
        let regions = self
            .devices
            .iter()
            .map(|d| {
                Ok((
                    d.id,
                    Region::register_shared(&d.ctx, map.clone(), Access::Local)?,
                ))
            })
            .collect::<std::io::Result<Vec<_>>>()?;
        Ok(PushPool(Arc::new(Staging {
            regions,
            map,
            slot_bytes,
            free: Mutex::new((0..slots).rev().collect()),
            returned: tokio::sync::Notify::new(),
        })))
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

    /// Write `slot` (its packed bytes) into `target` on `peer`, through a
    /// lane of the link to it. Returns once the NIC has finished (the bytes
    /// are in the peer's ring and its host was told). A write that failed
    /// hands its bytes back, to send another way.
    pub async fn push_write(
        &self,
        peer: NodeId,
        slot: PushSlot,
        target: &PushTarget,
    ) -> Result<(), (NestError, Vec<u8>)> {
        let fail = |e: NestError, slot: &PushSlot| (e, slot.as_ref().to_vec());
        let link = match self.link(peer).await {
            Ok(l) => l,
            Err(e) => return Err(fail(e, &slot)),
        };
        let lane = link.lanes
            [link.next.fetch_add(1, Ordering::Relaxed) as usize % link.lanes.len()]
        .clone();
        // The rank host's device at the far end of this lane shares its
        // subnet with ours.
        let Some(rkey) = target
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
        else {
            let e = NestError::Invalid("no ring registration in this lane's subnet".into());
            return Err(fail(e, &slot));
        };
        let Some(lkey) = slot
            .s
            .regions
            .iter()
            .find(|(d, _)| *d == lane.dev.id)
            .map(|(_, r)| r.lkey())
        else {
            let e = NestError::Unavailable("no push staging on that device".into());
            return Err(fail(e, &slot));
        };
        if lane.dead.load(Ordering::SeqCst) {
            let e = NestError::Unavailable("RDMA link failed".into());
            return Err(fail(e, &slot));
        }
        let n = self.push.next.fetch_add(1, Ordering::Relaxed);
        let id = wr(WR_PUSH, lane.id, n);
        let (tx, rx) = oneshot::channel();
        let (ptr, len) = (slot.s.map.addr() as *mut u8, slot.len);
        // SAFETY: the offset is within the mapping (a slot of it).
        let ptr = unsafe { ptr.add(slot.offset()) };
        self.push
            .in_flight
            .lock()
            .insert(id, (slot, len as u32, tx));
        // SAFETY: the staging slot stays in `in_flight` until this write's
        // completion (success or flush); the remote range is a ring slot
        // the rank host registered and granted to us for this write.
        let posted = unsafe {
            lane.qp.post_write_imm(
                id,
                ptr,
                len as u32,
                lkey,
                target.addr,
                rkey,
                IMM_PUSH | target.token,
            )
        };
        if let Err(e) = posted {
            let slot = self.push.in_flight.lock().remove(&id).map(|x| x.0);
            self.fail_lane(&lane);
            let e = NestError::Unavailable(e.to_string());
            return Err(match slot {
                Some(s) => fail(e, &s),
                None => (e, Vec::new()),
            });
        }
        match tokio::time::timeout(PUSH_TIMEOUT, rx).await {
            Ok(Ok(Ok(()))) => {
                self.stats.push_writes.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .push_bytes
                    .fetch_add(len as u64, Ordering::Relaxed);
                Ok(())
            }
            Ok(Ok(Err(slot))) => Err(fail(
                NestError::Unavailable("push write failed".into()),
                &slot,
            )),
            Ok(Err(_)) => Err((
                NestError::Unavailable("push write failed".into()),
                Vec::new(),
            )),
            Err(_) => {
                // The NIC may still read the slot: never reuse it.
                let mut back = Vec::new();
                if let Some((slot, _, _)) = self.push.in_flight.lock().remove(&id) {
                    back = slot.as_ref().to_vec();
                    std::mem::forget(slot);
                }
                self.fail_lane(&lane);
                Err((NestError::Unavailable("push write timed out".into()), back))
            }
        }
    }

    /// A push write's completion (on the completion thread): its bytes.
    pub(crate) fn push_written(&self, wr_id: u64, ok: bool) -> Option<u32> {
        let (slot, n, tx) = self.push.in_flight.lock().remove(&wr_id)?;
        let _ = tx.send(if ok { Ok(()) } else { Err(slot) });
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
