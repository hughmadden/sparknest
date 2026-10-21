//! Transports for remote reads in whole chunks: the RDMA fabric between
//! fabric hosts, TCP lanes when either end is off the fabric (ADR-048),
//! and between fabric hosts RDMA with the standby lanes on the LAN behind
//! it (ADR-057).

use futures::FutureExt;
use futures::future::BoxFuture;
use nest_types::{FileId, Generation, NestResult, NodeId};
use std::sync::Arc;

pub(crate) trait Bulk: Send + Sync + 'static {
    /// Largest single read.
    fn chunk(&self) -> usize;
    /// Read `len` bytes (≤ chunk) of exactly `generation` from `peer`.
    fn fetch(
        self: Arc<Self>,
        peer: NodeId,
        file: FileId,
        generation: Generation,
        offset: u64,
        len: usize,
    ) -> BoxFuture<'static, NestResult<Vec<u8>>>;
}

impl Bulk for nest_fabric::Fabric {
    fn chunk(&self) -> usize {
        nest_fabric::Fabric::chunk(self)
    }

    fn fetch(
        self: Arc<Self>,
        peer: NodeId,
        file: FileId,
        generation: Generation,
        offset: u64,
        len: usize,
    ) -> BoxFuture<'static, NestResult<Vec<u8>>> {
        async move {
            let b = self.read(peer, file, generation, offset, len).await?;
            // Copied out at once: the landing slot is free again.
            Ok(b.as_slice().to_vec())
        }
        .boxed()
    }
}

/// RDMA between fabric hosts, with the peer's standby lane on the LAN
/// behind it (ADR-057): a read goes over the lane only while the RDMA link
/// to that peer is down (setup failed or the link died, within the
/// fabric's retry backoff) and a lane to it is open; RDMA is tried again
/// when the backoff passes. A read that fails because the link died is
/// retried once on the lane. Slowness never moves a read.
pub(crate) struct Standby {
    pub(crate) fabric: Arc<nest_fabric::Fabric>,
    pub(crate) lanes: Arc<nest_fabric::tcp::Lanes>,
}

impl Standby {
    /// Use the lane to `peer` instead of RDMA right now.
    pub(crate) fn on_lane(&self, peer: NodeId) -> bool {
        self.fabric.link_failed(peer) && self.lanes.connected(peer) > 0
    }
}

impl Bulk for Standby {
    fn chunk(&self) -> usize {
        // Both transports read the same chunks (4 MiB).
        nest_fabric::Fabric::chunk(&self.fabric).min(self.lanes.chunk())
    }

    fn fetch(
        self: Arc<Self>,
        peer: NodeId,
        file: FileId,
        generation: Generation,
        offset: u64,
        len: usize,
    ) -> BoxFuture<'static, NestResult<Vec<u8>>> {
        async move {
            if !self.on_lane(peer) {
                match self.fabric.clone().fetch(peer, file, generation, offset, len).await {
                    Err(nest_types::NestError::Unavailable(why)) if self.on_lane(peer) => {
                        tracing::debug!(%peer, %why, "RDMA link down; reading over the standby lane");
                    }
                    r => return r,
                }
            }
            self.lanes.read(peer, file, generation, offset, len).await
        }
        .boxed()
    }
}

impl Bulk for nest_fabric::tcp::Lanes {
    fn chunk(&self) -> usize {
        nest_fabric::tcp::Lanes::chunk(self)
    }

    /// Lanes land in ordinary memory under a byte budget that every read,
    /// speculative or not, waits its turn for.
    fn fetch(
        self: Arc<Self>,
        peer: NodeId,
        file: FileId,
        generation: Generation,
        offset: u64,
        len: usize,
    ) -> BoxFuture<'static, NestResult<Vec<u8>>> {
        async move { self.read(peer, file, generation, offset, len).await }.boxed()
    }
}
