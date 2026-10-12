//! Transports for remote reads in whole chunks: the RDMA fabric between
//! fabric hosts, TCP lanes when either end is off the fabric (ADR-048).
//! Readahead works over either.

use futures::FutureExt;
use futures::future::BoxFuture;
use nest_types::{FileId, Generation, NestResult, NodeId};
use std::sync::Arc;

pub(crate) trait Bulk: Send + Sync + 'static {
    /// Largest single read.
    fn chunk(&self) -> usize;
    /// Whether a speculative read may start now: it must never take what a
    /// demanded read needs (PROPOSAL invariant 9).
    fn speculate(&self) -> bool;
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

    /// Fetched chunks hold a landing slot until copied out: speculate only
    /// while a quarter of them are spare.
    fn speculate(&self) -> bool {
        self.spare_landing() > self.landing_slots() / 4
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

impl Bulk for nest_fabric::tcp::Lanes {
    fn chunk(&self) -> usize {
        nest_fabric::tcp::Lanes::chunk(self)
    }

    /// Lanes land in ordinary memory under a byte budget that every read,
    /// speculative or not, waits its turn for.
    fn speculate(&self) -> bool {
        true
    }

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
