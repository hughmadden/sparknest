//! RDMA traffic counters per device port (ADR-051), for benchmarking tools
//! that line sparknest's share of the fabric up with the NIC's totals.
//!
//! Relaxed atomic adds on the completion path, where each operation's size
//! is already known; reading them takes no lock.

use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};

/// Cumulative since the daemon started; never decreasing.
#[derive(Default)]
pub struct PortCounters {
    bytes_sent: AtomicU64,
    bytes_received: AtomicU64,
    ops_sent: AtomicU64,
    ops_received: AtomicU64,
    packets_sent: AtomicU64,
    packets_received: AtomicU64,
}

/// One port's counters at a moment, in the order the stream sends them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct CounterValues {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub ops_sent: u64,
    pub ops_received: u64,
    /// Estimated: each operation counts `ceil(bytes / path_mtu)` (at least
    /// one), so `packets x header size` estimates the wire overhead.
    pub packets_sent: u64,
    pub packets_received: u64,
}

impl PortCounters {
    #[inline]
    pub(crate) fn sent(&self, bytes: u64, mtu: u32) {
        self.bytes_sent.fetch_add(bytes, Ordering::Relaxed);
        self.ops_sent.fetch_add(1, Ordering::Relaxed);
        self.packets_sent
            .fetch_add(packets(bytes, mtu), Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn received(&self, bytes: u64, mtu: u32) {
        self.bytes_received.fetch_add(bytes, Ordering::Relaxed);
        self.ops_received.fetch_add(1, Ordering::Relaxed);
        self.packets_received
            .fetch_add(packets(bytes, mtu), Ordering::Relaxed);
    }

    pub fn load(&self) -> CounterValues {
        CounterValues {
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            ops_sent: self.ops_sent.load(Ordering::Relaxed),
            ops_received: self.ops_received.load(Ordering::Relaxed),
            packets_sent: self.packets_sent.load(Ordering::Relaxed),
            packets_received: self.packets_received.load(Ordering::Relaxed),
        }
    }
}

#[inline]
fn packets(bytes: u64, mtu: u32) -> u64 {
    bytes.div_ceil(mtu.max(256) as u64).max(1)
}

/// A port as `/sys/class/infiniband` names it, and what it carries.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PortInfo {
    /// RDMA device, e.g. `mlx5_0` or `rocep1s0f0`.
    pub device: String,
    pub port: u8,
    /// Its network interface, e.g. `enp1s0f0np0`.
    pub netdev: String,
    /// The rail addresses sparknest uses on it (raptor has two on one port).
    pub addrs: Vec<String>,
    /// Active MTU of the port, bytes.
    pub path_mtu: u32,
}

/// Bytes for a verbs `ibv_mtu` value (1 = 256 … 5 = 4096).
pub(crate) fn mtu_bytes(enum_value: u32) -> u32 {
    128u32 << enum_value.clamp(1, 5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packets_round_up_and_count_at_least_one() {
        let c = PortCounters::default();
        c.sent(4 << 20, 4096);
        c.sent(64, 4096);
        c.received(4097, 4096);
        let v = c.load();
        assert_eq!(v.bytes_sent, (4 << 20) + 64);
        assert_eq!(v.ops_sent, 2);
        assert_eq!(v.packets_sent, 1024 + 1);
        assert_eq!(v.packets_received, 2);
        assert_eq!(mtu_bytes(5), 4096);
        assert_eq!(mtu_bytes(1), 256);
    }
}
