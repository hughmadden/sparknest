//! Retransmit floor for control connections (ADR-056).
//!
//! Linux waits at least 200 ms before retransmitting a lost segment. A Raft
//! heartbeat is one small segment with a budget shorter than that, so on a
//! fabric that drops packets under incast every lost one became a timeout.
//! `TCP_RTO_MIN_US` (Linux 6.x) lowers the floor for one socket; older
//! kernels (WSL, off-grid hosts) refuse it and keep the default.

use nix::{setsockopt_impl, sockopt_impl};
use std::os::fd::AsFd;
use std::time::Duration;

/// Not yet in the libc crate: include/uapi/linux/tcp.h.
const TCP_RTO_MIN_US: libc::c_int = 45;

sockopt_impl!(
    RtoMinUs,
    SetOnly,
    libc::IPPROTO_TCP,
    TCP_RTO_MIN_US,
    libc::c_int,
    nix::sys::socket::sockopt::SetStruct<libc::c_int>
);

/// The retransmit floor on sparknest's peer connections.
pub const RTO_MIN: Duration = Duration::from_millis(5);

/// Lower a connection's retransmit floor; false where the kernel refuses.
pub fn set_rto_min(fd: &impl AsFd, rto: Duration) -> bool {
    let us = rto.as_micros().min(libc::c_int::MAX as u128) as libc::c_int;
    nix::sys::socket::setsockopt(fd, RtoMinUs, &us).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_floor_is_set_where_the_kernel_supports_it() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let s = std::net::TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let set = set_rto_min(&s, RTO_MIN);
        // Linux 6.x and later honour it; the build hosts all do.
        let (major, minor) = kernel();
        if (major, minor) >= (6, 11) {
            assert!(set, "kernel {major}.{minor} refused TCP_RTO_MIN_US");
        }
    }

    fn kernel() -> (u32, u32) {
        let r = nix::sys::utsname::uname().unwrap();
        let rel = r.release().to_string_lossy().into_owned();
        let mut p = rel
            .split(|c: char| !c.is_ascii_digit())
            .filter_map(|x| x.parse().ok());
        (p.next().unwrap_or(0), p.next().unwrap_or(0))
    }
}
