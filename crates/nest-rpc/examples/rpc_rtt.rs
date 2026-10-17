//! Control-call round trips between two hosts (benchmarks/M12-CONTROL-PATH.md).
//!
//!   rpc_rtt serve LISTEN
//!   rpc_rtt ping  ADDR SECONDS RATE [BYTES]
//!
//! The pinger sends RATE small requests a second (BYTES of payload, 200 by
//! default: about a heartbeat) on one authenticated connection, as the
//! control plane does, and prints the latency distribution, the share over
//! 150 ms (a heartbeat's budget) and the worst. Both ends use the same
//! socket options as sparknestd (TCP_NODELAY, the 5 ms retransmit floor).

use bytes::Bytes;
use futures::FutureExt;
use nest_rpc::{Rpc, RpcConfig};
use nest_types::NodeId;
use std::sync::Arc;
use std::time::{Duration, Instant};

const SERVICE: u8 = 9;

fn cfg(node: u64, listen: &str) -> RpcConfig {
    RpcConfig {
        node: NodeId(node),
        cluster: "rpc-rtt".into(),
        secret: b"rpc-rtt".to_vec(),
        listen: listen.parse().unwrap(),
        connect_timeout: Duration::from_secs(2),
    }
}

#[tokio::main]
async fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("serve") => {
            let rpc = Rpc::bind(cfg(1, &a[2])).await.unwrap();
            rpc.register(
                SERVICE,
                Arc::new(|_p: NodeId, b: Bytes| async move { Ok(b) }.boxed()),
            );
            eprintln!("serving on {}", rpc.local_addr());
            loop {
                tokio::time::sleep(Duration::from_secs(3600)).await;
            }
        }
        Some("ping") => {
            let secs: u64 = a[3].parse().unwrap();
            let rate: u64 = a[4].parse().unwrap();
            let bytes: usize = a.get(5).map_or(200, |s| s.parse().unwrap());
            let rpc = Rpc::bind(cfg(2, "0.0.0.0:0")).await.unwrap();
            rpc.set_peer(NodeId(1), a[2].parse().unwrap());
            let body = Bytes::from(vec![7u8; bytes]);
            rpc.call(NodeId(1), SERVICE, body.clone(), Duration::from_secs(5))
                .await
                .expect("first call");
            let every = Duration::from_nanos(1_000_000_000 / rate.max(1));
            let mut tick = tokio::time::interval(every);
            let end = Instant::now() + Duration::from_secs(secs);
            let mut calls = Vec::new();
            while Instant::now() < end {
                tick.tick().await;
                let (rpc, body) = (rpc.clone(), body.clone());
                calls.push(tokio::spawn(async move {
                    let t = Instant::now();
                    let r = rpc
                        .call(NodeId(1), SERVICE, body, Duration::from_secs(5))
                        .await;
                    (t.elapsed(), r.is_ok())
                }));
            }
            let mut lat = Vec::new();
            let mut failed = 0;
            for c in calls {
                let (d, ok) = c.await.unwrap();
                if ok {
                    lat.push(d);
                } else {
                    failed += 1;
                }
            }
            lat.sort();
            let q = |p: f64| lat[((lat.len() - 1) as f64 * p) as usize];
            let over = lat
                .iter()
                .filter(|d| **d > Duration::from_millis(150))
                .count();
            println!(
                "calls={} failed={failed} p50={:?} p99={:?} p99.9={:?} p99.99={:?} max={:?} over150ms={over}",
                lat.len(),
                q(0.5),
                q(0.99),
                q(0.999),
                q(0.9999),
                lat.last().unwrap(),
            );
        }
        _ => eprintln!("usage: rpc_rtt serve LISTEN | ping ADDR SECONDS RATE [BYTES]"),
    }
}
