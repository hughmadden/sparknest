//! Throughput of TCP lanes between two hosts (benchmarks/M10-LAN-TCP.md).
//!
//!   lanes_bench serve LISTEN FILE            serve FILE as file 1
//!   lanes_bench read  ADDR SIZE LANES INFLIGHT [ROUNDS]
//!
//! The reader dials LANES lanes to ADDR and reads SIZE bytes of file 1 in
//! 4 MiB chunks, INFLIGHT at a time, ROUNDS times (the first warms the
//! server's page cache).

use futures::FutureExt;
use futures::future::BoxFuture;
use futures::stream::{FuturesUnordered, StreamExt};
use nest_fabric::tcp::{Lanes, LanesConfig};
use nest_fabric::{ReadSource, ServeFile};
use nest_types::{FileId, Generation, NestError, NodeId};
use std::sync::Arc;
use std::time::{Duration, Instant};

struct One(Arc<std::fs::File>);

impl ReadSource for One {
    fn open_now(&self, _: FileId, _: Generation) -> Option<Result<ServeFile, NestError>> {
        Some(Ok(ServeFile {
            file: self.0.clone(),
            guard: Box::new(()),
        }))
    }
    fn open(&self, f: FileId, g: Generation) -> BoxFuture<'static, Result<ServeFile, NestError>> {
        let r = self.open_now(f, g).unwrap();
        async move { r }.boxed()
    }
}

const CHUNK: usize = 4 << 20;

#[tokio::main]
async fn main() {
    let a: Vec<String> = std::env::args().collect();
    let secret = b"lanes-bench".to_vec();
    match a.get(1).map(String::as_str) {
        Some("serve") => {
            let f = Arc::new(std::fs::File::open(&a[3]).unwrap());
            let src: Arc<dyn ReadSource> = Arc::new(One(f));
            let lanes = Lanes::start(
                NodeId(1),
                "bench".into(),
                secret,
                LanesConfig::default(),
                Some(a[2].parse().unwrap()),
            )
            .await
            .unwrap();
            lanes.set_source(Arc::downgrade(&src));
            eprintln!("serving {} on {}", a[3], a[2]);
            loop {
                tokio::time::sleep(Duration::from_secs(3600)).await;
            }
        }
        Some("read") => {
            let size: u64 = a[3].parse().unwrap();
            let per_peer: usize = a[4].parse().unwrap();
            let inflight: usize = a[5].parse().unwrap();
            let rounds: usize = a.get(6).map_or(3, |r| r.parse().unwrap());
            let cfg = LanesConfig {
                per_peer,
                inflight_bytes: (inflight * CHUNK) as u64,
                ..LanesConfig::default()
            };
            let lanes = Lanes::start(NodeId(2), "bench".into(), secret, cfg, None)
                .await
                .unwrap();
            lanes.dial(NodeId(1), a[2].parse().unwrap());
            while lanes.connected(NodeId(1)) < per_peer {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            for round in 0..rounds {
                let t = Instant::now();
                let mut next = 0u64;
                let mut got = 0u64;
                let mut q = FuturesUnordered::new();
                while next < size || !q.is_empty() {
                    while q.len() < inflight && next < size {
                        q.push(lanes.read(NodeId(1), FileId(1), Generation(1), next, CHUNK));
                        next += CHUNK as u64;
                    }
                    if let Some(r) = q.next().await {
                        got += r.unwrap().len() as u64;
                    }
                }
                let s = t.elapsed().as_secs_f64();
                println!(
                    "lanes={per_peer} inflight={inflight} round={round} {:.2} GB in {:.2} s = {:.2} GB/s ({:.1} Gb/s)",
                    got as f64 / 1e9,
                    s,
                    got as f64 / s / 1e9,
                    got as f64 * 8.0 / s / 1e9
                );
            }
        }
        _ => eprintln!(
            "usage: lanes_bench serve LISTEN FILE | read ADDR SIZE LANES INFLIGHT [ROUNDS]"
        ),
    }
}
