//! Remote writes, RDMA against the control connection, in one process over
//! this host's NIC (benchmarks/M12-REMOTE-WRITES.md).
//!
//!   write_bench [MIB] [SIZE_KIB]
//!
//! Two fabric nodes; node 2 sends MIB MiB of SIZE_KIB writes to node 1, at
//! 1, 8 and 32 in flight: over RDMA (the owner pulls them) and as request
//! bodies on the nest-rpc connection Raft uses. The owner's sink only
//! counts, so this measures the transports, not a disk. Meanwhile a probe
//! measures the round trip of a 64-byte call on that same control
//! connection, which is what heartbeats queue behind.

use bytes::Bytes;
use futures::FutureExt;
use futures::future::BoxFuture;
use nest_fabric::{Fabric, FabricConfig, WriteOp, WriteSink};
use nest_rpc::{Rpc, RpcConfig};
use nest_types::{Epoch, FileId, NestError, NodeId};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Accepts every write; touches the bytes once, as a pwrite would.
struct Discard;

impl WriteSink for Discard {
    fn write(&self, _: WriteOp, data: Bytes) -> BoxFuture<'static, Result<u32, NestError>> {
        let n = data.len() as u32;
        std::hint::black_box(data.iter().step_by(4096).fold(0u8, |a, b| a ^ b));
        async move { Ok(n) }.boxed()
    }
}

const BENCH: u8 = 200;
const PROBE: u8 = 201;

async fn node(id: u64) -> (Rpc, Arc<Fabric>) {
    let rpc = Rpc::bind(RpcConfig {
        node: NodeId(id),
        cluster: "write-bench".into(),
        secret: b"s".to_vec(),
        listen: "127.0.0.1:0".parse().unwrap(),
        connect_timeout: Duration::from_secs(2),
    })
    .await
    .unwrap();
    let f = Fabric::start(NodeId(id), FabricConfig::default(), rpc.clone())
        .unwrap()
        .expect("no RoCE rail on this host");
    (rpc, f)
}

fn op(i: u64, size: usize) -> WriteOp {
    WriteOp {
        file: FileId(5),
        epoch: Epoch(1),
        writer: (NodeId(2), 1),
        offset: i * size as u64,
        append: false,
    }
}

/// Control-call round trips while `load` runs: (p50, p99, max) in us.
async fn probe_while<F: std::future::Future<Output = ()>>(
    rpc: &Rpc,
    load: F,
) -> (f64, (u64, u64, u64)) {
    let stop = Arc::new(AtomicBool::new(false));
    let probe = {
        let (rpc, stop) = (rpc.clone(), stop.clone());
        tokio::spawn(async move {
            let mut v = Vec::new();
            while !stop.load(Ordering::Relaxed) {
                let t = Instant::now();
                rpc.call(
                    NodeId(1),
                    PROBE,
                    Bytes::from_static(&[0u8; 64]),
                    Duration::from_secs(5),
                )
                .await
                .unwrap();
                v.push(t.elapsed().as_micros() as u64);
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            v
        })
    };
    let t = Instant::now();
    load.await;
    let secs = t.elapsed().as_secs_f64();
    stop.store(true, Ordering::Relaxed);
    let mut v = probe.await.unwrap();
    v.sort();
    let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
    (secs, (q(0.5), q(0.99), *v.last().unwrap()))
}

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mib: u64 = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(4096);
    let size: usize = a
        .get(2)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1024)
        << 10;
    let (rpc1, f1) = node(1).await;
    let (rpc2, f2) = node(2).await;
    rpc1.set_peer(NodeId(2), rpc2.local_addr());
    rpc2.set_peer(NodeId(1), rpc1.local_addr());
    let sink: Arc<dyn WriteSink> = Arc::new(Discard);
    f1.set_sink(Arc::downgrade(&sink));
    rpc1.register(
        BENCH,
        Arc::new(|_: NodeId, b: Bytes| {
            let n = b.len() as u32;
            std::hint::black_box(b.iter().step_by(4096).fold(0u8, |a, b| a ^ b));
            async move { Ok(Bytes::copy_from_slice(&n.to_le_bytes())) }.boxed()
                as BoxFuture<'static, Result<Bytes, String>>
        }),
    );
    rpc1.register(
        PROBE,
        Arc::new(|_: NodeId, _: Bytes| {
            async move { Ok(Bytes::new()) }.boxed() as BoxFuture<'static, Result<Bytes, String>>
        }),
    );
    let data = Bytes::from(vec![7u8; size]);
    let count = (mib << 20) / size as u64;
    // Link up and warm both paths.
    f2.write(NodeId(1), op(0, size), &data).await.unwrap();
    rpc2.call(NodeId(1), BENCH, data.clone(), Duration::from_secs(5))
        .await
        .unwrap();

    let (_, idle) = probe_while(&rpc2, tokio::time::sleep(Duration::from_millis(500))).await;
    println!(
        "control call, idle: p50 {} us, p99 {} us, max {} us",
        idle.0, idle.1, idle.2
    );
    println!("{size} B writes, {mib} MiB each run");
    for inflight in [1usize, 8, 32] {
        let (secs, lat) = probe_while(&rpc2, async {
            let next = Arc::new(std::sync::atomic::AtomicU64::new(0));
            let tasks: Vec<_> = (0..inflight)
                .map(|_| {
                    let (f2, data, next) = (f2.clone(), data.clone(), next.clone());
                    tokio::spawn(async move {
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= count {
                                return;
                            }
                            f2.write(NodeId(1), op(i, size), &data).await.unwrap();
                        }
                    })
                })
                .collect();
            for t in tasks {
                t.await.unwrap();
            }
        })
        .await;
        println!(
            "  RDMA, {inflight:>2} in flight: {:6.2} GB/s; control call p50 {} us, p99 {} us, max {} us",
            (count * size as u64) as f64 / secs / 1e9,
            lat.0,
            lat.1,
            lat.2
        );
        let (secs, lat) = probe_while(&rpc2, async {
            let next = Arc::new(std::sync::atomic::AtomicU64::new(0));
            let tasks: Vec<_> = (0..inflight)
                .map(|_| {
                    let (rpc2, data, next) = (rpc2.clone(), data.clone(), next.clone());
                    tokio::spawn(async move {
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= count {
                                return;
                            }
                            rpc2.call(NodeId(1), BENCH, data.clone(), Duration::from_secs(10))
                                .await
                                .unwrap();
                        }
                    })
                })
                .collect();
            for t in tasks {
                t.await.unwrap();
            }
        })
        .await;
        println!(
            "  TCP,  {inflight:>2} in flight: {:6.2} GB/s; control call p50 {} us, p99 {} us, max {} us",
            (count * size as u64) as f64 / secs / 1e9,
            lat.0,
            lat.1,
            lat.2
        );
    }
    f1.shutdown();
    f2.shutdown();
}
