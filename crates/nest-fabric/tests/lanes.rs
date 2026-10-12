//! TCP lanes (ADR-048) over loopback: a dialling host and an accepting one
//! read from each other over the lanes the dialler opened.

use futures::FutureExt;
use futures::future::BoxFuture;
use nest_fabric::tcp::{Lanes, LanesConfig};
use nest_fabric::{ReadSource, ServeFile};
use nest_types::{FileId, Generation, NestError, NodeId};
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;

/// Serves files from a map; any other generation is stale.
struct Files(HashMap<FileId, (Generation, Arc<std::fs::File>)>);

impl ReadSource for Files {
    fn open_now(
        &self,
        file: FileId,
        generation: Generation,
    ) -> Option<Result<ServeFile, NestError>> {
        Some(match self.0.get(&file) {
            Some((g, f)) if *g == generation => Ok(ServeFile {
                file: f.clone(),
                guard: Box::new(()),
            }),
            Some(_) => Err(NestError::Stale),
            None => Err(NestError::NotFound),
        })
    }

    fn open(
        &self,
        file: FileId,
        generation: Generation,
    ) -> BoxFuture<'static, Result<ServeFile, NestError>> {
        let r = self.open_now(file, generation).unwrap();
        async move { r }.boxed()
    }
}

fn pattern(seed: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

fn files(dir: &std::path::Path, list: &[(u64, u64, Vec<u8>)]) -> Arc<dyn ReadSource> {
    std::fs::create_dir_all(dir).unwrap();
    let mut m = HashMap::new();
    for (id, g, data) in list {
        let p = dir.join(format!("{id}"));
        std::fs::File::create(&p).unwrap().write_all(data).unwrap();
        m.insert(
            FileId(*id),
            (Generation(*g), Arc::new(std::fs::File::open(&p).unwrap())),
        );
    }
    Arc::new(Files(m))
}

async fn pair(cfg: LanesConfig) -> (Arc<Lanes>, Arc<Lanes>) {
    let core = Lanes::start(
        NodeId(1),
        "lanes-test".into(),
        b"s".to_vec(),
        cfg.clone(),
        Some("127.0.0.1:0".parse().unwrap()),
    )
    .await
    .unwrap();
    let offgrid = Lanes::start(
        NodeId(8),
        "lanes-test".into(),
        b"s".to_vec(),
        cfg.clone(),
        None,
    )
    .await
    .unwrap();
    offgrid.dial(NodeId(1), core.local_addr().unwrap());
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while offgrid.connected(NodeId(1)) < cfg.per_peer || core.connected(NodeId(8)) < cfg.per_peer {
        assert!(
            std::time::Instant::now() < deadline,
            "lanes did not come up"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    (core, offgrid)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn lanes_carry_reads_both_ways() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = LanesConfig {
        per_peer: 4,
        chunk: 1 << 20,
        ..LanesConfig::default()
    };
    let (core, offgrid) = pair(cfg).await;
    let big = pattern(1, (5 << 20) + 12345);
    let core_src = files(&dir.path().join("c"), &[(7, 3, big.clone())]);
    let small = pattern(2, 4000);
    let off_src = files(&dir.path().join("o"), &[(9, 1, small.clone())]);
    core.set_source(Arc::downgrade(&core_src));
    offgrid.set_source(Arc::downgrade(&off_src));

    // The off-grid host reads the whole file, many chunks in flight.
    let reads: Vec<_> = (0..big.len().div_ceil(1 << 20))
        .map(|i| {
            offgrid.read(
                NodeId(1),
                FileId(7),
                Generation(3),
                (i << 20) as u64,
                1 << 20,
            )
        })
        .collect();
    let mut got = Vec::new();
    for r in futures::future::join_all(reads).await {
        got.extend_from_slice(&r.unwrap());
    }
    assert_eq!(got, big, "chunks differ");
    // The core host reads from the off-grid host over the lanes it dialled.
    let r = core
        .read(NodeId(8), FileId(9), Generation(1), 100, 1 << 20)
        .await
        .unwrap();
    assert_eq!(r, small[100..]);
    // Failures come back as such.
    assert_eq!(
        offgrid
            .read(NodeId(1), FileId(7), Generation(2), 0, 10)
            .await,
        Err(NestError::Stale)
    );
    assert_eq!(
        offgrid
            .read(NodeId(1), FileId(70), Generation(1), 0, 10)
            .await,
        Err(NestError::NotFound)
    );
    // Past the end: nothing.
    let r = offgrid
        .read(NodeId(1), FileId(7), Generation(3), big.len() as u64, 10)
        .await;
    assert_eq!(r.unwrap(), Vec::<u8>::new());
    assert!(
        core.stats
            .served_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
            >= big.len() as u64
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn lanes_come_back_after_the_accepting_host_restarts() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = LanesConfig {
        per_peer: 2,
        chunk: 1 << 20,
        ..LanesConfig::default()
    };
    let (core, offgrid) = pair(cfg.clone()).await;
    let data = pattern(3, 3000);
    let src = files(dir.path(), &[(5, 1, data.clone())]);
    core.set_source(Arc::downgrade(&src));
    assert_eq!(
        offgrid
            .read(NodeId(1), FileId(5), Generation(1), 0, 3000)
            .await
            .unwrap(),
        data
    );

    let addr = core.local_addr().unwrap();
    core.shutdown();
    drop(core);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        offgrid
            .read(NodeId(1), FileId(5), Generation(1), 0, 10)
            .await
            .is_err()
    );
    let core = Lanes::start(
        NodeId(1),
        "lanes-test".into(),
        b"s".to_vec(),
        cfg,
        Some(addr),
    )
    .await
    .unwrap();
    core.set_source(Arc::downgrade(&src));
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while offgrid.connected(NodeId(1)) < 2 {
        assert!(std::time::Instant::now() < deadline, "lanes not redialled");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        offgrid
            .read(NodeId(1), FileId(5), Generation(1), 0, 3000)
            .await
            .unwrap(),
        data
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_wrong_secret_gets_no_lane() {
    let core = Lanes::start(
        NodeId(1),
        "lanes-test".into(),
        b"right".to_vec(),
        LanesConfig::default(),
        Some("127.0.0.1:0".parse().unwrap()),
    )
    .await
    .unwrap();
    let other = Lanes::start(
        NodeId(8),
        "lanes-test".into(),
        b"wrong".to_vec(),
        LanesConfig::default(),
        None,
    )
    .await
    .unwrap();
    other.dial(NodeId(1), core.local_addr().unwrap());
    tokio::time::sleep(Duration::from_millis(800)).await;
    assert_eq!(other.connected(NodeId(1)), 0);
    assert_eq!(core.connected(NodeId(8)), 0);
}
