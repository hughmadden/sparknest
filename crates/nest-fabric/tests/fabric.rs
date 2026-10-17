//! The read protocol between two fabric instances in one process (two
//! "nodes" over RDMA loopback on this machine's NIC). Skips without RoCE.

use futures::FutureExt;
use futures::future::BoxFuture;
use nest_fabric::{Fabric, FabricConfig, ReadSource, ServeFile, WriteError, WriteOp, WriteSink};
use nest_rpc::{Rpc, RpcConfig};
use nest_types::{Epoch, FileId, Generation, NestError, NodeId};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

fn byte(i: u64) -> u8 {
    (i.wrapping_mul(2654435761) >> 13) as u8
}

const SIZE: u64 = 64 << 20;

/// A 64 MiB file of `byte(i)` (written once, reused: the fabric serves
/// real files through io_uring).
fn pattern_file() -> &'static PathBuf {
    static P: OnceLock<PathBuf> = OnceLock::new();
    P.get_or_init(|| {
        let p = std::env::temp_dir().join("nest-fabric-pattern-v1-64m.bin");
        if std::fs::metadata(&p).map(|m| m.len()).ok() != Some(SIZE) {
            let tmp = p.with_extension(format!("{}.tmp", std::process::id()));
            let data: Vec<u8> = (0..SIZE).map(byte).collect();
            std::fs::write(&tmp, data).unwrap();
            std::fs::rename(&tmp, &p).unwrap();
        }
        p
    })
}

fn open_pattern(file: FileId, generation: Generation) -> Result<ServeFile, NestError> {
    if file != FileId(7) {
        return Err(NestError::NotFound);
    }
    if generation != Generation(3) {
        return Err(NestError::Stale);
    }
    Ok(ServeFile {
        file: Arc::new(std::fs::File::open(pattern_file()).unwrap()),
        guard: Box::new(()),
    })
}

/// Serves "file" 7, generation 3 from the pattern file, already open.
struct Pattern;

impl ReadSource for Pattern {
    fn open_now(
        &self,
        file: FileId,
        generation: Generation,
    ) -> Option<Result<ServeFile, NestError>> {
        Some(open_pattern(file, generation))
    }

    fn open(
        &self,
        file: FileId,
        generation: Generation,
    ) -> BoxFuture<'static, Result<ServeFile, NestError>> {
        async move { open_pattern(file, generation) }.boxed()
    }
}

async fn node(id: u64, cfg: FabricConfig) -> Option<(Rpc, Arc<Fabric>)> {
    let rpc = Rpc::bind(RpcConfig {
        node: NodeId(id),
        cluster: "fabric-test".into(),
        secret: b"s".to_vec(),
        listen: "127.0.0.1:0".parse().unwrap(),
        connect_timeout: Duration::from_secs(2),
    })
    .await
    .unwrap();
    let f = Fabric::start(NodeId(id), cfg, rpc.clone()).unwrap()?;
    Some((rpc, f))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn read_protocol_over_loopback() {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_env_filter("warn")
        .try_init();
    let cfg = FabricConfig {
        chunk: 4 << 20,
        client_slots: 16,
        server_slots: 8,
        window: 8,
        devices: vec![],
        ..FabricConfig::default()
    };
    let Some((rpc1, f1)) = node(1, cfg.clone()).await else {
        return;
    };
    let (rpc2, f2) = node(2, cfg).await.unwrap();
    rpc2.set_peer(NodeId(1), rpc1.local_addr());
    rpc1.set_peer(NodeId(2), rpc2.local_addr());
    let src: Arc<dyn ReadSource> = Arc::new(Pattern);
    f1.set_source(Arc::downgrade(&src));

    // Exact bytes at assorted offsets and lengths, including the tail.
    for (off, len) in [
        (0u64, 4096usize),
        (12345, 1 << 20),
        (SIZE - 1000, 4 << 20),
        (SIZE, 4096),
        (3 << 20, 4 << 20),
    ] {
        let b = f2
            .read(NodeId(1), FileId(7), Generation(3), off, len)
            .await
            .unwrap();
        let want = (SIZE.saturating_sub(off) as usize).min(len);
        assert_eq!(b.len(), want);
        assert!(
            b.as_slice()
                .iter()
                .enumerate()
                .all(|(k, v)| *v == byte(off + k as u64)),
            "mismatch at {off}"
        );
    }
    // Errors travel back as the right kind.
    assert_eq!(
        f2.read(NodeId(1), FileId(7), Generation(2), 0, 16)
            .await
            .err(),
        Some(NestError::Stale)
    );
    assert_eq!(
        f2.read(NodeId(1), FileId(8), Generation(3), 0, 16)
            .await
            .err(),
        Some(NestError::NotFound)
    );

    // Many concurrent reads (more than slots and window): bounded, correct.
    let t = Instant::now();
    let reads = 256u64;
    let mut tasks = Vec::new();
    for i in 0..reads {
        let f2 = f2.clone();
        tasks.push(tokio::spawn(async move {
            let off = (i * (4 << 20)) % SIZE;
            let b = f2
                .read(NodeId(1), FileId(7), Generation(3), off, 4 << 20)
                .await
                .unwrap();
            assert_eq!(b.as_slice()[4095], byte(off + 4095));
            b.len()
        }));
    }
    let mut total = 0usize;
    for t in tasks {
        total += t.await.unwrap();
    }
    let secs = t.elapsed().as_secs_f64();
    eprintln!(
        "loopback: {} MiB in {:.3}s = {:.2} GB/s",
        total >> 20,
        secs,
        total as f64 / secs / 1e9
    );
    assert_eq!(
        f2.stats.reads.load(std::sync::atomic::Ordering::Relaxed),
        5 + reads
    );
    f1.shutdown();
    f2.shutdown();
}

/// Nothing is open: every request is checked off-thread first, slowly (a
/// busy metadata store), before its io_uring read.
struct Slow;

impl ReadSource for Slow {
    fn open_now(&self, _: FileId, _: Generation) -> Option<Result<ServeFile, NestError>> {
        None
    }

    fn open(
        &self,
        file: FileId,
        generation: Generation,
    ) -> BoxFuture<'static, Result<ServeFile, NestError>> {
        async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            open_pattern(file, generation)
        }
        .boxed()
    }
}

/// A reader that gives up on a read (readahead cancelling a chunk) must
/// not let the next read of that landing slot take the late answer as its
/// own: that surfaced as short reads (a tail chunk's length) and wrong bytes.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn cancelled_reads_never_answer_later_ones() {
    let cfg = FabricConfig {
        chunk: 4 << 20,
        client_slots: 8,
        server_slots: 8,
        window: 8,
        devices: vec![],
        ..FabricConfig::default()
    };
    let Some((rpc1, f1)) = node(1, cfg.clone()).await else {
        return;
    };
    let (rpc2, f2) = node(2, cfg).await.unwrap();
    rpc2.set_peer(NodeId(1), rpc1.local_addr());
    rpc1.set_peer(NodeId(2), rpc2.local_addr());
    let src: Arc<dyn ReadSource> = Arc::new(Slow);
    f1.set_source(Arc::downgrade(&src));
    // Warm the link up.
    f2.read(NodeId(1), FileId(7), Generation(3), 0, 4096)
        .await
        .unwrap();
    for round in 0..4 {
        // Tail reads (1000 bytes each) abandoned while the peer works on them.
        for _ in 0..8 {
            let _ = tokio::time::timeout(
                Duration::from_millis(2),
                f2.read(NodeId(1), FileId(7), Generation(3), SIZE - 1000, 4 << 20),
            )
            .await;
        }
        // Full chunks right after, reusing the same slots.
        let mut tasks = Vec::new();
        for i in 0..16u64 {
            let f2 = f2.clone();
            tasks.push(tokio::spawn(async move {
                let off = i * (4 << 20);
                let b = f2
                    .read(NodeId(1), FileId(7), Generation(3), off, 4 << 20)
                    .await
                    .unwrap();
                assert_eq!(b.len(), 4 << 20, "round {round} read {i}: short");
                assert!(
                    b.as_slice()
                        .iter()
                        .enumerate()
                        .step_by(4099)
                        .all(|(k, v)| *v == byte(off + k as u64)),
                    "round {round} read {i}: wrong bytes"
                );
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
    }
}

/// Every read is served through the io_uring path: bursts of page-sized
/// reads (their own staging tier), the tail, errors, and chunks.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn reads_are_served_through_io_uring() {
    let cfg = FabricConfig {
        chunk: 4 << 20,
        client_slots: 16,
        server_slots: 8,
        window: 8,
        devices: vec![],
        ..FabricConfig::default()
    };
    let Some((rpc1, f1)) = node(1, cfg.clone()).await else {
        return;
    };
    let (rpc2, f2) = node(2, cfg).await.unwrap();
    rpc2.set_peer(NodeId(1), rpc1.local_addr());
    rpc1.set_peer(NodeId(2), rpc2.local_addr());
    let src: Arc<dyn ReadSource> = Arc::new(Pattern);
    f1.set_source(Arc::downgrade(&src));
    let fast = || {
        f1.stats
            .served_fast
            .load(std::sync::atomic::Ordering::Relaxed)
    };
    let before = fast();
    let mut tasks = Vec::new();
    for i in 0..512u64 {
        let f2 = f2.clone();
        tasks.push(tokio::spawn(async move {
            let off = (i * 7_777_777) % (SIZE - 4096);
            let b = f2
                .read(NodeId(1), FileId(7), Generation(3), off, 4096)
                .await
                .unwrap();
            assert_eq!(b.len(), 4096);
            assert!(
                b.as_slice()
                    .iter()
                    .enumerate()
                    .all(|(k, v)| *v == byte(off + k as u64))
            );
        }));
    }
    for t in tasks {
        t.await.unwrap();
    }
    // Page-sized reads have their own tier (1024 staging slots): even a
    // burst is served fast.
    let concurrent = fast() - before;
    assert_eq!(concurrent, 512, "only {concurrent} of 512 served fast");
    // One at a time, every small read is.
    let n = fast();
    for i in 0..64u64 {
        let off = i * 1_000_003;
        let b = f2
            .read(NodeId(1), FileId(7), Generation(3), off, 4096)
            .await
            .unwrap();
        assert_eq!(b.as_slice()[4095], byte(off + 4095));
    }
    assert_eq!(fast() - n, 64);
    // The tail, errors and a large read behave as before.
    let b = f2
        .read(NodeId(1), FileId(7), Generation(3), SIZE - 100, 4096)
        .await
        .unwrap();
    assert_eq!(b.len(), 100);
    assert_eq!(
        f2.read(NodeId(1), FileId(7), Generation(2), 0, 16)
            .await
            .err(),
        Some(NestError::Stale)
    );
    let n = fast();
    let b = f2
        .read(NodeId(1), FileId(7), Generation(3), 0, 4 << 20)
        .await
        .unwrap();
    assert_eq!(b.len(), 4 << 20);
    assert!(
        b.as_slice()
            .iter()
            .enumerate()
            .step_by(4099)
            .all(|(k, v)| *v == byte(k as u64))
    );
    assert_eq!(fast(), n + 1);
}

/// Page reads, kernel-sized reads and chunks at once: each takes its tier,
/// lengths come back right (the length now travels as the completion's
/// byte count), and bytes land in the right slots.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn mixed_sizes_share_the_tiers() {
    let cfg = FabricConfig {
        chunk: 4 << 20,
        client_slots: 8,
        server_slots: 4,
        small_tiers: vec![
            nest_fabric::Tier {
                size: 4 << 10,
                landing: 16,
                staging: 8,
            },
            nest_fabric::Tier {
                size: 128 << 10,
                landing: 8,
                staging: 4,
            },
        ],
        window: 16,
        devices: vec![],
        ..FabricConfig::default()
    };
    let Some((rpc1, f1)) = node(1, cfg.clone()).await else {
        return;
    };
    let (rpc2, f2) = node(2, cfg).await.unwrap();
    rpc2.set_peer(NodeId(1), rpc1.local_addr());
    rpc1.set_peer(NodeId(2), rpc2.local_addr());
    let src: Arc<dyn ReadSource> = Arc::new(Pattern);
    f1.set_source(Arc::downgrade(&src));
    let mut tasks = Vec::new();
    for i in 0..600u64 {
        let f2 = f2.clone();
        tasks.push(tokio::spawn(async move {
            let len = match i % 3 {
                0 => 4096,
                1 => 100 << 10,
                _ => 4 << 20,
            };
            let off = (i * 3_333_331) % (SIZE - len as u64);
            let b = f2
                .read(NodeId(1), FileId(7), Generation(3), off, len)
                .await
                .unwrap();
            assert_eq!(b.len(), len, "read {i}");
            assert!(
                b.as_slice()
                    .iter()
                    .enumerate()
                    .step_by(997)
                    .all(|(k, v)| *v == byte(off + k as u64)),
                "read {i}: wrong bytes"
            );
        }));
    }
    for t in tasks {
        t.await.unwrap();
    }
    // A tail shorter than the request, and an empty read at the end.
    let b = f2
        .read(NodeId(1), FileId(7), Generation(3), SIZE - 10, 4096)
        .await
        .unwrap();
    assert_eq!(b.len(), 10);
    let b = f2
        .read(NodeId(1), FileId(7), Generation(3), SIZE, 4096)
        .await
        .unwrap();
    assert_eq!(b.len(), 0);
}

/// A reservation's read answers through its callback, on the completion
/// thread: right bytes and lengths, the tail, errors; `None` before a link
/// exists.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reservations_answer_by_callback() {
    let Some((rpc1, f1)) = node(1, FabricConfig::default()).await else {
        return;
    };
    let (rpc2, f2) = node(2, FabricConfig::default()).await.unwrap();
    rpc2.set_peer(NodeId(1), rpc1.local_addr());
    rpc1.set_peer(NodeId(2), rpc2.local_addr());
    let src: Arc<dyn ReadSource> = Arc::new(Pattern);
    f1.set_source(Arc::downgrade(&src));
    assert!(f2.reserve(NodeId(1), 4096).is_none(), "no link yet");
    f2.read(NodeId(1), FileId(7), Generation(3), 0, 16)
        .await
        .unwrap();
    async fn get(
        f: &Arc<Fabric>,
        generation: u64,
        off: u64,
        len: usize,
    ) -> Result<Vec<u8>, NestError> {
        let r = f
            .reserve(NodeId(1), len)
            .expect("link, slots and budget free");
        let (tx, rx) = tokio::sync::oneshot::channel();
        r.send(
            FileId(7),
            Generation(generation),
            off,
            Box::new(move |res| {
                let _ = tx.send(res.map(|b| b.to_vec()));
            }),
        );
        rx.await.unwrap()
    }
    let mut tasks = Vec::new();
    for t in 0..8u64 {
        let f = f2.clone();
        tasks.push(tokio::spawn(async move {
            for i in 0..200u64 {
                let off = ((t * 1000 + i) * 104_729) % (SIZE - 8192);
                let b = get(&f, 3, off, 4096).await.unwrap();
                assert_eq!(b.len(), 4096);
                assert!(
                    b.iter()
                        .enumerate()
                        .all(|(k, v)| *v == byte(off + k as u64))
                );
            }
        }));
    }
    for t in tasks {
        t.await.unwrap();
    }
    assert_eq!(get(&f2, 3, SIZE - 5, 4096).await.unwrap().len(), 5);
    assert_eq!(get(&f2, 2, 0, 16).await.err(), Some(NestError::Stale));
    assert!(f2.stats.read_now.load(std::sync::atomic::Ordering::Relaxed) >= 1601);
}

/// Replacing a link must release the old lanes' QPs and rings once their
/// grace passes; before the fix every re-link kept them for the life of the
/// process (130k QPs on a busy host after one large replication).
#[tokio::test(flavor = "multi_thread")]
async fn replaced_links_release_their_lanes() {
    let cfg = FabricConfig {
        chunk: 1 << 20,
        client_slots: 4,
        server_slots: 4,
        window: 4,
        devices: vec![],
        lane_grace: Duration::ZERO,
        ..FabricConfig::default()
    };
    let Some((rpc1, f1)) = node(1, cfg.clone()).await else {
        return;
    };
    let (rpc2, f2) = node(2, cfg).await.unwrap();
    rpc2.set_peer(NodeId(1), rpc1.local_addr());
    rpc1.set_peer(NodeId(2), rpc2.local_addr());
    let src: Arc<dyn ReadSource> = Arc::new(Pattern);
    f1.set_source(Arc::downgrade(&src));
    let read = || f2.read(NodeId(1), FileId(7), Generation(3), 0, 4096);
    read().await.unwrap();
    let steady = f2.lane_count();
    for _ in 0..50 {
        f2.drop_link(NodeId(1));
        read().await.unwrap();
    }
    f2.drop_link(NodeId(1));
    f1.drop_link(NodeId(2));
    assert_eq!(f2.lane_count(), 0, "client lanes leaked");
    assert_eq!(f1.lane_count(), 0, "server lanes leaked");
    read().await.unwrap();
    assert_eq!(f2.lane_count(), steady);
}

/// A peer that cannot be linked is not re-dialled by every read: after a
/// failed setup, reads fail at once until the backoff passes, and the wait
/// doubles. Before this, each read dialled again (~5,000 setups a minute
/// against hosts that could not pair).
#[tokio::test(flavor = "multi_thread")]
async fn failed_link_setups_back_off() {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_env_filter("warn")
        .try_init();
    let cfg = FabricConfig {
        chunk: 1 << 20,
        client_slots: 4,
        server_slots: 4,
        window: 4,
        devices: vec![],
        link_retry: Duration::from_millis(300),
        link_retry_max: Duration::from_secs(5),
        ..FabricConfig::default()
    };
    let Some((rpc1, f1)) = node(1, cfg.clone()).await else {
        return;
    };
    let (rpc2, f2) = node(2, cfg).await.unwrap();
    rpc1.set_peer(NodeId(2), rpc2.local_addr());
    let src: Arc<dyn ReadSource> = Arc::new(Pattern);
    f1.set_source(Arc::downgrade(&src));
    // Peer 1 unreachable: nothing listens there.
    let closed = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    rpc2.set_peer(NodeId(1), closed.local_addr().unwrap());
    drop(closed);
    let read = || f2.read(NodeId(1), FileId(7), Generation(3), 0, 4096);

    assert!(read().await.is_err());
    assert_eq!(f2.link_attempts(), 1);
    for _ in 0..200 {
        assert!(read().await.is_err());
    }
    assert_eq!(f2.link_attempts(), 1, "reads re-dialled during the backoff");

    tokio::time::sleep(Duration::from_millis(350)).await;
    assert!(read().await.is_err());
    assert_eq!(f2.link_attempts(), 2);
    // The wait doubled: still backing off after the first interval.
    tokio::time::sleep(Duration::from_millis(350)).await;
    assert!(read().await.is_err());
    assert_eq!(f2.link_attempts(), 2, "the wait did not double");

    // Reachable again: the next attempt after the backoff links and reads.
    rpc2.set_peer(NodeId(1), rpc1.local_addr());
    tokio::time::sleep(Duration::from_millis(350)).await;
    read().await.unwrap();
    assert_eq!(f2.link_attempts(), 3);
    read().await.unwrap();
    assert_eq!(f2.link_attempts(), 3);
}

/// ADR-051: every byte read counts once, as sent on the serving host's
/// port and received on the reader's; requests count as control messages.
#[tokio::test(flavor = "multi_thread")]
async fn port_counters_match_the_bytes_moved() {
    let cfg = FabricConfig {
        chunk: 1 << 20,
        client_slots: 4,
        server_slots: 4,
        window: 4,
        devices: vec![],
        ..FabricConfig::default()
    };
    let Some((rpc1, f1)) = node(1, cfg.clone()).await else {
        return;
    };
    let (rpc2, f2) = node(2, cfg).await.unwrap();
    rpc2.set_peer(NodeId(1), rpc1.local_addr());
    rpc1.set_peer(NodeId(2), rpc2.local_addr());
    let src: Arc<dyn ReadSource> = Arc::new(Pattern);
    f1.set_source(Arc::downgrade(&src));
    let ports = f2.ports();
    assert!(!ports.is_empty());
    assert!(
        ports
            .iter()
            .all(|p| p.path_mtu >= 1024 && !p.netdev.is_empty())
    );

    let total = |f: &Fabric| {
        let mut v = Vec::new();
        f.port_counters(&mut v);
        v.iter().fold(
            nest_fabric::counters::CounterValues::default(),
            |mut a, c| {
                a.bytes_sent += c.bytes_sent;
                a.bytes_received += c.bytes_received;
                a.ops_sent += c.ops_sent;
                a.ops_received += c.ops_received;
                a.packets_sent += c.packets_sent;
                a.packets_received += c.packets_received;
                a
            },
        )
    };
    // Link up first (its setup sends nothing over RDMA).
    f2.read(NodeId(1), FileId(7), Generation(3), 0, 4096)
        .await
        .unwrap();
    let (s0, r0) = (total(&f1), total(&f2));
    let sizes = [1 << 20, 1 << 20, 4096, 100_000, 1];
    for (i, n) in sizes.iter().enumerate() {
        let b = f2
            .read(NodeId(1), FileId(7), Generation(3), (i as u64) << 20, *n)
            .await
            .unwrap();
        assert_eq!(b.len(), *n);
    }
    let bytes: u64 = sizes.iter().map(|n| *n as u64).sum();
    let (s1, r1) = (total(&f1), total(&f2));
    let mtu = ports[0].path_mtu as u64;
    let packets: u64 = sizes.iter().map(|n| (*n as u64).div_ceil(mtu)).sum();
    // Serving host: the data out, the requests in.
    assert_eq!(s1.bytes_sent - s0.bytes_sent, bytes);
    assert_eq!(s1.ops_sent - s0.ops_sent, sizes.len() as u64);
    assert_eq!(s1.packets_sent - s0.packets_sent, packets);
    assert_eq!(s1.ops_received - s0.ops_received, sizes.len() as u64);
    // Reader: the data in, one request message out per read.
    assert_eq!(r1.bytes_received - r0.bytes_received, bytes);
    assert_eq!(r1.packets_received - r0.packets_received, packets);
    assert_eq!(r1.ops_sent - r0.ops_sent, sizes.len() as u64);
    assert_eq!(
        r1.bytes_sent - r0.bytes_sent,
        s1.bytes_received - s0.bytes_received,
        "request bytes sent and received differ"
    );
}

// ------------------------------------------------------------ writes (ADR-057)

/// Owns "file" 5 in epoch 4: applies writes the way the data layer's owner
/// does (epoch check, append offset chosen here), optionally slowly, and
/// records what it was handed.
struct Owner {
    file: std::fs::File,
    path: PathBuf,
    end: std::sync::Mutex<()>,
    delay: Duration,
    applied: std::sync::atomic::AtomicU64,
    /// Most writes being applied at once.
    busy: std::sync::atomic::AtomicU64,
    peak: std::sync::atomic::AtomicU64,
}

impl Owner {
    fn new(dir: &std::path::Path, delay: Duration) -> Arc<Owner> {
        let path = dir.join("owned");
        Arc::new(Owner {
            file: std::fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .read(true)
                .write(true)
                .open(&path)
                .unwrap(),
            path,
            end: Default::default(),
            delay,
            applied: Default::default(),
            busy: Default::default(),
            peak: Default::default(),
        })
    }
}

struct Shared(Arc<Owner>);

impl WriteSink for Shared {
    fn write(&self, op: WriteOp, data: bytes::Bytes) -> BoxFuture<'static, Result<u32, NestError>> {
        let me = self.0.clone();
        async move {
            use std::os::unix::fs::FileExt;
            use std::sync::atomic::Ordering::SeqCst;
            if op.file != FileId(5) {
                return Err(NestError::NotFound);
            }
            if op.epoch != Epoch(4) {
                return Err(NestError::Stale);
            }
            let now = me.busy.fetch_add(1, SeqCst) + 1;
            me.peak.fetch_max(now, SeqCst);
            if !me.delay.is_zero() {
                // Holds the staging slot (in `data`) meanwhile.
                tokio::time::sleep(me.delay).await;
            }
            {
                let _end = me.end.lock().unwrap();
                let off = if op.append {
                    me.file.metadata().unwrap().len()
                } else {
                    op.offset
                };
                me.file.write_all_at(&data, off).unwrap();
            }
            me.busy.fetch_sub(1, SeqCst);
            me.applied.fetch_add(1, SeqCst);
            Ok(data.len() as u32)
        }
        .boxed()
    }
}

fn wop(writer: u64, offset: u64, append: bool) -> WriteOp {
    WriteOp {
        file: FileId(5),
        epoch: Epoch(4),
        writer: (NodeId(writer), 1),
        offset,
        append,
    }
}

async fn linked(cfg: FabricConfig, ids: &[u64]) -> Option<Vec<(Rpc, Arc<Fabric>)>> {
    let mut v = Vec::new();
    for id in ids {
        v.push(node(*id, cfg.clone()).await?);
    }
    for (rpc, _) in &v {
        for (other, _) in &v {
            if rpc.local_addr() != other.local_addr() {
                rpc.set_peer(other.node(), other.local_addr());
            }
        }
    }
    Some(v)
}

/// A remote write lands with exactly its bytes, at its offset, across the
/// write tiers (a page, 1 MiB, an odd size); the owner's errors come back
/// as final; nothing is applied twice.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn writes_land_with_the_right_bytes() {
    let Some(n) = linked(FabricConfig::default(), &[1, 2]).await else {
        return;
    };
    let (f1, f2) = (n[0].1.clone(), n[1].1.clone());
    let dir = tempfile::tempdir().unwrap();
    let owner = Owner::new(dir.path(), Duration::ZERO);
    let sink: Arc<dyn WriteSink> = Arc::new(Shared(owner.clone()));
    f1.set_sink(Arc::downgrade(&sink));
    assert_eq!(f2.max_write(), 1 << 20);
    let mut want = vec![0u8; 3 << 20];
    let mut writes = 0u64;
    for (off, len) in [
        (0u64, 1usize << 20),
        (1 << 20, 4096),
        ((1 << 20) + 4096, 100_000),
        (2 << 20, 1 << 20),
        (5, 1),
    ] {
        let data: Vec<u8> = (0..len as u64).map(|i| byte(i + off * 7)).collect();
        let r = f2.write(NodeId(1), wop(2, off, false), &data).await;
        assert_eq!(r, Ok(len as u32), "write at {off}");
        want[off as usize..off as usize + len].copy_from_slice(&data);
        writes += 1;
    }
    assert!(f2.can_write(NodeId(1)));
    assert_eq!(f2.link_protocol(NodeId(1)), Some(nest_fabric::PROTOCOL));
    assert_eq!(std::fs::read(&owner.path).unwrap(), want);
    // The owner's refusals are final.
    let stale = WriteOp {
        epoch: Epoch(3),
        ..wop(2, 0, false)
    };
    assert_eq!(
        f2.write(NodeId(1), stale, b"x").await,
        Err(WriteError::Failed(NestError::Stale))
    );
    let other = WriteOp {
        file: FileId(6),
        ..wop(2, 0, false)
    };
    assert_eq!(
        f2.write(NodeId(1), other, b"x").await,
        Err(WriteError::Failed(NestError::NotFound))
    );
    // Too large for RDMA: unsent (the caller uses TCP).
    assert!(matches!(
        f2.write(NodeId(1), wop(2, 0, false), &vec![0u8; (1 << 20) + 1])
            .await,
        Err(WriteError::Unsent(_))
    ));
    use std::sync::atomic::Ordering::Relaxed;
    assert_eq!(owner.applied.load(Relaxed), writes);
    assert_eq!(f2.stats.writes.load(Relaxed), writes);
    assert_eq!(f1.stats.pulled.load(Relaxed), writes);
    // Reads still work on the same link.
    let src: Arc<dyn ReadSource> = Arc::new(Pattern);
    f1.set_source(Arc::downgrade(&src));
    let b = f2
        .read(NodeId(1), FileId(7), Generation(3), 100, 4096)
        .await
        .unwrap();
    assert_eq!(b.as_slice()[0], byte(100));
}

/// Appends from two writers at once: the owner picks each offset, so every
/// record lands whole, once, and none overlap.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_appends_from_two_writers() {
    let Some(n) = linked(FabricConfig::default(), &[1, 2, 3]).await else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let owner = Owner::new(dir.path(), Duration::ZERO);
    let sink: Arc<dyn WriteSink> = Arc::new(Shared(owner.clone()));
    n[0].1.set_sink(Arc::downgrade(&sink));
    let rec = 10_000usize;
    let per = 100usize;
    let mut tasks = Vec::new();
    for w in [1usize, 2] {
        for t in 0..4 {
            let f = n[w].1.clone();
            tasks.push(tokio::spawn(async move {
                for i in 0..per / 4 {
                    // Record: writer, then its sequence number, repeated.
                    let tag = [w as u8, (t * (per / 4) + i) as u8];
                    let data: Vec<u8> = tag.iter().copied().cycle().take(rec).collect();
                    let r = f.write(NodeId(1), wop(w as u64 + 1, 0, true), &data).await;
                    assert_eq!(r, Ok(rec as u32));
                }
            }));
        }
    }
    for t in tasks {
        t.await.unwrap();
    }
    let got = std::fs::read(&owner.path).unwrap();
    assert_eq!(got.len(), 2 * per * rec, "lost or duplicated appends");
    let mut seen: Vec<(u8, u8)> = got
        .chunks(rec)
        .map(|c| {
            assert!(
                c.chunks(2).all(|p| p == &c[..2]),
                "an append was split or overlapped"
            );
            (c[0], c[1])
        })
        .collect();
    seen.sort();
    let mut want: Vec<(u8, u8)> = [1u8, 2]
        .iter()
        .flat_map(|w| (0..per as u8).map(move |i| (*w, i)))
        .collect();
    want.sort();
    assert_eq!(seen, want);
}

/// An owner with every write staging slot busy makes further writers wait
/// for one (no per-writer reservation, no failure).
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn writers_wait_for_the_owners_staging_slots() {
    let cfg = FabricConfig {
        write_tiers: vec![nest_fabric::Tier {
            size: 1 << 20,
            landing: 32,
            staging: 2,
        }],
        ..FabricConfig::default()
    };
    let Some(n) = linked(cfg, &[1, 2, 3]).await else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let owner = Owner::new(dir.path(), Duration::from_millis(40));
    let sink: Arc<dyn WriteSink> = Arc::new(Shared(owner.clone()));
    n[0].1.set_sink(Arc::downgrade(&sink));
    let writes = 24u64;
    let t0 = Instant::now();
    let mut tasks = Vec::new();
    for i in 0..writes {
        let f = n[1 + (i % 2) as usize].1.clone();
        tasks.push(tokio::spawn(async move {
            let data = vec![i as u8; 256 << 10];
            f.write(NodeId(1), wop(2, i * (256 << 10), false), &data)
                .await
        }));
    }
    for t in tasks {
        assert_eq!(t.await.unwrap(), Ok(256 << 10));
    }
    use std::sync::atomic::Ordering::SeqCst;
    // Two slots: never more than two applied at once, so it took at least
    // writes / 2 delays.
    assert!(owner.peak.load(SeqCst) <= 2, "{}", owner.peak.load(SeqCst));
    assert!(t0.elapsed() >= Duration::from_millis(40) * (writes as u32 / 2));
    assert!(n[0].1.stats.pull_waits.load(SeqCst) > 0, "nothing waited");
    let got = std::fs::read(&owner.path).unwrap();
    assert_eq!(got.len(), (writes as usize) << 18);
    for (i, c) in got.chunks(256 << 10).enumerate() {
        assert!(c.iter().all(|b| *b == i as u8), "write {i}");
    }
}

/// A host on protocol 2 (before writes) still pairs for reads; writes to
/// it are refused as unsent (the caller uses TCP), in both directions.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn protocol_2_peers_pair_for_reads_but_not_writes() {
    let Some((rpc1, old)) = node(
        1,
        FabricConfig {
            protocol: 2,
            ..FabricConfig::default()
        },
    )
    .await
    else {
        return;
    };
    let (rpc2, new) = node(2, FabricConfig::default()).await.unwrap();
    rpc2.set_peer(NodeId(1), rpc1.local_addr());
    rpc1.set_peer(NodeId(2), rpc2.local_addr());
    let src: Arc<dyn ReadSource> = Arc::new(Pattern);
    old.set_source(Arc::downgrade(&src));
    new.set_source(Arc::downgrade(&src));
    let dir = tempfile::tempdir().unwrap();
    let owner = Owner::new(dir.path(), Duration::ZERO);
    let sink: Arc<dyn WriteSink> = Arc::new(Shared(owner.clone()));
    old.set_sink(Arc::downgrade(&sink));
    new.set_sink(Arc::downgrade(&sink));
    // The new host dials the old: it falls back to 2 and reads.
    let b = new
        .read(NodeId(1), FileId(7), Generation(3), 0, 4096)
        .await
        .unwrap();
    assert_eq!(b.as_slice()[4095], byte(4095));
    assert_eq!(new.link_protocol(NodeId(1)), Some(2));
    assert!(!new.can_write(NodeId(1)));
    assert!(matches!(
        new.write(NodeId(1), wop(2, 0, false), b"x").await,
        Err(WriteError::Unsent(_))
    ));
    assert_eq!(old.max_write(), 0);
    // The old host dials the new (a fresh link): protocol 2 both ways.
    old.drop_link(NodeId(2));
    new.drop_link(NodeId(1));
    let b = old
        .read(NodeId(2), FileId(7), Generation(3), 4096, 4096)
        .await
        .unwrap();
    assert_eq!(b.as_slice()[0], byte(4096));
    assert_eq!(new.link_protocol(NodeId(1)), Some(2));
    assert!(matches!(
        new.write(NodeId(1), wop(2, 0, false), b"x").await,
        Err(WriteError::Unsent(_))
    ));
    assert_eq!(
        owner.applied.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "a write went over a protocol-2 link"
    );
}

/// A write whose owner never answers fails as an unknown outcome (never
/// resent), and its link is retired; the outgoing slot is not reused.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn unanswered_writes_fail_as_unknown() {
    let cfg = FabricConfig {
        write_timeout: Duration::from_millis(500),
        ..FabricConfig::default()
    };
    let Some(n) = linked(cfg, &[1, 2]).await else {
        return;
    };
    struct Never;
    impl WriteSink for Never {
        fn write(&self, _: WriteOp, _: bytes::Bytes) -> BoxFuture<'static, Result<u32, NestError>> {
            futures::future::pending().boxed()
        }
    }
    let sink: Arc<dyn WriteSink> = Arc::new(Never);
    n[0].1.set_sink(Arc::downgrade(&sink));
    let f2 = n[1].1.clone();
    let unknown = |r: Result<u32, WriteError>| match r {
        Err(WriteError::Failed(NestError::Io(why))) => {
            assert!(why.contains("outcome unknown"), "{why}")
        }
        other => panic!("{other:?}"),
    };
    // No answer within the timeout.
    let t0 = Instant::now();
    unknown(f2.write(NodeId(1), wop(2, 0, true), b"once").await);
    assert!(t0.elapsed() >= Duration::from_millis(500));
    assert!(!f2.can_write(NodeId(1)), "the lane was not retired");
    // A fresh link; then the owner's side goes away mid-write (a restart).
    let w = tokio::spawn({
        let f2 = f2.clone();
        async move { f2.write(NodeId(1), wop(2, 0, true), b"twice").await }
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    n[0].1.shutdown();
    unknown(
        tokio::time::timeout(Duration::from_secs(10), w)
            .await
            .expect("the writer waited for ever")
            .unwrap(),
    );
}
