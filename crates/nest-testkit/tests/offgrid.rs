//! Off-grid hosts (ADR-047): learners on the LAN only, which the fabric
//! hosts have no address for. Everything they exchange travels over the
//! connections the off-grid host opens.

use nest_data::vfs::oflags;
use nest_meta::Command;
use nest_store::ObjectKey;
use nest_testkit::TestCluster;
use nest_types::*;
use std::time::Duration;

const RO: i32 = 0;

async fn ready(n: u64) -> TestCluster {
    let c = TestCluster::start(n).await;
    c.eventually("all nodes serving", Duration::from_secs(8), |c| {
        (1..=n).all(|i| c.node(i).data.caught_up() && c.node(i).data.session().is_some())
    })
    .await;
    c
}

async fn with_offgrid(n: u64, id: u64) -> TestCluster {
    let mut c = ready(n).await;
    c.add_offgrid(id).await;
    c.eventually("off-grid host serving", Duration::from_secs(8), |c| {
        c.node(id).data.caught_up() && c.node(id).data.session().is_some()
    })
    .await;
    c
}

async fn read_all(c: &TestCluster, node: u64, file: FileId) -> Vec<u8> {
    let v = &c.node(node).vfs;
    let (fh, _) = v.open(file, RO).await.unwrap();
    let mut out = Vec::new();
    loop {
        let b = v.read(fh, out.len() as u64, 1 << 20).await.unwrap();
        if b.is_empty() {
            break;
        }
        out.extend_from_slice(&b);
    }
    v.release(fh, None).await;
    out
}

async fn file(c: &TestCluster, node: u64, name: &str, content: &[u8]) -> FileId {
    let v = &c.node(node).vfs;
    let (a, fh, _) = v
        .create(
            FileId::ROOT,
            name.as_bytes(),
            0o644,
            oflags::WRONLY | oflags::EXCL,
        )
        .await
        .unwrap();
    v.write(fh, 0, content.to_vec()).await.unwrap();
    v.release(fh, None).await;
    c.eventually("file stable", Duration::from_secs(5), |c| {
        c.attr(node, a.id)
            .is_some_and(|a| a.gen_state == GenState::Stable)
    })
    .await;
    c.converge().await;
    a.id
}

/// Copy the current generation from `from` to `to` and publish it.
async fn replicate(c: &TestCluster, f: FileId, from: u64, to: u64) {
    let a = c.attr(from, f).unwrap();
    let key = ObjectKey::new(f, a.generation);
    let bytes = std::fs::read(c.node(from).data.store().path(key)).unwrap();
    let st = c.node(to).data.store().begin_staging(key).unwrap();
    std::os::unix::fs::FileExt::write_all_at(st.file(), &bytes, 0).unwrap();
    c.node(to).data.store().commit_staging(st).unwrap();
    c.node(to)
        .meta
        .propose(Command::PublishReplica {
            file: f,
            generation: a.generation,
            store: NodeId(to).live_store(),
        })
        .await
        .unwrap();
    c.converge().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn an_offgrid_host_joins_as_a_learner_reads_writes_and_owns() {
    let c = with_offgrid(3, 4).await;
    // A learner, and the fabric hosts cannot dial it.
    let m = c.node(1).meta.metrics().membership_config;
    assert!(m.nodes().any(|(id, _)| *id == 4));
    assert!(!m.membership().voter_ids().any(|v| v == 4));
    for i in 1..=3 {
        assert_eq!(c.node(i).rpc.peer_addr(NodeId(4)), None);
        assert_eq!(c.node(i).vfs.offgrid_hosts(), [NodeId(4)].into());
    }

    // It reads what a fabric host wrote...
    let f = file(&c, 1, "from-fabric", b"weights from raptor").await;
    assert_eq!(read_all(&c, 4, f).await, b"weights from raptor");
    // ...writes a file it owns, which a fabric host reads from it...
    let g = file(&c, 4, "from-offgrid", b"written on kakapo").await;
    let a = c.attr(1, g).unwrap();
    assert!(
        c.node(1)
            .meta
            .open_reader()
            .map(|r| nest_meta::query::replicas(&r, g).unwrap())
            .unwrap()
            .iter()
            .any(|r| r.store == NodeId(4).live_store() && r.generation == a.generation)
    );
    assert_eq!(read_all(&c, 2, g).await, b"written on kakapo");
    // ...and rewrites a fabric host's file (fenced across the cluster).
    let v4 = &c.node(4).vfs;
    let (fh, _) = v4.open(f, oflags::RDWR).await.unwrap();
    v4.write(fh, 0, b"WEIGHTS".to_vec()).await.unwrap();
    v4.release(fh, None).await;
    c.converge().await;
    assert_eq!(read_all(&c, 3, f).await, b"WEIGHTS from raptor");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn an_offgrid_host_does_not_count_toward_quorum() {
    let mut c = with_offgrid(3, 4).await;
    c.stop(2).await;
    c.stop(3).await;
    // One voter of three is left, plus the learner: nothing commits.
    let v4 = &c.node(4).vfs;
    let r = tokio::time::timeout(
        Duration::from_secs(10),
        v4.create(FileId::ROOT, b"no-quorum", 0o644, oflags::WRONLY),
    )
    .await;
    assert!(
        !matches!(r, Ok(Ok(_))),
        "a write committed with one voter of three and a learner"
    );
    c.restart_many(&[2, 3]).await;
    c.eventually("quorum back", Duration::from_secs(15), |c| {
        c.node(4).data.caught_up() && c.node(1).data.caught_up()
    })
    .await;
    file(&c, 4, "with-quorum", b"ok").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn reads_prefer_the_offgrid_hosts_own_copy_and_fabric_holders_elsewhere() {
    let c = with_offgrid(3, 4).await;
    let f = file(&c, 1, "model", &vec![7u8; 256 << 10]).await;
    replicate(&c, f, 1, 2).await;
    replicate(&c, f, 1, 4).await;

    // The off-grid host reads its own copy although two others exist.
    let before = c.node(4).vfs.local_read_bytes();
    assert_eq!(read_all(&c, 4, f).await.len(), 256 << 10);
    assert_eq!(c.node(4).vfs.local_read_bytes() - before, 256 << 10);

    // A fabric host without a copy reads from fabric holders first.
    let order = c.node(3).vfs.read_order(f).unwrap();
    assert_eq!(order.len(), 3, "{order:?}");
    assert_eq!(order.last(), Some(&NodeId(4)), "{order:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn an_offgrid_host_rejoins_after_a_fabric_restart_and_its_own() {
    let mut c = with_offgrid(3, 4).await;
    let f = file(&c, 1, "a", b"one").await;
    assert_eq!(read_all(&c, 4, f).await, b"one");
    // The leader's host restarts: the off-grid host dials in again.
    let leader = c.wait_leader().await;
    c.stop(leader).await;
    c.restart(leader).await;
    let g = file(&c, leader, "b", b"two").await;
    c.eventually("off-grid host caught up", Duration::from_secs(10), |c| {
        c.lookup(4, FileId::ROOT, "b").is_some()
    })
    .await;
    assert_eq!(read_all(&c, 4, g).await, b"two");
    // The off-grid host restarts: it waits for the cluster and catches up.
    c.stop(4).await;
    c.restart(4).await;
    c.eventually(
        "off-grid host serving again",
        Duration::from_secs(10),
        |c| c.node(4).data.caught_up() && c.node(4).data.session().is_some(),
    )
    .await;
    let h = file(&c, 2, "c", b"three").await;
    c.eventually("sees new files", Duration::from_secs(10), |c| {
        c.lookup(4, FileId::ROOT, "c").is_some()
    })
    .await;
    assert_eq!(read_all(&c, 4, h).await, b"three");
}
