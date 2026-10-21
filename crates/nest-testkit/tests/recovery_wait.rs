//! Reads ride out a leader change or a short loss of metadata quorum
//! (ADR-056): they keep retrying for the cluster's read recovery wait, a
//! setting every host reads live, instead of failing with EIO at once.

use nest_data::vfs::oflags;
use nest_meta::Command;
use nest_testkit::TestCluster;
use nest_types::*;
use std::time::{Duration, Instant};

async fn ready(n: u64) -> TestCluster {
    let c = TestCluster::start(n).await;
    c.eventually(
        "all nodes serving with sessions",
        Duration::from_secs(8),
        |c| (1..=n).all(|i| c.node(i).data.caught_up() && c.node(i).data.session().is_some()),
    )
    .await;
    c
}

async fn file(c: &TestCluster, node: u64, name: &str, content: &[u8]) -> FileId {
    let v = &c.node(node).vfs;
    let (a, fh, _) = v
        .create(FileId::ROOT, name.as_bytes(), 0o644, oflags::WRONLY)
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

async fn set_wait(c: &TestCluster, ms: Option<u64>) {
    c.node(1)
        .meta
        .propose(Command::SetSetting {
            name: Setting::ReadRecoveryWait.name().into(),
            value: ms,
        })
        .await
        .unwrap();
    c.converge().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_setting_reaches_every_host() {
    let c = ready(3).await;
    for i in 1..=3 {
        assert_eq!(
            c.node(i).vfs.setting(Setting::ReadRecoveryWait),
            Setting::ReadRecoveryWait.default_value()
        );
    }
    set_wait(&c, Some(1234)).await;
    c.eventually("every host sees 1234 ms", Duration::from_secs(5), |c| {
        (1..=3).all(|i| c.node(i).vfs.setting(Setting::ReadRecoveryWait) == 1234)
    })
    .await;
    set_wait(&c, None).await;
    c.eventually(
        "every host back on the default",
        Duration::from_secs(5),
        |c| {
            (1..=3).all(|i| {
                c.node(i).vfs.setting(Setting::ReadRecoveryWait)
                    == Setting::ReadRecoveryWait.default_value()
            })
        },
    )
    .await;
}

/// A host cut off from the others for less than the wait loses its read
/// lease, yet a read of its own copy completes once it is back.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_read_waits_out_a_short_loss_of_quorum() {
    let c = ready(3).await;
    set_wait(&c, Some(6_000)).await;
    let f = file(&c, 2, "weights", b"expert shard").await;
    let v2 = &c.node(2).vfs;
    let (fh, _) = v2.open(f, 0).await.unwrap();
    c.partition(&[2], &[1, 3]);
    c.eventually("node 2's lease lapsed", Duration::from_secs(5), |c| {
        !c.node(2).data.lease_valid()
    })
    .await;
    let heal_after = Duration::from_millis(1500);
    let t = Instant::now();
    let read = v2.read(fh, 0, 64);
    let heal = async {
        tokio::time::sleep(heal_after).await;
        c.heal();
    };
    let (r, ()) = tokio::join!(read, heal);
    assert_eq!(r.unwrap(), b"expert shard");
    assert!(t.elapsed() >= heal_after, "served while cut off");
    v2.release(fh, None).await;
}

/// The leader steps away mid-read: readers on other hosts carry on.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_read_waits_out_a_leader_change() {
    let mut c = ready(3).await;
    let leader = c.wait_leader().await;
    let reader = (1..=3).find(|i| *i != leader).unwrap();
    let f = file(&c, reader, "weights", b"still here").await;
    let v = c.node(reader).vfs.clone();
    let (fh, _) = v.open(f, 0).await.unwrap();
    c.stop(leader).await;
    // Reads during the election succeed (they wait for the new leader).
    for _ in 0..20 {
        assert_eq!(v.read(fh, 0, 64).await.unwrap(), b"still here");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    v.release(fh, None).await;
}

/// Past the wait, the read fails (EIO at the mount) rather than hanging.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_read_fails_once_the_wait_is_spent() {
    let c = ready(3).await;
    set_wait(&c, Some(1_000)).await;
    let f = file(&c, 2, "weights", b"x").await;
    let v2 = &c.node(2).vfs;
    let (fh, _) = v2.open(f, 0).await.unwrap();
    c.partition(&[2], &[1, 3]);
    c.eventually("node 2's lease lapsed", Duration::from_secs(5), |c| {
        !c.node(2).data.lease_valid()
    })
    .await;
    let t = Instant::now();
    let r = v2.read(fh, 0, 64).await;
    let took = t.elapsed();
    assert!(matches!(r, Err(NestError::Unavailable(_))), "{r:?}");
    assert!(took >= Duration::from_millis(900), "gave up after {took:?}");
    assert!(took < Duration::from_secs(4), "waited {took:?}");
    c.heal();
    v2.release(fh, None).await;
}

/// A copy whose only source cannot serve right now (cut off, its lease
/// lapsed) keeps trying instead of skipping the file, and completes once
/// the source is back (ADR-067).
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_copy_waits_out_a_source_that_cannot_serve() {
    let c = ready(3).await;
    set_wait(&c, Some(500)).await;
    let f = file(&c, 2, "shard", &[7u8; 64 << 10]).await;
    c.partition(&[2], &[1, 3]);
    c.eventually("node 2's lease lapsed", Duration::from_secs(5), |c| {
        !c.node(2).data.lease_valid()
    })
    .await;
    let v3 = c.node(3).vfs.clone();
    let t = Instant::now();
    let copy = v3.replicate_into(f, NodeId(3).live_store());
    let heal = async {
        tokio::time::sleep(Duration::from_millis(2_000)).await;
        c.heal();
    };
    let (r, ()) = tokio::join!(copy, heal);
    // Longer than the read recovery wait set above: copies wait longer.
    assert!(r.is_ok(), "{r:?}");
    assert!(t.elapsed() >= Duration::from_millis(2_000));
    c.converge().await;
    let (_, copies) = v3.copies(f).unwrap();
    assert!(copies.iter().any(|c| c.store == "n3"), "{copies:?}");
}
