//! The LAN standby (ADR-057): when every fabric connection is cut the
//! cluster keeps its quorum over the standby connections, and goes back to
//! the fabric when it returns.

use nest_data::vfs::oflags;
use nest_testkit::TestCluster;
use nest_types::*;
use std::time::Duration;

async fn write_file(c: &TestCluster, node: u64, name: &str, content: &[u8]) -> FileId {
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
    a.id
}

async fn read_all(c: &TestCluster, node: u64, file: FileId) -> Vec<u8> {
    let v = &c.node(node).vfs;
    let (fh, _) = v.open(file, 0).await.unwrap();
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn quorum_survives_losing_the_fabric() {
    let c = TestCluster::start_with_standby(3, nest_testkit::fast_tuning()).await;
    c.eventually("serving", Duration::from_secs(8), |c| {
        (1..=3).all(|i| c.node(i).data.caught_up() && c.node(i).data.session().is_some())
    })
    .await;
    c.eventually("standbys up", Duration::from_secs(5), |c| {
        (1..=3u64).all(|a| {
            (1..=3u64)
                .filter(|b| *b != a)
                .all(|b| c.node(a).rpc.path(NodeId(b)).is_some_and(|p| p.standby_up))
        })
    })
    .await;
    let before = write_file(&c, 1, "before", b"on the fabric").await;
    let leader = c.wait_leader().await;
    let term = c.node(leader).meta.metrics().current_term;

    // The switch goes away: every primary connection closes and none can
    // be dialled again.
    c.cut_fabric();
    // Requests move peer by peer as they are made: the leader's heartbeats
    // move its peers at once.
    c.eventually(
        "the leader's peers on the standby",
        Duration::from_secs(5),
        |c| {
            (1..=3u64)
                .filter(|b| *b != leader)
                .all(|b| c.node(leader).rpc.path(NodeId(b)).unwrap().on_standby)
        },
    )
    .await;
    // Metadata still commits, from every host, and reads of a remote copy
    // still work (one request at a time over the control connection: no
    // RDMA in these tests).
    for i in 1..=3 {
        let f = write_file(&c, i, &format!("during-{i}"), b"over the LAN").await;
        c.converge().await;
        for j in 1..=3 {
            assert_eq!(read_all(&c, j, f).await, b"over the LAN");
        }
    }
    assert_eq!(read_all(&c, 2, before).await, b"on the fabric");
    for a in 1..=3u64 {
        for b in (1..=3u64).filter(|b| *b != a) {
            let p = c.node(a).rpc.path(NodeId(b)).unwrap();
            if p.on_standby {
                assert_eq!(p.failovers, 1, "{a}->{b}: {p:?}");
            }
        }
    }
    // The leader rode it out: no election was needed (a failover takes
    // well under the election timeout).
    let leader2 = c.wait_leader().await;
    let term2 = c.node(leader2).meta.metrics().current_term;
    assert_eq!(
        (leader2, term2),
        (leader, term),
        "the cut forced an election"
    );

    // The fabric returns: every peer goes back to it after `return_after`.
    c.heal();
    c.eventually(
        "every peer back on the fabric",
        Duration::from_secs(8),
        |c| {
            (1..=3u64).all(|a| {
                (1..=3u64).filter(|b| *b != a).all(|b| {
                    let p = c.node(a).rpc.path(NodeId(b)).unwrap();
                    !p.on_standby && p.returns == p.failovers
                })
            })
        },
    )
    .await;
    let f = write_file(&c, 3, "after", b"back on the fabric").await;
    c.converge().await;
    assert_eq!(read_all(&c, 1, f).await, b"back on the fabric");
}
