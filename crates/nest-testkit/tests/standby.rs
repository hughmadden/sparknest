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
    // `nest net` shows it: every host answers, the leader's peers on the
    // LAN with a reason, and Raft heartbeats measured.
    let net = c.node(1).placer.net().await.unwrap();
    assert_eq!(net.len(), 3);
    for h in &net {
        assert!(h.error.is_none(), "{}: {:?}", h.name, h.error);
    }
    let lr = net
        .iter()
        .find(|h| h.node == NodeId(leader))
        .and_then(|h| h.report.clone())
        .unwrap();
    assert_eq!(lr.leader, Some(NodeId(leader)));
    assert_eq!(lr.peers.len(), 2);
    for p in &lr.peers {
        assert!(p.control_on_standby, "{p:?}");
        assert!(p.standby_up && p.failovers >= 1, "{p:?}");
        assert!(p.reason.is_some(), "{p:?}");
        assert!(p.raft.as_ref().is_some_and(|r| r.heartbeat.ok > 0), "{p:?}");
    }
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

/// ADR-057 item 3: with the RDMA link to a peer down (setup refused, the
/// old link gone), reads of its copy go over the standby TCP lane on the
/// LAN, not one request at a time over the control connection. (Pairing
/// again after the backoff is nest-fabric's failed_link_setups_back_off.)
/// Needs a RoCE rail here: skips without one.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn reads_use_the_standby_lane_while_rdma_is_down() {
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    if nest_fabric::discover(&[]).is_empty() {
        return;
    }
    let mut t = nest_testkit::fast_tuning();
    t.fabric = Some(nest_fabric::FabricConfig {
        chunk: 4 << 20,
        client_slots: 32,
        server_slots: 16,
        window: 16,
        devices: vec![],
        link_retry: std::time::Duration::from_secs(5),
        ..nest_fabric::FabricConfig::default()
    });
    let c = TestCluster::start_with_standby(2, t).await;
    c.eventually("serving", Duration::from_secs(8), |c| {
        (1..=2).all(|i| c.node(i).data.caught_up() && c.node(i).data.session().is_some())
    })
    .await;
    let (f2, l2) = (c.node(2).fabric().unwrap(), c.node(2).vfs.lanes().unwrap());
    c.eventually("standby lane up", Duration::from_secs(5), |c| {
        c.node(2).vfs.lanes().unwrap().connected(NodeId(1)) >= 1
    })
    .await;
    let data: Vec<u8> = (0..(24u32 << 20)).map(|i| (i % 251) as u8).collect();
    let file = write_file(&c, 1, "model", &data).await;
    c.eventually("stable", Duration::from_secs(5), |c| {
        c.attr(2, file)
            .is_some_and(|a| a.gen_state == GenState::Stable)
    })
    .await;

    assert!(read_all(&c, 2, file).await == data);
    let rdma = f2.stats.read_bytes.load(Ordering::Relaxed);
    assert!(rdma >= data.len() as u64, "first read over RDMA ({rdma})");
    assert_eq!(l2.stats.read_bytes.load(Ordering::Relaxed), 0);

    // Node 1 refuses RDMA links from now on, and node 2's link is gone.
    c.node(1).rpc.register(
        nest_rpc::service::FABRIC,
        Arc::new(|_p: NodeId, _b: bytes::Bytes| {
            Box::pin(async { Err::<bytes::Bytes, _>("RDMA is down".to_string()) })
                as futures::future::BoxFuture<'static, Result<bytes::Bytes, String>>
        }),
    );
    f2.drop_link(NodeId(1));
    assert!(read_all(&c, 2, file).await == data);
    assert!(f2.link_failed(NodeId(1)));
    assert_eq!(
        f2.stats.read_bytes.load(Ordering::Relaxed),
        rdma,
        "read over RDMA with the link down"
    );
    let lane = l2.stats.read_bytes.load(Ordering::Relaxed);
    assert!(
        lane >= data.len() as u64,
        "standby lane carried {lane} bytes"
    );
}
