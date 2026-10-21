//! Leadership moves on purpose (ADR-057): a stopping leader hands off to a
//! caught-up voter instead of leaving the others to time out and elect, and
//! the leader hands off to the cluster's preferred leader once that host
//! has settled, but not again while it flaps.

use nest_meta::Command;
use nest_testkit::{TestCluster, fast_tuning};
use nest_types::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Elections only after 2-3 s of silence: a new leader within a second of
/// the old one stopping can only come from a handoff.
fn slow_elections() -> sparknestd::Tuning {
    let mut t = fast_tuning();
    t.heartbeat_ms = 100;
    t.election_min_ms = 2_000;
    t.election_max_ms = 3_000;
    t.lease = Duration::from_millis(3_000);
    t
}

fn term(c: &TestCluster, id: u64) -> u64 {
    c.node(id).meta.metrics().current_term
}

async fn set_preferred(c: &TestCluster, via: u64, node: Option<u64>) {
    c.node(via)
        .meta
        .propose(Command::SetSetting {
            name: Setting::PreferredLeader.name().into(),
            value: node,
        })
        .await
        .unwrap();
    c.converge().await;
}

/// Every running node agrees on a leader among them; returns it.
fn agreed_leader(c: &TestCluster) -> Option<u64> {
    let ids = c.running();
    let l = c.node(ids[0]).meta.leader()?.0;
    (ids.contains(&l)
        && ids
            .iter()
            .all(|i| c.node(*i).meta.leader() == Some(NodeId(l))))
    .then_some(l)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_stopping_leader_hands_off_without_an_election_timeout() {
    let mut c = TestCluster::start_with(3, slow_elections()).await;
    let old = c.wait_leader().await;
    c.eventually("all agree on the leader", Duration::from_secs(10), |c| {
        agreed_leader(c) == Some(old)
    })
    .await;
    c.converge().await;
    let before = term(&c, old);
    let t = Instant::now();
    c.stop(old).await;
    let stopped_in = t.elapsed();
    c.eventually(
        "a new leader within a second",
        Duration::from_secs(1),
        |c| agreed_leader(c).is_some_and(|l| l != old),
    )
    .await;
    let took = t.elapsed();
    let new = agreed_leader(&c).unwrap();
    for i in c.running() {
        assert_eq!(
            term(&c, i),
            before + 1,
            "node {i}: one election, the handoff"
        );
    }
    assert!(took < Duration::from_secs(1), "new leader after {took:?}");
    assert!(
        stopped_in < Duration::from_millis(2_500),
        "stop took {stopped_in:?}"
    );
    // The new leader writes.
    c.node(new)
        .meta
        .propose(Command::SetSetting {
            name: Setting::ReadRecoveryWait.name().into(),
            value: Some(1_000),
        })
        .await
        .unwrap();
    // It reports round trips to the others.
    let peers = c.node(new).meta.raft_peers();
    let other = c.running().into_iter().find(|i| *i != new).unwrap();
    let p = peers.iter().find(|p| p.peer == NodeId(other)).unwrap();
    assert!(p.voter);
    c.eventually("heartbeats measured", Duration::from_secs(3), |c| {
        c.node(new)
            .meta
            .raft_peers()
            .iter()
            .find(|p| p.peer == NodeId(other))
            .is_some_and(|p| p.heartbeat.ok > 0 && p.matched.is_some() && p.lag == Some(0))
    })
    .await;
}

/// A voter on 0.4.4 cannot decode the handoff: it is skipped, even when it
/// is the preferred leader, and the other voter takes over.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_peer_on_an_older_release_is_skipped() {
    let mut c = TestCluster::start_with(3, slow_elections()).await;
    let old = c.wait_leader().await;
    let others: Vec<u64> = c.running().into_iter().filter(|i| *i != old).collect();
    let (legacy, modern) = (others[0], others[1]);
    c.node(legacy).meta.emulate_0_4_4_raft(true);
    set_preferred(&c, old, Some(legacy)).await;
    c.eventually("all agree on the leader", Duration::from_secs(10), |c| {
        agreed_leader(c) == Some(old)
    })
    .await;
    let before = term(&c, old);
    let t = Instant::now();
    c.stop(old).await;
    c.eventually(
        "the current-release voter leads within a second",
        Duration::from_secs(1),
        |c| agreed_leader(c) == Some(modern),
    )
    .await;
    assert!(t.elapsed() < Duration::from_secs(1));
    assert_eq!(term(&c, modern), before + 1);
    // The old-release voter keeps following (it accepted the transfer vote).
    assert_eq!(c.node(legacy).meta.leader(), Some(NodeId(modern)));
}

/// With every other voter on an older release the stopping leader just
/// stops, and the others elect a leader as before.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn no_capable_voter_means_an_ordinary_election() {
    let mut c = TestCluster::start(3).await;
    let old = c.wait_leader().await;
    for i in c.running() {
        if i != old {
            c.node(i).meta.emulate_0_4_4_raft(true);
        }
    }
    let t = Instant::now();
    c.stop(old).await;
    assert!(
        t.elapsed() < Duration::from_secs(2),
        "stop took {:?}",
        t.elapsed()
    );
    c.eventually("an elected leader", Duration::from_secs(5), |c| {
        agreed_leader(c).is_some_and(|l| l != old)
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn the_preferred_leader_takes_over_once_settled_and_healthy() {
    let c = TestCluster::start(3).await;
    let old = c.wait_leader().await;
    let want = c.running().into_iter().find(|i| *i != old).unwrap();
    // Its link is degraded: no handoff.
    let healthy = Arc::new(AtomicBool::new(false));
    for i in c.running() {
        let h = healthy.clone();
        c.node(i).meta.set_peer_health(Arc::new(move |n| {
            n != NodeId(want) || h.load(Ordering::Relaxed)
        }));
    }
    set_preferred(&c, old, Some(want)).await;
    let settle = c.tuning.handoff.settle;
    tokio::time::sleep(settle * 2 + Duration::from_millis(500)).await;
    assert_eq!(agreed_leader(&c), Some(old), "moved to an unhealthy host");
    healthy.store(true, Ordering::Relaxed);
    let t = Instant::now();
    c.eventually("the preferred host leads", Duration::from_secs(10), |c| {
        agreed_leader(c) == Some(want)
    })
    .await;
    assert!(
        t.elapsed() >= settle,
        "did not wait to settle: {:?}",
        t.elapsed()
    );
    // And stays leader.
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert_eq!(agreed_leader(&c), Some(want));
    // Unset: nothing moves.
    set_preferred(&c, want, None).await;
    tokio::time::sleep(settle * 2).await;
    assert_eq!(agreed_leader(&c), Some(want));
}

/// The preferred host takes over, restarts, and is back quickly: it does
/// not get leadership again until the cooldown has passed.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_flapping_preferred_leader_is_not_readopted_within_the_cooldown() {
    let mut t = fast_tuning();
    t.handoff.settle = Duration::from_millis(500);
    t.handoff.cooldown = Duration::from_secs(8);
    let mut c = TestCluster::start_with(3, t).await;
    let old = c.wait_leader().await;
    let want = c.running().into_iter().find(|i| *i != old).unwrap();
    set_preferred(&c, old, Some(want)).await;
    c.eventually("the preferred host leads", Duration::from_secs(10), |c| {
        agreed_leader(c) == Some(want)
    })
    .await;
    let took_over = Instant::now();
    c.stop(want).await;
    c.eventually("another leader", Duration::from_secs(5), |c| {
        agreed_leader(c).is_some_and(|l| l != want)
    })
    .await;
    let interim = agreed_leader(&c).unwrap();
    c.restart(want).await;
    c.eventually("it is back and following", Duration::from_secs(10), |c| {
        c.node(want).meta.leader() == Some(NodeId(interim))
    })
    .await;
    // Ready to lead again, well within the cooldown: no handoff.
    let watch_until = took_over + Duration::from_secs(6);
    while Instant::now() < watch_until {
        assert_eq!(
            c.node(interim).meta.leader(),
            Some(NodeId(interim)),
            "leadership bounced back {:?} after the last takeover",
            took_over.elapsed()
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    // After the cooldown it is adopted again.
    c.eventually(
        "the preferred host leads after the cooldown",
        Duration::from_secs(10),
        |c| agreed_leader(c) == Some(want),
    )
    .await;
    assert!(took_over.elapsed() >= Duration::from_secs(8));
}

/// With control traffic on the LAN by preference (ADR-068, the default),
/// every peer's requests are on the standby connection, and that is not a
/// degraded link: the preferred leader still takes over. (0.6.0 counted any
/// peer on the standby as degraded, so production's preferred leader never
/// got leadership back after its restart.)
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn the_preferred_leader_takes_over_with_control_on_the_lan() {
    let c = TestCluster::start_with_standby(3, fast_tuning()).await;
    let old = c.wait_leader().await;
    let want = c.running().into_iter().find(|i| *i != old).unwrap();
    // Every pair on the LAN, by preference.
    c.eventually("control on the LAN", Duration::from_secs(10), |c| {
        c.running().iter().all(|a| {
            c.running().iter().filter(|b| *b != a).all(|b| {
                c.node(*a)
                    .rpc
                    .path(NodeId(*b))
                    .is_some_and(|p| p.on_standby && !p.degraded)
            })
        })
    })
    .await;
    set_preferred(&c, old, Some(want)).await;
    c.eventually("the preferred host leads", Duration::from_secs(15), |c| {
        agreed_leader(c) == Some(want)
    })
    .await;
}
