//! Artifact ownership survives a leader change and daemon restart without
//! releasing another process's mapped model or permitting stale publication.
use nest_meta::{Command, ModelCommand, ModelReply, Reply, composition};
use nest_store::ObjectKey;
use nest_testkit::TestCluster;
use nest_types::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn preparation_pins_and_usage_survive_leader_loss() {
    let mut c = TestCluster::start(3).await;
    let now = Timestamp::now();
    let Reply::Attr(snapshot) = c
        .node(1)
        .meta
        .propose(Command::Mkdir {
            parent: FileId::ROOT,
            name: b"snapshot".to_vec(),
            perm: 0o755,
            now,
        })
        .await
        .unwrap()
    else {
        panic!("mkdir")
    };
    let Reply::Created(created) = c
        .node(1)
        .meta
        .propose(Command::Create {
            parent: snapshot.id,
            name: b"fixture".to_vec(),
            perm: 0o644,
            node: NodeId(1),
            exclusive: true,
            now,
        })
        .await
        .unwrap()
    else {
        panic!("create")
    };
    let a = created.attr;
    let object = ObjectKey::new(a.id, a.generation);
    c.node(1).data.wait_ready(object).await;
    std::fs::write(c.node(1).data.store().path(object), b"fixture").unwrap();
    c.node(1)
        .meta
        .propose(Command::Finalize {
            file: a.id,
            epoch: a.epoch,
            size: 7,
            mtime: now,
            now,
        })
        .await
        .unwrap();
    c.node(1)
        .meta
        .propose(Command::Seal {
            file: a.id,
            sealed: true,
            now,
        })
        .await
        .unwrap();
    let Reply::Model(ModelReply::Epoch(epoch)) = c
        .node(1)
        .meta
        .propose(Command::Model(ModelCommand::Claim {
            key: "fixture".into(),
            owner: "worker".into(),
        }))
        .await
        .unwrap()
    else {
        panic!("claim")
    };
    for op in [
        ModelCommand::BindSnapshot {
            key: "fixture".into(),
            owner: "worker".into(),
            epoch,
            root: snapshot.id,
            output: true,
            files: vec![(b"fixture".to_vec(), a.id, a.generation)],
        },
        ModelCommand::Publish {
            key: "fixture".into(),
            owner: "worker".into(),
            epoch,
            publication: "verified-output".into(),
        },
        ModelCommand::Use {
            key: "fixture".into(),
            epoch,
            lease: "model-ranks".into(),
        },
    ] {
        c.node(1).meta.propose(Command::Model(op)).await.unwrap();
    }
    c.converge().await;
    let leader = c.wait_leader().await;
    c.stop(leader).await;
    let survivor = c.wait_leader().await;
    assert!(matches!(
        c.node(survivor)
            .meta
            .propose(Command::Model(ModelCommand::Reset {
                epoch,
                key: "fixture".into(),
                when_idle: false,
            }))
            .await,
        Err(NestError::Busy(_))
    ));
    assert!(matches!(
        c.node(survivor)
            .meta
            .propose(Command::Seal {
                file: a.id,
                sealed: false,
                now,
            })
            .await,
        Err(NestError::Busy(_))
    ));
    c.restart(leader).await;
    c.converge().await;
    for id in 1..=3 {
        let reader = c.node(id).meta.open_reader().unwrap();
        let entry = composition::get(&reader, "fixture").unwrap().unwrap();
        assert_eq!(entry.state, "ready");
        assert_eq!(entry.leases, ["model-ranks"]);
        assert_eq!(entry.files, [(a.id, a.generation)]);
        assert_eq!(
            composition::snapshots(&reader, "fixture").unwrap()[0].root,
            snapshot.id
        );
    }
    assert!(matches!(
        c.node(survivor)
            .meta
            .propose(Command::Rename {
                parent: FileId::ROOT,
                name: b"snapshot".to_vec(),
                new_parent: FileId::ROOT,
                new_name: b"renamed".to_vec(),
                flags: Default::default(),
                now,
            })
            .await,
        Err(NestError::Busy(_))
    ));
    // Losing the sole physical copy must be reported by startup reconciliation
    // even while the generation is pinned. Its existing user remains protected.
    let path = c.node(1).data.store().path(object);
    c.stop(1).await;
    std::fs::remove_file(path).unwrap();
    c.restart(1).await;
    c.eventually(
        "pinned physical loss reported",
        std::time::Duration::from_secs(8),
        |c| {
            let reader = c.node(2).meta.open_reader().unwrap();
            composition::get(&reader, "fixture")
                .unwrap()
                .is_some_and(|e| e.state == "unavailable" && e.leases == ["model-ranks"])
        },
    )
    .await;
    assert!(matches!(
        c.node(2)
            .meta
            .propose(Command::Model(ModelCommand::Use {
                key: "fixture".into(),
                epoch,
                lease: "new-load".into(),
            }))
            .await,
        Err(NestError::Unavailable(_))
    ));
    // The consumer's independent lifecycle still owns its lease after every
    // daemon has lost/recovered its process-local state.
    c.node(1)
        .meta
        .propose(Command::Model(ModelCommand::Reset {
            epoch,
            key: "fixture".into(),
            when_idle: true,
        }))
        .await
        .unwrap();
    c.node(2)
        .meta
        .propose(Command::Model(ModelCommand::Release {
            key: "fixture".into(),
            epoch,
            lease: "model-ranks".into(),
        }))
        .await
        .unwrap();
    c.converge().await;
    let reader = c.node(3).meta.open_reader().unwrap();
    assert_eq!(
        composition::get(&reader, "fixture").unwrap().unwrap().state,
        "retired"
    );
    assert!(
        composition::snapshots(&reader, "fixture")
            .unwrap()
            .is_empty()
    );
}
