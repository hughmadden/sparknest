//! Preparation ownership and generation pins must survive worker/daemon death.
mod common;
use common::Db;
use nest_meta::{Command, ModelCommand, ModelReply, Reply, composition};
use nest_types::*;

fn run(db: &mut Db, op: ModelCommand) -> Result<Reply, NestError> {
    db.run(Command::Model(op)).0
}

fn claim(db: &mut Db, key: &str, owner: &str) -> u64 {
    match run(
        db,
        ModelCommand::Claim {
            key: key.into(),
            owner: owner.into(),
        },
    )
    .unwrap()
    {
        Reply::Model(ModelReply::Epoch(epoch)) => epoch,
        r => panic!("{r:?}"),
    }
}

fn pinned(db: &mut Db, key: &str) -> (FileAttr, u64) {
    let a = db.file(FileId::ROOT, key, 1, 8);
    db.ok(Command::Seal {
        file: a.id,
        sealed: true,
        now: Timestamp(1),
    });
    let epoch = claim(db, key, "worker");
    run(
        db,
        ModelCommand::Pin {
            key: key.into(),
            owner: "worker".into(),
            epoch,
            files: vec![(a.id, a.generation)],
        },
    )
    .unwrap();
    (db.attr(a.id), epoch)
}

#[test]
fn pins_fence_unseal_last_unlink_and_last_replica() {
    let mut db = Db::new();
    let (a, epoch) = pinned(&mut db, "fixture");
    for cmd in [
        Command::Seal {
            file: a.id,
            sealed: false,
            now: Timestamp(2),
        },
        Command::Unlink {
            parent: FileId::ROOT,
            name: b"fixture".to_vec(),
            now: Timestamp(2),
        },
        Command::RetireReplica {
            file: a.id,
            generation: a.generation,
            store: NodeId(1).live_store(),
            allow_last: true,
        },
    ] {
        assert!(matches!(db.run(cmd).0, Err(NestError::Busy(_))));
    }
    assert_eq!(db.lookup(FileId::ROOT, "fixture"), Some(a.id));
    run(
        &mut db,
        ModelCommand::Stop {
            key: "fixture".into(),
            owner: "worker".into(),
            epoch,
            receipt: "worker-stopped".into(),
        },
    )
    .unwrap();
    db.ok(Command::Seal {
        file: a.id,
        sealed: false,
        now: Timestamp(3),
    });
}

#[test]
fn competing_worker_and_stale_publication_are_refused() {
    let mut db = Db::new();
    let (_, epoch) = pinned(&mut db, "fixture");
    assert_eq!(claim(&mut db, "fixture", "worker"), epoch);
    assert!(matches!(
        run(
            &mut db,
            ModelCommand::Claim {
                key: "fixture".into(),
                owner: "other".into()
            }
        ),
        Err(NestError::Busy(_))
    ));
    assert_eq!(
        run(
            &mut db,
            ModelCommand::Publish {
                key: "fixture".into(),
                owner: "worker".into(),
                epoch: epoch + 1,
                publication: "payload".into()
            }
        ),
        Err(NestError::Stale)
    );
    assert_eq!(
        composition::get(&db.c, "fixture").unwrap().unwrap().state,
        "preparing"
    );
}

#[test]
fn active_lease_blocks_reset_and_deferred_reset_waits_for_release() {
    let mut db = Db::new();
    let (_, epoch) = pinned(&mut db, "fixture");
    run(
        &mut db,
        ModelCommand::Publish {
            key: "fixture".into(),
            owner: "worker".into(),
            epoch,
            publication: "payload".into(),
        },
    )
    .unwrap();
    run(
        &mut db,
        ModelCommand::Use {
            key: "fixture".into(),
            epoch,
            lease: "model-ranks".into(),
        },
    )
    .unwrap();
    assert!(matches!(
        run(
            &mut db,
            ModelCommand::Reset {
                epoch,
                key: "fixture".into(),
                when_idle: false
            }
        ),
        Err(NestError::Busy(_))
    ));
    run(
        &mut db,
        ModelCommand::Reset {
            epoch,
            key: "fixture".into(),
            when_idle: true,
        },
    )
    .unwrap();
    assert_eq!(
        composition::get(&db.c, "fixture").unwrap().unwrap().state,
        "ready"
    );
    run(
        &mut db,
        ModelCommand::Release {
            key: "fixture".into(),
            epoch,
            lease: "model-ranks".into(),
        },
    )
    .unwrap();
    let state = composition::get(&db.c, "fixture").unwrap().unwrap();
    assert_eq!(state.state, "retired");
    assert_eq!(state.publication.as_deref(), Some("payload"));
    assert!(state.files.is_empty());
}

#[test]
fn reset_rebuild_retains_reproducibility_evidence() {
    let mut db = Db::new();
    let (a, epoch) = pinned(&mut db, "fixture");
    run(
        &mut db,
        ModelCommand::Publish {
            key: "fixture".into(),
            owner: "worker".into(),
            epoch,
            publication: "payload".into(),
        },
    )
    .unwrap();
    run(
        &mut db,
        ModelCommand::Reset {
            epoch,
            key: "fixture".into(),
            when_idle: false,
        },
    )
    .unwrap();
    let next = claim(&mut db, "fixture", "worker");
    assert!(next > epoch);
    run(
        &mut db,
        ModelCommand::Pin {
            key: "fixture".into(),
            owner: "worker".into(),
            epoch: next,
            files: vec![(a.id, a.generation)],
        },
    )
    .unwrap();
    assert!(
        matches!(run(&mut db, ModelCommand::Publish { key: "fixture".into(), owner: "worker".into(), epoch: next, publication: "different".into() }), Err(NestError::Invalid(s)) if s.contains("NON_REPRODUCIBLE_PREPARATION"))
    );
    assert_eq!(
        run(
            &mut db,
            ModelCommand::Publish {
                key: "fixture".into(),
                owner: "worker".into(),
                epoch,
                publication: "payload".into()
            }
        ),
        Err(NestError::Stale)
    );
    run(
        &mut db,
        ModelCommand::Publish {
            key: "fixture".into(),
            owner: "worker".into(),
            epoch: next,
            publication: "payload".into(),
        },
    )
    .unwrap();
}

#[test]
fn pin_validation_is_atomic_and_requires_sealed_current_files() {
    let mut db = Db::new();
    let (a, epoch) = pinned(&mut db, "fixture");
    let b = db.file(FileId::ROOT, "mutable", 1, 8);
    assert!(
        run(
            &mut db,
            ModelCommand::Pin {
                key: "fixture".into(),
                owner: "worker".into(),
                epoch,
                files: vec![(a.id, a.generation), (b.id, b.generation)]
            }
        )
        .is_err()
    );
    assert_eq!(
        composition::get(&db.c, "fixture").unwrap().unwrap().files,
        vec![(a.id, a.generation)]
    );
    assert!(matches!(
        run(
            &mut db,
            ModelCommand::Reset {
                epoch,
                key: "fixture".into(),
                when_idle: false
            }
        ),
        Err(NestError::Busy(_))
    ));
}

#[test]
fn deferred_reset_cannot_release_a_live_worker_through_a_usage_token() {
    let mut db = Db::new();
    let (_, epoch) = pinned(&mut db, "fixture");
    run(
        &mut db,
        ModelCommand::Reset {
            epoch,
            key: "fixture".into(),
            when_idle: true,
        },
    )
    .unwrap();
    assert_eq!(
        run(
            &mut db,
            ModelCommand::Release {
                key: "fixture".into(),
                epoch,
                lease: "unknown".into()
            }
        ),
        Err(NestError::Stale)
    );
    let a = composition::get(&db.c, "fixture").unwrap().unwrap();
    assert_eq!(a.state, "preparing");
    assert_eq!(a.files.len(), 1);
    run(
        &mut db,
        ModelCommand::CancelReset {
            epoch,
            reset: 1,
            key: "fixture".into(),
        },
    )
    .unwrap();
    run(
        &mut db,
        ModelCommand::Publish {
            key: "fixture".into(),
            owner: "worker".into(),
            epoch,
            publication: "payload".into(),
        },
    )
    .unwrap();
    assert_eq!(
        composition::get(&db.c, "fixture").unwrap().unwrap().state,
        "ready"
    );
}

#[test]
fn snapshot_restart_keeps_usage_leases_and_dependency_pins() {
    let mut db = Db::new();
    let (a, epoch) = pinned(&mut db, "fixture");
    run(
        &mut db,
        ModelCommand::Publish {
            key: "fixture".into(),
            owner: "worker".into(),
            epoch,
            publication: "payload".into(),
        },
    )
    .unwrap();
    run(
        &mut db,
        ModelCommand::Use {
            key: "fixture".into(),
            epoch,
            lease: "model-ranks".into(),
        },
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let snapshot = dir.path().join("meta.sqlite");
    db.c.execute("VACUUM INTO ?1", [snapshot.to_str().unwrap()])
        .unwrap();
    let c = nest_meta::open_write(&snapshot).unwrap();
    assert_eq!(
        composition::get(&db.c, "fixture").unwrap(),
        composition::get(&c, "fixture").unwrap()
    );
    assert!(composition::is_pinned(&c, a.id).unwrap());
    assert!(nest_meta::fsck::check(&c).unwrap().is_empty());
    assert!(matches!(
        nest_meta::apply(
            &c,
            &Command::Model(ModelCommand::Reset {
                epoch,
                key: "fixture".into(),
                when_idle: false
            })
        )
        .unwrap()
        .0,
        Err(NestError::Busy(_))
    ));
}

#[test]
fn shared_dependencies_survive_reset_and_only_redundant_copies_can_be_evicted() {
    let mut db = Db::new();
    let (a, epoch) = pinned(&mut db, "first");
    let other = claim(&mut db, "second", "worker-2");
    run(
        &mut db,
        ModelCommand::Pin {
            key: "second".into(),
            owner: "worker-2".into(),
            epoch: other,
            files: vec![(a.id, a.generation)],
        },
    )
    .unwrap();
    run(
        &mut db,
        ModelCommand::Publish {
            key: "first".into(),
            owner: "worker".into(),
            epoch,
            publication: "payload".into(),
        },
    )
    .unwrap();
    run(
        &mut db,
        ModelCommand::Reset {
            epoch,
            key: "first".into(),
            when_idle: false,
        },
    )
    .unwrap();
    assert!(composition::is_pinned(&db.c, a.id).unwrap());
    db.ok(Command::PublishReplica {
        file: a.id,
        generation: a.generation,
        store: NodeId(2).live_store(),
    });
    db.ok(Command::RetireReplica {
        file: a.id,
        generation: a.generation,
        store: NodeId(1).live_store(),
        allow_last: true,
    });
    assert!(matches!(
        db.run(Command::RetireReplica {
            file: a.id,
            generation: a.generation,
            store: NodeId(2).live_store(),
            allow_last: true
        })
        .0,
        Err(NestError::Busy(_))
    ));
}

#[test]
fn usage_tokens_are_single_use_even_without_rebuilding() {
    let mut db = Db::new();
    let (_, epoch) = pinned(&mut db, "fixture");
    run(
        &mut db,
        ModelCommand::Publish {
            key: "fixture".into(),
            owner: "worker".into(),
            epoch,
            publication: "payload".into(),
        },
    )
    .unwrap();
    let acquire = ModelCommand::Use {
        key: "fixture".into(),
        epoch,
        lease: "first-load".into(),
    };
    run(&mut db, acquire.clone()).unwrap();
    assert!(matches!(
        run(&mut db, acquire.clone()),
        Err(NestError::Busy(_))
    ));
    let release = ModelCommand::Release {
        key: "fixture".into(),
        epoch,
        lease: "first-load".into(),
    };
    run(&mut db, release.clone()).unwrap();
    assert!(matches!(run(&mut db, acquire), Err(NestError::Busy(_))));
    run(
        &mut db,
        ModelCommand::Use {
            key: "fixture".into(),
            epoch,
            lease: "second-load".into(),
        },
    )
    .unwrap();
    run(&mut db, release).unwrap();
    assert_eq!(
        composition::get(&db.c, "fixture").unwrap().unwrap().leases,
        ["second-load"]
    );
}

#[test]
fn reset_cancellation_matches_epoch_and_unique_reset_id() {
    let mut db = Db::new();
    let (_, epoch) = pinned(&mut db, "fixture");
    for reset in 1..=2 {
        assert_eq!(
            run(
                &mut db,
                ModelCommand::Reset {
                    key: "fixture".into(),
                    epoch,
                    when_idle: true
                }
            )
            .unwrap(),
            Reply::Model(ModelReply::Reset(reset))
        );
        if reset == 1 {
            run(
                &mut db,
                ModelCommand::CancelReset {
                    key: "fixture".into(),
                    epoch,
                    reset,
                },
            )
            .unwrap();
        }
    }
    assert_eq!(
        run(
            &mut db,
            ModelCommand::CancelReset {
                key: "fixture".into(),
                epoch,
                reset: 1
            }
        ),
        Err(NestError::Stale)
    );
    assert_eq!(
        composition::get(&db.c, "fixture")
            .unwrap()
            .unwrap()
            .reset_id,
        Some(2)
    );
    run(
        &mut db,
        ModelCommand::Stop {
            key: "fixture".into(),
            owner: "worker".into(),
            epoch,
            receipt: "stopped".into(),
        },
    )
    .unwrap();
    let next = claim(&mut db, "fixture", "new-worker");
    run(
        &mut db,
        ModelCommand::Reset {
            key: "fixture".into(),
            epoch: next,
            when_idle: true,
        },
    )
    .unwrap();
    assert_eq!(
        run(
            &mut db,
            ModelCommand::CancelReset {
                key: "fixture".into(),
                epoch,
                reset: 2
            }
        ),
        Err(NestError::Stale)
    );
    assert_eq!(
        composition::get(&db.c, "fixture")
            .unwrap()
            .unwrap()
            .reset_id,
        Some(3)
    );
}

#[test]
fn physical_loss_is_reported_without_releasing_active_users_or_pins() {
    let mut db = Db::new();
    let (a, epoch) = pinned(&mut db, "fixture");
    run(
        &mut db,
        ModelCommand::Publish {
            key: "fixture".into(),
            owner: "worker".into(),
            epoch,
            publication: "payload".into(),
        },
    )
    .unwrap();
    run(
        &mut db,
        ModelCommand::Use {
            key: "fixture".into(),
            epoch,
            lease: "existing-load".into(),
        },
    )
    .unwrap();
    db.ok(Command::LostReplica {
        file: a.id,
        generation: a.generation,
        store: NodeId(1).live_store(),
    });
    let entry = composition::get(&db.c, "fixture").unwrap().unwrap();
    assert_eq!(entry.state, "unavailable");
    assert_eq!(entry.leases, ["existing-load"]);
    assert_eq!(entry.files, [(a.id, a.generation)]);
    assert!(db.stores(a.id).is_empty());
    assert!(matches!(
        run(
            &mut db,
            ModelCommand::Use {
                key: "fixture".into(),
                epoch,
                lease: "new-load".into()
            }
        ),
        Err(NestError::Unavailable(_))
    ));
    assert!(matches!(
        run(
            &mut db,
            ModelCommand::Claim {
                key: "fixture".into(),
                owner: "new-worker".into()
            }
        ),
        Err(NestError::Busy(_))
    ));
    assert!(matches!(
        run(
            &mut db,
            ModelCommand::Reset {
                key: "fixture".into(),
                epoch,
                when_idle: false
            }
        ),
        Err(NestError::Busy(_))
    ));
    run(
        &mut db,
        ModelCommand::Release {
            key: "fixture".into(),
            epoch,
            lease: "existing-load".into(),
        },
    )
    .unwrap();
    run(
        &mut db,
        ModelCommand::Reset {
            key: "fixture".into(),
            epoch,
            when_idle: false,
        },
    )
    .unwrap();
}

#[test]
fn duplicate_loss_report_cannot_delete_an_unpublished_replacement() {
    let mut db = Db::new();
    let a = db.file(FileId::ROOT, "fixture", 1, 7);
    let loss = Command::LostReplica {
        file: a.id,
        generation: a.generation,
        store: NodeId(1).live_store(),
    };
    let (first, effects) = db.run(loss.clone());
    first.unwrap();
    assert_eq!(effects.len(), 1);
    // A transfer may now land replacement bytes. They are not yet advertised;
    // the duplicate report must not emit another generation deletion effect.
    let (again, effects) = db.run(loss);
    again.unwrap();
    assert!(effects.is_empty());
    db.ok(Command::PublishReplica {
        file: a.id,
        generation: a.generation,
        store: NodeId(1).live_store(),
    });
    assert_eq!(db.stores(a.id), [1]);
}
