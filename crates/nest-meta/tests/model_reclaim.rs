//! Reclamation proves exact derived import ownership before ordinary unlink.
mod common;
use common::Db;
use nest_meta::{Command, Effect, ModelCommand, ModelReply, Reply, composition};
use nest_types::*;

const KEY: &str = "fixture-key";
fn run(db: &mut Db, cmd: ModelCommand) -> Result<Reply, NestError> {
    db.run(Command::Model(cmd)).0
}
fn claim(db: &mut Db, key: &str) -> u64 {
    match run(
        db,
        ModelCommand::Claim {
            key: key.into(),
            owner: "worker".into(),
        },
    )
    .unwrap()
    {
        Reply::Model(ModelReply::Epoch(epoch)) => epoch,
        other => panic!("{other:?}"),
    }
}
fn import(db: &mut Db, epoch: u64, parent: FileId, name: &str) -> FileAttr {
    let file = match db.ok(Command::ReserveFileIds { count: 1 }) {
        Reply::FileIds(file) => file,
        other => panic!("{other:?}"),
    };
    run(
        db,
        ModelCommand::ImportOutput {
            key: KEY.into(),
            owner: "worker".into(),
            epoch,
            parent,
            name: name.into(),
            file,
            perm: 0o644,
            size: 8,
            mtime: Timestamp(2),
            node: NodeId(1),
            now: Timestamp(2),
        },
    )
    .unwrap();
    db.attr(file)
}
fn stop(db: &mut Db, epoch: u64) {
    run(
        db,
        ModelCommand::Stop {
            key: KEY.into(),
            owner: "worker".into(),
            epoch,
            receipt: "stopped".into(),
        },
    )
    .unwrap();
}
fn fixture(db: &mut Db) -> (u64, FileId, FileAttr) {
    let epoch = claim(db, KEY);
    let root = db.mkdir(FileId::ROOT, "output");
    let a = import(db, epoch, root, "weights.safetensors");
    (epoch, root, a)
}
fn reclaim(epoch: u64, parent: FileId, name: &str, a: &FileAttr) -> ModelCommand {
    ModelCommand::ReclaimOutputShard {
        key: KEY.into(),
        epoch,
        parent,
        name: name.into(),
        file: a.id,
        generation: a.generation,
        now: Timestamp(4),
    }
}
fn refuses_without_effect(db: &mut Db, cmd: ModelCommand) -> NestError {
    let (result, fx) = db.run(Command::Model(cmd));
    assert!(fx.is_empty());
    result.unwrap_err()
}

#[test]
fn live_workers_ready_users_and_stale_epochs_cannot_reclaim() {
    let mut db = Db::new();
    let (epoch, root, a) = fixture(&mut db);
    let cmd = reclaim(epoch, root, "weights.safetensors", &a);
    assert!(matches!(
        refuses_without_effect(&mut db, cmd.clone()),
        NestError::Busy(_)
    ));
    run(
        &mut db,
        ModelCommand::Pin {
            key: KEY.into(),
            owner: "worker".into(),
            epoch,
            files: vec![(a.id, a.generation)],
        },
    )
    .unwrap();
    run(
        &mut db,
        ModelCommand::Publish {
            key: KEY.into(),
            owner: "worker".into(),
            epoch,
            publication: "fixture-publication".into(),
        },
    )
    .unwrap();
    assert!(matches!(
        refuses_without_effect(&mut db, cmd.clone()),
        NestError::Busy(_)
    ));
    run(
        &mut db,
        ModelCommand::Use {
            key: KEY.into(),
            epoch,
            lease: "active-user".into(),
        },
    )
    .unwrap();
    run(
        &mut db,
        ModelCommand::Reset {
            key: KEY.into(),
            epoch,
            when_idle: true,
        },
    )
    .unwrap();
    assert!(matches!(
        refuses_without_effect(&mut db, cmd),
        NestError::Busy(_)
    ));
    assert_eq!(
        refuses_without_effect(&mut db, reclaim(epoch + 1, root, "weights.safetensors", &a)),
        NestError::Stale
    );
    assert_eq!(db.lookup(root, "weights.safetensors"), Some(a.id));
}

#[test]
fn unowned_input_and_forged_alias_proofs_are_refused() {
    let mut db = Db::new();
    let (epoch, root, a) = fixture(&mut db);
    let input = db.file(FileId::ROOT, "source.safetensors", 1, 8);
    db.ok(Command::Seal {
        file: input.id,
        sealed: true,
        now: Timestamp(3),
    });
    stop(&mut db, epoch);
    assert!(matches!(
        refuses_without_effect(
            &mut db,
            reclaim(epoch, FileId::ROOT, "source.safetensors", &input)
        ),
        NestError::NotPermitted(_)
    ));
    assert!(
        refuses_without_effect(&mut db, reclaim(epoch, root, "other.safetensors", &a))
            .to_string()
            .contains("ownership")
    );
    assert_eq!(
        db.lookup(FileId::ROOT, "source.safetensors"),
        Some(input.id)
    );
}

#[test]
fn retired_owned_shard_is_unlinked_once_with_normal_delete_effects() {
    let mut db = Db::new();
    let (epoch, root, a) = fixture(&mut db);
    stop(&mut db, epoch);
    run(
        &mut db,
        ModelCommand::Reset {
            key: KEY.into(),
            epoch,
            when_idle: false,
        },
    )
    .unwrap();
    let proof = composition::owned_outputs(&db.c, KEY).unwrap();
    let cmd = reclaim(epoch, root, "weights.safetensors", &a);
    let before = db.attr(a.id);
    let (result, effects) = db.run(Command::Model(cmd.clone()));
    assert_eq!(result, Ok(Reply::Attr(before)));
    assert!(effects.contains(&Effect::FileDeleted { file: a.id }));
    assert!(effects.contains(&Effect::ReplicaInvalidated {
        file: a.id,
        generation: a.generation,
        store: NodeId(1).live_store()
    }));
    assert_eq!(db.lookup(root, "weights.safetensors"), None);
    assert_eq!(db.run(Command::Model(cmd)), (Ok(Reply::Done), vec![]));
    assert_eq!(composition::owned_outputs(&db.c, KEY).unwrap(), proof);
}

#[test]
fn a_replaced_alias_or_changed_generation_is_never_deleted() {
    let mut db = Db::new();
    let (epoch, root, a) = fixture(&mut db);
    stop(&mut db, epoch);
    let cmd = reclaim(epoch, root, "weights.safetensors", &a);
    db.ok(Command::Unlink {
        parent: root,
        name: b"weights.safetensors".to_vec(),
        now: Timestamp(3),
    });
    let replacement = db.file(root, "weights.safetensors", 1, 12);
    assert_eq!(refuses_without_effect(&mut db, cmd), NestError::Stale);
    assert_eq!(db.lookup(root, "weights.safetensors"), Some(replacement.id));

    let mut db = Db::new();
    let (epoch, root, a) = fixture(&mut db);
    stop(&mut db, epoch);
    db.ok(Command::Seal {
        file: a.id,
        sealed: false,
        now: Timestamp(3),
    });
    assert!(matches!(
        refuses_without_effect(&mut db, reclaim(epoch, root, "weights.safetensors", &a)),
        NestError::Busy(_)
    ));
    db.ok(Command::AcquireOwner {
        file: a.id,
        node: NodeId(1),
        expect_gen: a.generation,
        truncate: true,
        now: Timestamp(4),
    });
    assert_eq!(
        refuses_without_effect(&mut db, reclaim(epoch, root, "weights.safetensors", &a)),
        NestError::Stale
    );
    assert_eq!(db.lookup(root, "weights.safetensors"), Some(a.id));
}

#[test]
fn shared_generation_pins_and_bound_roots_keep_the_ordinary_guards() {
    for bind_root in [false, true] {
        let mut db = Db::new();
        let (epoch, root, a) = fixture(&mut db);
        let shared_epoch = claim(&mut db, "shared-key");
        let protect = if bind_root {
            ModelCommand::BindSnapshot {
                key: "shared-key".into(),
                owner: "worker".into(),
                epoch: shared_epoch,
                root,
                output: false,
                files: vec![(b"weights.safetensors".to_vec(), a.id, a.generation)],
            }
        } else {
            ModelCommand::Pin {
                key: "shared-key".into(),
                owner: "worker".into(),
                epoch: shared_epoch,
                files: vec![(a.id, a.generation)],
            }
        };
        run(&mut db, protect).unwrap();
        stop(&mut db, epoch);
        assert!(matches!(
            refuses_without_effect(&mut db, reclaim(epoch, root, "weights.safetensors", &a)),
            NestError::Busy(_)
        ));
        assert_eq!(db.lookup(root, "weights.safetensors"), Some(a.id));
        assert!(composition::is_pinned(&db.c, a.id).unwrap());
    }
}

#[test]
fn owned_assets_and_both_historical_receipt_references_are_preserved() {
    let mut db = Db::new();
    let (epoch, root, a) = fixture(&mut db);
    let export_receipt = import(&mut db, epoch, root, "export.safetensors");
    let qualification_receipt = import(&mut db, epoch, root, "qualification.safetensors");
    let config = import(&mut db, epoch, root, "config.json");
    let asset = import(&mut db, epoch, root, "tokenizer.bin");
    let files = [
        ("weights.safetensors", &a),
        ("export.safetensors", &export_receipt),
        ("qualification.safetensors", &qualification_receipt),
        ("config.json", &config),
        ("tokenizer.bin", &asset),
    ]
    .map(|(name, a)| (name.as_bytes().to_vec(), a.id, a.generation))
    .to_vec();
    run(
        &mut db,
        ModelCommand::BindSnapshot {
            key: KEY.into(),
            owner: "worker".into(),
            epoch,
            root,
            output: true,
            files: files.clone(),
        },
    )
    .unwrap();
    run(
        &mut db,
        ModelCommand::PublishComplete {
            key: KEY.into(),
            owner: "worker".into(),
            epoch,
            root,
            files,
            publication: "fixture-publication".into(),
            export_receipt: (export_receipt.id, export_receipt.generation),
            qualification_receipt: (qualification_receipt.id, qualification_receipt.generation),
            qualification_policy: "fixture-policy".into(),
        },
    )
    .unwrap();
    run(
        &mut db,
        ModelCommand::Reset {
            key: KEY.into(),
            epoch,
            when_idle: false,
        },
    )
    .unwrap();
    for (name, a) in [
        ("export.safetensors", &export_receipt),
        ("qualification.safetensors", &qualification_receipt),
        ("config.json", &config),
        ("tokenizer.bin", &asset),
    ] {
        assert!(matches!(
            refuses_without_effect(&mut db, reclaim(epoch, root, name, a)),
            NestError::NotPermitted(_)
        ));
        assert_eq!(db.lookup(root, name), Some(a.id));
    }
    run(&mut db, reclaim(epoch, root, "weights.safetensors", &a)).unwrap();
}

#[test]
fn catalog_entries_include_keys_without_module_names() {
    let mut db = Db::new();
    let first = claim(&mut db, "old-key");
    claim(&mut db, "current-key");
    let rows = composition::entries(&db.c).unwrap();
    assert_eq!(
        rows.iter().map(|a| a.key.as_str()).collect::<Vec<_>>(),
        ["current-key", "old-key"]
    );
    assert_eq!(rows[1].epoch, first);
    assert!(composition::modules(&db.c).unwrap().is_empty());
}

#[test]
fn reclaiming_an_owned_alias_preserves_external_input_aliases_and_object_bytes() {
    let mut db = Db::new();
    let (epoch, root, a) = fixture(&mut db);
    db.ok(Command::Link {
        file: a.id,
        parent: FileId::ROOT,
        name: b"shared-input.safetensors".to_vec(),
        now: Timestamp(3),
    });
    stop(&mut db, epoch);
    let before = db.attr(a.id);
    assert_eq!(before.nlink, 2);
    let (result, effects) = db.run(Command::Model(reclaim(
        epoch,
        root,
        "weights.safetensors",
        &a,
    )));
    assert_eq!(result, Ok(Reply::Attr(before)));
    assert!(!effects.iter().any(|e| matches!(
        e,
        Effect::ReplicaInvalidated { .. } | Effect::FileDeleted { .. }
    )));
    assert_eq!(
        db.lookup(FileId::ROOT, "shared-input.safetensors"),
        Some(a.id)
    );
    assert_eq!(db.attr(a.id).nlink, 1);
    assert_eq!(db.stores(a.id), [1]);
    assert_eq!(db.lookup(root, "weights.safetensors"), None);
}
