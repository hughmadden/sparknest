//! A sealed generation is insufficient if a snapshot's names can be replaced.
mod common;
use common::Db;
use nest_meta::{Command, ModelCommand, ModelReply, RenameFlags, Reply, composition};
use nest_types::*;

fn model(db: &mut Db, command: ModelCommand) -> Result<Reply, NestError> {
    db.run(Command::Model(command)).0
}

fn claim(db: &mut Db, key: &str) -> u64 {
    let Reply::Model(ModelReply::Epoch(epoch)) = model(
        db,
        ModelCommand::Claim {
            key: key.into(),
            owner: "worker".into(),
        },
    )
    .unwrap() else {
        panic!("claim")
    };
    epoch
}

fn sealed(db: &mut Db, root: FileId, name: &str) -> FileAttr {
    let file = db.file(root, name, 1, 8);
    db.ok(Command::Seal {
        file: file.id,
        sealed: true,
        now: Timestamp(1),
    });
    db.attr(file.id)
}

fn bind(
    db: &mut Db,
    key: &str,
    epoch: u64,
    root: FileId,
    output: bool,
    files: &[(&str, &FileAttr)],
) -> Result<Reply, NestError> {
    model(
        db,
        ModelCommand::BindSnapshot {
            key: key.into(),
            owner: "worker".into(),
            epoch,
            root,
            output,
            files: files
                .iter()
                .map(|(name, f)| (name.as_bytes().to_vec(), f.id, f.generation))
                .collect(),
        },
    )
}

fn fixture(db: &mut Db) -> (FileId, FileId, FileAttr, u64) {
    let parent = db.mkdir(FileId::ROOT, "artifacts");
    let root = db.mkdir(parent, "snapshot");
    let file = sealed(db, root, "weights.bin");
    // A second alias makes this a namespace test, rather than last-link fencing.
    db.ok(Command::Link {
        file: file.id,
        parent: FileId::ROOT,
        name: b"source.bin".to_vec(),
        now: Timestamp(1),
    });
    let epoch = claim(db, "fixture");
    bind(db, "fixture", epoch, root, true, &[("weights.bin", &file)]).unwrap();
    (parent, root, file, epoch)
}

#[test]
fn bound_snapshot_fences_inventory_and_ancestor_moves() {
    let mut db = Db::new();
    let (parent, root, file, _) = fixture(&mut db);
    let other = db.mkdir(FileId::ROOT, "other");
    for command in [
        Command::SetAttr {
            file: root,
            perm: Some(0),
            atime: None,
            mtime: None,
            now: Timestamp(2),
        },
        Command::SetAttr {
            file: parent,
            perm: Some(0),
            atime: None,
            mtime: None,
            now: Timestamp(2),
        },
        Command::SetAttr {
            file: file.id,
            perm: Some(0),
            atime: None,
            mtime: None,
            now: Timestamp(2),
        },
        Command::Unlink {
            parent: root,
            name: b"weights.bin".to_vec(),
            now: Timestamp(2),
        },
        Command::Create {
            parent: root,
            name: b"extra".to_vec(),
            perm: 0o644,
            node: NodeId(1),
            exclusive: true,
            now: Timestamp(2),
        },
        Command::Mkdir {
            parent: root,
            name: b"extra".to_vec(),
            perm: 0o755,
            now: Timestamp(2),
        },
        Command::Symlink {
            parent: root,
            name: b"extra".to_vec(),
            target: b"weights.bin".to_vec(),
            now: Timestamp(2),
        },
        Command::Link {
            file: file.id,
            parent: root,
            name: b"extra".to_vec(),
            now: Timestamp(2),
        },
        Command::Rename {
            parent: root,
            name: b"weights.bin".to_vec(),
            new_parent: other,
            new_name: b"moved".to_vec(),
            flags: RenameFlags::default(),
            now: Timestamp(2),
        },
        Command::Rename {
            parent: FileId::ROOT,
            name: b"source.bin".to_vec(),
            new_parent: root,
            new_name: b"weights.bin".to_vec(),
            flags: RenameFlags::default(),
            now: Timestamp(2),
        },
        Command::Rename {
            parent,
            name: b"snapshot".to_vec(),
            new_parent: other,
            new_name: b"snapshot".to_vec(),
            flags: RenameFlags::default(),
            now: Timestamp(2),
        },
        Command::Rename {
            parent: FileId::ROOT,
            name: b"artifacts".to_vec(),
            new_parent: other,
            new_name: b"artifacts".to_vec(),
            flags: RenameFlags::default(),
            now: Timestamp(2),
        },
        Command::Rename {
            parent: FileId::ROOT,
            name: b"other".to_vec(),
            new_parent: FileId::ROOT,
            new_name: b"artifacts".to_vec(),
            flags: RenameFlags {
                exchange: true,
                noreplace: false,
            },
            now: Timestamp(2),
        },
    ] {
        assert!(
            matches!(db.run(command.clone()).0, Err(NestError::Busy(_))),
            "{command:?}"
        );
    }
    assert_eq!(db.lookup(root, "weights.bin"), Some(file.id));
    // Unrelated siblings and aliases remain ordinary mutable namespace entries.
    db.mkdir(parent, "unrelated");
    db.ok(Command::Unlink {
        parent: FileId::ROOT,
        name: b"source.bin".to_vec(),
        now: Timestamp(2),
    });
}

#[test]
fn binding_checks_complete_names_generations_and_owner_atomically() {
    let mut db = Db::new();
    let root = db.mkdir(FileId::ROOT, "snapshot");
    let a = sealed(&mut db, root, "a");
    let b = sealed(&mut db, root, "b");
    let epoch = claim(&mut db, "fixture");
    let mut stale = b.clone();
    stale.generation = Generation(b.generation.0 + 1);
    for files in [
        vec![],
        vec![("a", &a)],
        vec![("a", &a), ("a", &b)],
        vec![("a", &a), ("wrong", &b)],
        vec![("a", &a), ("b", &stale)],
    ] {
        assert!(bind(&mut db, "fixture", epoch, root, true, &files).is_err());
        assert!(composition::snapshots(&db.c, "fixture").unwrap().is_empty());
        assert!(
            composition::get(&db.c, "fixture")
                .unwrap()
                .unwrap()
                .files
                .is_empty()
        );
    }
    assert_eq!(
        bind(
            &mut db,
            "fixture",
            epoch + 1,
            root,
            true,
            &[("a", &a), ("b", &b)]
        ),
        Err(NestError::Stale)
    );
    bind(
        &mut db,
        "fixture",
        epoch,
        root,
        true,
        &[("a", &a), ("b", &b)],
    )
    .unwrap();
    bind(
        &mut db,
        "fixture",
        epoch,
        root,
        true,
        &[("b", &b), ("a", &a)],
    )
    .unwrap();
    assert!(
        bind(
            &mut db,
            "fixture",
            epoch,
            root,
            false,
            &[("a", &a), ("b", &b)]
        )
        .is_err()
    );
    assert_eq!(
        composition::snapshots(&db.c, "fixture").unwrap()[0]
            .files
            .len(),
        2
    );
}

#[test]
fn binding_rejects_nested_mutable_and_symlink_inputs() {
    let mut db = Db::new();
    let root = db.mkdir(FileId::ROOT, "snapshot");
    let a = db.file(root, "a", 1, 8);
    let epoch = claim(&mut db, "fixture");
    assert!(bind(&mut db, "fixture", epoch, root, false, &[("a", &a)]).is_err());
    db.ok(Command::Seal {
        file: a.id,
        sealed: true,
        now: Timestamp(1),
    });
    db.mkdir(root, "nested");
    assert!(bind(&mut db, "fixture", epoch, root, false, &[("a", &a)]).is_err());
    db.ok(Command::Rmdir {
        parent: root,
        name: b"nested".to_vec(),
        now: Timestamp(1),
    });
    db.ok(Command::Symlink {
        parent: root,
        name: b"alias".to_vec(),
        target: b"a".to_vec(),
        now: Timestamp(1),
    });
    assert!(bind(&mut db, "fixture", epoch, root, false, &[("a", &a)]).is_err());
    db.ok(Command::Unlink {
        parent: root,
        name: b"alias".to_vec(),
        now: Timestamp(1),
    });
    bind(&mut db, "fixture", epoch, root, false, &[("a", &a)]).unwrap();
    assert!(matches!(
        model(
            &mut db,
            ModelCommand::Publish {
                key: "fixture".into(),
                owner: "worker".into(),
                epoch,
                publication: "input-only".into(),
            }
        ),
        Err(NestError::Invalid(_))
    ));
}

#[test]
fn snapshot_bindings_survive_database_snapshot_and_reopen() {
    let mut db = Db::new();
    let (_, root, _, _) = fixture(&mut db);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("meta.sqlite");
    db.c.execute("VACUUM INTO ?1", [path.to_str().unwrap()])
        .unwrap();
    let restored = nest_meta::open_write(&path).unwrap();
    assert_eq!(
        composition::snapshots(&db.c, "fixture").unwrap(),
        composition::snapshots(&restored, "fixture").unwrap()
    );
    assert!(nest_meta::fsck::check(&restored).unwrap().is_empty());
    assert!(matches!(
        nest_meta::apply(
            &restored,
            &Command::Unlink {
                parent: root,
                name: b"weights.bin".to_vec(),
                now: Timestamp(2),
            }
        )
        .unwrap()
        .0,
        Err(NestError::Busy(_))
    ));
    restored
        .execute("DELETE FROM model_snapshot_files WHERE key = 'fixture'", [])
        .unwrap();
    assert!(!nest_meta::fsck::check(&restored).unwrap().is_empty());
}

#[test]
fn schema_v4_migration_preserves_existing_generation_pins() {
    let mut db = Db::new();
    let a = sealed(&mut db, FileId::ROOT, "source.bin");
    let epoch = claim(&mut db, "fixture");
    model(
        &mut db,
        ModelCommand::Pin {
            key: "fixture".into(),
            owner: "worker".into(),
            epoch,
            files: vec![(a.id, a.generation)],
        },
    )
    .unwrap();
    db.c.execute_batch(
        "DROP TABLE model_modules; DROP TABLE model_publications; DROP TABLE model_owned_outputs; DROP TABLE model_snapshot_files; DROP TABLE model_snapshots; PRAGMA user_version = 4;",
    )
    .unwrap();
    nest_meta::schema::init(&db.c).unwrap();
    assert_eq!(
        composition::get(&db.c, "fixture").unwrap().unwrap().files,
        [(a.id, a.generation)]
    );
    assert!(composition::snapshots(&db.c, "fixture").unwrap().is_empty());
    assert!(nest_meta::fsck::check(&db.c).unwrap().is_empty());
}

#[test]
fn shared_snapshot_remains_protected_until_every_artifact_retires() {
    let mut db = Db::new();
    let (_, root, file, epoch) = fixture(&mut db);
    let second = claim(&mut db, "second");
    bind(
        &mut db,
        "second",
        second,
        root,
        false,
        &[("weights.bin", &file)],
    )
    .unwrap();
    model(
        &mut db,
        ModelCommand::Stop {
            key: "fixture".into(),
            owner: "worker".into(),
            epoch,
            receipt: "stopped".into(),
        },
    )
    .unwrap();
    assert!(matches!(
        db.run(Command::Unlink {
            parent: root,
            name: b"weights.bin".to_vec(),
            now: Timestamp(2)
        })
        .0,
        Err(NestError::Busy(_))
    ));
    model(
        &mut db,
        ModelCommand::Stop {
            key: "second".into(),
            owner: "worker".into(),
            epoch: second,
            receipt: "stopped".into(),
        },
    )
    .unwrap();
    db.ok(Command::Unlink {
        parent: root,
        name: b"weights.bin".to_vec(),
        now: Timestamp(2),
    });
}

#[test]
fn usage_and_deferred_reset_keep_the_bound_path_until_release() {
    let mut db = Db::new();
    let (_, root, _, epoch) = fixture(&mut db);
    model(
        &mut db,
        ModelCommand::Publish {
            key: "fixture".into(),
            owner: "worker".into(),
            epoch,
            publication: "verified".into(),
        },
    )
    .unwrap();
    model(
        &mut db,
        ModelCommand::Use {
            key: "fixture".into(),
            epoch,
            lease: "consumer".into(),
        },
    )
    .unwrap();
    model(
        &mut db,
        ModelCommand::Reset {
            key: "fixture".into(),
            epoch,
            when_idle: true,
        },
    )
    .unwrap();
    assert!(!composition::snapshots(&db.c, "fixture").unwrap().is_empty());
    assert!(matches!(
        db.run(Command::Unlink {
            parent: root,
            name: b"weights.bin".to_vec(),
            now: Timestamp(2)
        })
        .0,
        Err(NestError::Busy(_))
    ));
    model(
        &mut db,
        ModelCommand::Release {
            key: "fixture".into(),
            epoch,
            lease: "consumer".into(),
        },
    )
    .unwrap();
    assert!(composition::snapshots(&db.c, "fixture").unwrap().is_empty());
    db.ok(Command::Unlink {
        parent: root,
        name: b"weights.bin".to_vec(),
        now: Timestamp(2),
    });
}
