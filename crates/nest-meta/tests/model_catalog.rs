//! Catalog contracts use generic fixtures and exercise refusal before success.
mod common;
use common::Db;
use nest_meta::{Command, ModelCommand, ModelReply, Reply, composition};
use nest_types::*;

fn run(db: &mut Db, cmd: ModelCommand) -> Result<Reply, NestError> {
    db.run(Command::Model(cmd)).0
}
fn claim(db: &mut Db) -> u64 {
    match run(
        db,
        ModelCommand::Claim {
            key: "fixture-key".into(),
            owner: "worker".into(),
        },
    )
    .unwrap()
    {
        Reply::Model(ModelReply::Epoch(epoch)) => epoch,
        other => panic!("{other:?}"),
    }
}
fn declaration(revision: u64) -> ModelCommand {
    ModelCommand::PutModule {
        name: "fixture-module".into(),
        spec: "{\"schema\":1}".into(),
        profile: "{\"groups\":{}}".into(),
        key: "fixture-key".into(),
        qualification_policy: "fixture-policy".into(),
        expected_revision: revision,
    }
}
fn sealed(db: &mut Db, root: FileId, name: &str) -> FileAttr {
    let a = db.file(root, name, 1, 8);
    db.ok(Command::Seal {
        file: a.id,
        sealed: true,
        now: Timestamp(2),
    });
    db.attr(a.id)
}
fn publication(epoch: u64, root: FileId, a: &FileAttr, receipt: &FileAttr) -> ModelCommand {
    ModelCommand::PublishComplete {
        key: "fixture-key".into(),
        owner: "worker".into(),
        epoch,
        root,
        files: vec![(b"weights.bin".to_vec(), a.id, a.generation)],
        publication: "fixture-publication".into(),
        export_receipt: (receipt.id, receipt.generation),
        qualification_receipt: (receipt.id, receipt.generation),
        qualification_policy: "fixture-policy".into(),
    }
}

#[test]
fn module_updates_require_exact_cas_and_do_not_modify_artifacts() {
    let mut db = Db::new();
    assert_eq!(run(&mut db, declaration(1)), Err(NestError::Stale));
    assert_eq!(run(&mut db, declaration(0)), Ok(Reply::Revision(1)));
    assert_eq!(run(&mut db, declaration(0)), Err(NestError::Stale));
    assert_eq!(run(&mut db, declaration(1)), Ok(Reply::Revision(2)));
    let module = composition::module(&db.c, "fixture-module")
        .unwrap()
        .unwrap();
    assert_eq!(module.revision, 2);
    assert_eq!(composition::modules(&db.c).unwrap(), [module]);
    assert!(composition::get(&db.c, "fixture-key").unwrap().is_none());
}

#[test]
fn complete_publish_requires_exact_output_and_pinned_verified_references() {
    let mut db = Db::new();
    let root = db.mkdir(FileId::ROOT, "output");
    let a = sealed(&mut db, root, "weights.bin");
    let receipt = sealed(&mut db, FileId::ROOT, "receipt");
    let epoch = claim(&mut db);
    let publish = publication(epoch, root, &a, &receipt);
    assert!(
        run(&mut db, publish.clone()).is_err(),
        "pins alone cannot publish"
    );
    run(
        &mut db,
        ModelCommand::BindSnapshot {
            key: "fixture-key".into(),
            owner: "worker".into(),
            epoch,
            root,
            output: true,
            files: vec![(b"weights.bin".to_vec(), a.id, a.generation)],
        },
    )
    .unwrap();
    assert!(
        run(&mut db, publish.clone()).is_err(),
        "receipt must be pinned"
    );
    run(
        &mut db,
        ModelCommand::Pin {
            key: "fixture-key".into(),
            owner: "worker".into(),
            epoch,
            files: vec![(receipt.id, receipt.generation)],
        },
    )
    .unwrap();
    let mut wrong = publish.clone();
    if let ModelCommand::PublishComplete { files, .. } = &mut wrong {
        files.clear();
    }
    assert!(run(&mut db, wrong).is_err());
    run(&mut db, publish).unwrap();
    let p = composition::complete_publication(&db.c, "fixture-key")
        .unwrap()
        .unwrap();
    assert_eq!(p.epoch, epoch);
    assert_eq!(p.qualification_policy, "fixture-policy");
    run(
        &mut db,
        ModelCommand::Reset {
            key: "fixture-key".into(),
            epoch,
            when_idle: false,
        },
    )
    .unwrap();
    let next = claim(&mut db);
    assert!(next > epoch);
    assert_eq!(
        composition::complete_publication(&db.c, "fixture-key")
            .unwrap()
            .unwrap(),
        p,
        "reset preserves verification evidence"
    );
    assert!(
        run(&mut db, publication(epoch, root, &a, &receipt)).is_err(),
        "stale worker cannot publish"
    );
}

#[test]
fn owned_import_never_claims_an_existing_input_or_an_alias_collision() {
    let mut db = Db::new();
    let input = sealed(&mut db, FileId::ROOT, "input");
    let root = db.mkdir(FileId::ROOT, "derived");
    let epoch = claim(&mut db);
    let import = |file, name: &[u8], epoch| ModelCommand::ImportOutput {
        key: "fixture-key".into(),
        owner: "worker".into(),
        epoch,
        parent: root,
        name: name.to_vec(),
        file,
        perm: 0o644,
        size: 8,
        mtime: Timestamp(3),
        node: NodeId(1),
        now: Timestamp(3),
    };
    assert!(run(&mut db, import(input.id, b"weights.bin", epoch)).is_err());
    assert!(
        composition::owned_outputs(&db.c, "fixture-key")
            .unwrap()
            .is_empty()
    );
    let file = match db.ok(Command::ReserveFileIds { count: 1 }) {
        Reply::FileIds(f) => f,
        other => panic!("{other:?}"),
    };
    assert_eq!(
        run(&mut db, import(file, b"weights.bin", epoch + 1)),
        Err(NestError::Stale)
    );
    let (_, effects) = db.run(Command::Model(import(file, b"weights.bin", epoch)));
    assert!(
        !effects.is_empty(),
        "normal import invalidation effects propagate"
    );
    let owned = composition::owned_outputs(&db.c, "fixture-key").unwrap();
    assert_eq!(owned.len(), 1);
    assert_eq!(
        (owned[0].file, owned[0].generation, owned[0].epoch),
        (file, Generation(1), epoch)
    );
    assert!(db.attr(file).sealed);
    let second = match db.ok(Command::ReserveFileIds { count: 1 }) {
        Reply::FileIds(f) => f,
        other => panic!("{other:?}"),
    };
    assert_eq!(
        run(&mut db, import(second, b"weights.bin", epoch)),
        Err(NestError::Exists)
    );
    assert!(nest_meta::query::getattr(&db.c, second).unwrap().is_none());
    assert_eq!(
        composition::owned_outputs(&db.c, "fixture-key").unwrap(),
        owned
    );
    run(
        &mut db,
        ModelCommand::Stop {
            key: "fixture-key".into(),
            owner: "worker".into(),
            epoch,
            receipt: "worker-stopped".into(),
        },
    )
    .unwrap();
    assert_eq!(
        composition::owned_outputs(&db.c, "fixture-key").unwrap(),
        owned
    );
    assert_eq!(db.lookup(FileId::ROOT, "input"), Some(input.id));
}

#[test]
fn snapshot_round_trip_migrates_v5_and_preserves_catalog_and_ownership() {
    let mut db = Db::new();
    let epoch = claim(&mut db);
    let dir = tempfile::tempdir().unwrap();
    let legacy = dir.path().join("legacy.sqlite");
    db.c.execute_batch("DROP TABLE model_modules; DROP TABLE model_publications; DROP TABLE model_owned_outputs; PRAGMA user_version = 5;").unwrap();
    db.c.execute("VACUUM INTO ?1", [legacy.to_str().unwrap()])
        .unwrap();
    let restored = nest_meta::open_write(&legacy).unwrap();
    assert_eq!(
        composition::get(&restored, "fixture-key")
            .unwrap()
            .unwrap()
            .epoch,
        epoch
    );
    assert!(composition::modules(&restored).unwrap().is_empty());
    assert!(nest_meta::fsck::check(&restored).unwrap().is_empty());
    db.c = restored;
    run(&mut db, declaration(0)).unwrap();
    let root = db.mkdir(FileId::ROOT, "derived");
    let file = match db.ok(Command::ReserveFileIds { count: 1 }) {
        Reply::FileIds(f) => f,
        other => panic!("{other:?}"),
    };
    run(
        &mut db,
        ModelCommand::ImportOutput {
            key: "fixture-key".into(),
            owner: "worker".into(),
            epoch,
            parent: root,
            name: b"weights.bin".to_vec(),
            file,
            perm: 0o644,
            size: 8,
            mtime: Timestamp(3),
            node: NodeId(1),
            now: Timestamp(3),
        },
    )
    .unwrap();
    let a = db.attr(file);
    let receipt = sealed(&mut db, FileId::ROOT, "receipt");
    run(
        &mut db,
        ModelCommand::BindSnapshot {
            key: "fixture-key".into(),
            owner: "worker".into(),
            epoch,
            root,
            output: true,
            files: vec![(b"weights.bin".to_vec(), a.id, a.generation)],
        },
    )
    .unwrap();
    run(
        &mut db,
        ModelCommand::Pin {
            key: "fixture-key".into(),
            owner: "worker".into(),
            epoch,
            files: vec![(receipt.id, receipt.generation)],
        },
    )
    .unwrap();
    run(&mut db, publication(epoch, root, &a, &receipt)).unwrap();
    let path = dir.path().join("current.sqlite");
    db.c.execute("VACUUM INTO ?1", [path.to_str().unwrap()])
        .unwrap();
    let restored = nest_meta::open_write(&path).unwrap();
    assert_eq!(
        composition::modules(&db.c).unwrap(),
        composition::modules(&restored).unwrap()
    );
    assert_eq!(
        composition::owned_outputs(&db.c, "fixture-key").unwrap(),
        composition::owned_outputs(&restored, "fixture-key").unwrap()
    );
    assert_eq!(
        composition::complete_publication(&db.c, "fixture-key").unwrap(),
        composition::complete_publication(&restored, "fixture-key").unwrap()
    );
    assert!(nest_meta::fsck::check(&restored).unwrap().is_empty());
    restored
        .execute(
            "DELETE FROM model_pins WHERE file = ?1",
            [receipt.id.0 as i64],
        )
        .unwrap();
    assert!(
        !nest_meta::fsck::check(&restored).unwrap().is_empty(),
        "complete publication receipt pins must survive snapshots"
    );
}

#[test]
fn fsck_rejects_unproven_ownership_epochs() {
    let mut db = Db::new();
    claim(&mut db);
    db.c.execute("INSERT INTO model_owned_outputs (key, epoch, parent, name, file, gen) VALUES ('fixture-key', 2, 1, X'61', 9, 1)", []).unwrap();
    assert!(
        !nest_meta::fsck::check(&db.c).unwrap().is_empty(),
        "future worker epoch is not an import proof"
    );
}
