//! Management failures use synthetic namespaces and in-process metadata only.
//! These tests make no claim about Docker, RDMA, loader or numerical readiness.
use nest_meta::{Command, ModelCommand, ModelReply, Reply, composition};
use nest_place::composition::Manager;
use nest_testkit::TestCluster;
use nest_types::{FileId, NestError};
use serde_json::{Value, json};

fn digest(byte: char) -> String {
    format!("sha256:{}", byte.to_string().repeat(64))
}
fn spec(name: &str, source: &str) -> Value {
    json!({
        "schema":"sparknest.model-composition.v1","name":name,
        "base":{"source":source,"content_manifest":digest('a')},"alternates":[],
        "output":{"loader_profile":"base-compatible-indexed-v1","precision":"match-base","configuration":"base-unchanged","tokenizer":"base-unchanged","non_weight_mismatch":"reject","namespace":"/generic-cache"},
        "preparation":{"trigger":"explicit-prepare-or-external-use-wrapper","adapter_image":format!("generic-worker@{}",digest('b')),"recipe":digest('c'),"parameters":{},"lossy_conversion":"deny","qualification_profile":digest('d'),"activation_calibration":"preserve-and-qualify","max_runtime_seconds":10,"max_ram_bytes":268435456,"cpu_threads":1,"device":"cpu","worker_pool":"cpu"},
        "cache":{"publication":"sealed-regular-files","reuse_unchanged":true,"prefer_reflink":false,"max_new_physical_bytes":1048576,"account_inflight":true,"eviction":"least-recently-used-unpinned","retain_dependencies":true}
    })
}
fn profile() -> Value {
    json!({"schema":"sparknest.operand-profile.v1","groups":{"operand":["weight"]},"static_activation_calibration":false})
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn declarative_modules_cas_deduplicate_and_refuse_unassigned_preparation() {
    let cluster = TestCluster::start(1).await;
    let manager = Manager::new(cluster.node(1).vfs.clone(), None, None);
    let first = manager
        .put(
            "first",
            serde_json::from_value(spec("ignored", "/generic-base")).unwrap(),
            serde_json::from_value(profile()).unwrap(),
            0,
        )
        .await
        .unwrap();
    let second = manager
        .put(
            "second",
            serde_json::from_value(spec("ignored", "/other-alias")).unwrap(),
            serde_json::from_value(profile()).unwrap(),
            0,
        )
        .await
        .unwrap();
    assert_eq!(
        first.key, second.key,
        "acquisition paths and friendly names are not preparation identity"
    );
    assert_eq!(manager.list().await.unwrap().len(), 2);
    assert!(matches!(
        manager
            .put(
                "first",
                serde_json::from_value(spec("first", "/generic-base")).unwrap(),
                serde_json::from_value(profile()).unwrap(),
                0
            )
            .await,
        Err(NestError::Stale)
    ));
    let error = manager.prepare("first", true).await.unwrap_err();
    assert!(error.to_string().contains("assigned CPU preparation pool"));
    let status = manager.status("first").await.unwrap();
    assert!(
        status
            .message
            .unwrap()
            .contains("assigned CPU preparation pool")
    );
    assert!(
        status.artifact.is_none(),
        "resource refusal must precede durable claim"
    );
    assert!(manager.resolve("first").await.is_err());
    assert!(manager.acquire("first", "consumer").await.is_err());
    assert!(
        matches!(manager.qualify("first").await,Err(NestError::Invalid(message)) if message.contains("QUALIFICATION_REQUIRED"))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pending_owner_survives_reset_dry_run_and_failed_stop_reconciliation() {
    let cluster = TestCluster::start(1).await;
    let manager = Manager::new(cluster.node(1).vfs.clone(), None, None);
    let first = manager
        .put(
            "first",
            serde_json::from_value(spec("first", "/generic-base")).unwrap(),
            serde_json::from_value(profile()).unwrap(),
            0,
        )
        .await
        .unwrap();
    manager
        .put(
            "second",
            serde_json::from_value(spec("second", "/generic-base")).unwrap(),
            serde_json::from_value(profile()).unwrap(),
            0,
        )
        .await
        .unwrap();
    let Reply::Model(ModelReply::Epoch(epoch)) = cluster
        .node(1)
        .meta
        .propose(Command::Model(ModelCommand::Claim {
            key: first.key.clone(),
            owner: "bounded-worker".into(),
        }))
        .await
        .unwrap()
    else {
        panic!("claim");
    };
    let report = manager.reset("first", false, true).await.unwrap();
    assert!(report.blocked);
    assert!(report.dry_run);
    assert!(!report.deletes_files);
    assert_eq!(report.shared_modules, ["first", "second"]);
    assert!(matches!(
        manager.reset("first", false, false).await,
        Err(NestError::Busy(_))
    ));
    let queued = manager.reset("first", true, false).await.unwrap();
    assert!(queued.reset.is_some());
    assert!(matches!(
        manager
            .cancel_reset("first", epoch + 1, queued.reset.unwrap())
            .await,
        Err(NestError::Stale)
    ));
    manager
        .cancel_reset("first", epoch, queued.reset.unwrap())
        .await
        .unwrap();
    assert!(manager.stop("first").await.is_err());
    assert!(manager.recover("first").await.is_err());
    let reader = cluster.node(1).meta.open_reader().unwrap();
    let artifact = composition::get(&reader, &first.key).unwrap().unwrap();
    assert_eq!(artifact.owner.as_deref(), Some("bounded-worker"));
    assert_eq!(artifact.state, "preparing");
    assert!(!artifact.reset_pending);
    assert!(manager.resolve("first").await.is_err());
    assert!(cluster.node(1).vfs.getattr(FileId::ROOT).await.is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn legacy_ready_cannot_resolve_and_complete_ready_keeps_usage_during_deferred_reset() {
    use nest_types::{Generation, NodeId, Timestamp};
    let cluster = TestCluster::start(1).await;
    let manager = Manager::new(
        cluster.node(1).vfs.clone(),
        Some("/generic-mount".into()),
        None,
    );
    let declaration = manager
        .put(
            "ready",
            serde_json::from_value(spec("ready", "/generic-base")).unwrap(),
            serde_json::from_value(profile()).unwrap(),
            0,
        )
        .await
        .unwrap();
    let root = cluster
        .node(1)
        .vfs
        .mkdir(FileId::ROOT, b"generic-output", 0o700)
        .await
        .unwrap()
        .id;
    let mut files = Vec::new();
    for name in ["weights.safetensors", "export.json", "qualification.json"] {
        let now = Timestamp::now();
        let Reply::Created(created) = cluster
            .node(1)
            .meta
            .propose(Command::Create {
                parent: root,
                name: name.as_bytes().to_vec(),
                perm: 0o444,
                node: NodeId(1),
                exclusive: true,
                now,
            })
            .await
            .unwrap()
        else {
            panic!("create fixture");
        };
        let attr = created.attr;
        let object = nest_store::ObjectKey::new(attr.id, attr.generation);
        cluster.node(1).data.wait_ready(object).await;
        let path = cluster.node(1).data.store().path(object);
        tokio::task::spawn_blocking(move || std::fs::write(path, b"generic"))
            .await
            .unwrap()
            .unwrap();
        cluster
            .node(1)
            .meta
            .propose(Command::Finalize {
                file: attr.id,
                epoch: attr.epoch,
                size: 7,
                mtime: now,
                now,
            })
            .await
            .unwrap();
        cluster.node(1).vfs.seal(attr.id, true).await.unwrap();
        files.push((name.as_bytes().to_vec(), attr.id, Generation(1)));
    }
    let key = declaration.key;
    let Reply::Model(ModelReply::Epoch(epoch)) = cluster
        .node(1)
        .meta
        .propose(Command::Model(ModelCommand::Claim {
            key: key.clone(),
            owner: "fixture-worker".into(),
        }))
        .await
        .unwrap()
    else {
        panic!("claim");
    };
    cluster
        .node(1)
        .meta
        .propose(Command::Model(ModelCommand::BindSnapshot {
            key: key.clone(),
            owner: "fixture-worker".into(),
            epoch,
            root,
            output: true,
            files: files.clone(),
        }))
        .await
        .unwrap();
    cluster
        .node(1)
        .meta
        .propose(Command::Model(ModelCommand::Publish {
            key: key.clone(),
            owner: "fixture-worker".into(),
            epoch,
            publication: "generic-output-identity".into(),
        }))
        .await
        .unwrap();
    assert!(
        manager.resolve("ready").await.is_err(),
        "legacy publication must never bypass manager qualification"
    );
    assert!(manager.acquire("ready", "consumer").await.is_err());
    // New epoch uses the same real sealed inventory. This synthetic metadata
    // attestation tests lease/resolve plumbing, not checkpoint qualification.
    manager.reset("ready", false, false).await.unwrap();
    let Reply::Model(ModelReply::Epoch(epoch)) = cluster
        .node(1)
        .meta
        .propose(Command::Model(ModelCommand::Claim {
            key: key.clone(),
            owner: "fixture-worker".into(),
        }))
        .await
        .unwrap()
    else {
        panic!("claim");
    };
    cluster
        .node(1)
        .meta
        .propose(Command::Model(ModelCommand::BindSnapshot {
            key: key.clone(),
            owner: "fixture-worker".into(),
            epoch,
            root,
            output: true,
            files: files.clone(),
        }))
        .await
        .unwrap();
    cluster
        .node(1)
        .meta
        .propose(Command::Model(ModelCommand::PublishComplete {
            key: key.clone(),
            owner: "fixture-worker".into(),
            epoch,
            root,
            files: files.clone(),
            publication: "generic-output-identity".into(),
            export_receipt: (files[1].1, files[1].2),
            qualification_receipt: (files[2].1, files[2].2),
            qualification_policy: declaration.qualification_policy,
        }))
        .await
        .unwrap();
    let resolved = manager.acquire("ready", "consumer").await.unwrap();
    assert_eq!(
        resolved.path,
        std::path::PathBuf::from("/generic-mount/generic-output")
    );
    manager.reset("ready", true, false).await.unwrap();
    assert!(manager.resolve("ready").await.is_err());
    assert!(manager.acquire("ready", "second-consumer").await.is_err());
    manager.release(&key, epoch, "consumer").await.unwrap();
    let status = manager.status("ready").await.unwrap();
    assert_eq!(status.phase, "retired");
    for (_, file, _) in files {
        assert!(
            cluster.node(1).vfs.getattr(file).await.is_ok(),
            "cache retirement deletes no fixture/source bytes"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn deferred_reset_reclaims_only_metadata_proven_new_shards_and_preserves_receipts_and_inputs()
{
    use nest_types::{Generation, Timestamp};
    let cluster = TestCluster::start(1).await;
    let manager = Manager::new(
        cluster.node(1).vfs.clone(),
        Some("/generic-mount".into()),
        None,
    );
    let declaration = manager
        .put(
            "cache",
            serde_json::from_value(spec("cache", "/generic-base")).unwrap(),
            serde_json::from_value(profile()).unwrap(),
            0,
        )
        .await
        .unwrap();
    let key = declaration.key;
    let Reply::Model(ModelReply::Epoch(epoch)) = cluster
        .node(1)
        .meta
        .propose(Command::Model(ModelCommand::Claim {
            key: key.clone(),
            owner: "fixture-worker".into(),
        }))
        .await
        .unwrap()
    else {
        panic!("claim");
    };
    let root = cluster
        .node(1)
        .vfs
        .mkdir(FileId::ROOT, b"generic-derived", 0o700)
        .await
        .unwrap()
        .id;
    let mut files = Vec::new();
    for name in ["changed.safetensors", "export.json", "qualification.json"] {
        let Reply::FileIds(file) = cluster
            .node(1)
            .meta
            .propose(Command::ReserveFileIds { count: 1 })
            .await
            .unwrap()
        else {
            panic!("reserve");
        };
        let path = cluster
            .node(1)
            .data
            .store()
            .path(nest_store::ObjectKey::new(file, Generation(1)));
        tokio::task::spawn_blocking(move || std::fs::write(path, b"generic"))
            .await
            .unwrap()
            .unwrap();
        let now = Timestamp::now();
        cluster
            .node(1)
            .meta
            .propose(Command::Model(ModelCommand::ImportOutput {
                key: key.clone(),
                owner: "fixture-worker".into(),
                epoch,
                parent: root,
                name: name.as_bytes().to_vec(),
                file,
                perm: 0o444,
                size: 7,
                mtime: now,
                node: cluster.node(1).data.id(),
                now,
            }))
            .await
            .unwrap();
        files.push((name.as_bytes().to_vec(), file, Generation(1)));
    }
    let source_root = cluster
        .node(1)
        .vfs
        .mkdir(FileId::ROOT, b"generic-base", 0o700)
        .await
        .unwrap()
        .id;
    let Reply::FileIds(source) = cluster
        .node(1)
        .meta
        .propose(Command::ReserveFileIds { count: 1 })
        .await
        .unwrap()
    else {
        panic!("reserve source");
    };
    let source_path = cluster
        .node(1)
        .data
        .store()
        .path(nest_store::ObjectKey::new(source, Generation(1)));
    tokio::task::spawn_blocking(move || std::fs::write(source_path, b"generic"))
        .await
        .unwrap()
        .unwrap();
    let now = Timestamp::now();
    cluster
        .node(1)
        .meta
        .propose(Command::Import {
            parent: source_root,
            name: b"source.safetensors".to_vec(),
            file: source,
            perm: 0o444,
            size: 7,
            mtime: now,
            node: cluster.node(1).data.id(),
            sealed: true,
            now,
        })
        .await
        .unwrap();
    cluster
        .node(1)
        .vfs
        .link(source, root, b"reused.safetensors")
        .await
        .unwrap();
    files.push((b"reused.safetensors".to_vec(), source, Generation(1)));
    cluster
        .node(1)
        .meta
        .propose(Command::Model(ModelCommand::BindSnapshot {
            key: key.clone(),
            owner: "fixture-worker".into(),
            epoch,
            root,
            output: true,
            files: files.clone(),
        }))
        .await
        .unwrap();
    cluster
        .node(1)
        .meta
        .propose(Command::Model(ModelCommand::PublishComplete {
            key: key.clone(),
            owner: "fixture-worker".into(),
            epoch,
            root,
            files: files.clone(),
            publication: "fixture-publication".into(),
            export_receipt: (files[1].1, files[1].2),
            qualification_receipt: (files[2].1, files[2].2),
            qualification_policy: declaration.qualification_policy,
        }))
        .await
        .unwrap();
    let preview = manager.reset("cache", false, true).await.unwrap();
    assert_eq!(preview.eligible_shard_bytes, 7);
    assert_eq!(preview.reclaimed_shard_bytes, 0);
    assert!(
        cluster
            .node(1)
            .vfs
            .lookup(root, b"changed.safetensors")
            .await
            .is_ok()
    );
    manager.acquire("cache", "consumer").await.unwrap();
    manager.reset("cache", true, false).await.unwrap();
    assert!(
        cluster
            .node(1)
            .vfs
            .lookup(root, b"changed.safetensors")
            .await
            .is_ok(),
        "active consumer retains its payload"
    );
    manager.release(&key, epoch, "consumer").await.unwrap();
    assert!(matches!(
        cluster
            .node(1)
            .vfs
            .lookup(root, b"changed.safetensors")
            .await,
        Err(NestError::NotFound)
    ));
    for name in [b"export.json".as_slice(), b"qualification.json".as_slice()] {
        assert!(cluster.node(1).vfs.lookup(root, name).await.is_ok());
    }
    assert!(
        cluster
            .node(1)
            .vfs
            .lookup(source_root, b"source.safetensors")
            .await
            .is_ok()
    );
    assert!(
        cluster
            .node(1)
            .vfs
            .lookup(root, b"reused.safetensors")
            .await
            .is_ok()
    );
    let reader = cluster.node(1).meta.open_reader().unwrap();
    assert_eq!(
        composition::owned_outputs(&reader, &key).unwrap().len(),
        3,
        "historical ownership is retained"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reset_all_includes_orphan_keys_and_continues_after_a_busy_entry() {
    let cluster = TestCluster::start(1).await;
    let manager = Manager::new(cluster.node(1).vfs.clone(), None, None);
    let module = manager
        .put(
            "busy",
            serde_json::from_value(spec("busy", "/generic-base")).unwrap(),
            serde_json::from_value(profile()).unwrap(),
            0,
        )
        .await
        .unwrap();
    for key in [&module.key, "zz-generic-orphan"] {
        let Reply::Model(ModelReply::Epoch(epoch)) = cluster
            .node(1)
            .meta
            .propose(Command::Model(ModelCommand::Claim {
                key: key.into(),
                owner: "fixture-worker".into(),
            }))
            .await
            .unwrap()
        else {
            panic!("claim");
        };
        if key == "zz-generic-orphan" {
            cluster
                .node(1)
                .meta
                .propose(Command::Model(ModelCommand::Stop {
                    key: key.into(),
                    owner: "fixture-worker".into(),
                    epoch,
                    receipt: "fixture-no-launch".into(),
                }))
                .await
                .unwrap();
        }
    }
    let preview = manager.reset_all(false, true).await.unwrap();
    assert_eq!(preview.len(), 2);
    let reports = manager.reset_all(false, false).await.unwrap();
    let busy = reports.iter().find(|r| r.key == module.key).unwrap();
    assert!(busy.blocked);
    assert_eq!(busy.state, "preparing");
    let orphan = reports
        .iter()
        .find(|r| r.key == "zz-generic-orphan")
        .unwrap();
    assert!(!orphan.blocked);
    assert_eq!(orphan.state, "retired");
    assert!(orphan.shared_modules.is_empty());
    assert_eq!(manager.cache_status().await.unwrap().len(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn raw_declaration_put_matches_typed_identity_and_rejects_duplicate_or_unknown_fields() {
    let cluster = TestCluster::start(1).await;
    let manager = Manager::new(cluster.node(1).vfs.clone(), None, None);
    let body =
        json!({"spec":spec("ignored","/generic-base"),"profile":profile(),"expected_revision":0});
    let raw = manager
        .put_json("raw", serde_json::to_vec(&body).unwrap().into())
        .await
        .unwrap();
    let typed = manager
        .put(
            "typed",
            serde_json::from_value(spec("ignored", "/generic-base")).unwrap(),
            serde_json::from_value(profile()).unwrap(),
            0,
        )
        .await
        .unwrap();
    assert_eq!(raw.key, typed.key);
    assert_eq!(raw.qualification_policy, typed.qualification_policy);
    let mut unknown = body.clone();
    unknown["unknown"] = true.into();
    assert!(
        manager
            .put_json("unknown", serde_json::to_vec(&unknown).unwrap().into())
            .await
            .is_err()
    );
    let duplicate = format!(
        "{{\"spec\":{},\"profile\":{},\"expected_revision\":0,\"expected_revision\":1}}",
        body["spec"], body["profile"]
    );
    assert!(
        manager
            .put_json("duplicate", duplicate.into())
            .await
            .is_err()
    );
    assert_eq!(
        manager.list().await.unwrap().len(),
        2,
        "invalid raw declarations never persist"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn another_node_can_inspect_but_cannot_qualify_or_recover_a_preparation_owner() {
    let cluster = TestCluster::start(2).await;
    let owner = Manager::new(cluster.node(1).vfs.clone(), None, None);
    let foreign = Manager::new(cluster.node(2).vfs.clone(), None, None);
    let mut declaration = spec("owned", "/generic-base");
    declaration["preparation"]["qualification_profile"] =
        nest_model::policy_identity().to_string().into();
    let module = owner
        .put(
            "owned",
            serde_json::from_value(declaration).unwrap(),
            serde_json::from_value(profile()).unwrap(),
            0,
        )
        .await
        .unwrap();
    cluster
        .node(1)
        .meta
        .propose(Command::Model(ModelCommand::Claim {
            key: module.key.clone(),
            owner: "cpu-1-fixture".into(),
        }))
        .await
        .unwrap();
    let status = foreign.status("owned").await.unwrap();
    assert_eq!(
        status.artifact.unwrap().owner.as_deref(),
        Some("cpu-1-fixture")
    );
    let error = foreign.qualify("owned").await.unwrap_err();
    assert!(error.to_string().contains("owning CPU node"));
    assert!(matches!(
        foreign.stop("owned").await,
        Err(NestError::NotPermitted(_))
    ));
    assert!(matches!(
        foreign.recover("owned").await,
        Err(NestError::NotPermitted(_))
    ));
    let reader = cluster.node(1).meta.open_reader().unwrap();
    let retained = composition::get(&reader, &module.key).unwrap().unwrap();
    assert_eq!(retained.state, "preparing");
    assert_eq!(retained.owner.as_deref(), Some("cpu-1-fixture"));
}
