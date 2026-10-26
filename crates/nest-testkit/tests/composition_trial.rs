//! Opt-in synthetic CPU trials. No downloaded weights, serving devices, live
//! mounts or daemons are used. Build the exact current worker into a LOCAL
//! image containing /usr/bin/timeout, then set SPARKNEST_COMPOSITION_TRIAL_IMAGE
//! to its immutable sha256 image ID. Run these ignored tests explicitly.
//!
//! The FUSE trial mounts only TestCluster's fresh temporary directory. Missing
//! Docker/FUSE is a failure, never a silently passing qualification. Optional
//! SPARKNEST_COMPOSITION_TRIAL_RECEIPTS retains evidence outside the temp tree.

use nest_meta::{composition as catalog, query};
use nest_model::{
    Digest, ExportLimits, ModuleConfig, Snapshot, policy_identity,
    worker::{Input, OperandProfile, Request, Response},
};
use nest_place::{
    composition::Manager,
    import::{ImportOptions, SealMode},
    model_worker::{Identity, Limits, Pool},
};
use nest_testkit::TestCluster;
use nest_types::{GenState, NestError};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

const FIRST: &str = "model-00001-of-00002.safetensors";
const SECOND: &str = "model-00002-of-00002.safetensors";
const INDEX: &str = "model.safetensors.index.json";
const RAM: u64 = 512 << 20;
const NEW_BYTES: u64 = 4 << 20;

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    tokio::task::spawn_blocking(f).await.unwrap()
}

fn json_file(root: &Path, name: &str, value: &Value) {
    fs::write(root.join(name), serde_json::to_vec(value).unwrap()).unwrap();
}

fn shard(path: &Path, operand: &str, value: u8, scale: f32) {
    let mut header = serde_json::to_vec(&json!({
        format!("{operand}.weight"): {"dtype":"U8","shape":[16],"data_offsets":[0,16]},
        format!("{operand}.scale"): {"dtype":"F32","shape":[1],"data_offsets":[16,20]}
    }))
    .unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend([value; 16]);
    bytes.extend(scale.to_le_bytes());
    fs::write(path, bytes).unwrap();
}

struct Fixture {
    base: PathBuf,
    alternate: PathBuf,
    base_manifest: Digest,
    alternate_manifest: Digest,
}

impl Fixture {
    fn new(root: &Path) -> Self {
        let base = root.join("synthetic-base");
        let alternate = root.join("synthetic-replacement");
        fs::create_dir_all(&base).unwrap();
        fs::create_dir_all(&alternate).unwrap();
        shard(&base.join(FIRST), "encoder", 1, 1.0);
        shard(&base.join(SECOND), "decoder", 3, 3.0);
        shard(
            &alternate.join("replacement.safetensors"),
            "encoder",
            9,
            2.0,
        );
        json_file(
            &base,
            "config.json",
            &json!({"architectures":[],"synthetic":true}),
        );
        json_file(
            &base,
            "tokenizer.json",
            &json!({"version":"1.0","synthetic":true}),
        );
        json_file(
            &base,
            INDEX,
            &json!({"metadata":{"total_size":40},"weight_map":{
                "encoder.weight":FIRST,"encoder.scale":FIRST,
                "decoder.weight":SECOND,"decoder.scale":SECOND
            }}),
        );
        let limits = limits();
        let base_manifest = Snapshot::inspect(&base, &limits)
            .unwrap()
            .content_manifest()
            .clone();
        let alternate_manifest =
            Snapshot::inspect_sidecar(&alternate, "replacement.safetensors", None, &limits)
                .unwrap()
                .content_manifest()
                .clone();
        Self {
            base,
            alternate,
            base_manifest,
            alternate_manifest,
        }
    }

    fn module(&self, pool: &Pool) -> ModuleConfig {
        serde_json::from_value(json!({
            "schema":"sparknest.model-composition.v1","name":"synthetic-composition",
            "base":{"source":"/synthetic-base","content_manifest":self.base_manifest},
            "alternates":[{"name":"replacement","kind":"replacement",
                "source":"/synthetic-replacement","content_manifest":self.alternate_manifest,
                "selection":"all","requires":{"base_manifest":self.base_manifest}}],
            "output":{"loader_profile":"base-compatible-indexed-v1","precision":"match-base",
                "configuration":"base-unchanged","tokenizer":"base-unchanged",
                "non_weight_mismatch":"reject","namespace":"/synthetic-prepared"},
            "preparation":{"trigger":"explicit-prepare-or-external-use-wrapper",
                "adapter_image":format!("synthetic-worker@{}",pool.image),
                "recipe":Digest::of(b"synthetic-exact-trial-v1"),"parameters":{},
                "lossy_conversion":"deny","qualification_profile":policy_identity(),
                "activation_calibration":"preserve-and-qualify","max_runtime_seconds":30,
                "max_ram_bytes":RAM,"cpu_threads":1,"device":"cpu","worker_pool":pool.name},
            "cache":{"publication":"sealed-regular-files","reuse_unchanged":true,
                "prefer_reflink":false,"max_new_physical_bytes":NEW_BYTES,"account_inflight":true,
                "eviction":"least-recently-used-unpinned","retain_dependencies":true}
        }))
        .unwrap()
    }

    fn inputs(&self) -> (Input, Vec<Input>) {
        (
            Input::Indexed {
                root: self.base.clone(),
            },
            vec![Input::Sidecar {
                root: self.alternate.clone(),
                tensors: "replacement.safetensors".into(),
                selection_manifest: None,
            }],
        )
    }
}

fn profile() -> OperandProfile {
    OperandProfile {
        schema: "sparknest.operand-profile.v1".into(),
        groups: BTreeMap::from([
            (
                "encoder".into(),
                vec!["encoder.weight".into(), "encoder.scale".into()],
            ),
            (
                "decoder".into(),
                vec!["decoder.weight".into(), "decoder.scale".into()],
            ),
        ]),
        static_activation_calibration: false,
    }
}

fn limits() -> ExportLimits {
    ExportLimits {
        max_output_bytes: NEW_BYTES,
        ..Default::default()
    }
}

fn admission() -> Limits {
    Limits {
        cpu_threads: 1,
        ram_bytes: RAM,
        new_bytes: NEW_BYTES,
        runtime_seconds: 30,
    }
}

fn pool(root: PathBuf) -> Pool {
    fs::create_dir_all(&root).unwrap();
    let image = std::env::var("SPARKNEST_COMPOSITION_TRIAL_IMAGE").expect(
        "set explicitly assigned local exact-worker sha256 image ID; images are never pulled",
    );
    Pool {
        name: "synthetic-cpu".into(),
        root,
        image: serde_json::from_value(json!(image)).expect("immutable sha256 image ID"),
        max_cpu_threads: 1,
        max_ram_bytes: RAM,
        max_new_bytes: NEW_BYTES,
        free_reserve_bytes: 16 << 20,
        max_runtime_seconds: 30,
        control_timeout_seconds: 10,
        max_pids: 64,
    }
}

fn records(pool: &Pool) -> Vec<Value> {
    let mut records = Vec::new();
    for entry in fs::read_dir(&pool.root).unwrap() {
        let path = entry.unwrap().path().join("control.json");
        if path.is_file() {
            let record: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
            assert_eq!(record["image"], json!(pool.image));
            assert_eq!(record["stopped"], true, "unconfirmed worker stop: {record}");
            assert_eq!(record["limits"]["cpu_threads"], 1);
            assert_eq!(record["limits"]["ram_bytes"], RAM);
            records.push(record);
        }
    }
    records.sort_by_key(|r| r["container"].as_str().unwrap().to_owned());
    records
}

fn retain(name: &str, value: &Value) {
    if let Ok(root) = std::env::var("SPARKNEST_COMPOSITION_TRIAL_RECEIPTS") {
        let root = PathBuf::from(root);
        assert!(root.is_absolute(), "receipt directory must be absolute");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join(name), serde_json::to_vec_pretty(value).unwrap()).unwrap();
    }
}

fn check_output(root: &Path) {
    let inspected = Snapshot::inspect(root, &limits()).unwrap();
    assert!(inspected.files().contains_key(FIRST));
    assert!(inspected.files().contains_key(SECOND));
    for (name, weight, scale) in [(FIRST, 9, 2.0_f32), (SECOND, 3, 3.0_f32)] {
        let bytes = fs::read(root.join(name)).unwrap();
        let start = 8 + u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!(&bytes[start..start + 16], &[weight; 16]);
        assert_eq!(&bytes[start + 16..], &scale.to_le_bytes());
        assert!(fs::symlink_metadata(root.join(name)).unwrap().is_file());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires explicitly assigned exact CPU Docker image; no GPU or model weights"]
async fn actual_docker_stage_assemble_and_exact_qualify() {
    let root = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(root.path());
    let pool = pool(root.path().join("pool"));
    let module = fixture.module(&pool);
    let identity = Identity {
        key: module
            .preparation_key(&profile().profile().unwrap())
            .unwrap(),
        epoch: 1,
        owner: "synthetic-standalone-trial".into(),
    };
    let (base, alternates) = fixture.inputs();
    let stage = pool
        .run(
            identity.clone(),
            Request::Stage {
                module: Box::new(module.clone()),
                profile: profile(),
                base,
                alternates,
                destination: PathBuf::new(),
                limits: limits(),
            },
            admission(),
        )
        .await
        .expect("real Docker Stage failed");
    let Response::Stage { staged } = stage.response.expect("worker rejected Stage") else {
        panic!("Stage returned another operation")
    };
    assert_eq!(staged.receipt.winners["encoder"], Some(0));
    assert_eq!(staged.receipt.winners["decoder"], None);
    assert!(staged.receipt.reused_files.contains(SECOND));
    assert!(
        !stage.staging.join(SECOND).exists(),
        "unchanged shard must stay unstaged"
    );
    let output = root.path().join("assembled");
    fs::create_dir(&output).unwrap();
    for name in staged.staged_files() {
        fs::rename(stage.staging.join(&name), output.join(name)).unwrap();
    }
    for name in &staged.receipt.reused_files {
        fs::hard_link(fixture.base.join(name), output.join(name)).unwrap();
    }
    check_output(&output);
    let (base, alternates) = fixture.inputs();
    let proof = pool
        .run(
            identity,
            Request::Qualify {
                module: Box::new(module),
                profile: profile(),
                base,
                alternates,
                output: output.clone(),
                limits: limits(),
            },
            admission(),
        )
        .await
        .expect("real Docker Qualify failed");
    let Response::Qualify { receipt } = proof.response.expect("worker rejected Qualify") else {
        panic!("Qualify returned another operation")
    };
    assert_eq!(receipt.policy, policy_identity());
    assert_eq!(receipt.compared_tensors, 4);
    assert_eq!(receipt.compared_bytes, 40);
    assert_eq!(receipt.comparisons.len(), 4);
    assert_eq!(records(&pool).len(), 2);
    // The exact current worker must reject changed bytes, even in a regular
    // assembled checkpoint. No caller-supplied boolean can publish this output.
    let shard = output.join(FIRST);
    let mut bytes = fs::read(&shard).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(shard, bytes).unwrap();
    let (base, alternates) = fixture.inputs();
    let bad = pool
        .run(
            Identity {
                key: receipt.preparation_key.clone(),
                epoch: 1,
                owner: "synthetic-standalone-trial".into(),
            },
            Request::Qualify {
                module: Box::new(fixture.module(&pool)),
                profile: profile(),
                base,
                alternates,
                output,
                limits: limits(),
            },
            admission(),
        )
        .await
        .expect("tamper rejection worker did not stop");
    assert!(
        bad.response.is_err(),
        "corrupted output passed qualification"
    );
    retain(
        "standalone.json",
        &json!({"qualification":receipt,"workers":records(&pool),
        "tampered_output_rejection":bad.response.unwrap_err()}),
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires explicitly assigned exact CPU Docker image and working user FUSE mount"]
async fn actual_manager_fuse_prepare_qualify_reset_rebuild_and_hit() {
    let c = TestCluster::start(1).await;
    c.eventually("trial ready", Duration::from_secs(5), |c| {
        c.node(1).data.caught_up()
    })
    .await;
    let mount = c.state_dir(1).join("synthetic-mount");
    fs::create_dir(&mount).unwrap();
    // Docker's daemon must traverse the inputs before binding them. Permit
    // access only on this synthetic mount, using existing host permission.
    let fuse_conf = fs::read_to_string("/etc/fuse.conf").expect("read existing FUSE permission");
    assert!(
        fuse_conf
            .lines()
            .any(|line| line.trim() == "user_allow_other"),
        "FUSE prerequisite: existing /etc/fuse.conf must permit user_allow_other; do not change host settings for this test"
    );
    let _mounted = nest_fuse::mount(
        c.node(1).vfs.clone(),
        &nest_fuse::MountConfig {
            mountpoint: mount.clone(),
            allow_other: true,
            ttl: Duration::from_secs(1),
            threads: 4,
            io_uring: false,
        },
    )
    .unwrap_or_else(|error| panic!("FUSE prerequisite blocked real composition trial: {error:#}"));
    manager_trial(&c, mount).await.unwrap();
}

async fn manager_trial(c: &TestCluster, mount: PathBuf) -> anyhow::Result<()> {
    let native = c.state_dir(1).join("synthetic-fixtures");
    let fixture = Fixture::new(&native);
    let pool = pool(c.state_dir(1).join("synthetic-worker-pool"));
    assert_eq!(
        fs::metadata(&pool.root)?.dev(),
        fs::metadata(c.node(1).data.store().root())?.dev(),
        "pool must share native store filesystem"
    );
    let module = fixture.module(&pool);
    for (src, dst) in [
        (fixture.base.clone(), "/synthetic-base"),
        (fixture.alternate.clone(), "/synthetic-replacement"),
    ] {
        nest_place::import::run(
            c.node(1).vfs.clone(),
            ImportOptions {
                src,
                dst: dst.into(),
                r#move: true,
                seal: SealMode::All,
                copy: false,
            },
            Arc::new(Default::default()),
        )
        .await?;
    }
    let manager = Manager::new(
        c.node(1).vfs.clone(),
        Some(mount.clone()),
        Some(pool.clone()),
    );
    let name = "synthetic-composition";
    let put = manager.put(name, module.clone(), profile(), 0).await?;
    assert_eq!(put.revision, 1);
    assert!(
        manager.put(name, module, profile(), 0).await.is_err(),
        "stale CAS accepted"
    );
    assert!(
        manager.resolve(name).await.is_err(),
        "unprepared module resolved"
    );
    let prepared = manager.prepare(name, true).await?;
    assert_eq!(prepared.phase, "awaiting-qualification", "{prepared:?}");
    assert!(
        manager.resolve(name).await.is_err(),
        "export alone became READY"
    );
    let reader = c.node(1).meta.open_reader()?;
    let snapshots = catalog::snapshots(&reader, &prepared.key)?;
    assert_eq!(snapshots.iter().filter(|s| !s.output).count(), 2);
    assert_eq!(snapshots.iter().filter(|s| s.output).count(), 1);
    for snapshot in &snapshots {
        for (_, file, generation) in &snapshot.files {
            let attr = query::getattr(&reader, *file)?.unwrap();
            assert_eq!(attr.generation, *generation);
            assert!(attr.sealed);
            assert_eq!(attr.gen_state, GenState::Stable);
        }
    }
    let input_file = snapshots.iter().find(|s| !s.output).unwrap().files[0].1;
    assert!(
        matches!(
            c.node(1).vfs.seal(input_file, false).await,
            Err(NestError::Busy(_))
        ),
        "prepared input generation was not pinned"
    );
    let qualified = manager.qualify(name).await?;
    assert_eq!(qualified.phase, "ready", "{qualified:?}");
    let resolved = manager.resolve(name).await?;
    let output = resolved.path.clone();
    blocking(move || check_output(&output)).await;
    let before = records(&pool);
    assert_eq!(
        before.len(),
        3,
        "prepare Stage + Inspect, then exact Qualify"
    );
    let hit = manager.prepare(name, true).await?;
    assert_eq!(hit.phase, "ready");
    assert_eq!(
        manager.resolve(name).await?.publication,
        resolved.publication
    );
    assert_eq!(records(&pool), before, "cache hit launched a worker");
    let held = manager.acquire(name, "synthetic-consumer").await?;
    assert_eq!(held.publication, resolved.publication);
    assert!(
        matches!(
            manager.reset(name, false, false).await,
            Err(NestError::Busy(_))
        ),
        "active usage did not refuse immediate reset"
    );
    let dry = manager.reset(name, false, true).await?;
    assert!(dry.blocked);
    assert!(!dry.deletes_files);
    let queued = manager.reset(name, true, false).await?;
    assert!(queued.blocked);
    assert!(
        manager.acquire(name, "late-user").await.is_err(),
        "pending reset admitted user"
    );
    assert!(
        manager.resolve(name).await.is_err(),
        "pending reset resolved"
    );
    let held_output = held.path.clone();
    blocking(move || check_output(&held_output)).await;
    manager
        .release(&held.key, held.epoch, "synthetic-consumer")
        .await?;
    assert_eq!(
        manager.status(name).await?.artifact.unwrap().state,
        "retired"
    );
    assert!(manager.resolve(name).await.is_err());
    let rebuilt = manager.prepare(name, true).await?;
    assert_eq!(rebuilt.key, resolved.key);
    assert_eq!(rebuilt.phase, "awaiting-qualification", "{rebuilt:?}");
    assert_eq!(manager.qualify(name).await?.phase, "ready");
    let rebuilt = manager.resolve(name).await?;
    assert!(rebuilt.epoch > resolved.epoch);
    assert_eq!(
        rebuilt.publication, resolved.publication,
        "same key changed publication"
    );
    let before_hit = records(&pool);
    assert_eq!(before_hit.len(), 6);
    manager.prepare(name, true).await?;
    let cached = manager.resolve(name).await?;
    assert_eq!(cached.publication, rebuilt.publication);
    assert_eq!(
        records(&pool),
        before_hit,
        "post-rebuild cache hit launched worker"
    );
    let output = cached.path.clone();
    blocking(move || check_output(&output)).await;
    retain(
        "manager-fuse.json",
        &json!({"first":resolved,"rebuilt":rebuilt,
        "deferred_reset":queued,"workers":before_hit,"cache_hit_worker_launches":0}),
    );
    Ok(())
}
