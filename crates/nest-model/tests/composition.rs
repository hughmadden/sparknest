use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use nest_model::{Digest, ExportLimits, ModuleConfig, Profile, Snapshot, export, resolve};
use serde_json::json;
use tempfile::TempDir;

const SHARD: &str = "model-00001-of-00001.safetensors";

fn snapshot(root: &Path, a: &[u8], b: &[u8]) {
    fs::create_dir_all(root).unwrap();
    let mut header = serde_json::to_vec(&json!({
        "a": {"dtype":"U8", "shape":[a.len()], "data_offsets":[0,a.len()]},
        "b": {"dtype":"U8", "shape":[b.len()], "data_offsets":[a.len(),a.len()+b.len()]}
    }))
    .unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(a);
    bytes.extend(b);
    fs::write(root.join(SHARD), bytes).unwrap();
    fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({
            "metadata":{"total_size":a.len()+b.len()},"weight_map":{"a":SHARD,"b":SHARD}
        }))
        .unwrap(),
    )
    .unwrap();
    fs::write(root.join("config.json"), b"{\"architectures\":[]}").unwrap();
    fs::write(root.join("tokenizer.json"), b"{}").unwrap();
}

fn profile(coupled: bool) -> Profile {
    Profile::new(
        if coupled {
            BTreeMap::from([("a".into(), BTreeSet::from(["a".into(), "b".into()]))])
        } else {
            BTreeMap::from([
                ("a".into(), BTreeSet::from(["a".into()])),
                ("b".into(), BTreeSet::from(["b".into()])),
            ])
        },
        false,
    )
    .unwrap()
}

fn config(base: &Snapshot, alternates: &[(&Snapshot, &str)]) -> ModuleConfig {
    let mut text = format!(
        r#"
schema = "sparknest.model-composition.v1"
name = "base-checkpoint"
[base]
source = "base-checkpoint"
content_manifest = "{}"
[output]
loader_profile = "base-compatible-indexed-v1"
precision = "match-base"
configuration = "base-unchanged"
tokenizer = "base-unchanged"
non_weight_mismatch = "reject"
namespace = "base-checkpoint"
[preparation]
trigger = "explicit-prepare-or-external-use-wrapper"
adapter_image = "converter@{}"
recipe = "{}"
lossy_conversion = "deny"
qualification_profile = "{}"
activation_calibration = "preserve-and-qualify"
max_runtime_seconds = 60
max_ram_bytes = 134217728
cpu_threads = 1
device = "cpu"
worker_pool = "cpu"
[cache]
publication = "sealed-regular-files"
reuse_unchanged = false
prefer_reflink = false
max_new_physical_bytes = 1048576
account_inflight = true
eviction = "least-recently-used-unpinned"
retain_dependencies = true
"#,
        base.content_manifest(),
        Digest::of(b"image"),
        Digest::of(b"recipe"),
        Digest::of(b"policy")
    );
    for (alternate, selection) in alternates {
        text.push_str(&format!(
            r#"
[[alternates]]
name = "alternate-checkpoint"
kind = "checkpoint"
source = "alternate-checkpoint"
content_manifest = "{}"
selection = "{}"
{}requires = {{ base_manifest = "{}" }}
"#,
            alternate.content_manifest(),
            selection,
            if *selection == "producer-manifest" {
                format!(
                    "selection_manifest = \"groups.json\"\nselection_manifest_digest = \"{}\"\n",
                    alternate.files()["groups.json"].digest
                )
            } else {
                String::new()
            },
            base.content_manifest()
        ));
    }
    ModuleConfig::parse(&text).unwrap()
}

fn inspect(root: &Path) -> Snapshot {
    Snapshot::inspect(root, &ExportLimits::default()).unwrap()
}

fn rewrite_shard(root: &Path, shard: &str, header: serde_json::Value, payload: &[u8]) {
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(payload);
    fs::write(root.join(shard), bytes).unwrap();
}

#[test]
fn strict_config_and_identity_separation() {
    let temp = TempDir::new().unwrap();
    snapshot(temp.path(), &[1], &[2]);
    let base = inspect(temp.path());
    let config = config(&base, &[]);
    let original = toml::to_string(&config).unwrap();
    assert!(ModuleConfig::parse(&original.replace("schema =", "unknown =")).is_err());
    assert!(
        ModuleConfig::parse(&original.replace(
            "sparknest.model-composition.v1",
            "sparknest.model-composition.v2"
        ))
        .is_err()
    );
    assert!(ModuleConfig::parse(&original.replace("cpu_threads = 1", "cpu_threads = 0")).is_err());
    assert!(ModuleConfig::parse(&original.replace("match-base", "float16")).is_err());
    let mut changed = config.clone();
    changed.name = "alternate-checkpoint".into();
    changed.base.source = "alternate-checkpoint".into();
    changed.output.namespace = "alternate-checkpoint".into();
    changed.preparation.qualification_profile = Digest::of(b"stricter");
    assert_eq!(
        config.preparation_key(&profile(false)).unwrap(),
        changed.preparation_key(&profile(false)).unwrap()
    );
    assert_ne!(
        config.qualification_key().unwrap(),
        changed.qualification_key().unwrap()
    );
    changed.preparation.recipe = Digest::of(b"new recipe");
    assert_ne!(
        config.preparation_key(&profile(false)).unwrap(),
        changed.preparation_key(&profile(false)).unwrap()
    );
}

#[test]
fn fallback_last_wins_and_bounded_full_shard_export() {
    let temp = TempDir::new().unwrap();
    let base_path = temp.path().join("base-checkpoint");
    let alt_path = temp.path().join("alternate-checkpoint");
    snapshot(&base_path, &[1, 2, 3], &[4, 5]);
    snapshot(&alt_path, &[8, 9, 10], &[11, 12]);
    fs::write(
        alt_path.join("groups.json"),
        br#"{"schema":"sparknest.model-selection.v1","groups":["a"]}"#,
    )
    .unwrap();
    let base = inspect(&base_path);
    let alternate = inspect(&alt_path);
    let config = config(
        &base,
        &[
            (&alternate, "producer-manifest"),
            (&alternate, "producer-manifest"),
        ],
    );
    let plan = resolve(
        &config,
        &profile(false),
        &base,
        &[&alternate, &alternate],
        &ExportLimits::default(),
    )
    .unwrap();
    assert_eq!(plan.winners()["a"], Some(1));
    assert_eq!(plan.winners()["b"], None);
    assert_eq!(plan.shadowed().len(), 1);
    let output = temp.path().join("output");
    let limits = ExportLimits {
        buffer_bytes: 1,
        ..ExportLimits::default()
    };
    let receipt = export(
        &config,
        &profile(false),
        &base,
        &[&alternate, &alternate],
        &output,
        &limits,
    )
    .unwrap();
    let prepared = inspect(&output);
    assert_eq!(prepared.tensor_bytes("a", 16).unwrap(), [8, 9, 10]);
    assert_eq!(prepared.tensor_bytes("b", 16).unwrap(), [4, 5]);
    assert_eq!(
        fs::read(output.join("config.json")).unwrap(),
        fs::read(base_path.join("config.json")).unwrap()
    );
    assert_eq!(fs::read(output.join("tokenizer.json")).unwrap(), b"{}");
    assert_eq!(
        &fs::read(output.join(SHARD)).unwrap()[..8],
        &fs::read(base_path.join(SHARD)).unwrap()[..8]
    );
    assert_eq!(
        receipt.preparation_key,
        config.preparation_key(&profile(false)).unwrap()
    );
    let index: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("model.safetensors.index.json")).unwrap())
            .unwrap();
    assert_eq!(
        index["metadata"]["sparknest_payload_manifest"],
        receipt.payload_manifest.to_string()
    );
    assert_eq!(
        index["metadata"]["sparknest_preparation_key"],
        receipt.preparation_key.to_string()
    );
    assert!(receipt.files.contains_key("model.safetensors.index.json"));
    assert!(!receipt.files.contains_key("composition.receipt.json"));
    assert!(
        export(
            &config,
            &profile(false),
            &base,
            &[&alternate, &alternate],
            &output,
            &limits
        )
        .is_err()
    );
}

#[test]
fn dependency_and_group_refusals() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1], &[2]);
    snapshot(&ap, &[3], &[4]);
    fs::write(
        ap.join("groups.json"),
        br#"{"schema":"sparknest.model-selection.v1","groups":["b"]}"#,
    )
    .unwrap();
    let base = inspect(&bp);
    let alt = inspect(&ap);
    let mut config = config(&base, &[(&alt, "producer-manifest")]);
    assert!(
        resolve(
            &config,
            &profile(true),
            &base,
            &[&alt],
            &ExportLimits::default()
        )
        .is_err()
    );
    config.alternates[0].content_manifest = Digest::of(b"wrong");
    assert!(
        resolve(
            &config,
            &profile(false),
            &base,
            &[&alt],
            &ExportLimits::default()
        )
        .is_err()
    );
    config.alternates[0].content_manifest = alt.content_manifest().clone();
    config.alternates[0].requires = nest_model::Reference::BaseManifest(Digest::of(b"wrong"));
    assert!(
        resolve(
            &config,
            &profile(false),
            &base,
            &[&alt],
            &ExportLimits::default()
        )
        .is_err()
    );
}

#[test]
fn untrusted_index_and_file_paths_refused() {
    let temp = TempDir::new().unwrap();
    snapshot(temp.path(), &[1], &[2]);
    let index_path = temp.path().join("model.safetensors.index.json");
    let original = fs::read_to_string(&index_path).unwrap();
    fs::write(
        &index_path,
        original.replace(SHARD, "../model-00001-of-00001.safetensors"),
    )
    .unwrap();
    assert!(Snapshot::inspect(temp.path(), &ExportLimits::default()).is_err());
    fs::write(&index_path, original).unwrap();
    fs::write(temp.path().join("checkpoint.py"), b"raise Exception()").unwrap();
    assert!(Snapshot::inspect(temp.path(), &ExportLimits::default()).is_err());
}

#[test]
fn malformed_headers_offsets_and_duplicate_keys_refused() {
    let temp = TempDir::new().unwrap();
    snapshot(temp.path(), &[1], &[2]);
    let shard = temp.path().join(SHARD);
    let bytes = fs::read(&shard).unwrap();
    fs::write(&shard, &bytes[..bytes.len() - 1]).unwrap();
    assert!(Snapshot::inspect(temp.path(), &ExportLimits::default()).is_err());
    fs::write(&shard, bytes).unwrap();
    fs::write(
        temp.path().join("model.safetensors.index.json"),
        format!(r#"{{"weight_map":{{"a":"{SHARD}","a":"{SHARD}","b":"{SHARD}"}}}}"#),
    )
    .unwrap();
    assert!(Snapshot::inspect(temp.path(), &ExportLimits::default()).is_err());
}

#[test]
fn base_only_is_exact_and_exports_are_deterministic() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    snapshot(&bp, &[1, 2], &[3, 4]);
    let base = inspect(&bp);
    let config = config(&base, &[]);
    let profile = profile(false);
    let one = temp.path().join("one");
    let two = temp.path().join("two");
    let receipt = export(
        &config,
        &profile,
        &base,
        &[],
        &one,
        &ExportLimits::default(),
    )
    .unwrap();
    export(
        &config,
        &profile,
        &base,
        &[],
        &two,
        &ExportLimits::default(),
    )
    .unwrap();
    for entry in fs::read_dir(&one).unwrap() {
        let name = entry.unwrap().file_name();
        assert_eq!(
            fs::read(one.join(&name)).unwrap(),
            fs::read(two.join(&name)).unwrap()
        );
    }
    assert_eq!(
        fs::read(bp.join(SHARD)).unwrap(),
        fs::read(one.join(SHARD)).unwrap()
    );
    let manifest = fs::read(one.join("composition.payload.json")).unwrap();
    assert_eq!(Digest::of(&manifest), receipt.payload_manifest);
    let payload: serde_json::Value = serde_json::from_slice(&manifest).unwrap();
    assert!(
        payload["files"]
            .get("model.safetensors.index.json")
            .is_none()
    );
    assert!(payload["files"].get("composition.receipt.json").is_none());
    for (name, identity) in &receipt.files {
        let bytes = fs::read(one.join(name)).unwrap();
        assert_eq!(Digest::of(&bytes), identity.digest);
        assert_eq!(bytes.len() as u64, identity.bytes);
    }
    assert_ne!(base.content_manifest(), inspect(&one).content_manifest());
    assert_eq!(receipt.qualification, "pending");
    assert!(receipt.winners.values().all(Option::is_none));
}

#[test]
fn ordered_alternates_and_lower_prefix_references() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    let ap2 = temp.path().join("second/alternate-checkpoint");
    snapshot(&bp, &[1], &[2]);
    snapshot(&ap, &[3], &[4]);
    snapshot(&ap2, &[5], &[6]);
    let base = inspect(&bp);
    let alt = inspect(&ap);
    let alt2 = inspect(&ap2);
    let profile = profile(false);
    let mut config = config(&base, &[(&alt, "all"), (&alt2, "all")]);
    let reverse = self::config(&base, &[(&alt2, "all"), (&alt, "all")]);
    assert_ne!(
        config.preparation_key(&profile).unwrap(),
        reverse.preparation_key(&profile).unwrap()
    );
    let prefix = config.prefix_key(&profile, 1).unwrap();
    config.alternates[1].requires = nest_model::Reference::LowerCompositionKey(prefix);
    let plan = resolve(
        &config,
        &profile,
        &base,
        &[&alt, &alt2],
        &ExportLimits::default(),
    )
    .unwrap();
    assert!(plan.winners().values().all(|w| *w == Some(1)));
    let output = temp.path().join("output");
    export(
        &config,
        &profile,
        &base,
        &[&alt, &alt2],
        &output,
        &ExportLimits::default(),
    )
    .unwrap();
    assert_eq!(inspect(&output).tensor_bytes("a", 1).unwrap(), [5]);
    config.alternates[1].requires =
        nest_model::Reference::LowerCompositionKey(Digest::of(b"wrong"));
    assert!(
        resolve(
            &config,
            &profile,
            &base,
            &[&alt, &alt2],
            &ExportLimits::default()
        )
        .is_err()
    );
}

#[test]
fn map_order_and_acquisition_hints_do_not_affect_preparation_key() {
    let temp = TempDir::new().unwrap();
    snapshot(temp.path(), &[1], &[2]);
    let base = inspect(temp.path());
    let mut config = config(&base, &[]);
    let mut other = config.clone();
    config
        .preparation
        .parameters
        .insert("a".into(), "0.10".into());
    config
        .preparation
        .parameters
        .insert("b".into(), "0.20".into());
    other
        .preparation
        .parameters
        .insert("b".into(), "0.20".into());
    other
        .preparation
        .parameters
        .insert("a".into(), "0.10".into());
    other.preparation.adapter_image = other
        .preparation
        .adapter_image
        .replace("converter@", "another-registry/converter@");
    other.preparation.max_runtime_seconds = 1;
    other.cache.max_new_physical_bytes = 2;
    assert_eq!(
        config.preparation_key(&profile(false)).unwrap(),
        other.preparation_key(&profile(false)).unwrap()
    );
    other
        .preparation
        .parameters
        .insert("a".into(), "0.100".into());
    assert_ne!(
        config.preparation_key(&profile(false)).unwrap(),
        other.preparation_key(&profile(false)).unwrap()
    );
    // Exact string spelling is intentional; no floating-point normalization.
    let text = toml::to_string(&config).unwrap();
    assert!(ModuleConfig::parse(&text.replace("0.10", "NaN")).is_err());
    assert!(ModuleConfig::parse(&text.replace("\"0.10\"", "0.10")).is_err());
}

#[test]
fn precision_lora_and_recalibration_explicitly_refuse() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1], &[2]);
    snapshot(&ap, &[3], &[4]);
    rewrite_shard(
        &ap,
        SHARD,
        json!({"a":{"dtype":"I8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}),
        &[3, 4],
    );
    let base = inspect(&bp);
    let alt = inspect(&ap);
    let mut config = config(&base, &[(&alt, "all")]);
    config.preparation.lossy_conversion = nest_model::LossyConversion::RequireQualifiedProfile;
    assert!(matches!(
        resolve(
            &config,
            &profile(false),
            &base,
            &[&alt],
            &ExportLimits::default()
        ),
        Err(nest_model::Error::Unsupported(_))
    ));
    config.alternates[0].kind = nest_model::SourceKind::Lora;
    assert!(matches!(
        resolve(
            &config,
            &profile(false),
            &base,
            &[&alt],
            &ExportLimits::default()
        ),
        Err(nest_model::Error::Unsupported(_))
    ));
    config.alternates.clear();
    config.preparation.activation_calibration = nest_model::ActivationCalibration::Recalibrate;
    assert!(matches!(
        resolve(
            &config,
            &profile(false),
            &base,
            &[],
            &ExportLimits::default()
        ),
        Err(nest_model::Error::Unsupported(_))
    ));
}

#[test]
fn complete_companion_groups_cannot_borrow_from_base() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1], &[2]);
    snapshot(&ap, &[3], &[4]);
    rewrite_shard(
        &ap,
        SHARD,
        json!({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}),
        &[3],
    );
    fs::write(
        ap.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"metadata":{"total_size":1},"weight_map":{"a":SHARD}})).unwrap(),
    )
    .unwrap();
    let base = inspect(&bp);
    let alt = inspect(&ap);
    let mut config = config(&base, &[(&alt, "all")]);
    assert!(
        resolve(
            &config,
            &profile(true),
            &base,
            &[&alt],
            &ExportLimits::default()
        )
        .is_err()
    );
    config.alternates[0].kind = nest_model::SourceKind::Replacement;
    assert!(
        resolve(
            &config,
            &profile(true),
            &base,
            &[&alt],
            &ExportLimits::default()
        )
        .is_err()
    );
    assert!(
        Profile::new(
            BTreeMap::from([
                ("a".into(), BTreeSet::from(["a".into()])),
                ("b".into(), BTreeSet::from(["a".into()]))
            ]),
            false
        )
        .is_err()
    );
    let incomplete = Profile::new(
        BTreeMap::from([("a".into(), BTreeSet::from(["a".into()]))]),
        false,
    )
    .unwrap();
    assert!(
        resolve(
            &config,
            &incomplete,
            &base,
            &[&alt],
            &ExportLimits::default()
        )
        .is_err()
    );
}

#[test]
fn selection_manifest_schema_and_unknown_groups_refuse() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1], &[2]);
    snapshot(&ap, &[3], &[4]);
    let base = inspect(&bp);
    for selection in [
        json!({"schema":"sparknest.model-selection.v1","groups":["a","a"]}),
        json!({"schema":"sparknest.model-selection.v1","groups":["unknown"]}),
        json!({"schema":"sparknest.model-selection.v2","groups":["a"]}),
        json!({"schema":"sparknest.model-selection.v1","groups":["a"],"extra":true}),
        json!({"schema":"sparknest.model-selection.v1","groups":[]}),
    ] {
        fs::write(
            ap.join("groups.json"),
            serde_json::to_vec(&selection).unwrap(),
        )
        .unwrap();
        let alt = inspect(&ap);
        let mut config = config(&base, &[(&alt, "producer-manifest")]);
        config.output.non_weight_mismatch = nest_model::NonWeightMismatch::UseBase;
        assert!(
            resolve(
                &config,
                &profile(false),
                &base,
                &[&alt],
                &ExportLimits::default()
            )
            .is_err()
        );
    }
}

#[test]
fn semantics_mismatch_requires_explicit_base_choice() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1], &[2]);
    snapshot(&ap, &[3], &[4]);
    fs::write(ap.join("tokenizer.json"), b"{\"changed\":true}").unwrap();
    let base = inspect(&bp);
    let alt = inspect(&ap);
    let mut config = config(&base, &[(&alt, "all")]);
    assert!(
        resolve(
            &config,
            &profile(false),
            &base,
            &[&alt],
            &ExportLimits::default()
        )
        .is_err()
    );
    let key = config.preparation_key(&profile(false)).unwrap();
    config.output.non_weight_mismatch = nest_model::NonWeightMismatch::UseBase;
    assert_ne!(key, config.preparation_key(&profile(false)).unwrap());
    let output = temp.path().join("output");
    let receipt = export(
        &config,
        &profile(false),
        &base,
        &[&alt],
        &output,
        &ExportLimits::default(),
    )
    .unwrap();
    assert_eq!(fs::read(output.join("tokenizer.json")).unwrap(), b"{}");
    assert_eq!(
        receipt.non_weight_mismatch,
        nest_model::NonWeightMismatch::UseBase
    );
    let calibrated = Profile::new(profile(false).groups().clone(), true).unwrap();
    assert!(
        resolve(
            &config,
            &calibrated,
            &base,
            &[&alt],
            &ExportLimits::default()
        )
        .unwrap()
        .requires_model_qualification
    );
}

#[test]
fn resource_and_mutation_refusals_happen_before_staging() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    snapshot(&bp, &[1], &[2]);
    let base = inspect(&bp);
    let config = config(&base, &[]);
    let output = temp.path().join("output");
    let limits = ExportLimits {
        max_output_bytes: 1,
        ..ExportLimits::default()
    };
    assert!(export(&config, &profile(false), &base, &[], &output, &limits).is_err());
    assert!(!output.exists());
    let limits = ExportLimits {
        metadata_bytes: 1,
        ..ExportLimits::default()
    };
    assert!(Snapshot::inspect(&bp, &limits).is_err());
    let limits = ExportLimits {
        max_files: 1,
        ..ExportLimits::default()
    };
    assert!(Snapshot::inspect(&bp, &limits).is_err());
    let limits = ExportLimits {
        max_tensors: 1,
        ..ExportLimits::default()
    };
    assert!(Snapshot::inspect(&bp, &limits).is_err());
    let limits = ExportLimits {
        buffer_bytes: 0,
        ..ExportLimits::default()
    };
    assert!(Snapshot::inspect(&bp, &limits).is_err());
    assert!(
        export(
            &config,
            &profile(false),
            &base,
            &[],
            &bp.join("output"),
            &ExportLimits::default()
        )
        .is_err()
    );
    let mut bytes = fs::read(bp.join(SHARD)).unwrap();
    *bytes.last_mut().unwrap() = 9;
    fs::write(bp.join(SHARD), bytes).unwrap();
    assert!(
        export(
            &config,
            &profile(false),
            &base,
            &[],
            &output,
            &ExportLimits::default()
        )
        .is_err()
    );
    assert!(!output.exists());
}

#[test]
fn source_sharding_can_differ_but_output_layout_is_base_exact() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1, 2], &[3]);
    snapshot(&ap, &[4, 5], &[6]);
    let one = "model-00001-of-00002.safetensors";
    let two = "model-00002-of-00002.safetensors";
    // Build a complete synthetic two-shard base from scratch, retaining the
    // existing single-shard fixture as a private unrelated file outside inputs.
    fs::rename(bp.join(SHARD), temp.path().join("fixture.safetensors")).unwrap();
    rewrite_shard(
        &bp,
        one,
        json!({"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}),
        &[1, 2],
    );
    rewrite_shard(
        &bp,
        two,
        json!({"b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}),
        &[3],
    );
    fs::write(
        bp.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"metadata":{"total_size":3},"weight_map":{"a":one,"b":two}}))
            .unwrap(),
    )
    .unwrap();
    let base = inspect(&bp);
    let alt = inspect(&ap);
    let config = config(&base, &[(&alt, "all")]);
    let output = temp.path().join("output");
    export(
        &config,
        &profile(false),
        &base,
        &[&alt],
        &output,
        &ExportLimits::default(),
    )
    .unwrap();
    let result = inspect(&output);
    assert_eq!(result.tensor_bytes("a", 2).unwrap(), [4, 5]);
    assert_eq!(result.tensor_bytes("b", 1).unwrap(), [6]);
    assert!(!output.join(SHARD).exists());
    assert!(output.join(one).exists());
    assert!(output.join(two).exists());
}

#[cfg(unix)]
#[test]
fn symlink_files_and_roots_refuse() {
    use std::os::unix::fs::symlink;
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    snapshot(&bp, &[1], &[2]);
    symlink(&bp, temp.path().join("alternate-checkpoint")).unwrap();
    assert!(
        Snapshot::inspect(
            &temp.path().join("alternate-checkpoint"),
            &ExportLimits::default()
        )
        .is_err()
    );
    symlink(bp.join(SHARD), bp.join("linked.safetensors")).unwrap();
    assert!(Snapshot::inspect(&bp, &ExportLimits::default()).is_err());
}

#[test]
fn tensor_header_schema_range_and_index_invariants() {
    let temp = TempDir::new().unwrap();
    for (a, b) in [
        (json!([0, 2]), json!([1, 2])),
        (json!([1, 2]), json!([2, 3])),
        (json!([0, 1]), json!([0, 1])),
    ] {
        let bp = temp.path().join(format!("{}/base-checkpoint", a));
        snapshot(&bp, &[1], &[2]);
        rewrite_shard(
            &bp,
            SHARD,
            json!({"a":{"dtype":"U8","shape":[1],"data_offsets":a},"b":{"dtype":"U8","shape":[1],"data_offsets":b}}),
            &[1, 2],
        );
        assert!(Snapshot::inspect(&bp, &ExportLimits::default()).is_err());
    }
    let bp = temp.path().join("base-checkpoint");
    snapshot(&bp, &[1], &[2]);
    rewrite_shard(
        &bp,
        SHARD,
        json!({"a":{"dtype":"U8","shape":[u64::MAX,2],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}),
        &[1, 2],
    );
    assert!(Snapshot::inspect(&bp, &ExportLimits::default()).is_err());
    rewrite_shard(
        &bp,
        SHARD,
        json!({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"extra":true},"b":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}),
        &[1, 2],
    );
    assert!(Snapshot::inspect(&bp, &ExportLimits::default()).is_err());
}

#[test]
fn selected_manifest_identity_is_bound_without_filename_in_key() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1], &[2]);
    snapshot(&ap, &[3], &[4]);
    let groups_a = br#"{"schema":"sparknest.model-selection.v1","groups":["a"]}"#;
    let groups_b = br#"{"schema":"sparknest.model-selection.v1","groups":["b"]}"#;
    fs::write(ap.join("groups.json"), groups_a).unwrap();
    fs::write(ap.join("other.json"), groups_b).unwrap();
    let base = inspect(&bp);
    let alt = inspect(&ap);
    let config = config(&base, &[(&alt, "producer-manifest")]);
    let mut other = config.clone();
    other.alternates[0].selection_manifest = Some("other.json".into());
    // The source contains both files: wrong selected-manifest identity must fail.
    assert!(
        resolve(
            &other,
            &profile(false),
            &base,
            &[&alt],
            &ExportLimits::default()
        )
        .is_err()
    );
    other.alternates[0].selection_manifest_digest = Some(Digest::of(groups_b));
    assert_ne!(
        config.preparation_key(&profile(false)).unwrap(),
        other.preparation_key(&profile(false)).unwrap()
    );
    assert_eq!(
        resolve(
            &other,
            &profile(false),
            &base,
            &[&alt],
            &ExportLimits::default()
        )
        .unwrap()
        .winners()["b"],
        Some(0)
    );
    other.alternates[0].selection_manifest_digest = Some(Digest::of(groups_a));
    assert_eq!(
        config.preparation_key(&profile(false)).unwrap(),
        other.preparation_key(&profile(false)).unwrap()
    );
}

#[test]
fn use_base_does_not_override_architecture_or_shapes() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1], &[2]);
    snapshot(&ap, &[3], &[4]);
    fs::write(
        ap.join("config.json"),
        b"{\"architectures\":[\"alternate-checkpoint\"]}",
    )
    .unwrap();
    let base = inspect(&bp);
    let alt = inspect(&ap);
    let mut config = config(&base, &[(&alt, "all")]);
    config.output.non_weight_mismatch = nest_model::NonWeightMismatch::UseBase;
    assert!(
        resolve(
            &config,
            &profile(false),
            &base,
            &[&alt],
            &ExportLimits::default()
        )
        .is_err()
    );
    fs::write(
        ap.join("config.json"),
        fs::read(bp.join("config.json")).unwrap(),
    )
    .unwrap();
    rewrite_shard(
        &ap,
        SHARD,
        json!({"a":{"dtype":"U8","shape":[1,1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}),
        &[3, 4],
    );
    let alt = inspect(&ap);
    config.alternates[0].content_manifest = alt.content_manifest().clone();
    assert!(
        resolve(
            &config,
            &profile(false),
            &base,
            &[&alt],
            &ExportLimits::default()
        )
        .is_err()
    );
}

#[test]
fn unknown_index_fields_and_stale_inventory_refuse() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    snapshot(&bp, &[1], &[2]);
    for value in [
        json!({"metadata":{"total_size":2},"weight_map":{"a":SHARD,"b":SHARD},"extra":true}),
        json!({"metadata":{"total_size":3},"weight_map":{"a":SHARD,"b":SHARD}}),
        json!({"metadata":{"total_size":2},"weight_map":{"a":SHARD}}),
        json!({"metadata":{"total_size":2},"weight_map":{"a":"model-+0001-of-00001.safetensors","b":SHARD}}),
        json!({"metadata":{"total_size":2},"weight_map":{"a":"model-😀001-of-00001.safetensors","b":SHARD}}),
    ] {
        fs::write(
            bp.join("model.safetensors.index.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
        assert!(Snapshot::inspect(&bp, &ExportLimits::default()).is_err());
    }
}

#[test]
fn unknown_source_tensor_and_missing_selection_file_refuse() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1], &[2]);
    snapshot(&ap, &[3], &[4]);
    rewrite_shard(
        &bp,
        SHARD,
        json!({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}),
        &[1],
    );
    fs::write(
        bp.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"metadata":{"total_size":1},"weight_map":{"a":SHARD}})).unwrap(),
    )
    .unwrap();
    let base = inspect(&bp);
    let alt = inspect(&ap);
    let mut config = config(&base, &[(&alt, "all")]);
    let single = Profile::new(
        BTreeMap::from([("a".into(), BTreeSet::from(["a".into()]))]),
        false,
    )
    .unwrap();
    assert!(resolve(&config, &single, &base, &[&alt], &ExportLimits::default()).is_err());
    config.alternates[0].selection = nest_model::Selection::ProducerManifest;
    config.alternates[0].selection_manifest = Some("groups.json".into());
    config.alternates[0].selection_manifest_digest = Some(Digest::of(b"missing"));
    assert!(resolve(&config, &single, &base, &[&alt], &ExportLimits::default()).is_err());
    // Missing manifest with an otherwise compatible, complete snapshot.
    let config = self::config(&alt, &[]);
    let mut config = config;
    config.alternates.push(nest_model::Alternate {
        name: "alternate-checkpoint".into(),
        kind: nest_model::SourceKind::Checkpoint,
        source: "alternate-checkpoint".into(),
        content_manifest: alt.content_manifest().clone(),
        selection: nest_model::Selection::ProducerManifest,
        selection_manifest: Some("groups.json".into()),
        selection_manifest_digest: Some(Digest::of(b"missing")),
        requires: nest_model::Reference::BaseManifest(alt.content_manifest().clone()),
    });
    assert!(
        resolve(
            &config,
            &profile(false),
            &alt,
            &[&alt],
            &ExportLimits::default()
        )
        .is_err()
    );
}

#[test]
fn coupled_operands_replace_together_in_last_wins_order() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1], &[2]);
    snapshot(&ap, &[3], &[4]);
    let base = inspect(&bp);
    let alt = inspect(&ap);
    let config = config(&base, &[(&alt, "all"), (&alt, "all")]);
    let output = temp.path().join("output");
    let receipt = export(
        &config,
        &profile(true),
        &base,
        &[&alt, &alt],
        &output,
        &ExportLimits::default(),
    )
    .unwrap();
    assert_eq!(receipt.winners.len(), 1);
    assert_eq!(receipt.winners["a"], Some(1));
    let result = inspect(&output);
    assert_eq!(result.tensor_bytes("a", 1).unwrap(), [3]);
    assert_eq!(result.tensor_bytes("b", 1).unwrap(), [4]);
}

#[test]
fn exact_fp8_payload_and_e8m0_scale_group() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1, 2], &[3]);
    snapshot(&ap, &[4, 5], &[6]);
    let header = json!({
        "a":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,2]},
        "b":{"dtype":"F8_E8M0","shape":[1],"data_offsets":[2,3]}
    });
    rewrite_shard(&bp, SHARD, header.clone(), &[1, 2, 3]);
    rewrite_shard(&ap, SHARD, header, &[4, 5, 6]);
    let base = inspect(&bp);
    let alt = inspect(&ap);
    let mut config = config(&base, &[(&alt, "all")]);
    let output = temp.path().join("output");
    export(
        &config,
        &profile(true),
        &base,
        &[&alt],
        &output,
        &ExportLimits::default(),
    )
    .unwrap();
    let result = inspect(&output);
    assert_eq!(result.tensor_bytes("a", 2).unwrap(), [4, 5]);
    assert_eq!(result.tensor_bytes("b", 1).unwrap(), [6]);
    assert_eq!(
        fs::read(output.join(SHARD)).unwrap(),
        fs::read(ap.join(SHARD)).unwrap()
    );
    // Equal byte widths are insufficient: coded scales cannot be relabelled.
    rewrite_shard(
        &ap,
        SHARD,
        json!({
            "a":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,2]},
            "b":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}
        }),
        &[4, 5, 6],
    );
    let alt = inspect(&ap);
    config.alternates[0].content_manifest = alt.content_manifest().clone();
    assert!(matches!(
        resolve(
            &config,
            &profile(true),
            &base,
            &[&alt],
            &ExportLimits::default()
        ),
        Err(nest_model::Error::Unsupported(_))
    ));
}

#[test]
fn e8m0_scale_companion_cannot_be_borrowed() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1, 2], &[3]);
    snapshot(&ap, &[4, 5], &[]);
    rewrite_shard(
        &bp,
        SHARD,
        json!({
            "a":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,2]},
            "b":{"dtype":"F8_E8M0","shape":[1],"data_offsets":[2,3]}
        }),
        &[1, 2, 3],
    );
    rewrite_shard(
        &ap,
        SHARD,
        json!({"a":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,2]}}),
        &[4, 5],
    );
    fs::write(
        ap.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({
            "metadata":{"total_size":2},"weight_map":{"a":SHARD}
        }))
        .unwrap(),
    )
    .unwrap();
    let base = inspect(&bp);
    let alt = inspect(&ap);
    let mut config = config(&base, &[(&alt, "all")]);
    config.alternates[0].kind = nest_model::SourceKind::Replacement;
    assert!(
        resolve(
            &config,
            &profile(true),
            &base,
            &[&alt],
            &ExportLimits::default()
        )
        .is_err()
    );
}

const FIRST: &str = "model-00001-of-00002.safetensors";
const SECOND: &str = "model-00002-of-00002.safetensors";

fn split_snapshot(root: &Path, a: &[u8], b: &[u8]) {
    fs::create_dir_all(root).unwrap();
    fs::write(root.join("config.json"), b"{\"architectures\":[]}").unwrap();
    fs::write(root.join("tokenizer.json"), b"{}").unwrap();
    rewrite_shard(
        root,
        FIRST,
        json!({"a":{"dtype":"U8","shape":[a.len()],"data_offsets":[0,a.len()]}}),
        a,
    );
    rewrite_shard(
        root,
        SECOND,
        json!({"b":{"dtype":"U8","shape":[b.len()],"data_offsets":[0,b.len()]}}),
        b,
    );
    fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({
            "metadata":{"total_size":a.len()+b.len()},"weight_map":{"a":FIRST,"b":SECOND}
        }))
        .unwrap(),
    )
    .unwrap();
}

#[cfg(unix)]
fn same_inode(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    let a = fs::metadata(a).unwrap();
    let b = fs::metadata(b).unwrap();
    (a.dev(), a.ino()) == (b.dev(), b.ino())
}

#[cfg(unix)]
#[test]
fn pinned_reuse_links_only_wholly_unchanged_files_and_admits_new_bytes() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    let b = vec![7; 32768];
    split_snapshot(&bp, &[1, 2], &b);
    split_snapshot(&ap, &[3, 4], &b);
    fs::write(
        ap.join("groups.json"),
        br#"{"schema":"sparknest.model-selection.v1","groups":["a"]}"#,
    )
    .unwrap();
    let base = inspect(&bp);
    let alt = inspect(&ap);
    let before: BTreeMap<_, _> = base
        .files()
        .keys()
        .map(|name| (name.clone(), fs::read(bp.join(name)).unwrap()))
        .collect();
    let mut config = config(&base, &[(&alt, "producer-manifest")]);
    config.cache.reuse_unchanged = true;
    let output = temp.path().join("output");
    let receipt = export(
        &config,
        &profile(false),
        &base,
        &[&alt],
        &output,
        &ExportLimits::default(),
    )
    .unwrap();
    for name in [SECOND, "config.json", "tokenizer.json"] {
        assert!(same_inode(&bp.join(name), &output.join(name)));
        assert!(receipt.reused_files.contains(name));
    }
    assert!(!same_inode(&bp.join(FIRST), &output.join(FIRST)));
    assert!(!same_inode(&ap.join(FIRST), &output.join(FIRST)));
    assert!(!same_inode(
        &bp.join("model.safetensors.index.json"),
        &output.join("model.safetensors.index.json")
    ));
    for (name, bytes) in before {
        assert_eq!(fs::read(bp.join(name)).unwrap(), bytes);
    }
    assert_eq!(receipt.copied_payload_bytes, 2);
    assert_eq!(receipt.reused_payload_bytes, b.len() as u64);
    assert_eq!(receipt.copied_file_bytes, base.files()[FIRST].bytes);
    assert_eq!(
        receipt.reused_file_bytes,
        receipt
            .reused_files
            .iter()
            .map(|name| base.files()[name].bytes)
            .sum::<u64>()
    );
    assert_eq!(
        receipt.new_metadata_bytes,
        [
            "model.safetensors.index.json",
            "composition.payload.json",
            "composition.receipt.json"
        ]
        .iter()
        .map(|name| fs::metadata(output.join(name)).unwrap().len())
        .sum::<u64>()
    );
    assert_eq!(
        receipt.logical_file_bytes,
        receipt.new_file_bytes + receipt.reused_file_bytes
    );
    assert_eq!(
        receipt.new_file_bytes,
        receipt.copied_file_bytes + receipt.new_metadata_bytes
    );
    assert!(receipt.new_file_bytes < receipt.logical_file_bytes);
    let total: u64 = fs::read_dir(&output)
        .unwrap()
        .map(|entry| entry.unwrap().metadata().unwrap().len())
        .sum();
    assert_eq!(receipt.logical_file_bytes, total);
    // Repeat using the same inspected pins: linking must not invalidate them,
    // and the unchanged large shard must not count toward new-file admission.
    config.cache.max_new_physical_bytes = receipt.new_file_bytes;
    let limits = ExportLimits {
        max_output_bytes: receipt.new_file_bytes,
        ..ExportLimits::default()
    };
    export(
        &config,
        &profile(false),
        &base,
        &[&alt],
        &temp.path().join("second"),
        &limits,
    )
    .unwrap();
    let too_small = ExportLimits {
        max_output_bytes: receipt.new_file_bytes - 1,
        ..limits.clone()
    };
    let refused = temp.path().join("refused");
    assert!(
        export(
            &config,
            &profile(false),
            &base,
            &[&alt],
            &refused,
            &too_small
        )
        .is_err()
    );
    assert!(!refused.exists());
    config.cache.reuse_unchanged = false;
    assert!(export(&config, &profile(false), &base, &[&alt], &refused, &limits).is_err());
    assert!(!refused.exists());
}

#[cfg(unix)]
#[test]
fn full_copy_remains_independent_and_mixed_shards_are_rebuilt_completely() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1], &[2]);
    snapshot(&ap, &[3], &[4]);
    fs::write(
        ap.join("groups.json"),
        br#"{"schema":"sparknest.model-selection.v1","groups":["a"]}"#,
    )
    .unwrap();
    let base = inspect(&bp);
    let alt = inspect(&ap);
    let mut config = config(&base, &[(&alt, "producer-manifest")]);
    let mut byte_identity = None;
    for reuse in [false, true] {
        config.cache.reuse_unchanged = reuse;
        let output = temp.path().join(if reuse { "reuse" } else { "copy" });
        let receipt = export(
            &config,
            &profile(false),
            &base,
            &[&alt],
            &output,
            &ExportLimits::default(),
        )
        .unwrap();
        let identity = (
            receipt.preparation_key.clone(),
            receipt.payload_manifest.clone(),
            receipt.files["model.safetensors.index.json"].clone(),
        );
        if let Some(previous) = &byte_identity {
            assert_eq!(&identity, previous);
        }
        byte_identity = Some(identity);
        assert!(!same_inode(&bp.join(SHARD), &output.join(SHARD)));
        assert_eq!(receipt.copied_payload_bytes, 2);
        assert_eq!(receipt.reused_payload_bytes, 0);
        assert_eq!(
            same_inode(&bp.join("tokenizer.json"), &output.join("tokenizer.json")),
            reuse
        );
        assert_eq!(inspect(&output).tensor_bytes("b", 1).unwrap(), [2]);
        if !reuse {
            assert_eq!(receipt.reused_file_bytes, 0);
            assert_eq!(receipt.new_file_bytes, receipt.logical_file_bytes);
        }
    }
}

#[cfg(unix)]
#[test]
fn reuse_refuses_cross_filesystem_links_without_copy_fallback() {
    use std::os::unix::fs::MetadataExt;
    let temp = TempDir::new().unwrap();
    let Ok(other) = tempfile::tempdir_in("/dev/shm") else {
        return;
    };
    if fs::metadata(temp.path()).unwrap().dev() == fs::metadata(other.path()).unwrap().dev() {
        return;
    }
    let bp = temp.path().join("base-checkpoint");
    snapshot(&bp, &[1], &[2]);
    let base = inspect(&bp);
    let mut config = config(&base, &[]);
    config.cache.reuse_unchanged = true;
    let output = other.path().join("output");
    let error = export(
        &config,
        &profile(false),
        &base,
        &[],
        &output,
        &ExportLimits::default(),
    )
    .unwrap_err();
    assert!(
        matches!(error,nest_model::Error::Io(e) if e.kind()==std::io::ErrorKind::CrossesDevices)
    );
    assert!(!output.join("composition.receipt.json").exists());
    assert_eq!(fs::read_dir(&output).unwrap().count(), 0);
    config.cache.reuse_unchanged = false;
    export(
        &config,
        &profile(false),
        &base,
        &[],
        &other.path().join("copy"),
        &ExportLimits::default(),
    )
    .unwrap();
}

#[test]
fn reuse_refuses_replaced_input_inode_before_staging() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    snapshot(&bp, &[1], &[2]);
    let base = inspect(&bp);
    let mut config = config(&base, &[]);
    config.cache.reuse_unchanged = true;
    // Identical bytes do not authorize replacing the verified pinned inode.
    let replacement = temp.path().join("replacement.json");
    fs::write(&replacement, fs::read(bp.join("config.json")).unwrap()).unwrap();
    fs::rename(replacement, bp.join("config.json")).unwrap();
    let output = temp.path().join("output");
    assert!(
        export(
            &config,
            &profile(false),
            &base,
            &[],
            &output,
            &ExportLimits::default()
        )
        .is_err()
    );
    assert!(!output.exists());
}

const SIDECAR: &str = "alternate-checkpoint.safetensors";

fn raw_sidecar(root: &Path, header: serde_json::Value, payload: &[u8], groups: Option<&[&str]>) {
    fs::create_dir_all(root).unwrap();
    rewrite_shard(root, SIDECAR, header, payload);
    if let Some(groups) = groups {
        fs::write(
            root.join("groups.json"),
            serde_json::to_vec(&json!({
                "schema":"sparknest.model-selection.v1", "groups":groups
            }))
            .unwrap(),
        )
        .unwrap();
    }
}

fn inspect_raw(root: &Path, selection: bool) -> Snapshot {
    Snapshot::inspect_sidecar(
        root,
        SIDECAR,
        selection.then_some("groups.json"),
        &ExportLimits::default(),
    )
    .unwrap()
}

fn replacement_config(base: &Snapshot, alternates: &[(&Snapshot, &str)]) -> ModuleConfig {
    let mut result = config(base, alternates);
    for alternate in &mut result.alternates {
        alternate.kind = nest_model::SourceKind::Replacement;
    }
    result
}

#[test]
fn raw_sparse_replacement_falls_back_and_preserves_base_metadata_and_identity_binding() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1, 2], &[3]);
    raw_sidecar(
        &ap,
        json!({"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}),
        &[8, 9],
        None,
    );
    let base = inspect(&bp);
    let alt = inspect_raw(&ap, false);
    assert!(alt.is_sidecar());
    assert!(!base.is_sidecar());
    assert_eq!(alt.files().len(), 1);
    let mut config = replacement_config(&base, &[(&alt, "all")]);
    let plan = resolve(
        &config,
        &profile(false),
        &base,
        &[&alt],
        &ExportLimits::default(),
    )
    .unwrap();
    assert_eq!(plan.winners()["a"], Some(0));
    assert_eq!(plan.winners()["b"], None);
    for reuse in [false, true] {
        config.cache.reuse_unchanged = reuse;
        let output = temp.path().join(if reuse { "reuse" } else { "copy" });
        let receipt = export(
            &config,
            &profile(false),
            &base,
            &[&alt],
            &output,
            &ExportLimits::default(),
        )
        .unwrap();
        let result = inspect(&output);
        assert!(!result.is_sidecar());
        assert_eq!(result.tensor_bytes("a", 2).unwrap(), [8, 9]);
        assert_eq!(result.tensor_bytes("b", 1).unwrap(), [3]);
        for name in ["config.json", "tokenizer.json"] {
            assert_eq!(
                fs::read(output.join(name)).unwrap(),
                fs::read(bp.join(name)).unwrap()
            );
        }
        let before = fs::read(bp.join(SHARD)).unwrap();
        let after = fs::read(output.join(SHARD)).unwrap();
        let prefix = 8 + u64::from_le_bytes(before[..8].try_into().unwrap()) as usize;
        assert_eq!(&before[..prefix], &after[..prefix]);
        let index: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("model.safetensors.index.json")).unwrap())
                .unwrap();
        assert_eq!(
            index["metadata"]["sparknest_payload_manifest"],
            receipt.payload_manifest.to_string()
        );
        assert_eq!(receipt.files[SHARD].digest, Digest::of(&after));
        assert_ne!(receipt.files[SHARD], base.files()[SHARD]);
    }
    assert_eq!(fs::read_dir(&ap).unwrap().count(), 1);
}

#[test]
fn raw_payload_and_scale_offsets_can_differ_from_canonical_output() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1, 2], &[3]);
    rewrite_shard(
        &bp,
        SHARD,
        json!({
            "a":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,2]},
            "b":{"dtype":"F8_E8M0","shape":[1],"data_offsets":[2,3]}
        }),
        &[1, 2, 3],
    );
    raw_sidecar(
        &ap,
        json!({
            "a":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[1,3]},
            "b":{"dtype":"F8_E8M0","shape":[1],"data_offsets":[0,1]}
        }),
        &[6, 4, 5],
        Some(&["a"]),
    );
    let base = inspect(&bp);
    let alt = inspect_raw(&ap, true);
    let config = replacement_config(&base, &[(&alt, "producer-manifest")]);
    let output = temp.path().join("output");
    export(
        &config,
        &profile(true),
        &base,
        &[&alt],
        &output,
        &ExportLimits::default(),
    )
    .unwrap();
    let result = inspect(&output);
    assert_eq!(result.tensor_bytes("a", 2).unwrap(), [4, 5]);
    assert_eq!(result.tensor_bytes("b", 1).unwrap(), [6]);
}

#[test]
fn raw_ordering_and_reference_and_kind_contracts() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    let ap2 = temp.path().join("second/alternate-checkpoint");
    snapshot(&bp, &[1], &[2]);
    let header = json!({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}});
    raw_sidecar(&ap, header.clone(), &[3], None);
    raw_sidecar(&ap2, header, &[4], None);
    let base = inspect(&bp);
    let alt = inspect_raw(&ap, false);
    let alt2 = inspect_raw(&ap2, false);
    let mut config = replacement_config(&base, &[(&alt, "all"), (&alt2, "all")]);
    config.alternates[1].requires =
        nest_model::Reference::LowerCompositionKey(config.prefix_key(&profile(false), 1).unwrap());
    let plan = resolve(
        &config,
        &profile(false),
        &base,
        &[&alt, &alt2],
        &ExportLimits::default(),
    )
    .unwrap();
    assert_eq!(plan.winners()["a"], Some(1));
    assert_eq!(plan.winners()["b"], None);
    assert_eq!(plan.shadowed().len(), 1);
    config.alternates[1].requires = nest_model::Reference::BaseManifest(Digest::of(b"wrong"));
    assert!(
        resolve(
            &config,
            &profile(false),
            &base,
            &[&alt, &alt2],
            &ExportLimits::default()
        )
        .is_err()
    );
    config.alternates.truncate(1);
    config.alternates[0].kind = nest_model::SourceKind::Checkpoint;
    assert!(
        resolve(
            &config,
            &profile(false),
            &base,
            &[&alt],
            &ExportLimits::default()
        )
        .is_err()
    );
    config.alternates[0].kind = nest_model::SourceKind::Lora;
    assert!(matches!(
        resolve(
            &config,
            &profile(false),
            &base,
            &[&alt],
            &ExportLimits::default()
        ),
        Err(nest_model::Error::Unsupported(_))
    ));
    let config = replacement_config(&alt, &[]);
    let single = Profile::new(
        BTreeMap::from([("a".into(), BTreeSet::from(["a".into()]))]),
        false,
    )
    .unwrap();
    assert!(resolve(&config, &single, &alt, &[], &ExportLimits::default()).is_err());
}

#[test]
fn raw_incomplete_companion_and_unknown_groups_and_tensors_refuse() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1], &[2]);
    raw_sidecar(
        &ap,
        json!({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}),
        &[3],
        Some(&["b"]),
    );
    let base = inspect(&bp);
    let alt = inspect_raw(&ap, true);
    for selection in ["all", "producer-manifest"] {
        let config = replacement_config(&base, &[(&alt, selection)]);
        assert!(
            resolve(
                &config,
                &profile(true),
                &base,
                &[&alt],
                &ExportLimits::default()
            )
            .is_err()
        );
    }
    let config = replacement_config(&base, &[(&alt, "producer-manifest")]);
    assert!(
        resolve(
            &config,
            &profile(false),
            &base,
            &[&alt],
            &ExportLimits::default()
        )
        .is_err()
    );
}

#[test]
fn raw_bundle_confinement_and_complete_byte_coverage() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("alternate-checkpoint");
    raw_sidecar(
        &root,
        json!({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}),
        &[1],
        None,
    );
    assert!(
        Snapshot::inspect_sidecar(
            &root,
            "../alternate-checkpoint.safetensors",
            None,
            &ExportLimits::default()
        )
        .is_err()
    );
    assert!(
        Snapshot::inspect_sidecar(
            &root,
            SIDECAR,
            Some("../groups.json"),
            &ExportLimits::default()
        )
        .is_err()
    );
    let original = fs::read(root.join(SIDECAR)).unwrap();
    for bytes in [
        original[..original.len() - 1].to_vec(),
        [original.clone(), vec![9]].concat(),
        vec![255; 8],
    ] {
        fs::write(root.join(SIDECAR), bytes).unwrap();
        assert!(Snapshot::inspect_sidecar(&root, SIDECAR, None, &ExportLimits::default()).is_err());
    }
    fs::write(root.join(SIDECAR), original).unwrap();
    fs::write(root.join("checkpoint.py"), b"raise Exception()").unwrap();
    assert!(Snapshot::inspect_sidecar(&root, SIDECAR, None, &ExportLimits::default()).is_err());
}

#[test]
fn raw_content_identity_uses_file_roles_not_acquisition_filenames() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("alternate-checkpoint");
    raw_sidecar(
        &root,
        json!({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}),
        &[1],
        Some(&["a"]),
    );
    let first = inspect_raw(&root, true);
    fs::rename(root.join(SIDECAR), root.join("renamed.safetensors")).unwrap();
    fs::rename(root.join("groups.json"), root.join("selection.json")).unwrap();
    let second = Snapshot::inspect_sidecar(
        &root,
        "renamed.safetensors",
        Some("selection.json"),
        &ExportLimits::default(),
    )
    .unwrap();
    assert_eq!(first.content_manifest(), second.content_manifest());
    let roles = BTreeMap::from([
        ("selection", first.files()["groups.json"].clone()),
        ("tensors", first.files()[SIDECAR].clone()),
    ]);
    assert_eq!(
        *second.content_manifest(),
        Digest::of(&serde_json::to_vec(&("sparknest.sidecar-input-manifest.v1", roles)).unwrap())
    );
}

#[test]
fn raw_unknown_tensor_cannot_extend_base_inventory() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1], &[]);
    rewrite_shard(
        &bp,
        SHARD,
        json!({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}),
        &[1],
    );
    fs::write(
        bp.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"metadata":{"total_size":1},"weight_map":{"a":SHARD}})).unwrap(),
    )
    .unwrap();
    raw_sidecar(
        &ap,
        json!({"b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}),
        &[2],
        None,
    );
    let base = inspect(&bp);
    let alt = inspect_raw(&ap, false);
    let config = replacement_config(&base, &[(&alt, "all")]);
    let single = Profile::new(
        BTreeMap::from([("a".into(), BTreeSet::from(["a".into()]))]),
        false,
    )
    .unwrap();
    assert!(
        matches!(resolve(&config,&single,&base,&[&alt],&ExportLimits::default()),Err(nest_model::Error::Invalid(message)) if message.contains("unknown alternate tensor"))
    );
}

#[test]
fn raw_shape_dtype_and_representation_mismatches_refuse() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1], &[2]);
    let base = inspect(&bp);
    for header in [
        json!({"a":{"dtype":"U8","shape":[1,1],"data_offsets":[0,1]}}),
        json!({"a":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}),
        json!({"__metadata__":{"format":"alternate"},"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}),
    ] {
        raw_sidecar(&ap, header, &[3], None);
        let alt = inspect_raw(&ap, false);
        let config = replacement_config(&base, &[(&alt, "all")]);
        assert!(
            resolve(
                &config,
                &profile(false),
                &base,
                &[&alt],
                &ExportLimits::default()
            )
            .is_err()
        );
    }
}

#[test]
fn raw_selection_is_strict_and_digest_bound() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1], &[2]);
    raw_sidecar(
        &ap,
        json!({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}),
        &[3],
        Some(&["a"]),
    );
    let base = inspect(&bp);
    let alt = inspect_raw(&ap, true);
    let mut config = replacement_config(&base, &[(&alt, "producer-manifest")]);
    config.alternates[0].selection_manifest_digest = Some(Digest::of(b"wrong"));
    assert!(
        resolve(
            &config,
            &profile(false),
            &base,
            &[&alt],
            &ExportLimits::default()
        )
        .is_err()
    );
    for selection in [
        br#"{"schema":"sparknest.model-selection.v2","groups":["a"]}"#.as_slice(),
        br#"{"schema":"sparknest.model-selection.v1","groups":["a","a"]}"#,
        br#"{"schema":"sparknest.model-selection.v1","groups":[""]}"#,
        br#"{"schema":"sparknest.model-selection.v1","groups":["a"],"extra":true}"#,
        br#"{"schema":"sparknest.model-selection.v1","groups":["a"],"groups":["b"]}"#,
    ] {
        fs::write(ap.join("groups.json"), selection).unwrap();
        assert!(
            Snapshot::inspect_sidecar(&ap, SIDECAR, Some("groups.json"), &ExportLimits::default())
                .is_err()
        );
    }
}

#[test]
fn raw_limits_and_header_schema_validation() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("alternate-checkpoint");
    raw_sidecar(
        &root,
        json!({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}),
        &[1, 2],
        Some(&["a"]),
    );
    for limits in [
        ExportLimits {
            max_files: 1,
            ..ExportLimits::default()
        },
        ExportLimits {
            max_tensors: 1,
            ..ExportLimits::default()
        },
        ExportLimits {
            metadata_bytes: 8,
            ..ExportLimits::default()
        },
    ] {
        assert!(Snapshot::inspect_sidecar(&root, SIDECAR, Some("groups.json"), &limits).is_err());
    }
    for header in [
        json!({"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}),
        json!({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}),
        json!({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"extra":true}}),
        json!({"a":{"dtype":"unknown","shape":[1],"data_offsets":[0,1]}}),
    ] {
        rewrite_shard(&root, SIDECAR, header, &[1, 2]);
        assert!(
            Snapshot::inspect_sidecar(
                &root,
                SIDECAR,
                Some("groups.json"),
                &ExportLimits::default()
            )
            .is_err()
        );
    }
}

#[cfg(unix)]
#[test]
fn raw_bundle_refuses_declared_symlinks_and_undeclared_json() {
    use std::os::unix::fs::symlink;
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("alternate-checkpoint");
    raw_sidecar(
        &root,
        json!({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}),
        &[1],
        None,
    );
    let actual = temp.path().join("actual.safetensors");
    fs::rename(root.join(SIDECAR), &actual).unwrap();
    symlink(&actual, root.join(SIDECAR)).unwrap();
    assert!(Snapshot::inspect_sidecar(&root, SIDECAR, None, &ExportLimits::default()).is_err());
    let root2 = temp.path().join("second/alternate-checkpoint");
    raw_sidecar(
        &root2,
        json!({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}),
        &[1],
        Some(&["a"]),
    );
    assert!(Snapshot::inspect_sidecar(&root2, SIDECAR, None, &ExportLimits::default()).is_err());
    let selection = temp.path().join("selection.json");
    fs::rename(root2.join("groups.json"), &selection).unwrap();
    symlink(&selection, root2.join("groups.json")).unwrap();
    assert!(
        Snapshot::inspect_sidecar(
            &root2,
            SIDECAR,
            Some("groups.json"),
            &ExportLimits::default()
        )
        .is_err()
    );
}

#[test]
fn raw_reuse_pins_reject_replaced_bundle_files_before_staging() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    snapshot(&bp, &[1], &[2]);
    raw_sidecar(
        &ap,
        json!({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}),
        &[3],
        None,
    );
    let base = inspect(&bp);
    let alt = inspect_raw(&ap, false);
    let mut config = replacement_config(&base, &[(&alt, "all")]);
    config.cache.reuse_unchanged = true;
    let replacement = temp.path().join("replacement.safetensors");
    fs::write(&replacement, fs::read(ap.join(SIDECAR)).unwrap()).unwrap();
    fs::rename(replacement, ap.join(SIDECAR)).unwrap();
    let output = temp.path().join("output");
    assert!(
        export(
            &config,
            &profile(false),
            &base,
            &[&alt],
            &output,
            &ExportLimits::default()
        )
        .is_err()
    );
    assert!(!output.exists());
}

#[cfg(unix)]
#[test]
fn detached_staging_crosses_filesystems_without_copying_reused_payloads() {
    use std::os::unix::fs::MetadataExt;
    let temp = TempDir::new().unwrap();
    let Ok(stage_parent) = tempfile::tempdir_in("/dev/shm") else {
        return;
    };
    if fs::metadata(temp.path()).unwrap().dev() == fs::metadata(stage_parent.path()).unwrap().dev()
    {
        return;
    }
    let bp = temp.path().join("base-checkpoint");
    let ap = temp.path().join("alternate-checkpoint");
    split_snapshot(&bp, &[1, 2], &vec![7; 32768]);
    split_snapshot(&ap, &[3, 4], &vec![7; 32768]);
    fs::write(
        ap.join("groups.json"),
        br#"{"schema":"sparknest.model-selection.v1","groups":["a"]}"#,
    )
    .unwrap();
    let base = inspect(&bp);
    let alt = inspect(&ap);
    let mut config = config(&base, &[(&alt, "producer-manifest")]);
    config.cache.reuse_unchanged = true;
    let staging = stage_parent.path().join("staging");
    let staged = nest_model::export_staged(
        &config,
        &profile(false),
        &base,
        &[&alt],
        &staging,
        &ExportLimits::default(),
    )
    .unwrap();
    assert!(!staging.join(SECOND).exists());
    assert!(!staging.join("config.json").exists());
    assert!(Snapshot::inspect(&staging, &ExportLimits::default()).is_err());
    assert_eq!(
        fs::read_dir(&staging).unwrap().count(),
        staged.staged_files().len()
    );
    let allocated: u64 = fs::read_dir(&staging)
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum();
    assert_eq!(allocated, staged.receipt.new_file_bytes);
    assert!(allocated < staged.receipt.logical_file_bytes);
    // Assemble ordinary complete files as the namespace manager must do. The
    // source and destination are separate filesystems; unchanged payloads never
    // visit the worker's native staging filesystem.
    let assembled = temp.path().join("assembled");
    fs::create_dir(&assembled).unwrap();
    for name in &staged.receipt.reused_files {
        fs::hard_link(bp.join(name), assembled.join(name)).unwrap();
    }
    for name in staged.staged_files() {
        fs::copy(staging.join(&name), assembled.join(&name)).unwrap();
    }
    let complete = inspect(&assembled);
    assert_eq!(complete.tensor_bytes("a", 2).unwrap(), [3, 4]);
    assert_eq!(complete.tensor_bytes("b", 32768).unwrap(), vec![7; 32768]);
    assert!(same_inode(&bp.join(SECOND), &assembled.join(SECOND)));
    let native = temp.path().join("native");
    let linked = export(
        &config,
        &profile(false),
        &base,
        &[&alt],
        &native,
        &ExportLimits::default(),
    )
    .unwrap();
    assert_eq!(linked.preparation_key, staged.receipt.preparation_key);
    assert_eq!(linked.payload_manifest, staged.receipt.payload_manifest);
    for name in linked
        .files
        .keys()
        .chain(std::iter::once(&"composition.receipt.json".to_string()))
    {
        assert_eq!(
            fs::read(native.join(name)).unwrap(),
            fs::read(assembled.join(name)).unwrap()
        );
    }
    // Admission is based on new regular-file bytes, independent of reused files.
    config.cache.max_new_physical_bytes = allocated - 1;
    let refused = stage_parent.path().join("refused");
    assert!(
        nest_model::export_staged(
            &config,
            &profile(false),
            &base,
            &[&alt],
            &refused,
            &ExportLimits::default()
        )
        .is_err()
    );
    assert!(!refused.exists());
}

#[test]
fn detached_base_only_stages_metadata_and_full_copy_still_emits_a_snapshot() {
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    snapshot(&bp, &[1], &[2]);
    let base = inspect(&bp);
    let mut config = config(&base, &[]);
    config.cache.reuse_unchanged = true;
    let staging = temp.path().join("staging");
    let staged = nest_model::export_staged(
        &config,
        &profile(false),
        &base,
        &[],
        &staging,
        &ExportLimits::default(),
    )
    .unwrap();
    assert_eq!(staged.receipt.copied_file_bytes, 0);
    assert_eq!(staged.receipt.copied_payload_bytes, 0);
    assert_eq!(
        staged.staged_files(),
        BTreeSet::from([
            "model.safetensors.index.json".into(),
            "composition.payload.json".into(),
            "composition.receipt.json".into()
        ])
    );
    // Existing destinations refuse without replacing evidence.
    let before = fs::read(staging.join("composition.receipt.json")).unwrap();
    assert!(
        nest_model::export_staged(
            &config,
            &profile(false),
            &base,
            &[],
            &staging,
            &ExportLimits::default()
        )
        .is_err()
    );
    assert_eq!(
        fs::read(staging.join("composition.receipt.json")).unwrap(),
        before
    );
    config.cache.reuse_unchanged = false;
    let copied = temp.path().join("copied");
    let staged = nest_model::export_staged(
        &config,
        &profile(false),
        &base,
        &[],
        &copied,
        &ExportLimits::default(),
    )
    .unwrap();
    assert!(staged.receipt.reused_files.is_empty());
    assert_eq!(inspect(&copied).tensor_bytes("a", 1).unwrap(), [1]);
}

#[test]
fn offline_worker_process_inspects_and_stages_without_publication() {
    use nest_model::worker::{Input, OperandProfile, Request, Response};
    use std::io::Write;
    use std::process::{Command, Stdio};
    let temp = TempDir::new().unwrap();
    let bp = temp.path().join("base-checkpoint");
    snapshot(&bp, &[1], &[2]);
    let base = inspect(&bp);
    let mut module = config(&base, &[]);
    module.cache.reuse_unchanged = true;
    let limits = ExportLimits {
        metadata_bytes: 1024 * 1024,
        ..ExportLimits::default()
    };
    let inspection = Request::Inspect {
        inputs: vec![Input::Indexed { root: bp.clone() }],
        limits: limits.clone(),
    }
    .run()
    .unwrap();
    let Response::Inspect { inputs } = inspection else {
        panic!("inspection")
    };
    assert_eq!(inputs[0].content_manifest, *base.content_manifest());
    let destination = temp.path().join("staging");
    let request = Request::Stage {
        module: Box::new(module),
        profile: OperandProfile {
            schema: "sparknest.operand-profile.v1".into(),
            groups: profile(false)
                .groups()
                .iter()
                .map(|(name, members)| (name.clone(), members.iter().cloned().collect()))
                .collect(),
            static_activation_calibration: false,
        },
        base: Input::Indexed { root: bp },
        alternates: vec![],
        destination: destination.clone(),
        limits,
    };
    let bytes = serde_json::to_vec(&request).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_nest-model-worker"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&bytes).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let Response::Stage { staged } = serde_json::from_slice(&output.stdout).unwrap() else {
        panic!("stage")
    };
    assert_eq!(staged.receipt.qualification, "pending");
    assert_eq!(staged.receipt.copied_payload_bytes, 0);
    assert!(!destination.join(SHARD).exists());
    assert!(Snapshot::inspect(&destination, &ExportLimits::default()).is_err());
    // The process accepts a finite strict protocol, never checkpoint code or a
    // caller-selected shell command. Invalid requests cannot touch new output.
    assert!(Request::read(br#"{"operation":"inspect","operation":"stage"}"#.as_slice()).is_err());
    let mut value = serde_json::to_value(request).unwrap();
    value["profile"]["groups"]["a"] = json!(["a", "a"]);
    value["destination"] = json!(temp.path().join("invalid"));
    assert!(
        Request::read(serde_json::to_vec(&value).unwrap().as_slice())
            .unwrap()
            .run()
            .is_err()
    );
    assert!(!temp.path().join("invalid").exists());
    value["profile"]["groups"]["a"] = json!(["a"]);
    value["module"]["preparation"]["max_ram_bytes"] = json!(1024);
    assert!(matches!(
        Request::read(serde_json::to_vec(&value).unwrap().as_slice())
            .unwrap()
            .run(),
        Err(nest_model::Error::Limit(_))
    ));
    assert!(!temp.path().join("invalid").exists());
}
