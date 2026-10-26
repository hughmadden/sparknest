use nest_model::{
    ActivationCalibration, Digest, ExportLimits, FileIdentity, LossyConversion, ModuleConfig,
    Profile, QualificationReceipt, Snapshot, export, policy_identity, publication_identity,
    qualify_exact,
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};
use tempfile::TempDir;

const SHARD: &str = "model-00001-of-00001.safetensors";
const INDEX: &str = "model.safetensors.index.json";
const PAYLOAD: &str = "composition.payload.json";
const EXPORT: &str = "composition.receipt.json";

fn write_json(root: &Path, name: &str, value: &Value) {
    fs::write(root.join(name), serde_json::to_vec(value).unwrap()).unwrap();
}

fn read_json(root: &Path, name: &str) -> Value {
    serde_json::from_slice(&fs::read(root.join(name)).unwrap()).unwrap()
}

fn shard(path: &Path, alternate: bool) {
    let mut header = serde_json::to_vec(&if alternate {
        json!({"a":{"dtype":"U8","shape":[19],"data_offsets":[0,19]}})
    } else {
        json!({
            "a":{"dtype":"U8","shape":[19],"data_offsets":[0,19]},
            "b":{"dtype":"U8","shape":[11],"data_offsets":[19,30]}
        })
    })
    .unwrap();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend([if alternate { 9 } else { 1 }; 19]);
    if !alternate {
        bytes.extend([2; 11]);
    }
    fs::write(path, bytes).unwrap();
}

fn profile(static_calibration: bool) -> Profile {
    Profile::new(
        BTreeMap::from([
            ("first".into(), BTreeSet::from(["a".into()])),
            ("second".into(), BTreeSet::from(["b".into()])),
        ]),
        static_calibration,
    )
    .unwrap()
}

struct Fixture {
    temp: TempDir,
    base: Snapshot,
    alternate: Snapshot,
    config: ModuleConfig,
    profile: Profile,
    output: PathBuf,
    limits: ExportLimits,
}

impl Fixture {
    fn new() -> Self {
        let temp = TempDir::new().unwrap();
        let base_path = temp.path().join("base");
        let alt_path = temp.path().join("replacement");
        fs::create_dir(&base_path).unwrap();
        fs::create_dir(&alt_path).unwrap();
        shard(&base_path.join(SHARD), false);
        shard(&alt_path.join("tensors.safetensors"), true);
        write_json(&base_path, "config.json", &json!({"architectures":[]}));
        write_json(&base_path, "tokenizer.json", &json!({"version":"1"}));
        write_json(
            &base_path,
            INDEX,
            &json!({"metadata":{"total_size":30},"weight_map":{"a":SHARD,"b":SHARD}}),
        );
        let limits = ExportLimits {
            metadata_bytes: 1024 * 1024,
            buffer_bytes: 3,
            ..ExportLimits::default()
        };
        let base = Snapshot::inspect(&base_path, &limits).unwrap();
        let alternate =
            Snapshot::inspect_sidecar(&alt_path, "tensors.safetensors", None, &limits).unwrap();
        let config: ModuleConfig = serde_json::from_value(json!({
            "schema":"sparknest.model-composition.v1", "name":"synthetic",
            "base":{"source":"base","content_manifest":base.content_manifest()},
            "alternates":[{
                "name":"replacement", "kind":"replacement", "source":"replacement",
                "content_manifest":alternate.content_manifest(), "selection":"all",
                "selection_manifest":null,"selection_manifest_digest":null,
                "requires":{"base_manifest":base.content_manifest()}
            }],
            "output":{
                "loader_profile":"base-compatible-indexed-v1","precision":"match-base",
                "configuration":"base-unchanged","tokenizer":"base-unchanged",
                "non_weight_mismatch":"reject","namespace":"synthetic"
            },
            "preparation":{
                "trigger":"explicit-prepare-or-external-use-wrapper",
                "adapter_image":format!("synthetic@{}",Digest::of(b"image")),
                "recipe":Digest::of(b"recipe"),"parameters":{},"lossy_conversion":"deny",
                "qualification_profile":policy_identity(),"activation_calibration":"preserve-and-qualify",
                "max_runtime_seconds":60,"max_ram_bytes":134217728,"cpu_threads":1,
                "device":"cpu","worker_pool":"cpu"
            },
            "cache":{
                "publication":"sealed-regular-files","reuse_unchanged":false,"prefer_reflink":false,
                "max_new_physical_bytes":1048576,"account_inflight":true,
                "eviction":"least-recently-used-unpinned","retain_dependencies":true
            }
        }))
        .unwrap();
        let profile = profile(false);
        let output = temp.path().join("output");
        export(&config, &profile, &base, &[&alternate], &output, &limits).unwrap();
        Self {
            temp,
            base,
            alternate,
            config,
            profile,
            output,
            limits,
        }
    }

    fn qualify(&self) -> nest_model::Result<QualificationReceipt> {
        let output = Snapshot::inspect(&self.output, &self.limits)?;
        qualify_exact(
            &self.config,
            &self.profile,
            &self.base,
            &[&self.alternate],
            &output,
            &self.limits,
        )
    }

    /// Rebind every declared output hash after tampering, so corruption tests
    /// reach the actual source comparison rather than just stale export hashes.
    fn rebind_output(&self) {
        let mut payload = read_json(&self.output, PAYLOAD);
        for (name, id) in payload["files"].as_object_mut().unwrap() {
            *id = serde_json::to_value(identity(&self.output.join(name))).unwrap();
        }
        write_json(&self.output, PAYLOAD, &payload);
        let mut index = read_json(&self.output, INDEX);
        index["metadata"]["sparknest_payload_manifest"] =
            json!(identity(&self.output.join(PAYLOAD)).digest);
        write_json(&self.output, INDEX, &index);
        let mut receipt = read_json(&self.output, EXPORT);
        receipt["payload_manifest"] = json!(identity(&self.output.join(PAYLOAD)).digest);
        for (name, id) in receipt["files"].as_object_mut().unwrap() {
            *id = serde_json::to_value(identity(&self.output.join(name))).unwrap();
        }
        write_json(&self.output, EXPORT, &receipt);
    }

    fn corrupt_tensor(&self, tensor_offset: usize) {
        let path = self.output.join(SHARD);
        let mut bytes = fs::read(&path).unwrap();
        let start = 8 + u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        bytes[start + tensor_offset] ^= 0x80;
        fs::write(path, bytes).unwrap();
        self.rebind_output();
    }
}

fn identity(path: &Path) -> FileIdentity {
    let bytes = fs::read(path).unwrap();
    FileIdentity {
        bytes: bytes.len() as u64,
        digest: Digest::of(&bytes),
    }
}

#[test]
fn built_in_policy_has_an_exact_versioned_digest() {
    assert_eq!(
        nest_model::EXACT_POLICY_NAME,
        "sparknest.exact-composition-cpu.v1"
    );
    assert_eq!(
        policy_identity().to_string(),
        "sha256:1eac557403f69d88496c1433083bbce9ad022bb3040337689d52009f45aa61c6"
    );
}

#[test]
fn valid_exact_proof_covers_winner_and_fallback_deterministically() {
    let f = Fixture::new();
    let proof = f.qualify().unwrap();
    assert_eq!(proof.policy, policy_identity());
    assert_eq!(
        proof.qualification_policy,
        f.config.qualification_key().unwrap()
    );
    assert_eq!(
        proof.preparation_key,
        f.config.preparation_key(&f.profile).unwrap()
    );
    assert_eq!(
        proof.output_payload_manifest,
        identity(&f.output.join(PAYLOAD)).digest
    );
    assert_eq!(proof.finished_index, identity(&f.output.join(INDEX)));
    assert_eq!(proof.compared_tensors, 2);
    assert_eq!(proof.compared_bytes, 30);
    assert!(proof.requires_model_qualification);
    assert_eq!(
        proof.comparisons[0].source_manifest,
        *f.alternate.content_manifest()
    );
    assert_eq!(
        proof.comparisons[1].source_manifest,
        *f.base.content_manifest()
    );
    assert_eq!(proof.comparisons[0].bytes, 19);
    assert_eq!(proof.comparisons[1].bytes, 11);
    assert_eq!(
        proof.publication_identity,
        publication_identity(
            &proof.preparation_key,
            &proof.output_payload_manifest,
            &proof.finished_index
        )
        .unwrap()
    );
    let bytes = serde_json::to_vec(&proof).unwrap();
    assert_eq!(bytes, serde_json::to_vec(&f.qualify().unwrap()).unwrap());
    let decoded: QualificationReceipt = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(decoded, proof);
    assert!(
        !String::from_utf8(bytes)
            .unwrap()
            .contains(f.temp.path().to_str().unwrap())
    );
}

#[test]
fn rejects_changed_winner_even_with_consistent_export_hashes() {
    let f = Fixture::new();
    f.corrupt_tensor(10);
    assert!(
        f.qualify()
            .unwrap_err()
            .to_string()
            .contains("tensor bytes differ")
    );
}

#[test]
fn rejects_fallback_corruption_even_with_consistent_export_hashes() {
    let f = Fixture::new();
    f.corrupt_tensor(25);
    assert!(
        f.qualify()
            .unwrap_err()
            .to_string()
            .contains("tensor bytes differ")
    );
}

#[test]
fn rejects_missing_and_extra_output_files() {
    for extra in [false, true] {
        let f = Fixture::new();
        if extra {
            write_json(
                &f.output,
                "hidden.json",
                &json!({"schema":"sparknest.model-selection.v1","groups":["first"]}),
            );
        } else {
            fs::remove_file(f.output.join("tokenizer.json")).unwrap();
        }
        assert!(f.qualify().is_err());
    }
}

#[test]
fn rejects_configuration_and_export_source_mismatches() {
    let mut f = Fixture::new();
    f.config.alternates[0].content_manifest = Digest::of(b"wrong input");
    assert!(f.qualify().is_err());
    let f = Fixture::new();
    let mut receipt = read_json(&f.output, EXPORT);
    receipt["base"] = json!(Digest::of(b"wrong base"));
    write_json(&f.output, EXPORT, &receipt);
    assert!(f.qualify().is_err());
}

#[test]
fn refuses_unknown_policy_and_claimed_pass_without_exact_policy() {
    for policy in [b"unknown".as_slice(), b"passed".as_slice()] {
        let mut f = Fixture::new();
        f.config.preparation.qualification_profile = Digest::of(policy);
        assert!(f.qualify().is_err());
    }
    let f = Fixture::new();
    let mut receipt = read_json(&f.output, EXPORT);
    receipt["qualification"] = json!("passed");
    write_json(&f.output, EXPORT, &receipt);
    assert!(f.qualify().is_err());
}

#[test]
fn refuses_static_calibration_lossy_conversion_and_recipe_parameters() {
    let mut f = Fixture::new();
    f.profile = profile(true);
    assert!(
        f.qualify()
            .unwrap_err()
            .to_string()
            .contains("static activation")
    );
    let mut f = Fixture::new();
    f.config.preparation.lossy_conversion = LossyConversion::RequireQualifiedProfile;
    assert!(f.qualify().is_err());
    let mut f = Fixture::new();
    f.config.preparation.activation_calibration = ActivationCalibration::Recalibrate;
    assert!(f.qualify().is_err());
    let mut f = Fixture::new();
    f.config
        .preparation
        .parameters
        .insert("convert".into(), "yes".into());
    assert!(f.qualify().is_err());
}

#[test]
fn operational_counters_and_copy_choice_do_not_change_publication_identity() {
    let f = Fixture::new();
    let first = f.qualify().unwrap();
    let mut receipt = read_json(&f.output, EXPORT);
    receipt["copied_payload_bytes"] = json!(999);
    receipt["reused_payload_bytes"] = json!(998);
    write_json(&f.output, EXPORT, &receipt);
    assert_eq!(first, f.qualify().unwrap());
    let mut config = f.config.clone();
    config.cache.reuse_unchanged = true;
    let output = f.temp.path().join("reused-output");
    export(
        &config,
        &f.profile,
        &f.base,
        &[&f.alternate],
        &output,
        &f.limits,
    )
    .unwrap();
    let snapshot = Snapshot::inspect(&output, &f.limits).unwrap();
    let reused = qualify_exact(
        &config,
        &f.profile,
        &f.base,
        &[&f.alternate],
        &snapshot,
        &f.limits,
    )
    .unwrap();
    assert_eq!(first, reused);
}

#[test]
fn rejects_header_or_base_asset_changes_even_with_rebound_hashes() {
    let f = Fixture::new();
    write_json(&f.output, "tokenizer.json", &json!({"version":"changed"}));
    f.rebind_output();
    assert!(f.qualify().is_err());
    let f = Fixture::new();
    let path = f.output.join(SHARD);
    let mut bytes = fs::read(&path).unwrap();
    let end = 8 + u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    // Harmless JSON whitespace still violates exact canonical base headers.
    bytes[end - 1] = b'\n';
    fs::write(path, bytes).unwrap();
    f.rebind_output();
    assert!(f.qualify().is_err());
}

#[test]
fn detects_inventory_changes_after_output_inspection_and_bounds_ram() {
    let mut f = Fixture::new();
    let snapshot = Snapshot::inspect(&f.output, &f.limits).unwrap();
    write_json(&f.output, "extra.json", &json!({}));
    assert!(
        qualify_exact(
            &f.config,
            &f.profile,
            &f.base,
            &[&f.alternate],
            &snapshot,
            &f.limits
        )
        .is_err()
    );
    fs::remove_file(f.output.join("extra.json")).unwrap();
    f.config.preparation.max_ram_bytes = 1;
    assert!(f.qualify().is_err());
}

#[test]
fn base_only_exact_proof_does_not_require_an_alternate() {
    let f = Fixture::new();
    let mut config = f.config.clone();
    config.alternates.clear();
    let output = f.temp.path().join("base-only");
    export(&config, &f.profile, &f.base, &[], &output, &f.limits).unwrap();
    let snapshot = Snapshot::inspect(&output, &f.limits).unwrap();
    let proof = qualify_exact(&config, &f.profile, &f.base, &[], &snapshot, &f.limits).unwrap();
    assert!(!proof.requires_model_qualification);
    assert_eq!(proof.compared_tensors, 2);
    assert_eq!(proof.compared_bytes, 30);
    assert!(
        proof
            .comparisons
            .iter()
            .all(|c| c.source_manifest == *f.base.content_manifest())
    );
}

#[test]
fn evidence_is_stored_outside_bound_snapshot_and_buffer_size_is_not_identity() {
    let f = Fixture::new();
    let proof = f.qualify().unwrap();
    let mut limits = f.limits.clone();
    limits.buffer_bytes = 1;
    let output = Snapshot::inspect(&f.output, &limits).unwrap();
    let single_byte = qualify_exact(
        &f.config,
        &f.profile,
        &f.base,
        &[&f.alternate],
        &output,
        &limits,
    )
    .unwrap();
    assert_eq!(proof, single_byte);
    let evidence = f.temp.path().join("qualification-evidence.json");
    fs::write(&evidence, serde_json::to_vec(&proof).unwrap()).unwrap();
    assert_eq!(proof, f.qualify().unwrap());
    fs::copy(&evidence, f.output.join("qualification-evidence.json")).unwrap();
    assert!(f.qualify().is_err());
}

#[test]
fn rejects_payload_index_and_winner_provenance_mismatches() {
    for field in ["payload", "index", "winner", "shadow", "image", "required"] {
        let f = Fixture::new();
        match field {
            "payload" => {
                let mut manifest = read_json(&f.output, PAYLOAD);
                manifest["files"][SHARD]["digest"] = json!(Digest::of(b"forged payload"));
                write_json(&f.output, PAYLOAD, &manifest);
            }
            "index" => {
                let mut index = read_json(&f.output, INDEX);
                index["metadata"]["sparknest_preparation_key"] = json!(Digest::of(b"forged key"));
                write_json(&f.output, INDEX, &index);
            }
            _ => {
                let mut receipt = read_json(&f.output, EXPORT);
                match field {
                    "winner" => receipt["winners"]["first"] = Value::Null,
                    "shadow" => {
                        receipt["shadowed"] =
                            json!([{"group":"first","previous_alternate":0,"winning_alternate":0}])
                    }
                    "image" => receipt["adapter_image_digest"] = json!(Digest::of(b"forged image")),
                    "required" => receipt["requires_model_qualification"] = json!(false),
                    _ => unreachable!(),
                }
                write_json(&f.output, EXPORT, &receipt);
            }
        }
        assert!(f.qualify().is_err(), "accepted invalid {field} provenance");
    }
}

#[test]
fn worker_qualify_roundtrip_returns_evidence_from_actual_comparisons() {
    let f = Fixture::new();
    let request = json!({
        "operation":"qualify", "module":f.config,
        "profile":{
            "schema":"sparknest.operand-profile.v1",
            "groups":{"first":["a"],"second":["b"]},
            "static_activation_calibration":false
        },
        "base":{"layout":"indexed","root":f.temp.path().join("base")},
        "alternates":[{"layout":"sidecar","root":f.temp.path().join("replacement"),
            "tensors":"tensors.safetensors","selection_manifest":null}],
        "output":f.output, "limits":f.limits
    });
    let bytes = serde_json::to_vec(&request).unwrap();
    let response = nest_model::worker::Request::read(bytes.as_slice())
        .unwrap()
        .run()
        .unwrap();
    let response = serde_json::to_value(response).unwrap();
    assert_eq!(response["operation"], "qualify");
    let proof: QualificationReceipt = serde_json::from_value(response["receipt"].clone()).unwrap();
    assert_eq!(proof, f.qualify().unwrap());
    f.corrupt_tensor(0);
    assert!(
        nest_model::worker::Request::read(bytes.as_slice())
            .unwrap()
            .run()
            .is_err()
    );
}

#[test]
fn inspected_inputs_cannot_bypass_tighter_qualification_limits() {
    let f = Fixture::new();
    let output = Snapshot::inspect(&f.output, &f.limits).unwrap();
    for limits in [
        ExportLimits {
            max_tensors: 1,
            ..f.limits.clone()
        },
        ExportLimits {
            max_files: 1,
            ..f.limits.clone()
        },
        ExportLimits {
            metadata_bytes: 1,
            ..f.limits.clone()
        },
    ] {
        assert!(
            qualify_exact(
                &f.config,
                &f.profile,
                &f.base,
                &[&f.alternate],
                &output,
                &limits
            )
            .is_err()
        );
    }
}
