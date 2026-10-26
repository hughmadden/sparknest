//! CPU proof of exact preparation under one built-in, versioned policy.
//!
//! Every tensor is compared byte for byte with its resolved winning source,
//! including base fallback. This is stronger than tolerance-based numerical
//! equality, preserving encoded scales, NaN payloads and signed zero exactly.
//! It makes no behavioral, inference-runtime, engine or hardware qualification
//! claim. Static activation calibration requires a different model gate.
//!
//! The caller owns immutable input and complete output namespace/generation
//! pins throughout inspection and proof, and output dependencies for their
//! entire lifetime. Store this evidence separately from the already-bound
//! output snapshot: adding it there changes the inventory and creates cycles.

use crate::{
    Digest, Error, ExportLimits, ExportReceipt, FileIdentity, LossyConversion, ModuleConfig, Plan,
    Profile, Result, Snapshot, invalid, resolve,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
};

const INDEX: &str = "model.safetensors.index.json";
const PAYLOAD: &str = "composition.payload.json";
const EXPORT: &str = "composition.receipt.json";

pub const EXACT_POLICY_NAME: &str = "sparknest.exact-composition-cpu.v1";

// These exact bytes are the public policy identity. Changing any rule requires
// a new policy version and digest, rather than accepting an operator pass bit.
const POLICY: &[u8] = b"sparknest.exact-composition-cpu.v1\n\
lossy=deny;conversion=deny;lora=deny;static-activation-calibration=deny\n\
base-layout-and-assets=exact;provenance=verified;every-resolved-tensor=byte-equal\n\
claims=preparation-only;behavioral-runtime-engine-hardware=unqualified\n";

/// Exact SHA-256 identity of the built-in policy, independent of local paths.
pub fn policy_identity() -> Digest {
    Digest::of(POLICY)
}

/// The same identity must be used for publication and reproducibility checks.
/// Copy/link accounting and the export receipt's own bytes are operational
/// evidence, so neither participates in this identity.
pub fn publication_identity(
    preparation_key: &Digest,
    payload_manifest: &Digest,
    finished_index: &FileIdentity,
) -> Result<Digest> {
    Digest::canonical(&(
        "sparknest.composition-publication.v1",
        preparation_key,
        payload_manifest,
        finished_index,
    ))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorComparison {
    pub tensor: String,
    pub source_manifest: Digest,
    pub bytes: u64,
}

/// Evidence from a successful exact comparison, never a caller-supplied pass
/// boolean. Consumers must bind BOTH policy scopes: `policy` is the built-in
/// verifier identity; `qualification_policy` is ModuleConfig::qualification_key
/// (including the lossy-conversion configuration) retained by the catalog.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationReceipt {
    pub schema: String,
    pub policy_name: String,
    pub policy: Digest,
    pub qualification_policy: Digest,
    pub preparation_key: Digest,
    pub output_payload_manifest: Digest,
    pub finished_index: FileIdentity,
    pub publication_identity: Digest,
    pub base_manifest: Digest,
    pub alternate_manifests: Vec<Digest>,
    /// Retains the resolver's requirement. This configured exact CPU gate
    /// satisfies it without making a behavioral/runtime qualification claim.
    pub requires_model_qualification: bool,
    pub comparisons: Vec<TensorComparison>,
    pub compared_tensors: u64,
    pub compared_bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PayloadManifest {
    schema: String,
    files: BTreeMap<String, FileIdentity>,
}

/// Verify a complete, inspected output under the exact built-in CPU policy.
/// Synchronous and bounded: use the admitted worker, never an async/FUSE path.
/// No checkpoint code, conversion, calibration or inference is executed.
pub fn qualify_exact(
    config: &ModuleConfig,
    profile: &Profile,
    base: &Snapshot,
    alternates: &[&Snapshot],
    output: &Snapshot,
    limits: &ExportLimits,
) -> Result<QualificationReceipt> {
    config.validate()?;
    limits.validate()?;
    if config.preparation.qualification_profile != policy_identity() {
        return Err(Error::Unsupported(
            "unknown exact qualification policy".into(),
        ));
    }
    if !matches!(config.preparation.lossy_conversion, LossyConversion::Deny) {
        return Err(Error::Unsupported(
            "exact policy requires lossy_conversion=deny".into(),
        ));
    }
    if profile.static_activation_calibration {
        return Err(Error::Unsupported(
            "static activation calibration requires an actual model qualification gate".into(),
        ));
    }
    // Two fixed buffers compare arbitrary-sized tensors. Retained metadata for
    // every input and the complete output is admitted before allocation. The
    // worker's cgroup additionally enforces actual process RSS and wall time.
    let admitted = (alternates.len() as u64)
        .checked_add(2)
        .and_then(|n| n.checked_mul(limits.metadata_bytes))
        .and_then(|n| {
            (limits.buffer_bytes as u64)
                .checked_mul(2)
                .and_then(|b| n.checked_add(b))
        });
    if admitted.is_none_or(|bytes| bytes > config.preparation.max_ram_bytes) {
        return Err(Error::Limit(
            "aggregate qualification memory exceeds RAM admission".into(),
        ));
    }
    check_pins(base, alternates, output, limits)?;
    let plan = resolve(config, profile, base, alternates, limits)?;
    let (payload_manifest, finished_index) =
        verify_output(config, &plan, base, alternates, output, limits)?;
    let mut expected_bytes = vec![0; limits.buffer_bytes];
    let mut actual_bytes = vec![0; limits.buffer_bytes];
    let mut comparisons = Vec::with_capacity(base.tensors.len());
    let mut compared_bytes = 0u64;
    // Ordered maps ensure stable evidence independent of input acquisition
    // paths, filesystem enumeration order and streaming buffer size.
    let sources: BTreeMap<_, _> = profile
        .groups()
        .iter()
        .flat_map(|(group, tensors)| {
            let source = plan.winners()[group].map(|i| alternates[i]).unwrap_or(base);
            tensors.iter().map(move |name| (name, source))
        })
        .collect();
    for (name, target) in &output.tensors {
        let source = sources[name];
        let bytes = target.spec.data_offsets[1] - target.spec.data_offsets[0];
        let mut expected = source.tensor_reader(&source.tensors[name])?;
        let mut actual = output.tensor_reader(target)?;
        let mut remaining = bytes;
        while remaining != 0 {
            let count = remaining.min(limits.buffer_bytes as u64) as usize;
            expected.read_exact(&mut expected_bytes[..count])?;
            actual.read_exact(&mut actual_bytes[..count])?;
            if expected_bytes[..count] != actual_bytes[..count] {
                return Err(invalid(format!(
                    "tensor bytes differ from intended source: {name}"
                )));
            }
            remaining -= count as u64;
        }
        compared_bytes = compared_bytes
            .checked_add(bytes)
            .ok_or_else(|| invalid("comparison byte count overflow"))?;
        comparisons.push(TensorComparison {
            tensor: name.clone(),
            source_manifest: source.content_manifest().clone(),
            bytes,
        });
    }
    check_pins(base, alternates, output, limits)?;
    Ok(QualificationReceipt {
        schema: "sparknest.exact-qualification-receipt.v1".into(),
        policy_name: EXACT_POLICY_NAME.into(),
        policy: policy_identity(),
        qualification_policy: config.qualification_key()?,
        publication_identity: publication_identity(
            &plan.preparation_key,
            &payload_manifest,
            &finished_index,
        )?,
        preparation_key: plan.preparation_key,
        output_payload_manifest: payload_manifest,
        finished_index,
        base_manifest: base.content_manifest().clone(),
        alternate_manifests: alternates
            .iter()
            .map(|s| s.content_manifest().clone())
            .collect(),
        requires_model_qualification: plan.requires_model_qualification,
        compared_tensors: comparisons.len() as u64,
        compared_bytes,
        comparisons,
    })
}

fn check_pins(
    base: &Snapshot,
    alternates: &[&Snapshot],
    output: &Snapshot,
    limits: &ExportLimits,
) -> Result<()> {
    for input in std::iter::once(base)
        .chain(alternates.iter().copied())
        .chain(std::iter::once(output))
    {
        // A public caller may have inspected with more permissive limits. Do
        // not let retained inventories bypass this qualification's admission.
        let metadata = input
            .files()
            .iter()
            .filter(|(name, _)| {
                name.ends_with(".json") || (input.is_sidecar() && !input.shards.contains_key(*name))
            })
            .map(|(_, id)| id.bytes)
            .chain(input.shards.values().map(|shard| shard.header.len() as u64))
            .try_fold(0u64, |total, bytes| total.checked_add(bytes));
        if input.files().len() > limits.max_files
            || input.tensors.len() > limits.max_tensors
            || metadata.is_none_or(|bytes| bytes > limits.metadata_bytes)
        {
            return Err(Error::Limit(
                "inspected snapshot exceeds qualification limits".into(),
            ));
        }
        input.revalidate_pins()?;
    }
    Ok(())
}

fn verify_output(
    config: &ModuleConfig,
    plan: &Plan,
    base: &Snapshot,
    alternates: &[&Snapshot],
    output: &Snapshot,
    limits: &ExportLimits,
) -> Result<(Digest, FileIdentity)> {
    if output.is_sidecar()
        || output.tensors.keys().ne(base.tensors.keys())
        || output.shards.keys().ne(base.shards.keys())
    {
        return Err(invalid(
            "output is not the complete base-compatible indexed layout",
        ));
    }
    for (name, tensor) in &output.tensors {
        let original = &base.tensors[name];
        if tensor.shard != original.shard
            || tensor.spec.dtype != original.spec.dtype
            || tensor.spec.shape != original.spec.shape
            || tensor.spec.data_offsets != original.spec.data_offsets
        {
            return Err(invalid("output tensor layout differs from base"));
        }
    }
    for (name, shard) in &output.shards {
        if shard.header != base.shards[name].header {
            return Err(invalid("output shard header differs from base"));
        }
    }
    if output.ancillary() != base.ancillary() {
        return Err(invalid("output ancillary base assets differ"));
    }
    let payload_names: BTreeSet<_> = base
        .ancillary()
        .keys()
        .copied()
        .chain(base.shards.keys().map(String::as_str))
        .collect();
    let expected_names: BTreeSet<_> = payload_names
        .iter()
        .copied()
        .chain([INDEX, PAYLOAD, EXPORT])
        .collect();
    if output
        .files()
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != expected_names
    {
        return Err(invalid(
            "output filename inventory differs from complete export",
        ));
    }
    let payload: PayloadManifest = output.read_json(PAYLOAD, limits)?;
    if payload.schema != "sparknest.payload-manifest.v1"
        || payload
            .files
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            != payload_names
        || payload
            .files
            .iter()
            .any(|(name, id)| output.files().get(name) != Some(id))
    {
        return Err(invalid(
            "output payload provenance differs from actual files",
        ));
    }
    let payload_manifest = output.files()[PAYLOAD].digest.clone();
    let mut expected_index: Value = base.read_json(INDEX, limits)?;
    expected_index["metadata"]["sparknest_preparation_key"] =
        Value::String(plan.preparation_key.to_string());
    expected_index["metadata"]["sparknest_payload_manifest"] =
        Value::String(payload_manifest.to_string());
    if output.read_json::<Value>(INDEX, limits)? != expected_index {
        return Err(invalid(
            "output index layout or preparation provenance differs",
        ));
    }
    let receipt: ExportReceipt = output.read_json(EXPORT, limits)?;
    let image_digest = config
        .preparation
        .adapter_image
        .rsplit_once('@')
        .ok_or_else(|| invalid("unpinned adapter image"))?
        .1;
    if receipt.schema != "sparknest.export-receipt.v1"
        || receipt.preparation_key != plan.preparation_key
        || receipt.payload_manifest != payload_manifest
        || receipt.base != *base.content_manifest()
        || receipt.alternates
            != alternates
                .iter()
                .map(|s| s.content_manifest().clone())
                .collect::<Vec<_>>()
        || receipt.recipe != config.preparation.recipe
        || receipt.adapter_image_digest.to_string() != image_digest
        || receipt.non_weight_mismatch != config.output.non_weight_mismatch
        || receipt.winners != *plan.winners()
        || receipt.shadowed.len() != plan.shadowed().len()
        || receipt
            .shadowed
            .iter()
            .zip(plan.shadowed())
            .any(|(actual, expected)| {
                actual.group != expected.group
                    || actual.previous_alternate != expected.previous_alternate
                    || actual.winning_alternate != expected.winning_alternate
            })
        || receipt.requires_model_qualification != plan.requires_model_qualification
        || receipt.qualification != "pending"
        || receipt.files.len() + 1 != output.files().len()
        || receipt.files.contains_key(EXPORT)
        || receipt
            .files
            .iter()
            .any(|(name, id)| output.files().get(name) != Some(id))
    {
        return Err(invalid(
            "export receipt provenance differs from resolved preparation",
        ));
    }
    Ok((payload_manifest, output.files()[INDEX].clone()))
}
