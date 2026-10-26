use crate::{
    ActivationCalibration, Digest, Error, ExportLimits, ModuleConfig, NonWeightMismatch, Reference,
    Result, Selection, Snapshot, SourceKind, invalid,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Trusted, versioned loader contract supplied by the owner, never inferred from
/// an alternate. Coupled/tied/fused operands and all quantization companions must
/// appear in the same group. Every base tensor must belong to exactly one group.
#[derive(Clone, Debug, Serialize)]
pub struct Profile {
    pub(crate) loader_profile: &'static str,
    pub(crate) groups: BTreeMap<String, BTreeSet<String>>,
    pub(crate) static_activation_calibration: bool,
}

impl Profile {
    pub fn new(
        groups: BTreeMap<String, BTreeSet<String>>,
        static_activation_calibration: bool,
    ) -> Result<Self> {
        let mut seen = BTreeSet::new();
        if groups.is_empty() {
            return Err(invalid("profile has no groups"));
        }
        for (group, tensors) in &groups {
            if group.is_empty() || tensors.is_empty() {
                return Err(invalid("empty operand group"));
            }
            for tensor in tensors {
                if tensor.is_empty() || !seen.insert(tensor) {
                    return Err(invalid("empty tensor or overlapping operand groups"));
                }
            }
        }
        Ok(Self {
            loader_profile: crate::LOADER_PROFILE,
            groups,
            static_activation_calibration,
        })
    }

    pub fn groups(&self) -> &BTreeMap<String, BTreeSet<String>> {
        &self.groups
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Shadowed {
    pub group: String,
    pub previous_alternate: usize,
    pub winning_alternate: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Plan {
    pub preparation_key: Digest,
    pub(crate) winners: BTreeMap<String, Option<usize>>,
    pub(crate) shadowed: Vec<Shadowed>,
    /// This is a requirement for the owner's later gate, never a qualification pass.
    pub requires_model_qualification: bool,
}

impl Plan {
    pub fn winners(&self) -> &BTreeMap<String, Option<usize>> {
        &self.winners
    }
    pub fn shadowed(&self) -> &[Shadowed] {
        &self.shadowed
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SelectionManifest {
    schema: String,
    groups: Vec<String>,
}

pub fn resolve(
    config: &ModuleConfig,
    profile: &Profile,
    base: &Snapshot,
    alternates: &[&Snapshot],
    limits: &ExportLimits,
) -> Result<Plan> {
    config.validate()?;
    limits.validate()?;
    if config.base.content_manifest != *base.content_manifest()
        || config.alternates.len() != alternates.len()
    {
        return Err(invalid("base identity or alternate input count mismatch"));
    }
    let inventory: BTreeSet<_> = profile
        .groups
        .values()
        .flatten()
        .map(String::as_str)
        .collect();
    if inventory != base.tensors.keys().map(String::as_str).collect() {
        return Err(invalid(
            "profile does not partition the complete base inventory",
        ));
    }
    if config.preparation.activation_calibration == ActivationCalibration::Recalibrate {
        return Err(Error::Unsupported("activation recalibration".into()));
    }
    if !config.preparation.parameters.is_empty() {
        return Err(Error::Unsupported(
            "exact adapter has no recipe parameters".into(),
        ));
    }
    let mut winners: BTreeMap<_, _> = profile.groups.keys().map(|g| (g.clone(), None)).collect();
    let mut shadowed = Vec::new();
    for (i, (declaration, input)) in config.alternates.iter().zip(alternates).enumerate() {
        if declaration.kind == SourceKind::Lora {
            return Err(Error::Unsupported("LoRA merge".into()));
        }
        if declaration.content_manifest != *input.content_manifest() {
            return Err(invalid("alternate content identity mismatch"));
        }
        let reference_matches = match &declaration.requires {
            Reference::BaseManifest(d) => *d == config.base.content_manifest,
            Reference::LowerCompositionKey(d) => *d == config.prefix_key(profile, i)?,
        };
        if !reference_matches {
            return Err(invalid("required reference mismatch"));
        }
        if input
            .tensors
            .keys()
            .any(|name| !base.tensors.contains_key(name))
        {
            return Err(invalid("unknown alternate tensor"));
        }
        if declaration.kind == SourceKind::Checkpoint
            && input.tensors.keys().ne(base.tensors.keys())
        {
            return Err(invalid(
                "full alternate checkpoint has incomplete inventory",
            ));
        }
        // Architecture/encoding semantics remain fixed even when the operator
        // explicitly chooses the base tokenizer or generation defaults.
        if input.files().get("config.json") != base.files().get("config.json") {
            return Err(invalid(
                "alternate architecture/configuration differs from base",
            ));
        }
        if config.output.non_weight_mismatch == NonWeightMismatch::Reject
            && input.ancillary() != base.ancillary()
        {
            return Err(invalid("alternate non-weight semantics differ from base"));
        }
        let selected: BTreeSet<String> = match declaration.selection {
            Selection::All => profile.groups.keys().cloned().collect(),
            Selection::ProducerManifest => {
                let name = declaration
                    .selection_manifest
                    .as_ref()
                    .ok_or_else(|| invalid("missing selection manifest"))?;
                if input.files().get(name).map(|id| &id.digest)
                    != declaration.selection_manifest_digest.as_ref()
                {
                    return Err(invalid("selected producer manifest identity mismatch"));
                }
                let manifest: SelectionManifest = input.read_json(name, limits)?;
                if manifest.schema != "sparknest.model-selection.v1" {
                    return Err(invalid("unknown selection schema"));
                }
                let count = manifest.groups.len();
                let groups: BTreeSet<_> = manifest.groups.into_iter().collect();
                if groups.len() != count || groups.is_empty() {
                    return Err(invalid("empty or duplicate group selection"));
                }
                groups
            }
        };
        // Validate even shadowed selections. An invalid whole-group declaration
        // cannot be repaired by borrowing an earlier or later companion.
        for group in selected {
            let members = profile
                .groups
                .get(&group)
                .ok_or_else(|| invalid("unknown selected group"))?;
            for member in members {
                let source = input
                    .tensors
                    .get(member)
                    .ok_or_else(|| invalid("selected group missing companion"))?;
                let target = &base.tensors[member];
                if source.spec.shape != target.spec.shape {
                    return Err(invalid("selected operand shape mismatch"));
                }
                if source.spec.dtype != target.spec.dtype {
                    return Err(Error::Unsupported(
                        "dtype/precision conversion; exact bytes required".into(),
                    ));
                }
                // The same tensor encodings can still have incompatible scale or
                // packing contracts. Require matching per-shard format metadata.
                if input.shards[&source.shard].metadata != base.shards[&target.shard].metadata {
                    return Err(Error::Unsupported(
                        "safetensors representation metadata mismatch".into(),
                    ));
                }
            }
            if let Some(previous_alternate) = winners.insert(group.clone(), Some(i)).flatten() {
                shadowed.push(Shadowed {
                    group,
                    previous_alternate,
                    winning_alternate: i,
                });
            }
        }
    }
    Ok(Plan {
        preparation_key: config.preparation_key(profile)?,
        winners,
        shadowed,
        requires_model_qualification: profile.static_activation_calibration
            || !alternates.is_empty(),
    })
}
