use crate::{Digest, Profile, Result, invalid};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SCHEMA: &str = "sparknest.model-composition.v1";
pub const LOADER_PROFILE: &str = "base-compatible-indexed-v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModuleConfig {
    pub schema: String,
    pub name: String,
    pub base: Source,
    #[serde(default)]
    pub alternates: Vec<Alternate>,
    pub output: Output,
    pub preparation: Preparation,
    pub cache: CachePolicy,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    /// Acquisition hint only; the manager resolves and pins the content identity.
    pub source: String,
    pub content_manifest: Digest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Alternate {
    pub name: String,
    pub kind: SourceKind,
    pub source: String,
    pub content_manifest: Digest,
    pub selection: Selection,
    pub selection_manifest: Option<String>,
    /// Selects the exact producer manifest within the pinned source, without
    /// making its acquisition filename part of preparation identity.
    pub selection_manifest_digest: Option<Digest>,
    pub requires: Reference,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum SourceKind {
    Replacement,
    Checkpoint,
    Lora,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Selection {
    All,
    ProducerManifest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reference {
    BaseManifest(Digest),
    LowerCompositionKey(Digest),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Output {
    pub loader_profile: String,
    pub precision: String,
    pub configuration: String,
    pub tokenizer: String,
    pub non_weight_mismatch: NonWeightMismatch,
    pub namespace: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum NonWeightMismatch {
    Reject,
    UseBase,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Preparation {
    pub trigger: String,
    pub adapter_image: String,
    pub recipe: Digest,
    /// Recipe parameters use exact strings, never ambiguous floating numbers.
    #[serde(default)]
    pub parameters: BTreeMap<String, String>,
    pub lossy_conversion: LossyConversion,
    pub qualification_profile: Digest,
    pub activation_calibration: ActivationCalibration,
    pub max_runtime_seconds: u64,
    pub max_ram_bytes: u64,
    pub cpu_threads: u32,
    pub device: String,
    pub worker_pool: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LossyConversion {
    Deny,
    RequireQualifiedProfile,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum ActivationCalibration {
    PreserveAndQualify,
    Recalibrate,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CachePolicy {
    pub publication: String,
    pub reuse_unchanged: bool,
    pub prefer_reflink: bool,
    pub max_new_physical_bytes: u64,
    pub account_inflight: bool,
    pub eviction: String,
    pub retain_dependencies: bool,
}

impl ModuleConfig {
    pub fn parse(text: &str) -> Result<Self> {
        let config: Self = toml::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    /// Revalidate public DTOs, including after a caller changes a parsed config.
    pub fn validate(&self) -> Result<()> {
        if self.schema != SCHEMA {
            return Err(invalid("unknown module schema"));
        }
        if self.name.is_empty() || self.base.source.is_empty() || self.output.namespace.is_empty() {
            return Err(invalid("empty module name, source or namespace"));
        }
        if self.output.loader_profile != LOADER_PROFILE
            || self.output.precision != "match-base"
            || self.output.configuration != "base-unchanged"
            || self.output.tokenizer != "base-unchanged"
        {
            return Err(invalid("unknown or unsupported output contract"));
        }
        let p = &self.preparation;
        if p.trigger != "explicit-prepare-or-external-use-wrapper"
            || p.device != "cpu"
            || p.worker_pool.is_empty()
            || p.max_runtime_seconds == 0
            || p.max_ram_bytes == 0
            || p.cpu_threads == 0
        {
            return Err(invalid("invalid CPU preparation contract or budget"));
        }
        let (_, digest) = p
            .adapter_image
            .rsplit_once('@')
            .ok_or_else(|| invalid("adapter image must be digest-pinned"))?;
        let _: Digest = serde_json::from_value(serde_json::Value::String(digest.into()))?;
        if p.adapter_image.starts_with('@')
            || p.parameters.iter().any(|(k, v)| {
                k.is_empty()
                    || v.is_empty()
                    || matches!(
                        v.to_ascii_lowercase().as_str(),
                        "nan" | "inf" | "+inf" | "-inf" | "infinity" | "+infinity" | "-infinity"
                    )
            })
        {
            return Err(invalid("invalid image or recipe parameter"));
        }
        let c = &self.cache;
        if c.publication != "sealed-regular-files"
            || c.eviction != "least-recently-used-unpinned"
            || !c.account_inflight
            || !c.retain_dependencies
            || c.max_new_physical_bytes == 0
        {
            return Err(invalid("unsupported cache contract"));
        }
        for a in &self.alternates {
            if a.name.is_empty() || a.source.is_empty() {
                return Err(invalid("empty alternate name/source"));
            }
            match (
                a.selection,
                &a.selection_manifest,
                &a.selection_manifest_digest,
            ) {
                (Selection::All, None, None) => {}
                (Selection::ProducerManifest, Some(path), Some(_)) => {
                    crate::indexed::safe_name(path)?
                }
                _ => return Err(invalid("selection and selection_manifest disagree")),
            }
        }
        Ok(())
    }

    pub fn preparation_key(&self, profile: &Profile) -> Result<Digest> {
        self.prefix_key(profile, self.alternates.len())
    }

    /// Identity of the lower prefix before alternate `count`. Requires references
    /// bind this identity, without materializing or hashing a prepared prefix.
    pub fn prefix_key(&self, profile: &Profile, count: usize) -> Result<Digest> {
        self.validate()?;
        if count > self.alternates.len() {
            return Err(invalid("prefix out of range"));
        }
        #[derive(Serialize)]
        struct KeyAlternate<'a> {
            kind: SourceKind,
            content: &'a Digest,
            selection: Selection,
            selection_manifest_digest: &'a Option<Digest>,
            requires: &'a Reference,
        }
        #[derive(Serialize)]
        struct Key<'a> {
            schema: &'static str,
            adapter: &'static str,
            base: &'a Digest,
            alternates: Vec<KeyAlternate<'a>>,
            profile: &'a Profile,
            image: &'a str,
            recipe: &'a Digest,
            parameters: &'a BTreeMap<String, String>,
            activation_calibration: ActivationCalibration,
            non_weight_mismatch: NonWeightMismatch,
        }
        // Paths (including selection manifest filename), labels, scheduling and
        // qualification-only policy cannot affect bytes. The input manifest binds
        // selection file bytes as well as tensors; the resolved profile binds groups.
        Digest::canonical(&Key {
            schema: "sparknest.preparation-key.v1",
            adapter: "exact-indexed-cpu.v1",
            base: &self.base.content_manifest,
            alternates: self.alternates[..count]
                .iter()
                .map(|a| KeyAlternate {
                    kind: a.kind,
                    content: &a.content_manifest,
                    selection: a.selection,
                    selection_manifest_digest: &a.selection_manifest_digest,
                    requires: &a.requires,
                })
                .collect(),
            profile,
            image: self
                .preparation
                .adapter_image
                .rsplit_once('@')
                .ok_or_else(|| invalid("unpinned adapter image"))?
                .1,
            recipe: &self.preparation.recipe,
            parameters: &self.preparation.parameters,
            activation_calibration: self.preparation.activation_calibration,
            non_weight_mismatch: self.output.non_weight_mismatch,
        })
    }

    pub fn qualification_key(&self) -> Result<Digest> {
        self.validate()?;
        Digest::canonical(&(
            "sparknest.qualification-policy.v1",
            &self.preparation.qualification_profile,
            self.preparation.lossy_conversion,
        ))
    }
}
