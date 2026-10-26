//! Narrow process boundary for an admitted offline CPU worker. The parent owns
//! cgroup/runtime/storage admission, input pins and all publication decisions.
use crate::{
    Digest, ExportLimits, FileIdentity, ModuleConfig, Profile, Result, Snapshot, StagedExport,
    export_staged, invalid,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    path::PathBuf,
};

/// Hard request ceiling before JSON allocation. Payloads never travel in this
/// protocol; a profile can describe large tensor inventories within this bound.
pub const MAX_REQUEST_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "layout", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Input {
    Indexed {
        root: PathBuf,
    },
    Sidecar {
        root: PathBuf,
        tensors: String,
        selection_manifest: Option<String>,
    },
}

impl Input {
    fn inspect(&self, limits: &ExportLimits) -> Result<Snapshot> {
        match self {
            Self::Indexed { root } => Snapshot::inspect(root, limits),
            Self::Sidecar {
                root,
                tensors,
                selection_manifest,
            } => Snapshot::inspect_sidecar(root, tensors, selection_manifest.as_deref(), limits),
        }
    }
}

/// Supplied by the operator/adapter, never inferred by the source checkpoint.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperandProfile {
    pub schema: String,
    pub groups: BTreeMap<String, Vec<String>>,
    pub static_activation_calibration: bool,
}

impl OperandProfile {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() as u64 > MAX_REQUEST_BYTES {
            return Err(crate::Error::Limit(
                "operand profile exceeds request ceiling".into(),
            ));
        }
        let profile: Self = crate::strict_json(bytes)?;
        profile.profile()?;
        Ok(profile)
    }

    pub fn profile(&self) -> Result<Profile> {
        if self.schema != "sparknest.operand-profile.v1" {
            return Err(invalid("unknown operand profile schema"));
        }
        let groups = self
            .groups
            .iter()
            .map(|(name, members)| {
                let unique: BTreeSet<_> = members.iter().cloned().collect();
                if unique.len() != members.len() {
                    return Err(invalid("duplicate operand profile member"));
                }
                Ok((name.clone(), unique))
            })
            .collect::<Result<_>>()?;
        Profile::new(groups, self.static_activation_calibration)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Request {
    Inspect {
        inputs: Vec<Input>,
        limits: ExportLimits,
    },
    Stage {
        module: Box<ModuleConfig>,
        profile: OperandProfile,
        base: Input,
        alternates: Vec<Input>,
        destination: PathBuf,
        limits: ExportLimits,
    },
    Qualify {
        module: Box<ModuleConfig>,
        profile: OperandProfile,
        base: Input,
        alternates: Vec<Input>,
        output: PathBuf,
        limits: ExportLimits,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InspectedInput {
    pub content_manifest: Digest,
    pub files: BTreeMap<String, FileIdentity>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Response {
    Inspect {
        inputs: Vec<InspectedInput>,
    },
    Stage {
        staged: Box<StagedExport>,
    },
    Qualify {
        receipt: Box<crate::QualificationReceipt>,
    },
}

fn inspect_inputs(
    module: &ModuleConfig,
    profile: OperandProfile,
    base: Input,
    alternates: Vec<Input>,
    limits: &ExportLimits,
    buffers: u64,
) -> Result<(Profile, Snapshot, Vec<Snapshot>)> {
    module.validate()?;
    limits.validate()?;
    if alternates.len() != module.alternates.len() {
        return Err(invalid("worker alternate count differs from module"));
    }
    let metadata = (alternates.len() as u64)
        .checked_add(2)
        .and_then(|n| n.checked_mul(limits.metadata_bytes))
        .and_then(|n| {
            (limits.buffer_bytes as u64)
                .checked_mul(buffers)
                .and_then(|b| n.checked_add(b))
        });
    if metadata.is_none_or(|n| n > module.preparation.max_ram_bytes) {
        return Err(crate::Error::Limit(
            "aggregate worker metadata exceeds RAM admission".into(),
        ));
    }
    let profile = profile.profile()?;
    let base = base.inspect(limits)?;
    let alternates = alternates
        .into_iter()
        .map(|input| input.inspect(limits))
        .collect::<Result<_>>()?;
    Ok((profile, base, alternates))
}

impl Request {
    /// Read one finite request, refusing duplicates/unknown fields and oversized
    /// requests before opening any checkpoint or creating staging.
    pub fn read(reader: impl Read) -> Result<Self> {
        let mut bytes = Vec::new();
        reader.take(MAX_REQUEST_BYTES + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_REQUEST_BYTES {
            return Err(crate::Error::Limit(
                "worker request exceeds byte ceiling".into(),
            ));
        }
        crate::strict_json(&bytes)
    }

    /// Synchronous and CPU-only: call in the dedicated worker process, never on
    /// a FUSE/management async executor. No networking or subprocess is started.
    pub fn run(self) -> Result<Response> {
        match self {
            Self::Qualify {
                module,
                profile,
                base,
                alternates,
                output,
                limits,
            } => {
                let (profile, base, alternates) =
                    inspect_inputs(&module, profile, base, alternates, &limits, 2)?;
                let output = Snapshot::inspect(&output, &limits)?;
                let references: Vec<_> = alternates.iter().collect();
                let receipt =
                    crate::qualify_exact(&module, &profile, &base, &references, &output, &limits)?;
                Ok(Response::Qualify {
                    receipt: Box::new(receipt),
                })
            }
            Self::Inspect { inputs, limits } => {
                limits.validate()?;
                if inputs.is_empty() {
                    return Err(invalid("worker inspection needs an input"));
                }
                let inputs = inputs
                    .into_iter()
                    .map(|input| {
                        let snapshot = input.inspect(&limits)?;
                        Ok(InspectedInput {
                            content_manifest: snapshot.content_manifest().clone(),
                            files: snapshot.files().clone(),
                        })
                    })
                    .collect::<Result<_>>()?;
                Ok(Response::Inspect { inputs })
            }
            Self::Stage {
                module,
                profile,
                base,
                alternates,
                destination,
                limits,
            } => {
                let (profile, base, alternates) =
                    inspect_inputs(&module, profile, base, alternates, &limits, 1)?;
                let references: Vec<_> = alternates.iter().collect();
                let staged =
                    export_staged(&module, &profile, &base, &references, &destination, &limits)?;
                Ok(Response::Stage {
                    staged: Box::new(staged),
                })
            }
        }
    }
}
