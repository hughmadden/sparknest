//! Explicit CPU preparation of ordinary immutable checkpoint namespaces.
//!
//! Input aliases are flattened into pinned regular-file links. Only changed
//! files enter native staging; imports adopt those bytes without copying a base
//! checkpoint to the coordinator. Export success is never qualification. An
//! explicit versioned bytewise CPU proof gates complete READY publication.
use crate::model_worker::{Identity, Limits, Pool, StopEvidence};
use nest_data::Vfs;
use nest_meta::{Command, ModelCommand, ModelReply, Reply, composition as catalog, query};
use nest_model::{
    Digest, ExportLimits, FileIdentity, ModuleConfig, StagedExport,
    worker::{Input, OperandProfile, Request, Response},
};
use nest_store::ObjectKey;
use nest_types::{
    FileAttr, FileId, FileKind, GenState, Generation, NestError, NestResult, Timestamp,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::sync::{Semaphore, watch};

const RECEIPT: &str = "composition.receipt.json";
const QUALIFICATION_REQUIRED: &str =
    "QUALIFICATION_REQUIRED: no trusted verifier is configured for this qualification policy";
type Inventory = Vec<(Vec<u8>, FileId, Generation)>;
type Running = HashMap<String, watch::Receiver<Option<String>>>;
fn operation_active(receiver: &watch::Receiver<Option<String>>) -> bool {
    receiver.borrow().is_none() && receiver.has_changed().is_ok()
}

struct StopGuard {
    key: String,
    stopping: Arc<Mutex<HashSet<String>>>,
}
impl Drop for StopGuard {
    fn drop(&mut self) {
        self.stopping.lock().remove(&self.key);
    }
}
fn check_cancel(cancel: &AtomicBool) -> NestResult<()> {
    if cancel.load(Ordering::Acquire) {
        Err(NestError::Busy("preparation cancellation requested".into()))
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModuleStatus {
    pub name: String,
    pub revision: u64,
    pub key: String,
    pub qualification_policy: String,
    pub phase: String,
    pub artifact: Option<catalog::ModelEntry>,
    pub output: Option<FileId>,
    pub message: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Resolved {
    pub name: String,
    pub key: String,
    pub epoch: u64,
    pub publication: String,
    pub qualification_policy: String,
    pub root: FileId,
    pub path: PathBuf,
    pub revision: String,
    pub files: Inventory,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResetReport {
    pub key: String,
    pub epoch: u64,
    pub dry_run: bool,
    pub when_idle: bool,
    pub blocked: bool,
    pub reset: Option<u64>,
    pub shared_modules: Vec<String>,
    pub leases: Vec<String>,
    /// True only when this call actually unlinked proven derived payloads.
    pub deletes_files: bool,
    pub eligible_shard_bytes: u64,
    /// Size of last namespace references actually removed; ordinary store
    /// effects reclaim replicas. Additional aliases can retain their objects.
    pub reclaimed_shard_bytes: u64,
    pub reclaim_blocked: Vec<String>,
    pub state: String,
    pub reset_pending: bool,
    pub retained_shard_bytes: u64,
}

pub struct Manager {
    vfs: Arc<Vfs>,
    mountpoint: Option<PathBuf>,
    pool: Option<Pool>,
    blocking: Arc<Semaphore>,
    operations: Arc<Semaphore>,
    running: Mutex<Running>,
    cancellation: Mutex<HashMap<String, Arc<AtomicBool>>>,
    stopping: Arc<Mutex<HashSet<String>>>,
}

fn io(e: impl std::fmt::Display) -> NestError {
    NestError::Io(e.to_string())
}
fn invalid(e: impl Into<String>) -> NestError {
    NestError::Invalid(e.into())
}
fn flat_name(name: &str) -> NestResult<()> {
    if name.is_empty() || matches!(name, "." | "..") || name.contains(['/', '\\', ':', '\0']) {
        return Err(invalid("snapshot requires confined flat filenames"));
    }
    Ok(())
}
fn namespace(path: &str) -> NestResult<()> {
    if !path.starts_with('/')
        || path == "/"
        || path
            .split('/')
            .skip(1)
            .any(|p| p.is_empty() || matches!(p, "." | "..") || p.contains(['\\', ':', '\0']))
    {
        return Err(invalid(
            "namespace must be an absolute confined non-root path",
        ));
    }
    Ok(())
}
fn ready(
    entry: &catalog::ModelEntry,
    publication: &catalog::ModelPublication,
    policy: &str,
    snapshots: &[catalog::ModelSnapshot],
) -> NestResult<()> {
    if entry.state != "ready" || entry.reset_pending {
        return Err(NestError::Busy(
            "artifact is not READY for new users".into(),
        ));
    }
    if entry.epoch != publication.epoch
        || publication.qualification_policy != policy
        || entry.publication.as_deref() != Some(publication.publication.as_str())
        || !snapshots
            .iter()
            .any(|s| s.output && s.root == publication.root)
    {
        return Err(invalid(
            "complete publication does not bind this epoch/output/qualification policy",
        ));
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    spec: ModuleConfig,
    profile: OperandProfile,
    expected_revision: u64,
}

enum DeclarationInput {
    Json(bytes::Bytes),
    Typed(Box<Declaration>),
}

struct CompiledDeclaration {
    command: ModelCommand,
    #[cfg(test)]
    processed_on: std::thread::ThreadId,
}

fn identities(spec: &ModuleConfig, profile: &OperandProfile) -> NestResult<(Digest, Digest)> {
    let profile = profile.profile().map_err(|e| invalid(e.to_string()))?;
    Ok((
        spec.preparation_key(&profile)
            .map_err(|e| invalid(e.to_string()))?,
        spec.qualification_key()
            .map_err(|e| invalid(e.to_string()))?,
    ))
}

// Called only inside the manager's bounded blocking worker. JSON decoding,
// profile validation, canonical hashing and serialization share this owner.
fn compile_declaration(name: String, input: DeclarationInput) -> NestResult<CompiledDeclaration> {
    let Declaration {
        mut spec,
        profile,
        expected_revision,
    } = match input {
        DeclarationInput::Json(body) => {
            if body.len() as u64 > nest_model::worker::MAX_REQUEST_BYTES {
                return Err(invalid("declaration body exceeds admitted byte ceiling"));
            }
            nest_model::strict_json(&body).map_err(|e| invalid(e.to_string()))?
        }
        DeclarationInput::Typed(declaration) => *declaration,
    };
    flat_name(&name)?;
    spec.name = name.clone();
    spec.validate().map_err(|e| invalid(e.to_string()))?;
    namespace(&spec.output.namespace)?;
    namespace(&spec.base.source)?;
    for alternate in &spec.alternates {
        namespace(&alternate.source)?;
    }
    for source in
        std::iter::once(&spec.base.source).chain(spec.alternates.iter().map(|a| &a.source))
    {
        if spec.output.namespace == *source
            || spec.output.namespace.starts_with(&format!("{source}/"))
        {
            return Err(invalid(
                "derived output namespace must be outside every input checkpoint tree",
            ));
        }
    }
    let (key, policy) = identities(&spec, &profile)?;
    Ok(CompiledDeclaration {
        command: ModelCommand::PutModule {
            name,
            spec: serde_json::to_string(&spec).map_err(io)?,
            profile: serde_json::to_string(&profile).map_err(io)?,
            key: key.to_string(),
            qualification_policy: policy.to_string(),
            expected_revision,
        },
        #[cfg(test)]
        processed_on: std::thread::current().id(),
    })
}

fn decode_module(
    module: catalog::ModelModule,
) -> NestResult<(catalog::ModelModule, ModuleConfig, OperandProfile, Digest)> {
    let catalog::ModelModule {
        name,
        revision,
        spec,
        profile,
        key,
        qualification_policy,
    } = module;
    let decoded_spec: ModuleConfig = serde_json::from_str(&spec).map_err(io)?;
    let decoded_profile: OperandProfile = serde_json::from_str(&profile).map_err(io)?;
    let (computed, policy) = identities(&decoded_spec, &decoded_profile)?;
    if computed.to_string() != key {
        return Err(invalid("stored module preparation identity differs"));
    }
    if policy.to_string() != qualification_policy {
        return Err(invalid("stored module qualification identity differs"));
    }
    // Release large serialized inputs here rather than returning or cloning
    // them onto the async executor alongside their decoded inventories.
    Ok((
        catalog::ModelModule {
            name,
            revision,
            spec: String::new(),
            profile: String::new(),
            key,
            qualification_policy,
        },
        decoded_spec,
        decoded_profile,
        computed,
    ))
}

fn try_operation(pool: &Arc<Semaphore>) -> NestResult<tokio::sync::OwnedSemaphorePermit> {
    pool.clone().try_acquire_owned().map_err(|_| {
        NestError::Busy(
            "assigned CPU preparation pool already has an active manager operation".into(),
        )
    })
}

async fn admitted_claim<T>(
    pool: &Arc<Semaphore>,
    claim: impl std::future::Future<Output = NestResult<T>>,
) -> NestResult<(tokio::sync::OwnedSemaphorePermit, T)> {
    // Polling the claim is forbidden until this nonblocking admission succeeds.
    let permit = try_operation(pool)?;
    Ok((permit, claim.await?))
}

fn require_owner_node(node: nest_types::NodeId, owner: Option<&str>) -> NestResult<()> {
    if owner.is_some_and(|owner| owner.starts_with(&format!("cpu-{}-", node.0))) {
        Ok(())
    } else {
        Err(NestError::NotPermitted(
            "preparation phase operations require the owning CPU node".into(),
        ))
    }
}

async fn bounded_block<T: Send + 'static>(
    pool: Arc<Semaphore>,
    f: impl FnOnce() -> NestResult<T> + Send + 'static,
) -> NestResult<T> {
    let permit = pool.acquire_owned().await.map_err(io)?;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        f()
    })
    .await
    .map_err(io)?
}

impl Manager {
    pub fn new(vfs: Arc<Vfs>, mountpoint: Option<PathBuf>, pool: Option<Pool>) -> Arc<Self> {
        Arc::new(Self {
            vfs,
            mountpoint,
            pool,
            blocking: Arc::new(Semaphore::new(1)),
            operations: Arc::new(Semaphore::new(1)),
            running: Mutex::new(HashMap::new()),
            cancellation: Mutex::new(HashMap::new()),
            stopping: Arc::new(Mutex::new(HashSet::new())),
        })
    }
    /// All manager-owned SQLite and native filesystem work leaves the API task.
    async fn block<T: Send + 'static>(
        &self,
        f: impl FnOnce() -> NestResult<T> + Send + 'static,
    ) -> NestResult<T> {
        bounded_block(self.blocking.clone(), f).await
    }

    async fn read<T: Send + 'static>(
        &self,
        f: impl FnOnce(&rusqlite::Connection) -> NestResult<T> + Send + 'static,
    ) -> NestResult<T> {
        let data = self.vfs.data().clone();
        self.block(move || data.with_reader(|c| Ok(f(c))).map_err(io)?)
            .await
    }
    async fn command(&self, command: ModelCommand) -> NestResult<Reply> {
        self.vfs
            .data()
            .meta()
            .propose(Command::Model(command))
            .await
    }
    async fn module(&self, name: &str) -> NestResult<catalog::ModelModule> {
        let name = name.to_string();
        self.read(move |c| {
            catalog::module(c, &name)
                .map_err(io)?
                .ok_or(NestError::NotFound)
        })
        .await
    }
    pub async fn put(
        &self,
        name: &str,
        spec: ModuleConfig,
        profile: OperandProfile,
        expected_revision: u64,
    ) -> NestResult<ModuleStatus> {
        self.put_input(
            name,
            DeclarationInput::Typed(Box::new(Declaration {
                spec,
                profile,
                expected_revision,
            })),
        )
        .await
    }
    pub async fn put_json(&self, name: &str, body: bytes::Bytes) -> NestResult<ModuleStatus> {
        self.put_input(name, DeclarationInput::Json(body)).await
    }
    async fn put_input(&self, name: &str, input: DeclarationInput) -> NestResult<ModuleStatus> {
        let owned_name = name.to_string();
        let declaration = self
            .block(move || compile_declaration(owned_name, input))
            .await?;
        self.command(declaration.command).await?;
        self.status(name).await
    }
    pub async fn list(&self) -> NestResult<Vec<ModuleStatus>> {
        let modules = self.read(|c| catalog::modules(c).map_err(io)).await?;
        let mut result = Vec::new();
        for module in modules {
            result.push(self.status(&module.name).await?);
        }
        Ok(result)
    }
    pub async fn status(&self, name: &str) -> NestResult<ModuleStatus> {
        let module = self.module(name).await?;
        let key = module.key.clone();
        let (artifact, snapshots, publication) = self
            .read(move |c| {
                Ok((
                    catalog::get(c, &key).map_err(io)?,
                    catalog::snapshots(c, &key).map_err(io)?,
                    catalog::complete_publication(c, &key).map_err(io)?,
                ))
            })
            .await?;
        let output = snapshots.iter().find(|s| s.output).map(|s| s.root);
        let mut phase = artifact
            .as_ref()
            .map_or("unresolved", |a| a.state.as_str())
            .to_string();
        let mut message = None;
        if let Some(entry) = &artifact {
            if entry.state == "preparing" && output.is_some() {
                let completed = self
                    .running
                    .lock()
                    .get(&module.key)
                    .is_some_and(|rx| rx.borrow().as_deref() == Some(""));
                phase = if completed {
                    "awaiting-qualification"
                } else {
                    "verification-required"
                }
                .into();
                message = Some("QUALIFICATION_REQUIRED: explicit policy-bound proof is required before publication".into());
            } else if entry.state == "ready"
                && publication.as_ref().is_none_or(|p| {
                    ready(entry, p, &module.qualification_policy, &snapshots).is_err()
                })
            {
                phase = "qualification-required".into();
                message = Some("complete matching qualification publication is required".into());
            }
        }
        if let Some(error) = self
            .running
            .lock()
            .get(&module.key)
            .and_then(|rx| rx.borrow().clone())
            .filter(|e| !e.is_empty())
        {
            message = Some(error);
        }
        if self
            .running
            .lock()
            .get(&module.key)
            .is_some_and(|rx| rx.borrow().is_none() && rx.has_changed().is_err())
        {
            message=Some("RECOVERY_REQUIRED: preparation task ended without a completion result; ownership is retained".into());
        }
        Ok(ModuleStatus {
            name: module.name,
            revision: module.revision,
            key: module.key,
            qualification_policy: module.qualification_policy,
            phase,
            artifact,
            output,
            message,
        })
    }
    pub async fn resolve(&self, name: &str) -> NestResult<Resolved> {
        let module = self.module(name).await?;
        let key = module.key.clone();
        let policy = module.qualification_policy.clone();
        let (entry, publication, output, path) = self
            .read(move |c| {
                let entry = catalog::get(c, &key)
                    .map_err(io)?
                    .ok_or(NestError::NotFound)?;
                let publication = catalog::complete_publication(c, &key)
                    .map_err(io)?
                    .ok_or_else(|| {
                        NestError::Busy("complete qualified publication is required".into())
                    })?;
                let snapshots = catalog::snapshots(c, &key).map_err(io)?;
                ready(&entry, &publication, &policy, &snapshots)?;
                let output = snapshots
                    .into_iter()
                    .find(|s| s.output && s.root == publication.root)
                    .ok_or(NestError::Stale)?;
                let path = query::path_of(c, output.root)
                    .map_err(io)?
                    .ok_or(NestError::NotFound)?;
                Ok((
                    entry,
                    publication,
                    output,
                    String::from_utf8(path).map_err(io)?,
                ))
            })
            .await?;
        let mount = self.mountpoint.as_ref().ok_or_else(|| {
            NestError::Unavailable("composition mountpoint is not configured".into())
        })?;
        Ok(Resolved {
            name: module.name,
            key: module.key.clone(),
            epoch: entry.epoch,
            publication: publication.publication,
            qualification_policy: module.qualification_policy,
            root: output.root,
            path: mount.join(path.trim_start_matches('/')),
            revision: module.key,
            files: output.files,
        })
    }
    /// Run the specifically versioned exact-byte proof. Unknown policies and
    /// calibration-dependent operands remain pending; no external passed flag
    /// can substitute for this worker's comparisons.
    pub async fn qualify(self: &Arc<Self>, name: &str) -> NestResult<ModuleStatus> {
        let module = self.module(name).await?;
        let (module, spec, profile, computed) = self
            .block(move || {
                let decoded = decode_module(module)?;
                if decoded.1.preparation.qualification_profile != nest_model::policy_identity()
                    || decoded.2.static_activation_calibration
                    || !matches!(
                        decoded.1.preparation.lossy_conversion,
                        nest_model::LossyConversion::Deny
                    )
                {
                    return Err(invalid(QUALIFICATION_REQUIRED));
                }
                Ok(decoded)
            })
            .await?;
        let mut receiver = {
            let stopping = self.stopping.lock();
            if stopping.contains(&module.key) {
                return Err(NestError::Busy("preparation stop is in progress".into()));
            }
            let mut running = self.running.lock();
            if running.get(&module.key).is_some_and(operation_active) {
                return Err(NestError::Busy(
                    "preparation or qualification is already active".into(),
                ));
            }
            let (sender, receiver) = watch::channel(None);
            running.insert(module.key.clone(), receiver.clone());
            let cancel = Arc::new(AtomicBool::new(false));
            self.cancellation
                .lock()
                .insert(module.key.clone(), cancel.clone());
            let manager = self.clone();
            tokio::spawn(async move {
                let result = manager
                    .qualify_inner(module, spec, profile, computed, cancel)
                    .await;
                sender.send_replace(Some(
                    result.err().map_or_else(String::new, |e| e.to_string()),
                ));
            });
            receiver
        };
        while receiver.borrow().is_none() {
            receiver.changed().await.map_err(io)?;
        }
        let failure = receiver.borrow().clone().unwrap_or_default();
        if !failure.is_empty() {
            return Err(invalid(failure));
        }
        self.status(name).await
    }
    pub async fn acquire(&self, name: &str, lease: &str) -> NestResult<Resolved> {
        let resolved = self.resolve(name).await?;
        self.command(ModelCommand::Use {
            key: resolved.key.clone(),
            epoch: resolved.epoch,
            lease: lease.into(),
        })
        .await?;
        Ok(resolved)
    }
    /// Caller must establish that every owned consumer/rank stopped first.
    pub async fn release(&self, key: &str, epoch: u64, lease: &str) -> NestResult<()> {
        self.command(ModelCommand::Release {
            key: key.into(),
            epoch,
            lease: lease.into(),
        })
        .await?;
        let (_, blocked) = self.reclaim_retired(key, epoch).await?;
        for reason in blocked {
            tracing::warn!(key, %reason, "derived shard reclamation blocked after lease release");
        }
        Ok(())
    }
    pub async fn reset(
        &self,
        name: &str,
        when_idle: bool,
        dry_run: bool,
    ) -> NestResult<ResetReport> {
        let module = self.module(name).await?;
        self.reset_key(&module.key, when_idle, dry_run).await
    }
    pub async fn cache_status(&self) -> NestResult<Vec<catalog::ModelEntry>> {
        self.read(|c| catalog::entries(c).map_err(io)).await
    }
    pub async fn reset_all(&self, when_idle: bool, dry_run: bool) -> NestResult<Vec<ResetReport>> {
        let entries = self.cache_status().await?;
        let modules = self.read(|c| catalog::modules(c).map_err(io)).await?;
        let mut reports = Vec::new();
        for entry in entries {
            match self.reset_key(&entry.key, when_idle, dry_run).await {
                Ok(report) => reports.push(report),
                Err(error) => reports.push(ResetReport {
                    key: entry.key.clone(),
                    epoch: entry.epoch,
                    dry_run,
                    when_idle,
                    blocked: true,
                    reset: entry.reset_id,
                    shared_modules: modules
                        .iter()
                        .filter(|m| m.key == entry.key)
                        .map(|m| m.name.clone())
                        .collect(),
                    leases: entry.leases,
                    deletes_files: false,
                    eligible_shard_bytes: 0,
                    reclaimed_shard_bytes: 0,
                    reclaim_blocked: vec![error.to_string()],
                    state: entry.state,
                    reset_pending: entry.reset_pending,
                    retained_shard_bytes: self
                        .retained_shard_bytes(&entry.key, entry.epoch)
                        .await?,
                }),
            }
        }
        Ok(reports)
    }
    async fn retained_shard_bytes(&self, key: &str, epoch: u64) -> NestResult<u64> {
        let key = key.to_string();
        self.read(move |c| {
            let mut seen = BTreeSet::new();
            let mut retained = 0u64;
            for proof in catalog::owned_outputs(c, &key)
                .map_err(io)?
                .into_iter()
                .filter(|p| p.epoch == epoch && p.name.ends_with(b".safetensors"))
            {
                if !seen.insert((proof.file, proof.generation)) {
                    continue;
                }
                if let Some(attr) = query::getattr(c, proof.file).map_err(io)?
                    && attr.generation == proof.generation
                    && attr.nlink != 0
                {
                    retained = retained.checked_add(attr.size).ok_or(NestError::NoSpace)?;
                }
            }
            Ok(retained)
        })
        .await
    }
    pub async fn reset_key(
        &self,
        key: &str,
        when_idle: bool,
        dry_run: bool,
    ) -> NestResult<ResetReport> {
        let lookup_key = key.to_string();
        let entry = self
            .read(move |c| {
                catalog::get(c, &lookup_key)
                    .map_err(io)?
                    .ok_or(NestError::NotFound)
            })
            .await?;
        let (eligible_shard_bytes, _, mut reclaim_blocked) =
            self.reclaim_plan(key, entry.epoch).await?;
        let key = key.to_string();
        let shared_key = key.clone();
        let shared_modules = self
            .read(move |c| {
                Ok(catalog::modules(c)
                    .map_err(io)?
                    .into_iter()
                    .filter(|m| m.key == shared_key)
                    .map(|m| m.name)
                    .collect())
            })
            .await?;
        let blocked = entry.state == "preparing" || !entry.leases.is_empty();
        let reset = if dry_run {
            entry.reset_id
        } else {
            match self
                .command(ModelCommand::Reset {
                    key: key.clone(),
                    epoch: entry.epoch,
                    when_idle,
                })
                .await?
            {
                Reply::Model(ModelReply::Reset(id)) => Some(id),
                _ => return Err(io("unexpected reset reply")),
            }
        };
        let reclaimed_shard_bytes = if dry_run {
            0
        } else {
            let (bytes, blocked) = self.reclaim_retired(&key, entry.epoch).await?;
            reclaim_blocked.extend(blocked);
            bytes
        };
        let lookup = key.clone();
        let current = self
            .read(move |c| {
                catalog::get(c, &lookup)
                    .map_err(io)?
                    .ok_or(NestError::NotFound)
            })
            .await?;
        let retained_shard_bytes = self.retained_shard_bytes(&key, entry.epoch).await?;
        Ok(ResetReport {
            key: key.clone(),
            epoch: entry.epoch,
            dry_run,
            when_idle,
            blocked: blocked || !reclaim_blocked.is_empty(),
            reset,
            shared_modules,
            leases: entry.leases,
            deletes_files: reclaimed_shard_bytes != 0,
            eligible_shard_bytes,
            reclaimed_shard_bytes,
            reclaim_blocked,
            state: current.state,
            reset_pending: current.reset_pending,
            retained_shard_bytes,
        })
    }
    async fn reclaim_plan(
        &self,
        key: &str,
        epoch: u64,
    ) -> NestResult<(u64, Vec<catalog::ModelOwnedOutput>, Vec<String>)> {
        let key = key.to_string();
        self.read(move |c| {
            let entry = catalog::get(c, &key)
                .map_err(io)?
                .ok_or(NestError::NotFound)?;
            if entry.epoch != epoch {
                return Err(NestError::Stale);
            }
            let idle = entry.owner.is_none() && entry.leases.is_empty();
            let entries = catalog::entries(c).map_err(io)?;
            let publication = catalog::complete_publication(c, &key).map_err(io)?;
            let mut eligible = 0u64;
            let mut candidates = Vec::new();
            let mut blocked = Vec::new();
            for proof in catalog::owned_outputs(c, &key)
                .map_err(io)?
                .into_iter()
                .filter(|p| p.epoch == epoch && p.name.ends_with(b".safetensors"))
            {
                let label = format!("file {} generation {}", proof.file.0, proof.generation.0);
                let Some(alias) = query::lookup(c, proof.parent, &proof.name).map_err(io)? else {
                    continue;
                };
                if alias != proof.file {
                    blocked.push(format!("{label}: owned alias changed"));
                    continue;
                }
                let Some(attr) = query::getattr(c, proof.file).map_err(io)? else {
                    blocked.push(format!("{label}: generation missing"));
                    continue;
                };
                if attr.generation != proof.generation
                    || !attr.sealed
                    || attr.gen_state != GenState::Stable
                {
                    blocked.push(format!("{label}: current generation changed"));
                    continue;
                }
                if publication.as_ref().is_some_and(|p| {
                    p.export_receipt == (proof.file, proof.generation)
                        || p.qualification_receipt == (proof.file, proof.generation)
                }) {
                    blocked.push(format!("{label}: retained receipt"));
                    continue;
                }
                let mut shared = false;
                for other in entries.iter().filter(|e| e.key != key) {
                    if other.files.contains(&(proof.file, proof.generation))
                        || catalog::snapshots(c, &other.key)
                            .map_err(io)?
                            .iter()
                            .any(|s| s.root == proof.parent)
                    {
                        shared = true;
                        break;
                    }
                }
                if shared {
                    blocked.push(format!(
                        "{label}: another artifact retains the generation or root"
                    ));
                }
                if !idle {
                    blocked.push(format!(
                        "{label}: preparation owner or consumer lease remains"
                    ));
                }
                if idle && !shared && attr.nlink == 1 {
                    eligible = eligible.checked_add(attr.size).ok_or(NestError::NoSpace)?;
                }
                if attr.nlink > 1 {
                    blocked.push(format!(
                        "{label}: other namespace aliases retain physical bytes"
                    ));
                }
                candidates.push(proof);
            }
            Ok((eligible, candidates, blocked))
        })
        .await
    }
    async fn reclaim_retired(&self, key: &str, epoch: u64) -> NestResult<(u64, Vec<String>)> {
        let lookup = key.to_string();
        let entry = self
            .read(move |c| {
                catalog::get(c, &lookup)
                    .map_err(io)?
                    .ok_or(NestError::NotFound)
            })
            .await?;
        if entry.epoch != epoch {
            return Err(NestError::Stale);
        }
        if !matches!(entry.state.as_str(), "failed" | "retired") {
            return Ok((0, Vec::new()));
        }
        let (_, candidates, mut blocked) = self.reclaim_plan(key, epoch).await?;
        let mut reclaimed = 0u64;
        for proof in candidates {
            // The replicated command rechecks current epoch, ownership, exact
            // alias/generation, receipt exclusions and ordinary shared pins.
            match self
                .command(ModelCommand::ReclaimOutputShard {
                    key: key.into(),
                    epoch,
                    parent: proof.parent,
                    name: proof.name,
                    file: proof.file,
                    generation: proof.generation,
                    now: Timestamp::now(),
                })
                .await
            {
                Ok(Reply::Attr(attr)) if attr.nlink == 1 => {
                    reclaimed = reclaimed.checked_add(attr.size).ok_or(NestError::NoSpace)?
                }
                Ok(_) => {}
                Err(error) => blocked.push(format!("file {}: {error}", proof.file.0)),
            }
        }
        Ok((reclaimed, blocked))
    }
    pub async fn cancel_reset(
        &self,
        name: &str,
        epoch: u64,
        reset: u64,
    ) -> NestResult<ModuleStatus> {
        let module = self.module(name).await?;
        self.command(ModelCommand::CancelReset {
            key: module.key,
            epoch,
            reset,
        })
        .await?;
        self.status(name).await
    }
    pub async fn prepare(self: &Arc<Self>, name: &str, wait: bool) -> NestResult<ModuleStatus> {
        let module = self.module(name).await?;
        let mut receiver = {
            let stopping = self.stopping.lock();
            if stopping.contains(&module.key) {
                return Err(NestError::Busy("preparation stop is in progress".into()));
            }
            let mut running = self.running.lock();
            if let Some(receiver) = running.get(&module.key).filter(|rx| operation_active(rx)) {
                receiver.clone()
            } else {
                let (sender, receiver) = watch::channel(None);
                running.insert(module.key.clone(), receiver.clone());
                let cancel = Arc::new(AtomicBool::new(false));
                self.cancellation
                    .lock()
                    .insert(module.key.clone(), cancel.clone());
                let manager = self.clone();
                let module_name = module.name.clone();
                tokio::spawn(async move {
                    let result = manager.prepare_inner(module, cancel).await;
                    sender.send_replace(Some(
                        result.err().map_or_else(String::new, |e| e.to_string()),
                    ));
                    tracing::debug!(module=%module_name,"composition preparation task finished");
                });
                receiver
            }
        };
        if wait {
            while receiver.borrow().is_none() {
                receiver.changed().await.map_err(io)?;
            }
            if let Some(failure) = receiver
                .borrow()
                .as_ref()
                .filter(|message| !message.is_empty())
            {
                return Err(io(failure));
            }
        }
        self.status(name).await
    }
    async fn worker(
        &self,
        identity: &Identity,
        request: Request,
        limits: Limits,
    ) -> NestResult<(Response, PathBuf)> {
        let pool = self.pool.as_ref().ok_or_else(|| {
            NestError::Unavailable("assigned CPU preparation pool is required".into())
        })?;
        match pool.run(identity.clone(), request, limits).await {
            Ok(done) => match done.response {
                Ok(response) => Ok((response, done.staging)),
                Err(reason) => {
                    let reason = self.stopped(&done.stopped, reason).await?;
                    Err(NestError::Io(reason))
                }
            },
            Err(failure) => {
                let reason = if let Some(evidence) = failure.stopped {
                    self.stopped(&evidence, failure.reason).await?
                } else {
                    failure.reason
                };
                Err(NestError::Io(reason))
            }
        }
    }
    async fn stopped(&self, evidence: &StopEvidence, reason: String) -> NestResult<String> {
        let identity = evidence.identity();
        let (reason, receipt) = self
            .block(move || {
                let receipt = Digest::of(reason.as_bytes()).to_string();
                Ok((reason, receipt))
            })
            .await?;
        self.command(ModelCommand::Stop {
            key: identity.key.to_string(),
            owner: identity.owner.clone(),
            epoch: identity.epoch,
            receipt,
        })
        .await?;
        Ok(reason)
    }
    /// Explicit reconciliation; never infer worker death from a timeout.
    pub async fn recover(&self, name: &str) -> NestResult<ModuleStatus> {
        self.stop(name).await
    }
    pub async fn stop(&self, name: &str) -> NestResult<ModuleStatus> {
        let module = self.module(name).await?;
        {
            let mut stopping = self.stopping.lock();
            if !stopping.insert(module.key.clone()) {
                return Err(NestError::Busy(
                    "preparation stop is already in progress".into(),
                ));
            }
        }
        let _guard = StopGuard {
            key: module.key.clone(),
            stopping: self.stopping.clone(),
        };
        let active = self
            .running
            .lock()
            .get(&module.key)
            .filter(|rx| operation_active(rx))
            .cloned();
        if let Some(mut receiver) = active {
            if let Some(cancel) = self.cancellation.lock().get(&module.key) {
                cancel.store(true, Ordering::Release);
            }
            // Never abort an import or detach an unbounded worker. Its enforced
            // runtime and actual removal complete before the safe boundary.
            while receiver.borrow().is_none() {
                receiver.changed().await.map_err(io)?;
            }
        }
        let status = self.status(name).await?;
        let entry = status.artifact.ok_or(NestError::NotFound)?;
        if entry.state != "preparing" {
            return self.status(name).await;
        }
        require_owner_node(self.vfs.data().id(), entry.owner.as_deref())?;
        let _operation = try_operation(&self.operations)?;
        let identity = Identity {
            key: self
                .block(move || {
                    serde_json::from_value(serde_json::Value::String(status.key)).map_err(io)
                })
                .await?,
            epoch: entry.epoch,
            owner: entry.owner.ok_or(NestError::Stale)?,
        };
        let pool = self.pool.as_ref().ok_or_else(|| {
            NestError::Unavailable("assigned CPU pool is required to reconcile worker stop".into())
        })?;
        let evidence = pool.reconcile(identity).await.map_err(|f| io(f.reason))?;
        self.stopped(
            &evidence,
            "explicit worker stop reconciled; namespace and receipt evidence retained".into(),
        )
        .await?;
        self.status(name).await
    }
    async fn ensure_dir(&self, path: &str) -> NestResult<FileId> {
        namespace(path)?;
        let mut parent = FileId::ROOT;
        for name in path.trim_start_matches('/').split('/') {
            parent = match self.vfs.lookup(parent, name.as_bytes()).await {
                Ok(attr) if attr.kind == FileKind::Directory => attr.id,
                Ok(_) => return Err(NestError::NotDir),
                Err(NestError::NotFound) => {
                    self.vfs.mkdir(parent, name.as_bytes(), 0o700).await?.id
                }
                Err(e) => return Err(e),
            };
        }
        Ok(parent)
    }
    async fn inventory(
        &self,
        source: &str,
        max_files: usize,
    ) -> NestResult<BTreeMap<String, FileAttr>> {
        let source = source.to_string();
        self.read(move |c| {
            let root = query::resolve_follow(c, source.as_bytes())
                .map_err(io)?
                .ok_or(NestError::NotFound)?;
            let attr = query::getattr(c, root)
                .map_err(io)?
                .ok_or(NestError::NotFound)?;
            if attr.kind != FileKind::Directory {
                return Err(NestError::NotDir);
            }
            let entries = query::readdir(c, root, 0, max_files.saturating_add(1)).map_err(io)?;
            if entries.is_empty() || entries.len() > max_files {
                return Err(invalid(
                    "input flat inventory is empty or exceeds admitted file limit",
                ));
            }
            let mut files = BTreeMap::new();
            for (_, entry) in entries {
                let name = String::from_utf8(entry.name).map_err(io)?;
                flat_name(&name)?;
                let path = format!("{source}/{name}");
                let id = query::resolve_follow(c, path.as_bytes())
                    .map_err(io)?
                    .ok_or(NestError::NotFound)?;
                let attr = query::getattr(c, id)
                    .map_err(io)?
                    .ok_or(NestError::NotFound)?;
                if attr.kind != FileKind::Regular
                    || !attr.sealed
                    || attr.gen_state != GenState::Stable
                {
                    return Err(invalid(
                        "input needs sealed stable regular-file generations",
                    ));
                }
                files.insert(name, attr);
            }
            Ok(files)
        })
        .await
    }
    async fn bind(
        &self,
        identity: &Identity,
        root: FileId,
        output: bool,
        files: Inventory,
    ) -> NestResult<()> {
        self.command(ModelCommand::BindSnapshot {
            key: identity.key.to_string(),
            owner: identity.owner.clone(),
            epoch: identity.epoch,
            root,
            output,
            files,
        })
        .await?;
        Ok(())
    }
    async fn flatten(
        &self,
        identity: &Identity,
        source: &str,
        destination: &str,
        max_files: usize,
    ) -> NestResult<(PathBuf, BTreeMap<String, FileAttr>)> {
        let files = self.inventory(source, max_files).await?;
        // Fence the actual resolved generations before making any worker-visible
        // alias. A source alias can subsequently move without changing inputs.
        self.command(ModelCommand::Pin {
            key: identity.key.to_string(),
            owner: identity.owner.clone(),
            epoch: identity.epoch,
            files: files.values().map(|a| (a.id, a.generation)).collect(),
        })
        .await?;
        let root = self.ensure_dir(destination).await?;
        for (name, attr) in &files {
            self.vfs.link(attr.id, root, name.as_bytes()).await?;
        }
        self.bind(
            identity,
            root,
            false,
            files
                .iter()
                .map(|(n, a)| (n.as_bytes().to_vec(), a.id, a.generation))
                .collect(),
        )
        .await?;
        let mount = self
            .mountpoint
            .as_ref()
            .ok_or_else(|| NestError::Unavailable("composition mountpoint is required".into()))?;
        Ok((mount.join(destination.trim_start_matches('/')), files))
    }
    async fn import_output(
        &self,
        identity: &Identity,
        parent: FileId,
        name: &str,
        source: PathBuf,
        size: u64,
    ) -> NestResult<(Vec<u8>, FileId, Generation)> {
        flat_name(name)?;
        let file = match self
            .vfs
            .data()
            .meta()
            .propose(Command::ReserveFileIds { count: 1 })
            .await?
        {
            Reply::FileIds(file) => file,
            _ => return Err(io("unexpected reserved file id reply")),
        };
        let store = self.vfs.data().store().clone();
        let adopted_source = source.clone();
        self.block(move || {
            let md = std::fs::symlink_metadata(&source).map_err(io)?;
            if !md.file_type().is_file() || md.len() != size {
                return Err(invalid(
                    "staged file differs from admitted regular-file size",
                ));
            }
            let key = ObjectKey::new(file, Generation(1));
            store.link_from(&source, key).map_err(io)?;
            std::fs::File::open(store.path(key))
                .and_then(|f| f.sync_all())
                .map_err(io)?;
            if let Some(parent) = store.path(key).parent() {
                std::fs::File::open(parent)
                    .and_then(|f| f.sync_all())
                    .map_err(io)?;
            }
            Ok(())
        })
        .await?;
        let now = Timestamp::now();
        // ImportOutput records a fresh derived generation atomically with the
        // ordinary import effects. Never claim ownership of a reused base id.
        self.command(ModelCommand::ImportOutput {
            key: identity.key.to_string(),
            owner: identity.owner.clone(),
            epoch: identity.epoch,
            parent,
            name: name.as_bytes().to_vec(),
            file,
            perm: 0o444,
            size,
            mtime: now,
            node: self.vfs.data().id(),
            now,
        })
        .await?;
        // Only this newly generated task-owned staging alias is removed. The
        // committed sealed namespace/object links preserve bytes and receipts.
        self.block(move || {
            std::fs::remove_file(&adopted_source).map_err(io)?;
            let parent = adopted_source
                .parent()
                .ok_or_else(|| invalid("staged source has no parent"))?;
            std::fs::File::open(parent)
                .and_then(|f| f.sync_all())
                .map_err(io)
        })
        .await?;
        Ok((name.as_bytes().to_vec(), file, Generation(1)))
    }
    async fn assemble(
        &self,
        identity: &Identity,
        staged: StagedExport,
        staging: PathBuf,
        base: &BTreeMap<String, FileAttr>,
        destination: &str,
        limits: &ExportLimits,
    ) -> NestResult<(FileId, BTreeMap<String, FileIdentity>)> {
        let key = identity.key.clone();
        let root_copy = staging.clone();
        let limits = limits.clone();
        let (expected, reused_files) = self
            .block(move || {
                if staged.receipt.preparation_key != key
                    || staged.receipt.schema != "sparknest.export-receipt.v1"
                {
                    return Err(invalid(
                        "worker receipt schema/preparation identity mismatch",
                    ));
                }
                let receipt_bytes = serde_json::to_vec(&staged.receipt).map_err(io)?;
                let staged_names = staged.staged_files();
                let mut expected = staged.receipt.files;
                expected.insert(
                    RECEIPT.into(),
                    FileIdentity {
                        bytes: receipt_bytes.len() as u64,
                        digest: Digest::of(&receipt_bytes),
                    },
                );
                if expected.len() > limits.max_files {
                    return Err(invalid("output exceeds admitted file inventory"));
                }
                let mut actual = BTreeSet::new();
                let mut bytes = 0u64;
                for entry in std::fs::read_dir(&root_copy).map_err(io)? {
                    let entry = entry.map_err(io)?;
                    let name = entry
                        .file_name()
                        .into_string()
                        .map_err(|_| invalid("non UTF8 staged file"))?;
                    flat_name(&name)?;
                    let md = std::fs::symlink_metadata(entry.path()).map_err(io)?;
                    if !md.file_type().is_file()
                        || expected.get(&name).is_none_or(|id| id.bytes != md.len())
                    {
                        return Err(invalid("unexpected staged file or size"));
                    }
                    bytes = bytes.checked_add(md.len()).ok_or(NestError::NoSpace)?;
                    if bytes > limits.max_output_bytes {
                        return Err(NestError::NoSpace);
                    }
                    actual.insert(name);
                }
                if actual != staged_names
                    || std::fs::read(root_copy.join(RECEIPT)).map_err(io)? != receipt_bytes
                {
                    return Err(invalid(
                        "staging inventory/receipt differs from worker response",
                    ));
                }
                Ok((expected, staged.receipt.reused_files))
            })
            .await?;
        let root = self.ensure_dir(destination).await?;
        let mut inventory = Vec::new();
        for (name, id) in &expected {
            flat_name(name)?;
            if reused_files.contains(name) {
                let original = base
                    .get(name)
                    .ok_or_else(|| invalid("worker reused a file absent from pinned base"))?;
                if original.size != id.bytes {
                    return Err(invalid("worker reused base size differs"));
                }
                self.vfs.link(original.id, root, name.as_bytes()).await?;
                inventory.push((name.as_bytes().to_vec(), original.id, original.generation));
            } else {
                inventory.push(
                    self.import_output(identity, root, name, staging.join(name), id.bytes)
                        .await?,
                );
            }
        }
        self.bind(identity, root, true, inventory).await?;
        self.block(move || {
            drop(reused_files);
            Ok(())
        })
        .await?;
        Ok((root, expected))
    }
    async fn read_sealed(
        &self,
        file: FileId,
        generation: Generation,
        limit: u64,
    ) -> NestResult<Vec<u8>> {
        let attr = self.vfs.getattr(file).await?;
        if attr.generation != generation
            || !attr.sealed
            || attr.gen_state != GenState::Stable
            || attr.size > limit
        {
            return Err(invalid(
                "evidence must be a bounded sealed current generation",
            ));
        }
        let (handle, _) = self.vfs.open(file, 0).await?;
        let result = async {
            let mut bytes = Vec::new();
            while (bytes.len() as u64) < attr.size {
                let part = self
                    .vfs
                    .read(
                        handle,
                        bytes.len() as u64,
                        (attr.size - bytes.len() as u64).min(1024 * 1024) as u32,
                    )
                    .await?;
                if part.is_empty() {
                    return Err(io("short evidence read"));
                }
                bytes.extend(part);
            }
            Ok(bytes)
        }
        .await;
        self.vfs.release(handle, None).await;
        result
    }
    async fn qualify_inner(
        &self,
        module: catalog::ModelModule,
        spec: ModuleConfig,
        profile: OperandProfile,
        computed: Digest,
        cancel: Arc<AtomicBool>,
    ) -> NestResult<()> {
        check_cancel(&cancel)?;
        let key = module.key.clone();
        let (entry, mut snapshots) = self
            .read(move |c| {
                Ok((
                    catalog::get(c, &key)
                        .map_err(io)?
                        .ok_or(NestError::NotFound)?,
                    catalog::snapshots(c, &key).map_err(io)?,
                ))
            })
            .await?;
        if entry.state == "ready" {
            self.resolve(&module.name).await?;
            return Ok(());
        }
        if entry.state != "preparing" || entry.reset_pending {
            return Err(NestError::Busy(
                "qualification requires an owned prepared output without a pending reset".into(),
            ));
        }
        require_owner_node(self.vfs.data().id(), entry.owner.as_deref())?;
        let _operation = try_operation(&self.operations)?;
        let identity = Identity {
            key: computed,
            epoch: entry.epoch,
            owner: entry.owner.ok_or(NestError::Stale)?,
        };
        let output_index = snapshots.iter().position(|s| s.output).ok_or_else(|| {
            NestError::Busy("prepare a complete output before qualification".into())
        })?;
        let output = snapshots.remove(output_index);
        let root = output.root;
        let output_path = self
            .read(move |c| {
                String::from_utf8(
                    query::path_of(c, root)
                        .map_err(io)?
                        .ok_or(NestError::NotFound)?,
                )
                .map_err(io)
            })
            .await?;
        let prefix = output_path
            .strip_suffix("/snapshot")
            .ok_or_else(|| invalid("output does not have a managed immutable snapshot path"))?
            .to_string();
        let mount = self
            .mountpoint
            .as_ref()
            .ok_or_else(|| NestError::Unavailable("composition mountpoint is required".into()))?;
        let pool = self.pool.as_ref().ok_or_else(|| {
            NestError::Unavailable("assigned CPU preparation pool is required".into())
        })?;
        if pool.name != spec.preparation.worker_pool {
            return Err(invalid("module requires a different assigned CPU pool"));
        }
        let mut inputs = Vec::new();
        for path in std::iter::once(format!("{prefix}/inputs/base"))
            .chain((0..spec.alternates.len()).map(|i| format!("{prefix}/inputs/alternate-{i}")))
        {
            check_cancel(&cancel)?;
            let path_copy = path.clone();
            let snapshot_root = self
                .read(move |c| {
                    query::resolve(c, path_copy.as_bytes())
                        .map_err(io)?
                        .ok_or(NestError::NotFound)
                })
                .await?;
            let pinned = snapshots
                .iter()
                .find(|s| !s.output && s.root == snapshot_root)
                .ok_or_else(|| invalid("qualification input is not a bound immutable root"))?;
            let files = self
                .inventory(&path, ExportLimits::default().max_files)
                .await?;
            let inventory: Inventory = files
                .iter()
                .map(|(n, a)| (n.as_bytes().to_vec(), a.id, a.generation))
                .collect();
            if inventory != pinned.files {
                return Err(invalid("qualification input differs from bound inventory"));
            }
            let selection = if inputs.is_empty() {
                None
            } else {
                spec.alternates[inputs.len() - 1].selection_manifest.clone()
            };
            inputs.push(input(
                mount.join(path.trim_start_matches('/')),
                &files,
                selection,
            )?);
        }
        let base = inputs.remove(0);
        let limits = ExportLimits {
            max_output_bytes: spec.cache.max_new_physical_bytes,
            ..ExportLimits::default()
        };
        let (export_file, export_generation) = output
            .files
            .iter()
            .find(|(n, _, _)| n == RECEIPT.as_bytes())
            .map(|(_, f, g)| (*f, *g))
            .ok_or_else(|| invalid("output has no bound export receipt"))?;
        let export_bytes = self
            .read_sealed(export_file, export_generation, limits.metadata_bytes)
            .await?;
        let admission = Limits {
            cpu_threads: spec.preparation.cpu_threads,
            ram_bytes: spec.preparation.max_ram_bytes,
            new_bytes: spec.cache.max_new_physical_bytes.min(limits.metadata_bytes),
            runtime_seconds: spec.preparation.max_runtime_seconds,
        };
        let output_path = mount.join(output_path.trim_start_matches('/'));
        let request_limits = limits.clone();
        let (request, expected_base, expected_alternates, max_receipt_bytes) = self
            .block(move || {
                let expected_base = spec.base.content_manifest.clone();
                let expected_alternates = spec
                    .alternates
                    .iter()
                    .map(|a| a.content_manifest.clone())
                    .collect::<Vec<_>>();
                let max_receipt_bytes = request_limits
                    .metadata_bytes
                    .min(spec.cache.max_new_physical_bytes);
                Ok((
                    Request::Qualify {
                        module: Box::new(spec),
                        profile,
                        base,
                        alternates: inputs,
                        output: output_path,
                        limits: request_limits,
                    },
                    expected_base,
                    expected_alternates,
                    max_receipt_bytes,
                ))
            })
            .await?;
        check_cancel(&cancel)?;
        let (response, staging) = self.worker(&identity, request, admission).await?;
        check_cancel(&cancel)?;
        let Response::Qualify { receipt } = response else {
            return Err(invalid(
                "worker did not return exact qualification evidence",
            ));
        };
        let expected_key = identity.key.clone();
        let expected_policy = module.qualification_policy.clone();
        let (bytes, publication) = self
            .block(move || {
                let exported: nest_model::ExportReceipt =
                    serde_json::from_slice(&export_bytes).map_err(io)?;
                let finished_index = exported
                    .files
                    .get("model.safetensors.index.json")
                    .ok_or_else(|| invalid("export receipt has no finished index"))?;
                if receipt.schema != "sparknest.exact-qualification-receipt.v1"
                    || receipt.policy_name != nest_model::EXACT_POLICY_NAME
                    || receipt.policy != nest_model::policy_identity()
                    || receipt.qualification_policy.to_string() != expected_policy
                    || receipt.preparation_key != expected_key
                    || exported.preparation_key != expected_key
                    || receipt.base_manifest != expected_base
                    || receipt.alternate_manifests != expected_alternates
                    || receipt.output_payload_manifest != exported.payload_manifest
                    || receipt.finished_index != *finished_index
                    || receipt.publication_identity
                        != nest_model::publication_identity(
                            &expected_key,
                            &exported.payload_manifest,
                            finished_index,
                        )
                        .map_err(io)?
                {
                    return Err(invalid(
                        "qualification evidence does not bind the prepared output and policy",
                    ));
                }
                let bytes = serde_json::to_vec(&receipt).map_err(io)?;
                if bytes.len() as u64 > max_receipt_bytes {
                    return Err(NestError::NoSpace);
                }
                Ok((bytes, receipt.publication_identity.to_string()))
            })
            .await?;
        let size = bytes.len() as u64;
        let native = staging
            .parent()
            .ok_or_else(|| invalid("worker staging has no private parent"))?
            .join("qualification.json");
        let native_copy = native.clone();
        self.block(move || {
            use std::{io::Write, os::unix::fs::OpenOptionsExt};
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&native_copy)
                .map_err(io)?;
            file.write_all(&bytes)
                .and_then(|_| file.sync_all())
                .map_err(io)?;
            std::fs::File::open(native_copy.parent().unwrap())
                .and_then(|f| f.sync_all())
                .map_err(io)?;
            Ok(())
        })
        .await?;
        let evidence_root = self.ensure_dir(&format!("{prefix}/evidence")).await?;
        let name = format!("qualification-{:016x}.json", rand::random::<u64>());
        let (_, qualification_file, qualification_generation) = self
            .import_output(&identity, evidence_root, &name, native, size)
            .await?;
        self.command(ModelCommand::Pin {
            key: module.key.clone(),
            owner: identity.owner.clone(),
            epoch: identity.epoch,
            files: vec![(qualification_file, qualification_generation)],
        })
        .await?;
        check_cancel(&cancel)?;
        self.command(ModelCommand::PublishComplete {
            key: module.key,
            owner: identity.owner,
            epoch: identity.epoch,
            root: output.root,
            files: output.files,
            publication,
            export_receipt: (export_file, export_generation),
            qualification_receipt: (qualification_file, qualification_generation),
            qualification_policy: module.qualification_policy,
        })
        .await?;
        Ok(())
    }
    async fn prepare_inner(
        &self,
        module: catalog::ModelModule,
        cancel: Arc<AtomicBool>,
    ) -> NestResult<()> {
        check_cancel(&cancel)?;
        let (module, spec, profile, computed) = self.block(move || decode_module(module)).await?;
        let key = module.key.clone();
        let existing = self
            .read(move |c| catalog::get(c, &key).map_err(io))
            .await?;
        if let Some(entry) = existing {
            if entry.state == "ready" {
                self.resolve(&module.name).await?;
                return Ok(());
            }
            if entry.state == "preparing" {
                return Err(NestError::Busy("preparation owner already exists; inspect pending qualification or explicitly reconcile stop".into()));
            }
        }
        let pool = self.pool.as_ref().ok_or_else(|| {
            NestError::Unavailable("assigned CPU preparation pool is required".into())
        })?;
        if pool.name != spec.preparation.worker_pool {
            return Err(invalid("module requires a different assigned CPU pool"));
        }
        if self.mountpoint.is_none() {
            return Err(NestError::Unavailable(
                "composition mountpoint is required".into(),
            ));
        }
        let owner = format!(
            "cpu-{}-{:016x}",
            self.vfs.data().id().0,
            rand::random::<u64>()
        );
        let (_operation, reply) = admitted_claim(
            &self.operations,
            self.command(ModelCommand::Claim {
                key: module.key.clone(),
                owner: owner.clone(),
            }),
        )
        .await?;
        let epoch = match reply {
            Reply::Model(ModelReply::Epoch(epoch)) => epoch,
            _ => return Err(io("unexpected claim reply")),
        };
        let identity = Identity {
            key: computed,
            epoch,
            owner,
        };
        // The durable claim remains a blocker on any uncertain failure. Only
        // runner-generated stop evidence can release it after a launch.
        let admission = Limits {
            cpu_threads: spec.preparation.cpu_threads,
            ram_bytes: spec.preparation.max_ram_bytes,
            new_bytes: spec.cache.max_new_physical_bytes,
            runtime_seconds: spec.preparation.max_runtime_seconds,
        };
        let limits = ExportLimits {
            max_output_bytes: spec.cache.max_new_physical_bytes,
            ..ExportLimits::default()
        };
        let preparation = async {
            check_cancel(&cancel)?;
            let prefix = format!("{}/{}/{}", spec.output.namespace, &module.key[7..], epoch);
            let (base_root, base_files) = self
                .flatten(
                    &identity,
                    &spec.base.source,
                    &format!("{prefix}/inputs/base"),
                    limits.max_files,
                )
                .await?;
            let mut alternates = Vec::new();
            for (index, alternate) in spec.alternates.iter().enumerate() {
                check_cancel(&cancel)?;
                let (root, files) = self
                    .flatten(
                        &identity,
                        &alternate.source,
                        &format!("{prefix}/inputs/alternate-{index}"),
                        limits.max_files,
                    )
                    .await?;
                alternates.push(input(root, &files, alternate.selection_manifest.clone())?);
            }
            let request = Request::Stage {
                module: Box::new(spec),
                profile,
                base: Input::Indexed { root: base_root },
                alternates,
                destination: PathBuf::new(),
                limits: limits.clone(),
            };
            check_cancel(&cancel)?;
            Ok::<_, NestError>((request, base_files, prefix))
        }
        .await;
        let (request, base_files, prefix) = match preparation {
            Ok(preparation) => preparation,
            Err(error) => {
                // This task has not called the runner: there is no process to
                // reconcile, and input/output namespace evidence is retained.
                let (error, receipt) = self
                    .block(move || {
                        let receipt =
                            Digest::of(format!("no launch issued: {error}").as_bytes()).to_string();
                        Ok((error, receipt))
                    })
                    .await?;
                self.command(ModelCommand::Stop {
                    key: identity.key.to_string(),
                    owner: identity.owner.clone(),
                    epoch,
                    receipt,
                })
                .await?;
                return Err(error);
            }
        };
        let (response, staging) = self.worker(&identity, request, admission.clone()).await?;
        check_cancel(&cancel)?;
        let Response::Stage { staged } = response else {
            return Err(invalid("worker did not return staged export"));
        };
        let destination = format!("{prefix}/snapshot");
        let (_root, expected) = self
            .assemble(
                &identity,
                *staged,
                staging,
                &base_files,
                &destination,
                &limits,
            )
            .await?;
        // Inspect the complete ordinary namespace, including reused distributed
        // files. This reads through the mount without making native base copies.
        let root = self
            .mountpoint
            .as_ref()
            .unwrap()
            .join(destination.trim_start_matches('/'));
        let request = Request::Inspect {
            inputs: vec![Input::Indexed { root }],
            limits,
        };
        check_cancel(&cancel)?;
        let inspect_admission = Limits {
            new_bytes: admission
                .new_bytes
                .min(ExportLimits::default().metadata_bytes),
            ..admission
        };
        let (response, _) = self.worker(&identity, request, inspect_admission).await?;
        check_cancel(&cancel)?;
        let Response::Inspect { inputs } = response else {
            return Err(invalid("worker did not verify complete output"));
        };
        self.block(move || {
            if inputs.len() != 1 || inputs[0].files != expected {
                return Err(invalid(
                    "complete output identities differ from export receipt",
                ));
            }
            Ok(())
        })
        .await?;
        // Owner, bound roots and generation pins intentionally survive here.
        // Qualification runs only through the explicit policy-bound qualify operation.
        Ok(())
    }
}

fn input(
    root: PathBuf,
    files: &BTreeMap<String, FileAttr>,
    selection_manifest: Option<String>,
) -> NestResult<Input> {
    if files.contains_key("model.safetensors.index.json") {
        return Ok(Input::Indexed { root });
    }
    let tensors: Vec<_> = files
        .keys()
        .filter(|n| n.ends_with(".safetensors"))
        .collect();
    if tensors.len() != 1 {
        return Err(invalid(
            "replacement source needs one raw safetensors file or an indexed snapshot",
        ));
    }
    Ok(Input::Sidecar {
        root,
        tensors: tensors[0].clone(),
        selection_manifest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test(flavor = "current_thread")]
    async fn a_busy_cpu_operation_refuses_another_key_without_polling_its_claim() {
        async fn claim(c: &mut rusqlite::Connection, key: &str) -> NestResult<Reply> {
            let tx = c.transaction().map_err(io)?;
            let (reply, _) = nest_meta::apply(
                &tx,
                &Command::Model(ModelCommand::Claim {
                    key: key.into(),
                    owner: "cpu-1-fixture".into(),
                }),
            )
            .map_err(io)?;
            tx.commit().map_err(io)?;
            reply
        }
        let operations = Arc::new(Semaphore::new(1));
        let mut c = nest_meta::open_memory().unwrap();
        let (first, _) = admitted_claim(&operations, claim(&mut c, "first-key"))
            .await
            .unwrap();
        let second = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            admitted_claim(&operations, claim(&mut c, "second-key")),
        )
        .await
        .expect("busy admission must not queue behind an active owner");
        assert!(matches!(second, Err(NestError::Busy(_))));
        assert!(catalog::get(&c, "first-key").unwrap().is_some());
        assert!(
            catalog::get(&c, "second-key").unwrap().is_none(),
            "busy admission must leave no durable owner or pins for another key"
        );
        drop(first);
        let (_second, _) = admitted_claim(&operations, claim(&mut c, "second-key"))
            .await
            .unwrap();
        assert!(catalog::get(&c, "second-key").unwrap().is_some());
    }
    #[test]
    fn preparation_phase_owner_is_fenced_to_its_cpu_node() {
        assert!(require_owner_node(nest_types::NodeId(1), Some("cpu-1-fixture")).is_ok());
        for owner in [
            None,
            Some("cpu-2-fixture"),
            Some("cpu-11-fixture"),
            Some("unscoped-owner"),
        ] {
            assert!(matches!(
                require_owner_node(nest_types::NodeId(1), owner),
                Err(NestError::NotPermitted(_))
            ));
        }
    }
    #[tokio::test(flavor = "current_thread")]
    async fn declaration_decode_profile_validation_and_hashing_leave_the_caller_executor() {
        let caller = std::thread::current().id();
        let digest = format!("sha256:{}", "a".repeat(64));
        let body=serde_json::to_vec(&serde_json::json!({
            "spec":{
                "schema":"sparknest.model-composition.v1","name":"generic",
                "base":{"source":"/generic-input","content_manifest":digest},"alternates":[],
                "output":{"loader_profile":"base-compatible-indexed-v1","precision":"match-base","configuration":"base-unchanged","tokenizer":"base-unchanged","non_weight_mismatch":"reject","namespace":"/generic-output"},
                "preparation":{"trigger":"explicit-prepare-or-external-use-wrapper","adapter_image":format!("generic-worker@{digest}"),"recipe":digest,"parameters":{},"lossy_conversion":"deny","qualification_profile":digest,"activation_calibration":"preserve-and-qualify","max_runtime_seconds":10,"max_ram_bytes":268435456,"cpu_threads":1,"device":"cpu","worker_pool":"cpu"},
                "cache":{"publication":"sealed-regular-files","reuse_unchanged":true,"prefer_reflink":false,"max_new_physical_bytes":1048576,"account_inflight":true,"eviction":"least-recently-used-unpinned","retain_dependencies":true}
            },
            "profile":{"schema":"sparknest.operand-profile.v1","groups":{"operand":(0..8192).map(|i|format!("tensor-{i}")).collect::<Vec<_>>()},"static_activation_calibration":false},
            "expected_revision":0
        })).unwrap();
        let pool = Arc::new(Semaphore::new(1));
        let raw = bounded_block(pool.clone(), move || {
            compile_declaration("raw".into(), DeclarationInput::Json(body.into()))
        })
        .await
        .unwrap();
        assert_ne!(
            raw.processed_on, caller,
            "raw declaration decode/validation/key/serialization ran on caller"
        );
        let ModelCommand::PutModule {
            name,
            spec,
            profile,
            key,
            qualification_policy,
            ..
        } = raw.command
        else {
            panic!("declaration command");
        };
        let expected = key.clone();
        let stored = catalog::ModelModule {
            name,
            revision: 1,
            spec,
            profile,
            key,
            qualification_policy,
        };
        let (decoded, decoded_on) = bounded_block(pool.clone(), move || {
            Ok((decode_module(stored)?, std::thread::current().id()))
        })
        .await
        .unwrap();
        assert_ne!(
            decoded_on, caller,
            "stored declaration decode/hash ran on caller"
        );
        assert!(
            decoded.0.spec.is_empty() && decoded.0.profile.is_empty(),
            "large serialized metadata must not return to caller"
        );
        let (_, spec, profile, _) = decoded;
        assert_eq!(profile.groups["operand"].len(), 8192);
        let typed = bounded_block(pool, move || {
            compile_declaration(
                "typed".into(),
                DeclarationInput::Typed(Box::new(Declaration {
                    spec,
                    profile,
                    expected_revision: 0,
                })),
            )
        })
        .await
        .unwrap();
        assert_ne!(
            typed.processed_on, caller,
            "typed declaration validation/key/serialization ran on caller"
        );
        let ModelCommand::PutModule { key, .. } = typed.command else {
            panic!("typed command");
        };
        assert_eq!(
            key, expected,
            "raw and typed paths must share declaration identity"
        );
    }
    #[test]
    fn cancelled_actor_channel_is_recoverable_without_inventing_worker_stop() {
        let (sender, receiver) = watch::channel(None);
        assert!(operation_active(&receiver));
        drop(sender);
        assert!(!operation_active(&receiver));
        assert!(
            receiver.borrow().is_none(),
            "actor death does not synthesize successful completion"
        );
    }
    #[test]
    fn flat_namespace_rejects_escape_and_ambiguous_paths() {
        for name in ["", ".", "..", "a/b", "a\\b", "x:y", "x\0y"] {
            assert!(flat_name(name).is_err(), "{name:?}");
        }
        for path in ["/", "relative", "/a/../b", "/a//b", "/a/", "/a\\b"] {
            assert!(namespace(path).is_err(), "{path:?}");
        }
        assert!(namespace("/model-cache/generic").is_ok());
    }
    #[test]
    fn file_existence_or_legacy_ready_cannot_substitute_for_qualified_publication() {
        let entry = catalog::ModelEntry {
            key: "key".into(),
            epoch: 2,
            owner: None,
            state: "ready".into(),
            publication: Some("output".into()),
            reset_pending: false,
            reset_id: None,
            files: vec![],
            leases: vec![],
        };
        let mut publication = catalog::ModelPublication {
            epoch: 1,
            root: FileId(9),
            publication: "output".into(),
            export_receipt: (FileId(10), Generation(1)),
            qualification_receipt: (FileId(11), Generation(1)),
            qualification_policy: "policy".into(),
        };
        let snapshots = vec![catalog::ModelSnapshot {
            root: FileId(9),
            output: true,
            files: vec![],
        }];
        assert!(ready(&entry, &publication, "policy", &snapshots).is_err());
        publication.epoch = 2;
        assert!(ready(&entry, &publication, "different-policy", &snapshots).is_err());
        assert!(ready(&entry, &publication, "policy", &[]).is_err());
        assert!(ready(&entry, &publication, "policy", &snapshots).is_ok());
        let mut pending = entry.clone();
        pending.reset_pending = true;
        assert!(ready(&pending, &publication, "policy", &snapshots).is_err());
        pending = entry;
        pending.state = "preparing".into();
        assert!(ready(&pending, &publication, "policy", &snapshots).is_err());
    }
}
