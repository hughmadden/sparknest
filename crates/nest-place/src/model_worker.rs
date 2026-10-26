//! Parent-side CPU execution only. Artifact ownership/pins and publication stay
//! with the manager. A launch record survives cancellation and daemon death;
//! uncertainty never becomes stop evidence or frees a preparation owner.
use nest_model::{
    Digest,
    worker::{Input, MAX_REQUEST_BYTES, Request, Response},
};
use nest_types::{NestError, NestResult};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    process::Command,
};

/// An explicitly assigned, single-operation CPU pool. Concurrent calls refuse
/// rather than reserve a serving GPU or wait while holding another resource.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pool {
    pub name: String,
    pub root: PathBuf,
    pub image: Digest,
    pub max_cpu_threads: u32,
    pub max_ram_bytes: u64,
    pub max_new_bytes: u64,
    pub free_reserve_bytes: u64,
    pub max_runtime_seconds: u64,
    pub control_timeout_seconds: u64,
    pub max_pids: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub key: Digest,
    pub epoch: u64,
    pub owner: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub cpu_threads: u32,
    pub ram_bytes: u64,
    pub new_bytes: u64,
    pub runtime_seconds: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    identity: Identity,
    image: Digest,
    container: String,
    id: Option<String>,
    limits: Limits,
    stopped: bool,
}

/// Constructible only after this runner confirms removal of the exact owned
/// container, or establishes that no launch was issued. It is not model/rank
/// shutdown evidence and cannot authorize releasing an external usage lease.
#[derive(Debug)]
pub struct StopEvidence {
    identity: Identity,
    record: Option<PathBuf>,
}

impl StopEvidence {
    pub fn identity(&self) -> &Identity {
        &self.identity
    }
    /// No record means admission failed before any launch was issued.
    pub fn record(&self) -> Option<&Path> {
        self.record.as_deref()
    }
}

#[derive(Debug)]
pub struct Completion {
    pub response: Result<Response, String>,
    pub staging: PathBuf,
    pub stopped: StopEvidence,
}

#[derive(Debug)]
pub struct Failure {
    pub reason: String,
    /// None means the manager MUST retain preparation ownership and pins.
    pub stopped: Option<StopEvidence>,
}

fn io(error: impl std::fmt::Display) -> NestError {
    NestError::Io(error.to_string())
}

async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> NestResult<T> + Send + 'static,
) -> NestResult<T> {
    tokio::task::spawn_blocking(f).await.map_err(io)?
}

/// The pool lock bounds these large request operations to one active worker.
/// Move ownership across the blocking boundary instead of cloning a potentially
/// large operand profile on a control-plane executor thread.
async fn request_work<T: Send + 'static>(
    request: &mut Request,
    work: impl FnOnce(&mut Request) -> NestResult<T> + Send + 'static,
) -> NestResult<T> {
    let mut owned = std::mem::replace(
        request,
        Request::Inspect {
            inputs: Vec::new(),
            limits: Default::default(),
        },
    );
    let (result, original) = blocking(move || {
        let result = work(&mut owned);
        Ok((result, owned))
    })
    .await?;
    *request = original;
    result
}

fn container_name(identity: &Identity) -> String {
    format!(
        "sn-model-{}-{}-{}",
        &identity.key.to_string()[7..],
        identity.epoch,
        &Digest::of(identity.owner.as_bytes()).to_string()[7..]
    )
}

fn write_record(root: &Path, record: &Record) -> NestResult<()> {
    // Under the pool lock, a crash-left temporary is never authoritative.
    let temporary = root.join("control.new");
    if temporary.exists() {
        fs::remove_file(&temporary).map_err(io)?;
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temporary)
        .map_err(io)?;
    serde_json::to_writer(&mut file, record).map_err(io)?;
    file.flush().map_err(io)?;
    file.sync_all().map_err(io)?;
    fs::rename(&temporary, root.join("control.json")).map_err(io)?;
    File::open(root).and_then(|f| f.sync_all()).map_err(io)
}

fn directory(path: &Path) -> NestResult<PathBuf> {
    if !fs::symlink_metadata(path).map_err(io)?.is_dir() {
        return Err(NestError::Invalid(
            "worker root must be a non-symlink directory".into(),
        ));
    }
    let path = path.canonicalize().map_err(io)?;
    if path
        .as_os_str()
        .as_encoded_bytes()
        .iter()
        .any(|b| matches!(b, b',' | b':' | b'\n' | b'\r'))
    {
        return Err(NestError::Invalid(
            "worker bind path contains a reserved separator".into(),
        ));
    }
    Ok(path)
}

fn limits(pool: &Pool, requested: &Limits, request: &Request) -> NestResult<()> {
    if pool.name.is_empty()
        || !pool.root.is_absolute()
        || pool.control_timeout_seconds == 0
        || pool.max_pids == 0
        || requested.cpu_threads == 0
        || requested.cpu_threads > pool.max_cpu_threads
        || requested.ram_bytes == 0
        || requested.ram_bytes > pool.max_ram_bytes
        || requested.new_bytes == 0
        || requested.new_bytes > pool.max_new_bytes
        || requested.runtime_seconds == 0
        || requested.runtime_seconds > pool.max_runtime_seconds
    {
        return Err(NestError::Invalid(
            "worker exceeds assigned CPU pool admission".into(),
        ));
    }
    let export = match request {
        Request::Inspect { limits, .. }
        | Request::Stage { limits, .. }
        | Request::Qualify { limits, .. } => limits,
    };
    export.validate().map_err(io)?;
    if let Request::Stage {
        module,
        limits: export,
        ..
    }
    | Request::Qualify {
        module,
        limits: export,
        ..
    } = request
    {
        module.validate().map_err(io)?;
        if module.preparation.worker_pool != pool.name
            || module
                .preparation
                .adapter_image
                .rsplit_once('@')
                .map(|(_, image)| image)
                != Some(pool.image.to_string().as_str())
            || module.preparation.cpu_threads != requested.cpu_threads
            || module.preparation.max_ram_bytes != requested.ram_bytes
            || module.preparation.max_runtime_seconds != requested.runtime_seconds
            || (matches!(request, Request::Stage { .. })
                && (export.max_output_bytes > requested.new_bytes
                    || module.cache.max_new_physical_bytes > requested.new_bytes))
        {
            return Err(NestError::Invalid(
                "worker request differs from admitted module/image budgets".into(),
            ));
        }
    }
    Ok(())
}

async fn capped(reader: impl AsyncRead + Unpin, limit: u64) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    reader
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .await
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > limit {
        return Err("worker control output exceeded its byte ceiling".into());
    }
    Ok(bytes)
}

struct Output {
    success: bool,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

/// Never invoke a shell. CLI death is distinct from the container's death.
async fn command(
    program: &Path,
    args: &[String],
    input: &[u8],
    timeout: Duration,
    stdout_limit: u64,
) -> Result<Output, String> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| e.to_string())?;
    let mut stdin = child.stdin.take().ok_or("missing command input")?;
    let stdout = child.stdout.take().ok_or("missing command output")?;
    let stderr = child.stderr.take().ok_or("missing command diagnostics")?;
    let execution = async {
        let send = async move {
            stdin.write_all(input).await.map_err(|e| e.to_string())?;
            stdin.shutdown().await.map_err(|e| e.to_string())
        };
        let wait = async { child.wait().await.map_err(|e| e.to_string()) };
        let (_, status, stdout, stderr) = tokio::try_join!(
            send,
            wait,
            capped(stdout, stdout_limit),
            capped(stderr, 65536)
        )?;
        Ok(Output {
            success: status.success(),
            stdout,
            stderr,
        })
    };
    tokio::time::timeout(timeout, execution)
        .await
        .map_err(|_| "worker command deadline exceeded".to_string())?
}

fn failure(reason: impl std::fmt::Display) -> Failure {
    Failure {
        reason: reason.to_string(),
        stopped: None,
    }
}

fn valid_id(id: &str) -> bool {
    id.len() == 64
        && id
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

/// The root is assigned exclusively to this pool. Never follow record symlinks.
fn records(root: &Path) -> NestResult<Vec<(PathBuf, Record)>> {
    let mut records = Vec::new();
    for entry in fs::read_dir(root).map_err(io)? {
        let entry = entry.map_err(io)?;
        if entry.file_name() == "pool.lock" {
            continue;
        }
        if !entry.file_type().map_err(io)?.is_dir() {
            return Err(NestError::Busy("unknown worker pool record".into()));
        }
        let path = entry.path().join("control.json");
        if !fs::symlink_metadata(&path).map_err(io)?.is_file() {
            return Err(NestError::Busy("invalid worker record".into()));
        }
        let file = File::open(&path).map_err(io)?;
        if file.metadata().map_err(io)?.len() > 65536 {
            return Err(NestError::Invalid("oversized worker record".into()));
        }
        let record: Record = serde_json::from_reader(file).map_err(io)?;
        let prefix = format!("{}-", container_name(&record.identity));
        if !record.container.starts_with(&prefix)
            || entry.file_name().to_str() != Some(record.container.as_str())
            || record.id.as_deref().is_some_and(|id| !valid_id(id))
        {
            return Err(NestError::Invalid("worker record identity mismatch".into()));
        }
        records.push((entry.path(), record));
    }
    Ok(records)
}

struct PoolLock(File);

impl Drop for PoolLock {
    fn drop(&mut self) {
        // Explicit unlock also releases the shared open-file-description lock
        // while a concurrently forked child has not yet closed its CLOEXEC fd.
        let _ = self.0.unlock();
    }
}

fn pool_lock(root: &Path) -> NestResult<PoolLock> {
    directory(root)?;
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(nix::libc::O_NOFOLLOW)
        .open(root.join("pool.lock"))
        .map_err(io)?;
    lock.try_lock()
        .map_err(|_| NestError::Busy("CPU worker pool is in use".into()))?;
    Ok(PoolLock(lock))
}

async fn save(job: &Path, record: &Record) -> Result<(), Failure> {
    let job = job.to_owned();
    let bytes = serde_json::to_vec(record).map_err(failure)?;
    blocking(move || write_record(&job, &serde_json::from_slice(&bytes).map_err(io)?))
        .await
        .map_err(failure)
}

async fn inspect(
    docker: &Path,
    target: &str,
    timeout: Duration,
) -> Result<Option<serde_json::Value>, String> {
    let output = command(
        docker,
        &[
            "container".into(),
            "inspect".into(),
            "--format".into(),
            "{{json .}}".into(),
            target.into(),
        ],
        &[],
        timeout,
        1024 * 1024,
    )
    .await?;
    if output.success {
        return serde_json::from_slice(&output.stdout)
            .map(Some)
            .map_err(|e| e.to_string());
    }
    // A transport/permission/daemon failure never proves absence.
    let diagnostic = String::from_utf8_lossy(&output.stderr);
    if [
        format!("Error: No such container: {target}"),
        format!("Error response from daemon: No such container: {target}"),
    ]
    .iter()
    .any(|absent| diagnostic.trim() == absent)
    {
        Ok(None)
    } else {
        Err("worker inspection is uncertain; retain ownership".into())
    }
}

fn owned(value: &serde_json::Value, record: &Record) -> bool {
    let labels = &value["Config"]["Labels"];
    let id = value["Id"].as_str().unwrap_or("");
    valid_id(id)
        && record.id.as_deref().is_none_or(|expected| id == expected)
        && value["Name"].as_str() == Some(format!("/{}", record.container).as_str())
        && value["Image"].as_str() == Some(record.image.to_string().as_str())
        && labels["sparknest.model-owner"].as_str() == Some(record.identity.owner.as_str())
        && labels["sparknest.model-key"].as_str() == Some(record.identity.key.to_string().as_str())
        && labels["sparknest.model-epoch"].as_str()
            == Some(record.identity.epoch.to_string().as_str())
}

fn empty(value: &serde_json::Value) -> bool {
    value.is_null()
        || value.as_array().is_some_and(Vec::is_empty)
        || value.as_object().is_some_and(serde_json::Map::is_empty)
}

/// Compare daemon readback, not just CLI arguments, before authorizing start.
fn admitted(
    value: &serde_json::Value,
    record: &Record,
    pool: &Pool,
    mounts: &[(PathBuf, String, bool)],
) -> bool {
    let host = &value["HostConfig"];
    let config = &value["Config"];
    let mount_values = value["Mounts"].as_array();
    owned(value, record)
        && value["State"]["Status"] == "created"
        && value["State"]["Running"] == false
        && host["Runtime"] == "runc"
        && host["NetworkMode"] == "none"
        && host["ReadonlyRootfs"] == true
        && host["Privileged"] == false
        && host["Memory"].as_u64() == Some(record.limits.ram_bytes)
        && host["MemorySwap"].as_u64() == Some(record.limits.ram_bytes)
        && host["NanoCpus"].as_u64() == Some(u64::from(record.limits.cpu_threads) * 1_000_000_000)
        && host["PidsLimit"].as_u64() == Some(u64::from(pool.max_pids))
        && host["CapDrop"] == serde_json::json!(["ALL"])
        && empty(&host["CapAdd"])
        && host["SecurityOpt"] == serde_json::json!(["no-new-privileges"])
        && host["LogConfig"]["Type"] == "none"
        && host["RestartPolicy"]["Name"] == "no"
        && empty(&host["DeviceRequests"])
        && empty(&host["Devices"])
        && empty(&host["DeviceCgroupRules"])
        && empty(&host["Tmpfs"])
        && empty(&host["Binds"])
        && empty(&host["VolumesFrom"])
        && empty(&host["PortBindings"])
        && host["PublishAllPorts"] == false
        && host["PidMode"] == ""
        && host["IpcMode"] == "private"
        && empty(&config["Volumes"])
        && config["User"] == format!("{}:{}", nix::unistd::geteuid(), nix::unistd::getegid())
        && config["Entrypoint"] == serde_json::json!(["/usr/bin/timeout"])
        && config["Cmd"]
            == serde_json::json!([
                "--signal=KILL",
                record.limits.runtime_seconds.to_string(),
                "nest-model-worker"
            ])
        && mount_values.is_some_and(|actual| {
            actual.len() == mounts.len()
                && mounts.iter().all(|(source, target, readonly)| {
                    actual.iter().any(|m| {
                        m["Type"] == "bind"
                            && m["Source"].as_str() == source.to_str()
                            && m["Destination"] == target.as_str()
                            && m["RW"] == !readonly
                            && m["Propagation"] == "rprivate"
                    })
                })
        })
}

async fn remove_owned(
    docker: &Path,
    timeout: Duration,
    job: &Path,
    record: &mut Record,
) -> Result<StopEvidence, Failure> {
    let target = record.id.as_deref().unwrap_or(&record.container);
    match inspect(docker, target, timeout).await.map_err(failure)? {
        Some(value) => {
            if !owned(&value, record) {
                return Err(failure(
                    "container ownership mismatch; no container touched",
                ));
            }
            record.id = value["Id"].as_str().map(str::to_owned);
            save(job, record).await?;
            let id = record
                .id
                .as_ref()
                .ok_or_else(|| failure("missing exact container id"))?;
            let removed = command(
                docker,
                &["rm".into(), "--force".into(), id.clone()],
                &[],
                timeout,
                4096,
            )
            .await
            .map_err(failure)?;
            if !removed.success || String::from_utf8_lossy(&removed.stdout).trim() != id {
                return Err(failure(
                    "worker container removal is unconfirmed; retain ownership",
                ));
            }
            if inspect(docker, id, timeout)
                .await
                .map_err(failure)?
                .is_some()
            {
                return Err(failure("removed worker is still present; retain ownership"));
            }
        }
        None if record.id.is_some() => {}
        // A create RPC may still be in flight. Name absence cannot prove it will
        // never create a container. Retry reconciliation when the exact ID exists.
        None => {
            return Err(failure(
                "create outcome remains uncertain; retain ownership",
            ));
        }
    }
    record.stopped = true;
    save(job, record).await?;
    Ok(StopEvidence {
        identity: record.identity.clone(),
        record: Some(job.join("control.json")),
    })
}

/// Built-in exact-copy adapter only: its pinned exporter admits aggregate new
/// regular-file bytes before writing. This is not a kernel quota for arbitrary
/// converter images. All writable output is private native disk, never inputs.
fn output_bytes(
    root: &Path,
    max_files: usize,
    logical_cap: u64,
    allocated_cap: u64,
) -> NestResult<()> {
    let mut todo = vec![root.to_owned()];
    let (mut count, mut logical, mut allocated) = (0usize, 0u64, 0u64);
    while let Some(path) = todo.pop() {
        let metadata = fs::symlink_metadata(&path).map_err(io)?;
        count = count.checked_add(1).ok_or(NestError::NoSpace)?;
        if count > max_files {
            return Err(NestError::NoSpace);
        }
        allocated = allocated
            .checked_add(
                metadata
                    .blocks()
                    .checked_mul(512)
                    .ok_or(NestError::NoSpace)?,
            )
            .ok_or(NestError::NoSpace)?;
        if metadata.is_dir() {
            for entry in fs::read_dir(path).map_err(io)? {
                todo.push(entry.map_err(io)?.path());
            }
        } else if metadata.is_file() && metadata.nlink() == 1 {
            logical = logical
                .checked_add(metadata.len())
                .ok_or(NestError::NoSpace)?;
        } else {
            return Err(NestError::Invalid(
                "worker output contains a link or special file".into(),
            ));
        }
        if logical > logical_cap || allocated > allocated_cap {
            return Err(NestError::NoSpace);
        }
    }
    Ok(())
}

impl Pool {
    /// The caller pins all inputs and owns an exact preparation epoch. Dropping
    /// this future leaves an unfinished record; reconcile before releasing pins.
    /// Sequential operations on the same identity retain separate stop records.
    pub async fn run(
        &self,
        identity: Identity,
        mut request: Request,
        admission: Limits,
    ) -> Result<Completion, Failure> {
        self.run_with(Path::new("docker"), identity, &mut request, admission)
            .await
    }

    /// Explicit recovery for this identity only. A pool lock or Docker failure,
    /// unknown create ID, or label mismatch leaves ownership retained. A missed
    /// heartbeat and elapsed runtime are never stop evidence.
    pub async fn reconcile(&self, identity: Identity) -> Result<StopEvidence, Failure> {
        self.reconcile_with(Path::new("docker"), identity).await
    }

    async fn reconcile_with(
        &self,
        docker: &Path,
        identity: Identity,
    ) -> Result<StopEvidence, Failure> {
        let root = self.root.clone();
        let (lock, entries) = blocking(move || Ok((pool_lock(&root)?, records(&root)?)))
            .await
            .map_err(failure)?;
        let mut evidence = None;
        for (job, mut record) in entries {
            if record.identity != identity {
                continue;
            }
            evidence = Some(if record.stopped {
                StopEvidence {
                    identity: identity.clone(),
                    record: Some(job.join("control.json")),
                }
            } else {
                remove_owned(
                    docker,
                    Duration::from_secs(self.control_timeout_seconds),
                    &job,
                    &mut record,
                )
                .await?
            });
        }
        drop(lock);
        evidence.ok_or_else(|| failure("no durable record for exact worker identity"))
    }

    async fn run_with(
        &self,
        docker: &Path,
        identity: Identity,
        request: &mut Request,
        admission: Limits,
    ) -> Result<Completion, Failure> {
        // Serialize before admission too: an unfinished earlier operation with
        // this identity must not acquire false no-launch evidence from a retry.
        let root = self.root.clone();
        let lock = blocking(move || {
            let lock = pool_lock(&root)?;
            if records(&root)?.iter().any(|(_, record)| !record.stopped) {
                return Err(NestError::Busy(
                    "unfinished worker requires stop reconciliation".into(),
                ));
            }
            Ok(lock)
        })
        .await
        .map_err(failure)?;
        let fail = |reason: String| Failure {
            reason,
            stopped: Some(StopEvidence {
                identity: identity.clone(),
                record: None,
            }),
        };
        let pool = self.clone();
        let requested = admission.clone();
        request_work(request, move |request| limits(&pool, &requested, request))
            .await
            .map_err(|e| fail(e.to_string()))?;
        if identity.epoch == 0
            || identity.owner.is_empty()
            || identity.owner.len() > 512
            || identity.owner.chars().any(char::is_control)
        {
            return Err(fail("invalid worker ownership identity".into()));
        }
        let stages_output = matches!(request, Request::Stage { .. });
        let export = match request {
            Request::Inspect { limits, .. }
            | Request::Stage { limits, .. }
            | Request::Qualify { limits, .. } => limits,
        };
        let max_files = if stages_output {
            export
                .max_files
                .checked_add(8)
                .ok_or_else(|| fail("worker file-count overflow".into()))?
        } else {
            8
        };
        // Reserve worst-case block rounding and directory/metadata overhead,
        // separately from the site's required free-space reserve.
        let root = self.root.clone();
        let new_bytes = admission.new_bytes;
        let reserve = self.free_reserve_bytes;
        let allocated_cap = blocking(move || {
            let free = nix::sys::statvfs::statvfs(&root).map_err(io)?;
            let overhead = (max_files as u64)
                .checked_mul(free.block_size().max(4096))
                .and_then(|n| n.checked_mul(2))
                .ok_or(NestError::NoSpace)?;
            let allocated_cap = new_bytes.checked_add(overhead).ok_or(NestError::NoSpace)?;
            let required = allocated_cap
                .checked_add(reserve)
                .ok_or(NestError::NoSpace)?;
            if free
                .blocks_available()
                .checked_mul(free.fragment_size())
                .is_none_or(|n| n < required)
            {
                return Err(NestError::NoSpace);
            }
            Ok(allocated_cap)
        })
        .await
        .map_err(|e| fail(e.to_string()))?;
        let control_timeout = Duration::from_secs(self.control_timeout_seconds);
        let image = self.image.to_string();
        let inspected = command(
            docker,
            &[
                "image".into(),
                "inspect".into(),
                "--format".into(),
                "{{.Id}}".into(),
                image.clone(),
            ],
            &[],
            control_timeout,
            4096,
        )
        .await
        .map_err(&fail)?;
        if !inspected.success || String::from_utf8_lossy(&inspected.stdout).trim() != image {
            return Err(fail(
                "assigned worker image is not present at its pinned identity".into(),
            ));
        }
        let name = format!(
            "{}-{:016x}",
            container_name(&identity),
            rand::random::<u64>()
        );
        let pool_root = self.root.clone();
        let pool_root = blocking(move || directory(&pool_root))
            .await
            .map_err(|e| fail(e.to_string()))?;
        let job = pool_root.join(&name);
        let work = job.join("work");
        let mut mounts = if matches!(request, Request::Stage { .. }) {
            vec![(work.clone(), "/work".to_string(), false)]
        } else {
            Vec::new()
        };
        let (inputs, qualified_output): (Vec<&mut Input>, Option<&mut PathBuf>) = match request {
            Request::Inspect { inputs, .. } => (inputs.iter_mut().collect(), None),
            Request::Stage {
                base,
                alternates,
                destination,
                ..
            } => {
                *destination = PathBuf::from("/work/staging");
                (
                    std::iter::once(base).chain(alternates.iter_mut()).collect(),
                    None,
                )
            }
            Request::Qualify {
                base,
                alternates,
                output,
                ..
            } => (
                std::iter::once(base).chain(alternates.iter_mut()).collect(),
                Some(output),
            ),
        };
        let mut roots: Vec<_> = inputs
            .into_iter()
            .enumerate()
            .map(|(index, input)| {
                let root = match input {
                    Input::Indexed { root } | Input::Sidecar { root, .. } => root,
                };
                (root, format!("/inputs/{index}"))
            })
            .collect();
        if let Some(output) = qualified_output {
            roots.push((output, "/inputs/output".into()));
        }
        for (root, target) in roots {
            let source = root.clone();
            let source = blocking(move || directory(&source))
                .await
                .map_err(|e| fail(e.to_string()))?;
            if source.starts_with(&pool_root) || pool_root.starts_with(&source) {
                return Err(fail("worker pool and input roots overlap".into()));
            }
            *root = PathBuf::from(&target);
            mounts.push((source, target, true));
        }
        let bytes = request_work(request, |request| serde_json::to_vec(request).map_err(io))
            .await
            .map_err(|e| fail(e.to_string()))?;
        if bytes.len() as u64 > MAX_REQUEST_BYTES {
            return Err(fail("worker request exceeds byte ceiling".into()));
        }
        let mut record = Record {
            identity: identity.clone(),
            image: self.image.clone(),
            container: name.clone(),
            id: None,
            limits: admission.clone(),
            stopped: false,
        };
        let job_clone = job.clone();
        let record_bytes = serde_json::to_vec(&record).map_err(|e| fail(e.to_string()))?;
        blocking(move || {
            fs::DirBuilder::new()
                .mode(0o700)
                .create(&job_clone)
                .map_err(io)?;
            fs::DirBuilder::new()
                .mode(0o700)
                .create(job_clone.join("work"))
                .map_err(io)?;
            write_record(
                &job_clone,
                &serde_json::from_slice(&record_bytes).map_err(io)?,
            )?;
            File::open(
                job_clone
                    .parent()
                    .ok_or_else(|| NestError::Invalid("missing worker parent".into()))?,
            )
            .and_then(|f| f.sync_all())
            .map_err(io)
        })
        .await
        .map_err(|e| fail(e.to_string()))?;
        // After this durable journal, every error/abort retains ownership until
        // an exact owned container is confirmed removed by reconciliation.
        let mut args: Vec<String> = [
            "create",
            "--runtime=runc",
            "--no-healthcheck",
            "--log-driver=none",
            "--pull=never",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--restart=no",
            "--ipc=private",
            "--interactive",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        args.extend([
            "--name".into(),
            name,
            "--label".into(),
            format!("sparknest.model-owner={}", identity.owner),
            "--label".into(),
            format!("sparknest.model-key={}", identity.key),
            "--label".into(),
            format!("sparknest.model-epoch={}", identity.epoch),
            "--user".into(),
            format!("{}:{}", nix::unistd::geteuid(), nix::unistd::getegid()),
            "--memory".into(),
            admission.ram_bytes.to_string(),
            "--memory-swap".into(),
            admission.ram_bytes.to_string(),
            "--cpus".into(),
            admission.cpu_threads.to_string(),
            "--pids-limit".into(),
            self.max_pids.to_string(),
            "--env=NVIDIA_VISIBLE_DEVICES=void".into(),
            "--env=CUDA_VISIBLE_DEVICES=".into(),
            format!("--env=OMP_NUM_THREADS={}", admission.cpu_threads),
            format!("--env=RAYON_NUM_THREADS={}", admission.cpu_threads),
        ]);
        for (source, target, readonly) in &mounts {
            args.extend([
                "--mount".into(),
                format!(
                    "type=bind,src={},dst={target},bind-propagation=rprivate{}",
                    source.display(),
                    if *readonly { ",readonly" } else { "" }
                ),
            ]);
        }
        args.extend([
            "--entrypoint=/usr/bin/timeout".into(),
            image,
            "--signal=KILL".into(),
            admission.runtime_seconds.to_string(),
            "nest-model-worker".into(),
        ]);
        let created = command(docker, &args, &[], control_timeout, 4096)
            .await
            .map_err(failure)?;
        let id = String::from_utf8_lossy(&created.stdout).trim().to_string();
        if !created.success || !valid_id(&id) {
            let reason = format!(
                "worker create outcome is uncertain; retain ownership: {}",
                String::from_utf8_lossy(&created.stderr).trim()
            );
            return Err(failure(&reason));
        }
        record.id = Some(id.clone());
        save(&job, &record).await?;
        let actual = inspect(docker, &id, control_timeout)
            .await
            .map_err(failure)?;
        if actual
            .as_ref()
            .is_none_or(|value| !admitted(value, &record, self, &mounts))
        {
            // Never start a container whose daemon configuration differs. Even
            // here cleanup rechecks ownership before touching its exact ID.
            let stopped = remove_owned(docker, control_timeout, &job, &mut record)
                .await
                .ok();
            return Err(Failure {
                reason: "worker actual configuration differs from admission".into(),
                stopped,
            });
        }
        let execution = command(
            docker,
            &[
                "start".into(),
                "--attach".into(),
                "--interactive".into(),
                id,
            ],
            &bytes,
            Duration::from_secs(admission.runtime_seconds),
            MAX_REQUEST_BYTES,
        )
        .await;
        let stopped = remove_owned(docker, control_timeout, &job, &mut record).await?;
        let response = match execution {
            Ok(output) if output.success => {
                blocking(move || nest_model::strict_json(&output.stdout).map_err(io))
                    .await
                    .map_err(|e| e.to_string())
            }
            Ok(output) => Err(format!(
                "worker failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )),
            Err(error) => Err(error),
        };
        let work_clone = work.clone();
        let cap = admission.new_bytes;
        let verification =
            blocking(move || output_bytes(&work_clone, max_files, cap, allocated_cap)).await;
        drop(lock);
        Ok(Completion {
            response: verification.map_err(|e| e.to_string()).and(response),
            staging: work.join("staging"),
            stopped,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn setup(mode: &str) -> (tempfile::TempDir, Pool, PathBuf, Identity, Request, Limits) {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("pool")).unwrap();
        fs::create_dir(dir.path().join("input")).unwrap();
        let image = Digest::of(b"worker-image");
        let script = dir.path().join("docker-fixture");
        let source = format!(
            r#"#!/usr/bin/python3
import json, pathlib, sys, time
root = pathlib.Path({root:?})
a = sys.argv[1:]
mode = {mode:?}
id = 'a' * 64
state = root / 'container.json'
def arg(flag):
    return a[a.index(flag) + 1]
if a[0] == 'image':
    print({image:?} if mode != 'wrong-image' else 'sha256:' + 'b'*64)
elif a[0] == 'create':
    (root / 'create.json').write_text(json.dumps(a))
    labels = dict(v.split('=', 1) for i,v in enumerate(a) if i and a[i-1] == '--label')
    mounts = []
    for i,v in enumerate(a):
        if i and a[i-1] == '--mount':
            opts = dict(x.split('=',1) if '=' in x else (x, True) for x in v.split(','))
            mounts.append({{'Type':'bind','Source':opts['src'],'Destination':opts['dst'],'RW':'readonly' not in opts,'Propagation':'rprivate'}})
    data = {{'Id':id,'Image':{image:?},'Name':'/' + arg('--name'),
        'State':{{'Status':'created','Running':False}},
        'Config':{{'Labels':labels,'User':arg('--user'),'Volumes':None,
            'Entrypoint':['/usr/bin/timeout'],'Cmd':['--signal=KILL', a[-2], 'nest-model-worker']}},
        'HostConfig':{{'Runtime':'runc','NetworkMode':'none','ReadonlyRootfs':True,'Privileged':False,
            'Memory':int(arg('--memory')),'MemorySwap':int(arg('--memory-swap')),
            'NanoCpus':int(arg('--cpus'))*1000000000,'PidsLimit':int(arg('--pids-limit')),
            'CapDrop':['ALL'],'CapAdd':None,'SecurityOpt':['no-new-privileges'],
            'LogConfig':{{'Type':'none'}},'RestartPolicy':{{'Name':'no'}},
            'DeviceRequests':None,'Devices':[],'DeviceCgroupRules':None,'Tmpfs':None,'Binds':None,
            'VolumesFrom':None,'PortBindings':{{}},'PublishAllPorts':False,'PidMode':'','IpcMode':'private'}},'Mounts':mounts}}
    if mode == 'wrong-memory': data['HostConfig']['Memory'] += 1
    if mode == 'gpu': data['HostConfig']['DeviceRequests'] = [{{'Count':-1,'Capabilities':[['gpu']]}}]
    if mode == 'writable-input': data['Mounts'][0]['RW'] = True
    if mode == 'extra-mount': data['Mounts'].append(data['Mounts'][0])
    if mode == 'wrong-owner': data['Config']['Labels']['sparknest.model-owner'] = 'another-owner'
    state.write_text(json.dumps(data))
    if mode == 'create-hang': time.sleep(60)
    if mode == 'create-uncertain':
        print('synthetic create diagnostic', file=sys.stderr)
        sys.exit(1)
    if mode == 'create-missing': state.unlink(); sys.exit(1)
    print(id)
elif a[:2] == ['container','inspect']:
    if mode == 'inspect-unavailable': sys.exit(1)
    if not state.exists():
        print('Error: No such container: ' + a[-1], file=sys.stderr)
        sys.exit(1)
    print(state.read_text())
elif a[0] == 'start':
    request = json.load(sys.stdin)
    (root / 'started').write_text(json.dumps(request))
    if mode in ['hang','abort']:
        time.sleep(60)
    if mode == 'worker-error':
        print('synthetic worker failure', file=sys.stderr)
        sys.exit(1)
    if mode == 'stdout-flood':
        sys.stdout.write('a' * (65 * 1024 * 1024)); sys.exit(0)
    if mode == 'stderr-flood':
        sys.stderr.write('a' * 65537); sys.exit(1)
    print(json.dumps({{'operation':'inspect','inputs':[]}}))
elif a[0] == 'rm':
    (root / 'removed').write_text(a[-1])
    if mode == 'rm-hang': time.sleep(60)
    if mode == 'rm-fail': sys.exit(1)
    state.unlink()
    print(a[-1])
else:
    sys.exit(2)
"#,
            root = dir.path().to_str().unwrap(),
            image = image.to_string()
        );
        fs::write(&script, source).unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o700)).unwrap();
        let pool = Pool {
            name: "cpu".into(),
            root: dir.path().join("pool"),
            image,
            max_cpu_threads: 2,
            max_ram_bytes: 32 << 20,
            max_new_bytes: 32 << 20,
            free_reserve_bytes: 1 << 20,
            max_runtime_seconds: 10,
            control_timeout_seconds: 2,
            max_pids: 32,
        };
        let identity = Identity {
            key: Digest::of(b"composition"),
            epoch: 1,
            owner: "worker-owner".into(),
        };
        let request = Request::Inspect {
            inputs: vec![Input::Indexed {
                root: dir.path().join("input"),
            }],
            limits: nest_model::ExportLimits {
                max_files: 16,
                ..Default::default()
            },
        };
        let admission = Limits {
            cpu_threads: 1,
            ram_bytes: 16 << 20,
            new_bytes: 16 << 20,
            runtime_seconds: 1,
        };
        (dir, pool, script, identity, request, admission)
    }

    fn module_fixture(
        pool: &Pool,
        admission: &Limits,
        manifest: Digest,
    ) -> nest_model::ModuleConfig {
        serde_json::from_value(serde_json::json!({
            "schema": nest_model::SCHEMA, "name":"synthetic-model",
            "base":{"source":"synthetic-base","content_manifest":manifest},
            "alternates":[],
            "output":{"loader_profile":nest_model::LOADER_PROFILE,"precision":"match-base",
                "configuration":"base-unchanged","tokenizer":"base-unchanged",
                "non_weight_mismatch":"reject","namespace":"synthetic-output"},
            "preparation":{"trigger":"explicit-prepare-or-external-use-wrapper",
                "adapter_image":format!("worker@{}",pool.image),"recipe":Digest::of(b"exact-recipe"),
                "lossy_conversion":"deny","qualification_profile":nest_model::policy_identity(),
                "activation_calibration":"preserve-and-qualify","max_runtime_seconds":admission.runtime_seconds,
                "max_ram_bytes":admission.ram_bytes,"cpu_threads":admission.cpu_threads,
                "device":"cpu","worker_pool":pool.name},
            "cache":{"publication":"sealed-regular-files","reuse_unchanged":false,"prefer_reflink":false,
                "max_new_physical_bytes":admission.new_bytes,"account_inflight":true,
                "eviction":"least-recently-used-unpinned","retain_dependencies":true}
        })).unwrap()
    }

    #[test]
    fn stage_admits_both_declared_output_caps_independently() {
        let (dir, pool, _, _, _, admission) = setup("ok");
        for field in ["export", "cache"] {
            let mut module = module_fixture(&pool, &admission, Digest::of(b"synthetic-base"));
            let mut export = nest_model::ExportLimits {
                max_output_bytes: admission.new_bytes,
                ..Default::default()
            };
            if field == "export" {
                export.max_output_bytes += 1;
            } else {
                module.cache.max_new_physical_bytes += 1;
            }
            let request = Request::Stage {
                module: Box::new(module),
                profile: nest_model::worker::OperandProfile {
                    schema: "sparknest.operand-profile.v1".into(),
                    groups: std::collections::BTreeMap::from([("a".into(), vec!["a".into()])]),
                    static_activation_calibration: false,
                },
                base: Input::Indexed {
                    root: dir.path().join("input"),
                },
                alternates: vec![],
                destination: dir.path().join("output"),
                limits: export,
            };
            assert!(limits(&pool, &admission, &request).is_err(), "{field}");
        }
    }

    #[tokio::test]
    async fn qualification_is_readonly_and_does_not_reserve_stage_output_twice() {
        let (dir, pool, docker, identity, _, mut admission) = setup("ok");
        let module = module_fixture(&pool, &admission, Digest::of(b"synthetic-base"));
        let export = nest_model::ExportLimits {
            max_files: 16,
            max_output_bytes: admission.new_bytes,
            ..Default::default()
        };
        admission.new_bytes = 4096;
        let mut request = Request::Qualify {
            module: Box::new(module),
            profile: nest_model::worker::OperandProfile {
                schema: "sparknest.operand-profile.v1".into(),
                groups: std::collections::BTreeMap::from([("a".into(), vec!["a".into()])]),
                static_activation_calibration: false,
            },
            base: Input::Indexed {
                root: dir.path().join("input"),
            },
            alternates: vec![],
            output: dir.path().join("input"),
            limits: export,
        };
        pool.run_with(&docker, identity, &mut request, admission)
            .await
            .unwrap();
        let args: Vec<String> =
            serde_json::from_slice(&fs::read(dir.path().join("create.json")).unwrap()).unwrap();
        let binds: Vec<_> = args
            .iter()
            .filter(|arg| arg.starts_with("type=bind,"))
            .collect();
        assert_eq!(binds.len(), 2);
        assert!(binds.iter().all(|arg| arg.ends_with(",readonly")));
        assert!(binds.iter().any(|arg| arg.contains("dst=/inputs/output,")));
        assert!(!binds.iter().any(|arg| arg.contains("dst=/work,")));
    }

    #[tokio::test]
    async fn confirmed_removal_produces_private_evidence_and_allows_sequential_identity() {
        let (dir, pool, docker, identity, mut request, admission) = setup("ok");
        for _ in 0..2 {
            let done = pool
                .run_with(&docker, identity.clone(), &mut request, admission.clone())
                .await
                .unwrap();
            assert!(done.response.is_ok(), "{:?}", done.response);
            assert_eq!(done.stopped.identity(), &identity);
            let record: Record =
                serde_json::from_slice(&fs::read(done.stopped.record().unwrap()).unwrap()).unwrap();
            assert!(record.stopped);
            assert_eq!(
                fs::read_to_string(dir.path().join("removed")).unwrap(),
                "a".repeat(64)
            );
            assert_eq!(
                fs::metadata(done.stopped.record().unwrap())
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o077,
                0
            );
            // Every run consumes paths, just like the public owned Request API.
            if let Request::Inspect { inputs, .. } = &mut request {
                *inputs = vec![Input::Indexed {
                    root: dir.path().join("input"),
                }];
            }
        }
        assert_eq!(records(&pool.root).unwrap().len(), 2);
        let args: Vec<String> =
            serde_json::from_slice(&fs::read(dir.path().join("create.json")).unwrap()).unwrap();
        for flag in [
            "--runtime=runc",
            "--pull=never",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
        ] {
            assert!(args.iter().any(|arg| arg == flag), "{flag}");
        }
        let at = |flag: &str| args.iter().position(|s| s == flag).unwrap();
        assert_eq!(args[at("--memory") + 1], args[at("--memory-swap") + 1]);
        assert!(
            args.iter()
                .find(|s| s.contains("dst=/inputs/0"))
                .unwrap()
                .ends_with(",readonly")
        );
        assert!(!args.iter().any(|s| s.contains("dst=/work")));
        assert!(!args.iter().any(|s| s.contains("control.json")));
        pool.reconcile_with(&docker, identity).await.unwrap();
    }

    #[tokio::test]
    async fn worker_error_deadline_and_output_floods_still_require_removal() {
        for mode in ["worker-error", "hang", "stderr-flood", "stdout-flood"] {
            let (dir, pool, docker, identity, mut request, admission) = setup(mode);
            let done = pool
                .run_with(&docker, identity, &mut request, admission)
                .await
                .unwrap();
            assert!(done.response.is_err(), "{mode}");
            assert!(dir.path().join("removed").exists());
            assert!(records(&pool.root).unwrap()[0].1.stopped);
        }
    }

    #[tokio::test]
    async fn actual_caps_or_mount_mismatch_never_starts_and_cleans_only_owned_id() {
        for mode in ["wrong-memory", "gpu", "writable-input", "extra-mount"] {
            let (dir, pool, docker, identity, mut request, admission) = setup(mode);
            let error = pool
                .run_with(&docker, identity, &mut request, admission)
                .await
                .unwrap_err();
            assert!(
                error.reason.contains("actual configuration"),
                "{mode}: {}",
                error.reason
            );
            assert!(error.stopped.is_some());
            assert!(!dir.path().join("started").exists());
            assert_eq!(
                fs::read_to_string(dir.path().join("removed")).unwrap(),
                "a".repeat(64)
            );
        }
    }

    #[tokio::test]
    async fn wrong_owner_is_never_started_or_removed() {
        let (dir, pool, docker, identity, mut request, admission) = setup("wrong-owner");
        let error = pool
            .run_with(&docker, identity.clone(), &mut request, admission)
            .await
            .unwrap_err();
        assert!(error.stopped.is_none());
        assert!(
            pool.reconcile_with(&docker, identity)
                .await
                .unwrap_err()
                .stopped
                .is_none()
        );
        assert!(!dir.path().join("started").exists());
        assert!(!dir.path().join("removed").exists());
    }

    #[tokio::test]
    async fn unconfirmed_removal_blocks_retries_and_reconciles_exact_identity_only() {
        let (dir, pool, docker, identity, mut request, admission) = setup("rm-fail");
        let error = pool
            .run_with(&docker, identity.clone(), &mut request, admission.clone())
            .await
            .unwrap_err();
        assert!(error.stopped.is_none());
        assert!(!records(&pool.root).unwrap()[0].1.stopped);
        assert!(
            pool.reconcile_with(
                &docker,
                Identity {
                    epoch: 2,
                    ..identity.clone()
                }
            )
            .await
            .is_err()
        );
        let error = pool
            .run_with(&docker, identity.clone(), &mut request, admission)
            .await
            .unwrap_err();
        assert!(error.reason.contains("reconciliation"));
        assert!(error.stopped.is_none());
        let source = fs::read_to_string(&docker)
            .unwrap()
            .replace("mode = \"rm-fail\"", "mode = \"ok\"");
        fs::write(&docker, source).unwrap();
        let evidence = pool.reconcile_with(&docker, identity).await.unwrap();
        assert!(evidence.record().is_some());
        assert!(!dir.path().join("container.json").exists());
    }

    #[tokio::test]
    async fn removal_deadline_retains_ownership_until_explicit_reconciliation() {
        let (dir, pool, docker, identity, mut request, admission) = setup("rm-hang");
        let error = pool
            .run_with(&docker, identity.clone(), &mut request, admission)
            .await
            .unwrap_err();
        assert!(error.stopped.is_none());
        assert!(!records(&pool.root).unwrap()[0].1.stopped);
        let source = fs::read_to_string(&docker)
            .unwrap()
            .replace("mode = \"rm-hang\"", "mode = \"ok\"");
        fs::write(&docker, source).unwrap();
        pool.reconcile_with(&docker, identity).await.unwrap();
        assert!(!dir.path().join("container.json").exists());
    }

    #[tokio::test]
    async fn uncertain_create_with_container_is_reconciled_but_name_absence_is_not_death() {
        for mode in [
            "create-uncertain",
            "create-hang",
            "create-missing",
            "inspect-unavailable",
        ] {
            let (dir, pool, docker, identity, mut request, admission) = setup(mode);
            let error = pool
                .run_with(&docker, identity.clone(), &mut request, admission)
                .await
                .unwrap_err();
            assert!(error.stopped.is_none());
            if mode == "create-uncertain" {
                assert!(error.reason.contains("synthetic create diagnostic"));
            }
            let result = pool.reconcile_with(&docker, identity).await;
            assert_eq!(
                result.is_ok(),
                matches!(mode, "create-uncertain" | "create-hang"),
                "{mode}: {result:?}"
            );
            assert!(!dir.path().join("started").exists());
        }
    }

    #[tokio::test]
    async fn abort_retains_durable_owner_and_process_lock_excludes_overlapping_run() {
        let (dir, pool, docker, identity, mut request, admission) = setup("abort");
        let mut longer = admission.clone();
        longer.runtime_seconds = 10;
        let task = {
            let pool = pool.clone();
            let docker = docker.clone();
            let identity = identity.clone();
            tokio::spawn(
                async move { pool.run_with(&docker, identity, &mut request, longer).await },
            )
        };
        tokio::time::timeout(Duration::from_secs(3), async {
            while !dir.path().join("started").exists() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let (_, _, _, _, mut another, _) = setup("ok");
        let error = pool
            .run_with(&docker, identity.clone(), &mut another, admission)
            .await
            .unwrap_err();
        assert!(error.stopped.is_none());
        assert!(error.reason.contains("in use"));
        task.abort();
        let _ = task.await;
        assert!(!records(&pool.root).unwrap()[0].1.stopped);
        let evidence = pool.reconcile_with(&docker, identity).await.unwrap();
        assert!(evidence.record().is_some());
        assert_eq!(
            fs::read_to_string(dir.path().join("removed")).unwrap(),
            "a".repeat(64)
        );
    }

    #[tokio::test]
    async fn admission_refusal_has_no_launch_evidence_only_after_pool_is_known_idle() {
        let (dir, pool, docker, identity, mut request, mut admission) = setup("ok");
        admission.cpu_threads = 3;
        let error = pool
            .run_with(&docker, identity, &mut request, admission)
            .await
            .unwrap_err();
        assert!(error.stopped.unwrap().record().is_none());
        assert!(!dir.path().join("create.json").exists());
    }

    #[tokio::test]
    async fn mutable_or_unpinned_image_is_not_created() {
        let (dir, pool, docker, identity, mut request, admission) = setup("wrong-image");
        let error = pool
            .run_with(&docker, identity, &mut request, admission)
            .await
            .unwrap_err();
        assert!(error.stopped.is_some());
        assert!(!dir.path().join("create.json").exists());
        let mut json = serde_json::to_value(&pool).unwrap();
        json["image"] = serde_json::json!("worker:latest");
        assert!(serde_json::from_value::<Pool>(json).is_err());
        let mut json = serde_json::to_value(&pool).unwrap();
        json["gpus"] = serde_json::json!("all");
        assert!(serde_json::from_value::<Pool>(json).is_err());
    }

    #[test]
    fn output_verification_rejects_oversize_and_links() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("output"), b"12345").unwrap();
        assert!(output_bytes(dir.path(), 8, 4, 65536).is_err());
        assert!(output_bytes(dir.path(), 8, 5, 65536).is_ok());
        std::os::unix::fs::symlink("output", dir.path().join("alias")).unwrap();
        assert!(output_bytes(dir.path(), 8, 5, 65536).is_err());
    }

    /// Opt-in only: needs a local CPU image built from the exact worker source,
    /// with coreutils timeout, and access to Docker. Never pulls an image.
    #[tokio::test]
    #[ignore = "requires explicitly assigned CPU-only Docker smoke image"]
    async fn real_docker_cpu_smoke() {
        let (_, mut pool, _, identity, mut request, mut admission) = setup("ok");
        // Keep roots alive independently from setup's fixture script directory.
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("pool")).unwrap();
        fs::create_dir(root.path().join("input")).unwrap();
        pool.root = root.path().join("pool");
        pool.image = serde_json::from_value(serde_json::json!(
            std::env::var("SPARKNEST_WORKER_SMOKE_IMAGE").expect("assigned pinned image ID")
        ))
        .unwrap();
        pool.max_ram_bytes = 128 << 20;
        admission.ram_bytes = 128 << 20;
        admission.runtime_seconds = 5;
        if let Request::Inspect { inputs, .. } = &mut request {
            *inputs = vec![Input::Indexed {
                root: root.path().join("input"),
            }];
        }
        let input = root.path().join("input");
        let shard = "model-00001-of-00001.safetensors";
        let mut header = serde_json::to_vec(&serde_json::json!({
            "a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}
        }))
        .unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut tensor = (header.len() as u64).to_le_bytes().to_vec();
        tensor.extend(header);
        tensor.push(7);
        fs::write(input.join(shard), &tensor).unwrap();
        fs::write(input.join("config.json"), b"{}").unwrap();
        fs::write(
            input.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "metadata":{"total_size":1},"weight_map":{"a":shard}
            }))
            .unwrap(),
        )
        .unwrap();
        let export = nest_model::ExportLimits {
            metadata_bytes: 1 << 20,
            max_files: 16,
            max_tensors: 16,
            buffer_bytes: 65536,
            max_output_bytes: admission.new_bytes,
        };
        if let Request::Inspect { limits, .. } = &mut request {
            *limits = export.clone();
        }
        let done = pool
            .run(identity.clone(), request, admission.clone())
            .await
            .unwrap();
        let inspected = match done.response.unwrap() {
            Response::Inspect { inputs } => inputs,
            _ => panic!("wrong operation response"),
        };
        assert_eq!(inspected.len(), 1);
        let module = module_fixture(&pool, &admission, inspected[0].content_manifest.clone());
        let stage = Request::Stage {
            module: Box::new(module),
            profile: nest_model::worker::OperandProfile {
                schema: "sparknest.operand-profile.v1".into(),
                groups: std::collections::BTreeMap::from([("a".into(), vec!["a".into()])]),
                static_activation_calibration: false,
            },
            base: Input::Indexed {
                root: input.clone(),
            },
            alternates: vec![],
            destination: PathBuf::from("ignored-by-runner"),
            limits: export,
        };
        let done = pool.run(identity.clone(), stage, admission).await.unwrap();
        assert!(
            matches!(done.response, Ok(Response::Stage { .. })),
            "{:?}",
            done.response
        );
        assert_eq!(fs::read(done.staging.join(shard)).unwrap(), tensor);
        assert_eq!(fs::read(input.join(shard)).unwrap(), tensor);
        assert_eq!(done.stopped.identity(), &identity);
        assert!(
            records(&pool.root)
                .unwrap()
                .iter()
                .all(|(_, record)| record.stopped)
        );
    }
}
