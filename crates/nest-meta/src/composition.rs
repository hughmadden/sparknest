//! Replicated preparation ownership and immutable artifact references (ADR-072).
//! Payload verification belongs to workers; these commands fence publication
//! and deletion. No wall-clock expiry can prove that a mapped model is dead.

use crate::apply::{Fail, R, err};
use crate::{Command, Effect, Reply, query};
use nest_types::{
    FileId, FileKind, GenState, Generation, NestError, NodeId, ReplicaState, Timestamp,
};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Persisted inside Command::Model: append variants only (ADR-018).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelCommand {
    Claim {
        key: String,
        owner: String,
    },
    Pin {
        key: String,
        owner: String,
        epoch: u64,
        files: Vec<(FileId, Generation)>,
    },
    Publish {
        key: String,
        owner: String,
        epoch: u64,
        publication: String,
    },
    /// The owning worker has stopped and completed its cleanup. An unreachable
    /// worker must remain PREPARING; a missed heartbeat does not call Stop.
    Stop {
        key: String,
        owner: String,
        epoch: u64,
        receipt: String,
    },
    Use {
        key: String,
        epoch: u64,
        lease: String,
    },
    /// Called only after the lease's model containers and ranks have stopped.
    Release {
        key: String,
        epoch: u64,
        lease: String,
    },
    Reset {
        key: String,
        when_idle: bool,
        epoch: u64,
    },
    CancelReset {
        key: String,
        epoch: u64,
        reset: u64,
    },
    /// Atomically protect an exact flat namespace inventory and pin its sealed
    /// generations. Input acquisition aliases are resolved before this command.
    BindSnapshot {
        key: String,
        owner: String,
        epoch: u64,
        root: FileId,
        output: bool,
        files: Vec<(Vec<u8>, FileId, Generation)>,
    },
    /// Validated declarative data, never executable code. The manager validates
    /// the serialized spec/profile and computes their identities before apply.
    PutModule {
        name: String,
        spec: String,
        profile: String,
        key: String,
        qualification_policy: String,
        expected_revision: u64,
    },
    /// Manager attestation after export and numerical verification. Metadata
    /// checks exact inventory and durable references, not checkpoint numerics.
    PublishComplete {
        key: String,
        owner: String,
        epoch: u64,
        root: FileId,
        files: Vec<(Vec<u8>, FileId, Generation)>,
        publication: String,
        export_receipt: (FileId, Generation),
        qualification_receipt: (FileId, Generation),
        qualification_policy: String,
    },
    /// Import a NEW sealed derived generation and record ownership atomically.
    /// The complete object must already be durable under its reserved file id.
    /// Existing files/aliases are refused by the ordinary import primitive.
    ImportOutput {
        key: String,
        owner: String,
        epoch: u64,
        parent: FileId,
        name: Vec<u8>,
        file: FileId,
        perm: u32,
        size: u64,
        mtime: Timestamp,
        node: NodeId,
        now: Timestamp,
    },
    /// Explicit reclamation of one proven, retired/failed derived shard alias.
    /// Historical import proof is retained; retries never delete a replacement.
    /// Actual unlink returns the pre-unlink Attr; an absent alias returns Done.
    /// Attr size is logical bytes, not proof of immediate physical disk release.
    ReclaimOutputShard {
        key: String,
        epoch: u64,
        parent: FileId,
        name: Vec<u8>,
        file: FileId,
        generation: Generation,
        now: Timestamp,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelReply {
    Epoch(u64),
    Reset(u64),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelEntry {
    pub key: String,
    pub epoch: u64,
    pub owner: Option<String>,
    pub state: String,
    pub publication: Option<String>,
    pub reset_pending: bool,
    pub reset_id: Option<u64>,
    pub files: Vec<(FileId, Generation)>,
    pub leases: Vec<String>,
}

/// Query DTO, not a persisted log type. Names bind ordinary regular files;
/// this is neither an extent map nor a different data-plane representation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSnapshot {
    pub root: FileId,
    pub output: bool,
    pub files: Vec<(Vec<u8>, FileId, Generation)>,
}

/// Query DTOs are not persisted log fields. Module revisions can change
/// without rewriting an artifact's preparation or qualification identities.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelModule {
    pub name: String,
    pub revision: u64,
    pub spec: String,
    pub profile: String,
    pub key: String,
    pub qualification_policy: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPublication {
    pub epoch: u64,
    pub root: FileId,
    pub publication: String,
    pub export_receipt: (FileId, Generation),
    pub qualification_receipt: (FileId, Generation),
    pub qualification_policy: String,
}

/// Historical import proof, not permission to delete. ReclaimOutputShard
/// atomically checks current alias/generation, lifetime and receipt exclusions
/// before applying ordinary namespace and shared-reference guards.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelOwnedOutput {
    pub epoch: u64,
    pub parent: FileId,
    pub name: Vec<u8>,
    pub file: FileId,
    pub generation: Generation,
}

pub fn module(c: &Connection, name: &str) -> rusqlite::Result<Option<ModelModule>> {
    c.prepare_cached("SELECT name, revision, spec, profile, key, qualification_policy FROM model_modules WHERE name = ?1")?
        .query_row([name], module_from_row).optional()
}

fn module_from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<ModelModule> {
    Ok(ModelModule {
        name: r.get(0)?,
        revision: r.get::<_, i64>(1)? as u64,
        spec: r.get(2)?,
        profile: r.get(3)?,
        key: r.get(4)?,
        qualification_policy: r.get(5)?,
    })
}

pub fn modules(c: &Connection) -> rusqlite::Result<Vec<ModelModule>> {
    c.prepare_cached("SELECT name, revision, spec, profile, key, qualification_policy FROM model_modules ORDER BY name")?
        .query_map([], module_from_row)?.collect()
}

/// Most recently completed qualification. Historical records survive reset;
/// callers MUST also require the artifact to be READY in this exact epoch and
/// the qualification policy to match the requesting module.
pub fn complete_publication(
    c: &Connection,
    key: &str,
) -> rusqlite::Result<Option<ModelPublication>> {
    c.prepare_cached("SELECT epoch, root, publication, export_file, export_gen, qualification_file, qualification_gen, qualification_policy FROM model_publications WHERE key = ?1 ORDER BY epoch DESC LIMIT 1")?
        .query_row([key], |r| Ok(ModelPublication { epoch: r.get::<_, i64>(0)? as u64,
            root: FileId(r.get::<_, i64>(1)? as u64), publication: r.get(2)?,
            export_receipt: (FileId(r.get::<_, i64>(3)? as u64), Generation(r.get::<_, i64>(4)? as u64)),
            qualification_receipt: (FileId(r.get::<_, i64>(5)? as u64), Generation(r.get::<_, i64>(6)? as u64)),
            qualification_policy: r.get(7)?,
        })).optional()
}

pub fn owned_outputs(c: &Connection, key: &str) -> rusqlite::Result<Vec<ModelOwnedOutput>> {
    c.prepare_cached("SELECT epoch, parent, name, file, gen FROM model_owned_outputs WHERE key = ?1 ORDER BY epoch, parent, name, file")?
        .query_map([key], |r| Ok(ModelOwnedOutput { epoch: r.get::<_, i64>(0)? as u64,
            parent: FileId(r.get::<_, i64>(1)? as u64), name: r.get(2)?,
            file: FileId(r.get::<_, i64>(3)? as u64), generation: Generation(r.get::<_, i64>(4)? as u64),
        }))?.collect()
}

pub fn snapshots(c: &Connection, key: &str) -> rusqlite::Result<Vec<ModelSnapshot>> {
    let mut roots = c
        .prepare_cached("SELECT root, output FROM model_snapshots WHERE key = ?1 ORDER BY root")?
        .query_map([key], |r| {
            Ok(ModelSnapshot {
                root: FileId(r.get::<_, i64>(0)? as u64),
                output: r.get(1)?,
                files: Vec::new(),
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for root in &mut roots {
        root.files = c.prepare_cached(
            "SELECT name, file, gen FROM model_snapshot_files WHERE key = ?1 AND root = ?2 ORDER BY name",
        )?.query_map(params![key, root.root.0 as i64], |r| Ok((
            r.get(0)?, FileId(r.get::<_, i64>(1)? as u64), Generation(r.get::<_, i64>(2)? as u64),
        )))?.collect::<rusqlite::Result<_>>()?;
    }
    Ok(roots)
}

pub(crate) fn writable_directory(c: &Connection, dir: FileId) -> R<()> {
    let bound: bool = c.query_row(
        "SELECT EXISTS(SELECT 1 FROM model_snapshots WHERE root = ?1)",
        [dir.0 as i64],
        |r| r.get(0),
    )?;
    if bound {
        return err(NestError::Busy(
            "prepared snapshot inventory is immutable".into(),
        ));
    }
    Ok(())
}

/// Moving a root or any of its ancestors changes the resolved snapshot path.
pub(crate) fn movable_directory(c: &Connection, dir: FileId) -> R<()> {
    let protected: bool = c.query_row(
        "WITH RECURSIVE ancestors(id, parent) AS (\
         SELECT f.id, f.parent FROM model_snapshots s JOIN files f ON f.id = s.root \
         UNION SELECT f.id, f.parent FROM files f JOIN ancestors a ON f.id = a.parent) \
         SELECT EXISTS(SELECT 1 FROM ancestors WHERE id = ?1)",
        [dir.0 as i64],
        |r| r.get(0),
    )?;
    if protected {
        return err(NestError::Busy(
            "prepared snapshot path is immutable".into(),
        ));
    }
    Ok(())
}

fn release_pins(c: &Connection, key: &str) -> R<()> {
    c.execute("DELETE FROM model_snapshot_files WHERE key = ?1", [key])?;
    c.execute("DELETE FROM model_snapshots WHERE key = ?1", [key])?;
    c.execute("DELETE FROM model_pins WHERE key = ?1", [key])?;
    Ok(())
}

pub fn get(c: &Connection, key: &str) -> rusqlite::Result<Option<ModelEntry>> {
    let row = c.prepare_cached("SELECT epoch, owner, state, publication, reset_pending FROM model_artifacts WHERE key = ?1")?
        .query_row([key], |r| Ok(ModelEntry {
            key: key.into(), epoch: r.get::<_, i64>(0)? as u64, owner: r.get(1)?, state: r.get(2)?,
            publication: r.get(3)?, reset_pending: r.get::<_, i64>(4)? != 0, reset_id: match r.get::<_, i64>(4)? { 0 => None, n => Some(n as u64) }, files: Vec::new(), leases: Vec::new(),
        })).optional()?;
    let Some(mut row) = row else { return Ok(None) };
    row.files = c
        .prepare_cached("SELECT file, gen FROM model_pins WHERE key = ?1 ORDER BY file, gen")?
        .query_map([key], |r| {
            Ok((
                FileId(r.get::<_, i64>(0)? as u64),
                Generation(r.get::<_, i64>(1)? as u64),
            ))
        })?
        .collect::<rusqlite::Result<_>>()?;
    row.leases = c
        .prepare_cached(
            "SELECT lease FROM model_users WHERE key = ?1 AND released = 0 ORDER BY lease",
        )?
        .query_map([key], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(Some(row))
}

/// List every preparation key, including keys no longer named by a module.
/// Hydration uses the same query as `get`, keeping lease/pin semantics in one
/// place. Call inside the caller's read transaction for a consistent listing.
pub fn entries(c: &Connection) -> rusqlite::Result<Vec<ModelEntry>> {
    let keys = c
        .prepare_cached("SELECT key FROM model_artifacts ORDER BY key")?
        .query_map([], |r| r.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    keys.iter()
        .filter_map(|key| get(c, key).transpose())
        .collect()
}

pub fn is_pinned(c: &Connection, file: FileId) -> rusqlite::Result<bool> {
    c.prepare_cached("SELECT EXISTS(SELECT 1 FROM model_pins WHERE file = ?1)")?
        .query_row([file.0 as i64], |r| r.get(0))
}

fn entry(c: &Connection, key: &str) -> R<ModelEntry> {
    get(c, key)?.ok_or(Fail::Nest(NestError::NotFound))
}

fn token(value: &str) -> R<()> {
    if value.is_empty() || value.len() > 512 || value.chars().any(char::is_control) {
        return err(NestError::Invalid("invalid preparation identity".into()));
    }
    Ok(())
}

fn worker(c: &Connection, key: &str, owner: &str, epoch: u64) -> R<ModelEntry> {
    let a = entry(c, key)?;
    if a.epoch != epoch || a.owner.as_deref() != Some(owner) || a.state != "preparing" {
        return err(NestError::Stale);
    }
    Ok(a)
}

fn validate_file(c: &Connection, file: FileId, generation: Generation) -> R<()> {
    let a = query::getattr(c, file)?.ok_or(Fail::Nest(NestError::NotFound))?;
    if a.kind != FileKind::Regular
        || a.gen_state != GenState::Stable
        || !a.sealed
        || a.generation != generation
        || a.nlink == 0
    {
        return err(NestError::Invalid(
            "preparation requires named sealed current generations".into(),
        ));
    }
    if !query::replicas(c, file)?
        .iter()
        .any(|r| r.generation == generation && r.state == ReplicaState::Live)
    {
        return err(NestError::Unavailable(
            "pinned generation has no replica".into(),
        ));
    }
    Ok(())
}

fn retire(c: &Connection, key: &str) -> R<()> {
    // Retire the catalog reference before releasing its generation pins. The
    // manager may subsequently reclaim only its own unreferenced derived files.
    c.execute("UPDATE model_artifacts SET state = 'retired', owner = NULL, reset_pending = 0 WHERE key = ?1", [key])?;
    release_pins(c, key)?;
    Ok(())
}

pub(crate) fn exec(c: &Connection, command: &ModelCommand, fx: &mut Vec<Effect>) -> R<Reply> {
    match command {
        ModelCommand::ReclaimOutputShard {
            key,
            epoch,
            parent,
            name,
            file,
            generation,
            now,
        } => {
            let a = entry(c, key)?;
            if a.epoch != *epoch {
                return err(NestError::Stale);
            }
            if !matches!(a.state.as_str(), "failed" | "retired")
                || a.owner.is_some()
                || !a.leases.is_empty()
            {
                return err(NestError::Busy(
                    "artifact must be stopped and unused before reclamation".into(),
                ));
            }
            let proven: bool = c.query_row(
                "SELECT EXISTS(SELECT 1 FROM model_owned_outputs WHERE key = ?1 AND epoch = ?2 AND parent = ?3 AND name = ?4 AND file = ?5 AND gen = ?6)",
                params![key, *epoch as i64, parent.0 as i64, name, file.0 as i64, generation.0 as i64], |r| r.get(0),
            )?;
            if !proven {
                return err(NestError::NotPermitted(
                    "no exact derived import ownership proof".into(),
                ));
            }
            // v1 reclaims payload shards only. JSON/config/tokenizer assets and
            // every historical publication receipt stay as durable evidence,
            // even when an imported receipt has a shard-looking filename.
            if !name.ends_with(b".safetensors") {
                return err(NestError::NotPermitted(
                    "only derived safetensors shard payloads may be reclaimed".into(),
                ));
            }
            let receipt: bool = c.query_row(
                "SELECT EXISTS(SELECT 1 FROM model_publications WHERE (export_file = ?1 AND export_gen = ?2) OR (qualification_file = ?1 AND qualification_gen = ?2))",
                params![file.0 as i64, generation.0 as i64], |r| r.get(0),
            )?;
            if receipt {
                return err(NestError::NotPermitted(
                    "publication receipts must be retained".into(),
                ));
            }
            match query::lookup(c, *parent, name)? {
                None => return Ok(Reply::Done),
                Some(current) if current != *file => return err(NestError::Stale),
                Some(_) => {}
            }
            let attr = query::getattr(c, *file)?.ok_or(Fail::Nest(NestError::Stale))?;
            if attr.generation != *generation {
                return err(NestError::Stale);
            }
            if attr.kind != FileKind::Regular || !attr.sealed || attr.gen_state != GenState::Stable
            {
                return err(NestError::Busy(
                    "derived generation must remain sealed and stable".into(),
                ));
            }
            // Reuse namespace/root, shared-pin and last-link guards, and emit
            // the ordinary generation deletion effects in this savepoint.
            crate::apply::exec(
                c,
                &Command::Unlink {
                    parent: *parent,
                    name: name.clone(),
                    now: *now,
                },
                fx,
            )?;
            Ok(Reply::Attr(attr))
        }
        ModelCommand::PutModule {
            name,
            spec,
            profile,
            key,
            qualification_policy,
            expected_revision,
        } => {
            token(name)?;
            token(key)?;
            token(qualification_policy)?;
            // Bound allocations at the replicated catalog boundary. Parsing and
            // interpretation belong to the manager, not the metadata machine.
            if [spec, profile]
                .iter()
                .any(|v| v.is_empty() || v.len() > 64 * 1024 * 1024)
            {
                return err(NestError::Invalid(
                    "invalid serialized module declaration".into(),
                ));
            }
            let current = module(c, name)?.map_or(0, |m| m.revision);
            if current != *expected_revision {
                return err(NestError::Stale);
            }
            let revision = current
                .checked_add(1)
                .filter(|r| *r < i64::MAX as u64)
                .ok_or(Fail::Nest(NestError::Invalid(
                    "module revision exhausted".into(),
                )))?;
            c.execute("INSERT INTO model_modules (name, revision, spec, profile, key, qualification_policy) VALUES (?1, ?2, ?3, ?4, ?5, ?6) ON CONFLICT(name) DO UPDATE SET revision = excluded.revision, spec = excluded.spec, profile = excluded.profile, key = excluded.key, qualification_policy = excluded.qualification_policy",
                params![name, revision as i64, spec, profile, key, qualification_policy])?;
            Ok(Reply::Revision(revision))
        }
        ModelCommand::PublishComplete {
            key,
            owner,
            epoch,
            root,
            files,
            publication,
            export_receipt,
            qualification_receipt,
            qualification_policy,
        } => {
            let a = worker(c, key, owner, *epoch)?;
            token(publication)?;
            token(qualification_policy)?;
            let output = snapshots(c, key)?
                .into_iter()
                .find(|s| s.output)
                .ok_or(Fail::Nest(NestError::Invalid(
                    "no bound output snapshot".into(),
                )))?;
            let mut inventory = files.clone();
            inventory.sort();
            if output.root != *root || output.files != inventory {
                return err(NestError::Invalid(
                    "publication differs from bound OUTPUT inventory".into(),
                ));
            }
            for reference in [export_receipt, qualification_receipt] {
                if !a.files.contains(reference) {
                    return err(NestError::Invalid(
                        "verification receipt is not pinned".into(),
                    ));
                }
                validate_file(c, reference.0, reference.1)?;
            }
            // The legacy publication check also validates all generation pins,
            // retains the first identity and handles a queued reset atomically.
            exec(
                c,
                &ModelCommand::Publish {
                    key: key.clone(),
                    owner: owner.clone(),
                    epoch: *epoch,
                    publication: publication.clone(),
                },
                fx,
            )?;
            c.execute("INSERT INTO model_publications (key, epoch, root, publication, export_file, export_gen, qualification_file, qualification_gen, qualification_policy) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![key, *epoch as i64, root.0 as i64, publication, export_receipt.0.0 as i64, export_receipt.1.0 as i64, qualification_receipt.0.0 as i64, qualification_receipt.1.0 as i64, qualification_policy])?;
            Ok(Reply::Done)
        }
        ModelCommand::ImportOutput {
            key,
            owner,
            epoch,
            parent,
            name,
            file,
            perm,
            size,
            mtime,
            node,
            now,
        } => {
            worker(c, key, owner, *epoch)?;
            // Reuse ordinary import validation and effects inside the SAME
            // outer command savepoint. Recording proof can never happen after
            // a failed import or confer ownership on an existing input.
            let reply = crate::apply::exec(
                c,
                &Command::Import {
                    parent: *parent,
                    name: name.clone(),
                    file: *file,
                    perm: *perm,
                    size: *size,
                    mtime: *mtime,
                    node: *node,
                    sealed: true,
                    now: *now,
                },
                fx,
            )?;
            c.execute("INSERT INTO model_owned_outputs (key, epoch, parent, name, file, gen) VALUES (?1, ?2, ?3, ?4, ?5, 1)", params![key, *epoch as i64, parent.0 as i64, name, file.0 as i64])?;
            Ok(reply)
        }
        ModelCommand::Claim { key, owner } => {
            token(key)?;
            token(owner)?;
            let previous = get(c, key)?;
            if let Some(ref a) = previous {
                if a.state == "preparing" && a.owner.as_deref() == Some(owner) {
                    return Ok(Reply::Model(ModelReply::Epoch(a.epoch)));
                }
                if !matches!(a.state.as_str(), "failed" | "retired") {
                    return err(NestError::Busy("preparation already owned or ready".into()));
                }
            }
            let epoch = previous.map_or(1, |a| a.epoch.saturating_add(1));
            if epoch >= i64::MAX as u64 {
                return err(NestError::Invalid("preparation epoch exhausted".into()));
            }
            c.execute("INSERT INTO model_artifacts (key, epoch, owner, state) VALUES (?1, ?2, ?3, 'preparing') ON CONFLICT(key) DO UPDATE SET epoch = excluded.epoch, owner = excluded.owner, state = 'preparing', reset_pending = 0", params![key, epoch as i64, owner])?;
            Ok(Reply::Model(ModelReply::Epoch(epoch)))
        }
        ModelCommand::Pin {
            key,
            owner,
            epoch,
            files,
        } => {
            worker(c, key, owner, *epoch)?;
            for &(file, generation) in files {
                validate_file(c, file, generation)?;
            }
            for &(file, generation) in files {
                c.execute(
                    "INSERT OR IGNORE INTO model_pins (key, file, gen) VALUES (?1, ?2, ?3)",
                    params![key, file.0 as i64, generation.0 as i64],
                )?;
            }
            Ok(Reply::Done)
        }
        ModelCommand::Publish {
            key,
            owner,
            epoch,
            publication,
        } => {
            token(publication)?;
            let a = worker(c, key, owner, *epoch)?;
            let roots = snapshots(c, key)?;
            if !roots.is_empty() && !roots.iter().any(|r| r.output) {
                return err(NestError::Invalid("no bound output snapshot".into()));
            }
            if a.files.is_empty() {
                return err(NestError::Invalid("no pinned output generations".into()));
            }
            for &(file, generation) in &a.files {
                validate_file(c, file, generation)?;
            }
            if a.publication.as_ref().is_some_and(|p| p != publication) {
                return err(NestError::Invalid("NON_REPRODUCIBLE_PREPARATION".into()));
            }
            c.execute("UPDATE model_artifacts SET state = 'ready', owner = NULL, publication = ?2 WHERE key = ?1", params![key, publication])?;
            if a.reset_pending {
                retire(c, key)?;
            }
            Ok(Reply::Done)
        }
        ModelCommand::Stop {
            key,
            owner,
            epoch,
            receipt,
        } => {
            let a = worker(c, key, owner, *epoch)?;
            token(receipt)?;
            c.execute(
                "INSERT INTO model_receipts (key, epoch, receipt) VALUES (?1, ?2, ?3)",
                params![key, *epoch as i64, receipt],
            )?;
            c.execute(
                "UPDATE model_artifacts SET state = 'failed', owner = NULL WHERE key = ?1",
                [key],
            )?;
            release_pins(c, key)?;
            if a.reset_pending {
                retire(c, key)?;
            }
            Ok(Reply::Done)
        }
        ModelCommand::Use { key, epoch, lease } => {
            token(lease)?;
            let a = entry(c, key)?;
            if a.epoch != *epoch {
                return err(NestError::Stale);
            }
            if a.state == "unavailable" {
                return err(NestError::Unavailable(
                    "prepared artifact lost a required replica".into(),
                ));
            }
            if a.state != "ready" || a.reset_pending {
                return err(NestError::Busy(
                    "artifact is not available for new users".into(),
                ));
            }
            for &(file, generation) in &a.files {
                validate_file(c, file, generation)?;
            }
            let inserted = c.execute(
                "INSERT OR IGNORE INTO model_users (key, lease) VALUES (?1, ?2)",
                params![key, lease],
            )?;
            if inserted == 0 {
                return err(NestError::Busy(
                    "usage identity was already acquired; use a fresh unique token".into(),
                ));
            }
            Ok(Reply::Done)
        }
        ModelCommand::Release { key, epoch, lease } => {
            let a = entry(c, key)?;
            if a.epoch != *epoch || !matches!(a.state.as_str(), "ready" | "unavailable") {
                return err(NestError::Stale);
            }
            c.execute(
                "UPDATE model_users SET released = 1 WHERE key = ?1 AND lease = ?2",
                params![key, lease],
            )?;
            if a.reset_pending && a.leases.iter().all(|l| l == lease) {
                retire(c, key)?;
            }
            Ok(Reply::Done)
        }
        ModelCommand::Reset {
            key,
            when_idle,
            epoch,
        } => {
            let a = entry(c, key)?;
            if a.epoch != *epoch {
                return err(NestError::Stale);
            }
            let busy = a.state == "preparing" || !a.leases.is_empty();
            if busy && !when_idle {
                return err(NestError::Busy(
                    "artifact has a preparation owner or usage lease".into(),
                ));
            }
            if let Some(id) = a.reset_id {
                return Ok(Reply::Model(ModelReply::Reset(id)));
            }
            let sequence: i64 = c.query_row(
                "SELECT reset_seq FROM model_artifacts WHERE key = ?1",
                [key],
                |r| r.get(0),
            )?;
            let id = sequence
                .checked_add(1)
                .filter(|n| *n > 0)
                .ok_or(Fail::Nest(NestError::Invalid(
                    "reset sequence exhausted".into(),
                )))?;
            c.execute(
                "UPDATE model_artifacts SET reset_seq = ?2, reset_pending = ?3 WHERE key = ?1",
                params![key, id, if busy { id } else { 0 }],
            )?;
            if !busy {
                retire(c, key)?;
            }
            Ok(Reply::Model(ModelReply::Reset(id as u64)))
        }
        ModelCommand::CancelReset { key, epoch, reset } => {
            let a = entry(c, key)?;
            if a.epoch != *epoch || a.reset_id != Some(*reset) {
                return err(NestError::Stale);
            }
            c.execute(
                "UPDATE model_artifacts SET reset_pending = 0 WHERE key = ?1",
                [key],
            )?;
            Ok(Reply::Done)
        }
        ModelCommand::BindSnapshot {
            key,
            owner,
            epoch,
            root,
            output,
            files,
        } => {
            worker(c, key, owner, *epoch)?;
            let directory = query::getattr(c, *root)?.ok_or(Fail::Nest(NestError::NotFound))?;
            if *root == FileId::ROOT
                || directory.kind != FileKind::Directory
                || directory.nlink == 0
                || files.is_empty()
            {
                return err(NestError::Invalid(
                    "snapshot requires a named nonempty flat directory".into(),
                ));
            }
            let mut names = BTreeSet::new();
            for (name, file, generation) in files {
                nest_types::validate_name(name)?;
                if !names.insert(name) || query::lookup(c, *root, name)? != Some(*file) {
                    return err(NestError::Invalid(
                        "snapshot names differ from manifest".into(),
                    ));
                }
                validate_file(c, *file, *generation)?;
            }
            let count: i64 = c.query_row(
                "SELECT count(*) FROM dentries WHERE parent = ?1",
                [root.0 as i64],
                |r| r.get(0),
            )?;
            if count as usize != files.len() {
                return err(NestError::Invalid(
                    "snapshot manifest does not cover every entry".into(),
                ));
            }
            if let Some(previous) = snapshots(c, key)?.iter().find(|r| r.root == *root) {
                if previous.output != *output {
                    return err(NestError::Invalid("snapshot role already bound".into()));
                }
                return Ok(Reply::Done);
            }
            if *output && snapshots(c, key)?.iter().any(|r| r.output) {
                return err(NestError::Exists);
            }
            c.execute(
                "INSERT INTO model_snapshots (key, root, output) VALUES (?1, ?2, ?3)",
                params![key, root.0 as i64, output],
            )?;
            for (name, file, generation) in files {
                c.execute("INSERT INTO model_snapshot_files (key, root, name, file, gen) VALUES (?1, ?2, ?3, ?4, ?5)", params![key, root.0 as i64, name, file.0 as i64, generation.0 as i64])?;
                c.execute(
                    "INSERT OR IGNORE INTO model_pins (key, file, gen) VALUES (?1, ?2, ?3)",
                    params![key, file.0 as i64, generation.0 as i64],
                )?;
            }
            Ok(Reply::Done)
        }
    }
}
