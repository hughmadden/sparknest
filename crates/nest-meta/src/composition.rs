//! Replicated preparation ownership and immutable artifact references (ADR-072).
//! Payload verification belongs to workers; these commands fence publication
//! and deletion. No wall-clock expiry can prove that a mapped model is dead.

use crate::apply::{Fail, R, err};
use crate::{Reply, query};
use nest_types::{FileId, FileKind, GenState, Generation, NestError, ReplicaState};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

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
    c.execute("DELETE FROM model_pins WHERE key = ?1", [key])?;
    Ok(())
}

pub(crate) fn exec(c: &Connection, command: &ModelCommand) -> R<Reply> {
    match command {
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
            c.execute("DELETE FROM model_pins WHERE key = ?1", [key])?;
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
    }
}
