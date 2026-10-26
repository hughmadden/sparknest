//! Offline, synchronous preparation only. Call on a bounded blocking worker.
//!
//! The owner must pin immutable input files for inspection through export and
//! provide a private staging parent with no concurrent writers. This crate does
//! not publish, cache, seal, acquire leases, execute checkpoint code, or qualify
//! a model. Hashes here never participate in filesystem reads or wire metadata.

mod config;
mod indexed;
mod plan;

pub use config::*;
pub use indexed::{ExportLimits, ExportReceipt, FileIdentity, Snapshot, export};
pub use plan::{Plan, Profile, Shadowed, resolve};

use serde::{Deserialize, Serialize, de};
use sha2::{Digest as _, Sha256};
use std::fmt;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid model composition: {0}")]
    Invalid(String),
    #[error("unsupported preparation: {0}")]
    Unsupported(String),
    #[error("preparation resource limit: {0}")]
    Limit(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Toml(#[from] toml::de::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

/// A canonical lowercase SHA-256 content identity, never a path or revision name.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct Digest(String);

impl Digest {
    pub fn of(bytes: &[u8]) -> Self {
        Self::from_hasher(Sha256::new_with_prefix(bytes))
    }

    pub(crate) fn from_hasher(hasher: Sha256) -> Self {
        Self(format!("sha256:{:x}", hasher.finalize()))
    }

    pub(crate) fn canonical<T: Serialize>(value: &T) -> Result<Self> {
        Ok(Self::of(&serde_json::to_vec(value)?))
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D: de::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        if text.len() != 71
            || !text.starts_with("sha256:")
            || !text[7..]
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(de::Error::custom(
                "expected lowercase sha256 content digest",
            ));
        }
        Ok(Self(text))
    }
}
