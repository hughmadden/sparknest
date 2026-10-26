//! Composition command transport. Preparation and lifecycle rules belong to the daemon.
use crate::Client;
use anyhow::{Context, Result, bail};
use clap::Subcommand;
use hyper::Method;
use serde_json::{Value, json};
use std::path::PathBuf;

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Register a declarative module and its trusted complete operand profile.
    Put {
        name: String,
        #[arg(long)]
        module: PathBuf,
        #[arg(long)]
        profile: PathBuf,
        /// 0 creates a module; updates require its current revision.
        #[arg(long)]
        expected_revision: u64,
    },
    /// Inspect one module, or list all registered modules.
    Status { name: Option<String> },
    /// Prepare immutable files on the explicitly assigned CPU worker.
    Prepare {
        name: String,
        #[arg(long)]
        wait: bool,
    },
    /// Return a qualified READY snapshot's immutable path and revision.
    Resolve { name: String },
    /// Request qualification through the configured trusted verifier.
    Qualify { name: String },
    /// Inspect and retire composition cache entries, preserving source data.
    Cache {
        #[command(subcommand)]
        cmd: Cache,
    },
    /// Explicit preparation worker stop and recovery.
    Jobs {
        #[command(subcommand)]
        cmd: Jobs,
    },
    /// Acquire an artifact usage lease before loading; tokens are single-use.
    Acquire {
        name: String,
        #[arg(long)]
        lease: String,
    },
    /// Release only after all of the lease's model containers/ranks have stopped.
    Release {
        key: String,
        #[arg(long)]
        epoch: u64,
        #[arg(long)]
        lease: String,
    },
}

#[derive(Debug, Subcommand)]
pub(crate) enum Cache {
    Status {
        name: Option<String>,
    },
    Reset {
        #[arg(required_unless_present = "all")]
        name: Option<String>,
        #[arg(long, conflicts_with = "name")]
        all: bool,
        #[arg(long)]
        when_idle: bool,
        #[arg(long)]
        dry_run: bool,
    },
    ResetCancel {
        name: String,
        #[arg(long)]
        epoch: u64,
        #[arg(long)]
        reset: u64,
    },
}

#[derive(Debug, Subcommand)]
pub(crate) enum Jobs {
    Show {
        name: String,
    },
    /// Stop this preparation's exact worker, without releasing model usage leases.
    Stop {
        name: String,
    },
    /// Reconcile the exact owner's retained worker evidence.
    Recover {
        name: String,
    },
}

fn path(name: &str) -> String {
    format!("/v1/model/{}", crate::enc(name).replace('/', "%2F"))
}

pub(crate) async fn run(client: &Client, command: &Command, raw: bool) -> Result<()> {
    let value = match command {
        Command::Put {
            name,
            module,
            profile,
            expected_revision,
        } => {
            // These are declarative files on the client, never checkpoint code.
            let text = tokio::fs::read_to_string(module)
                .await
                .with_context(|| format!("reading {}", module.display()))?;
            let spec = nest_model::ModuleConfig::parse(&text)
                .context("parsing composition module TOML")?;
            let profile =
                nest_model::worker::OperandProfile::parse(&tokio::fs::read(profile).await?)
                    .context("parsing operand profile JSON")?;
            client.call(Method::PUT, &path(name), Some(json!({"spec": spec, "profile": profile, "expected_revision": expected_revision}))).await?
        }
        Command::Status { name } => match name {
            Some(name) => client.get(&path(name)).await?,
            None => client.get("/v1/model").await?,
        },
        Command::Cache {
            cmd: Cache::Status { name },
        } => match name {
            Some(name) => client.get(&path(name)).await?,
            None => client.get("/v1/model/cache/status").await?,
        },
        Command::Prepare { name, wait } => {
            client
                .post(&format!("{}/prepare", path(name)), json!({"wait":wait}))
                .await?
        }
        Command::Resolve { name } => client.get(&format!("{}/resolve", path(name))).await?,
        Command::Qualify { name } => {
            client
                .post(&format!("{}/qualify", path(name)), json!({}))
                .await?
        }
        Command::Cache {
            cmd:
                Cache::Reset {
                    name,
                    all,
                    when_idle,
                    dry_run,
                },
        } => {
            let endpoint = match (name, all) {
                (Some(name), false) => format!("{}/cache/reset", path(name)),
                (None, true) => "/v1/model/cache/reset".into(),
                _ => bail!("select a module NAME or --all"),
            };
            client
                .post(&endpoint, json!({"when_idle":when_idle,"dry_run":dry_run}))
                .await?
        }
        Command::Cache {
            cmd: Cache::ResetCancel { name, epoch, reset },
        } => {
            client
                .post(
                    &format!("{}/cache/reset-cancel", path(name)),
                    json!({"epoch":epoch,"reset":reset}),
                )
                .await?
        }
        Command::Jobs {
            cmd: Jobs::Show { name },
        } => client.get(&path(name)).await?,
        Command::Jobs {
            cmd: Jobs::Stop { name },
        } => {
            client
                .post(&format!("{}/jobs/stop", path(name)), json!({}))
                .await?
        }
        Command::Jobs {
            cmd: Jobs::Recover { name },
        } => {
            client
                .post(&format!("{}/jobs/recover", path(name)), json!({}))
                .await?
        }
        Command::Acquire { name, lease } => {
            client
                .post(&format!("{}/use", path(name)), json!({"lease":lease}))
                .await?
        }
        Command::Release { key, epoch, lease } => {
            client
                .post(
                    "/v1/model/usage/release",
                    json!({"key":key,"epoch":epoch,"lease":lease}),
                )
                .await?
        }
    };
    if raw {
        println!("{}", serde_json::to_string_pretty(&value)?);
    } else {
        print(&value);
    }
    Ok(())
}

fn print(value: &Value) {
    if let Some(items) = value.as_array() {
        for item in items {
            print(item);
        }
        return;
    }
    if let Some(path) = value["path"].as_str() {
        println!("{path}");
        println!("revision: {}", value["revision"].as_str().unwrap_or("?"));
        println!("preparation: {}", value["key"].as_str().unwrap_or("?"));
        return;
    }
    if let Some(name) = value["name"].as_str() {
        println!(
            "{}  revision {}  {}",
            name,
            value["revision"],
            value["phase"].as_str().unwrap_or("unknown")
        );
        println!("preparation: {}", value["key"].as_str().unwrap_or("?"));
        if let Some(message) = value["message"].as_str() {
            println!("{message}");
        }
        if let Some(artifact) = value["artifact"].as_object() {
            println!(
                "artifact epoch: {}  state: {}",
                artifact.get("epoch").unwrap_or(&Value::Null),
                artifact
                    .get("state")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
            );
        }
        return;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(value).unwrap_or_default()
    );
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    #[test]
    fn reset_requires_exactly_one_explicit_scope() {
        assert!(crate::Cli::try_parse_from(["nest", "model", "cache", "reset"]).is_err());
        assert!(
            crate::Cli::try_parse_from(["nest", "model", "cache", "reset", "fixture", "--all"])
                .is_err()
        );
        assert!(
            crate::Cli::try_parse_from(["nest", "model", "cache", "reset", "--all", "--dry-run"])
                .is_ok()
        );
        assert!(
            crate::Cli::try_parse_from([
                "nest",
                "model",
                "cache",
                "reset",
                "fixture",
                "--when-idle"
            ])
            .is_ok()
        );
    }
}
