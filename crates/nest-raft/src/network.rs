//! openraft network over `nest-rpc` (service RAFT), openraft 0.10 (v2).
//!
//! Snapshots are `meta.sqlite` copies on disk; they travel in chunks and
//! are reassembled into a file under the receiver's snapshot directory
//! before `install_full_snapshot`.

use crate::state_machine::SnapshotFile;
use crate::stats::{Kind, Outcome, RaftStats};
use crate::{Raft, TypeConfig, Vote};
use bytes::Bytes;
use futures::FutureExt;
use futures::future::BoxFuture;
use nest_rpc::{Handler, Rpc, RpcError, service};
use nest_types::NodeId;
use openraft::alias::{SnapshotMetaOf, SnapshotOf};
use openraft::error::{
    NetworkError, RPCError, RaftError, ReplicationClosed, StreamingError, Unreachable,
};
use openraft::network::{RPCOption, RaftNetworkFactory, v2::RaftNetworkV2};
use openraft::raft::{
    AppendEntriesRequest, AppendEntriesResponse, SnapshotResponse, TransferLeaderRequest,
    TransferLeaderResponse, VoteRequest, VoteResponse,
};
use openraft::{BasicNode, OptionalSend};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Snapshot bytes per RPC.
const SNAPSHOT_CHUNK: usize = 4 << 20;

/// Wire format: variants are only ever appended (a host on an older
/// release answers a variant it does not know with a decode error).
#[derive(Serialize, Deserialize)]
enum RaftReq {
    Append(AppendEntriesRequest<TypeConfig>),
    Vote(VoteRequest<TypeConfig>),
    PreVote(VoteRequest<TypeConfig>),
    SnapshotChunk(Chunk),
    /// The leader is handing leadership to `to_node_id` (ADR-057).
    TransferLeader(TransferLeaderRequest<TypeConfig>),
    /// What this host's Raft service understands; a host on 0.4.4 cannot
    /// decode it, which is the answer too.
    Features,
}

/// One piece of a snapshot file in transit.
#[derive(Serialize, Deserialize)]
struct Chunk {
    transfer: u64,
    vote: Vote,
    meta: SnapshotMetaOf<TypeConfig>,
    offset: u64,
    data: Vec<u8>,
    done: bool,
}

/// Every Raft message names the cluster incarnation (the re-found its
/// metadata descends from): hosts from different incarnations must never
/// exchange Raft traffic (ADR-026).
#[derive(Serialize, Deserialize)]
struct Envelope {
    incarnation: Option<String>,
    req: RaftReq,
}

#[derive(Serialize, Deserialize)]
enum RaftResp {
    Append(Result<AppendEntriesResponse<TypeConfig>, RaftError<TypeConfig>>),
    Vote(Result<VoteResponse<TypeConfig>, RaftError<TypeConfig>>),
    /// `None` until the last chunk has been installed.
    SnapshotChunk(Result<Option<SnapshotResponse<TypeConfig>>, String>),
    /// `Err` when this host's Raft has stopped.
    TransferLeader(Result<TransferLeaderResponse<TypeConfig>, String>),
    Features(Features),
}

/// The Raft requests a host's service understands beyond 0.4.4's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Features {
    pub transfer_leader: bool,
}

impl Features {
    pub const HERE: Features = Features {
        transfer_leader: true,
    };
}

/// Serves incoming Raft RPCs into the local Raft instance.
pub struct RaftService {
    raft: Raft,
    incarnation: Option<String>,
    snap_dir: PathBuf,
    incoming: Mutex<HashMap<(NodeId, u64), PathBuf>>,
    legacy: Arc<AtomicBool>,
}

impl RaftService {
    async fn snapshot_chunk(
        self: Arc<Self>,
        peer: NodeId,
        chunk: Chunk,
    ) -> Result<Option<SnapshotResponse<TypeConfig>>, String> {
        let Chunk {
            transfer,
            vote,
            meta,
            offset,
            data,
            done,
        } = chunk;
        let path = {
            let mut m = self.incoming.lock();
            if offset == 0 {
                let p = self
                    .snap_dir
                    .join(format!("incoming-{}-{transfer:016x}.sqlite", peer.0));
                m.insert((peer, transfer), p);
            }
            m.get(&(peer, transfer))
                .cloned()
                .ok_or("snapshot chunk for an unknown transfer")?
        };
        let p = path.clone();
        tokio::task::spawn_blocking(move || -> std::io::Result<()> {
            use std::os::unix::fs::FileExt;
            let f = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(offset == 0)
                .open(&p)?;
            f.write_all_at(&data, offset)?;
            if done {
                f.sync_all()?;
            }
            Ok(())
        })
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
        if !done {
            return Ok(None);
        }
        self.incoming.lock().remove(&(peer, transfer));
        let snapshot = SnapshotOf::<TypeConfig, SnapshotFile> {
            meta,
            snapshot: SnapshotFile(path),
        };
        self.raft
            .install_full_snapshot(vote, snapshot)
            .await
            .map(Some)
            .map_err(|e| e.to_string())
    }
}

struct RaftHandler(Arc<RaftService>);

impl Handler for RaftHandler {
    fn call(&self, peer: NodeId, body: Bytes) -> BoxFuture<'static, Result<Bytes, String>> {
        let me = self.0.clone();
        async move {
            let env: Envelope = nest_rpc::decode(&body).map_err(|e| e.to_string())?;
            if me.legacy.load(Ordering::Relaxed)
                && matches!(env.req, RaftReq::TransferLeader(_) | RaftReq::Features)
            {
                // What a 0.4.4 host answers to a variant it does not know.
                return Err("Found an enum discriminant that was not expected".into());
            }
            if env.incarnation != me.incarnation {
                return Err(format!(
                    "different cluster incarnation ({:?}, here {:?})",
                    env.incarnation, me.incarnation
                ));
            }
            let resp = match env.req {
                RaftReq::Append(r) => RaftResp::Append(me.raft.append_entries(r).await),
                RaftReq::Vote(r) => RaftResp::Vote(me.raft.vote(r).await),
                RaftReq::PreVote(r) => RaftResp::Vote(me.raft.pre_vote(r).await),
                RaftReq::SnapshotChunk(c) => {
                    RaftResp::SnapshotChunk(me.snapshot_chunk(peer, c).await)
                }
                RaftReq::TransferLeader(r) => RaftResp::TransferLeader(
                    me.raft
                        .handle_transfer_leader(r)
                        .await
                        .map_err(|e| e.to_string()),
                ),
                RaftReq::Features => RaftResp::Features(Features::HERE),
            };
            nest_rpc::encode(&resp)
                .map(Bytes::from)
                .map_err(|e| e.to_string())
        }
        .boxed()
    }
}

/// Serve the RAFT service. `legacy` set makes it answer the requests 0.4.4
/// does not know as 0.4.4 does (tests of mixed-version clusters).
pub fn register(
    rpc: &Rpc,
    raft: Raft,
    snap_dir: PathBuf,
    incarnation: Option<String>,
    legacy: Arc<AtomicBool>,
) {
    rpc.register(
        service::RAFT,
        Arc::new(RaftHandler(Arc::new(RaftService {
            raft,
            incarnation,
            snap_dir,
            incoming: Mutex::new(HashMap::new()),
            legacy,
        }))),
    );
}

pub struct NetworkFactory {
    pub rpc: Rpc,
    pub incarnation: Option<String>,
    pub stats: Arc<RaftStats>,
}

impl RaftNetworkFactory<TypeConfig> for NetworkFactory {
    type Network = Network;

    async fn new_client(&mut self, target: u64, node: &BasicNode) -> Network {
        if let Ok(addr) = node.addr.parse() {
            self.rpc.set_peer(NodeId(target), addr);
        }
        Network {
            rpc: self.rpc.clone(),
            target,
            incarnation: self.incarnation.clone(),
            stats: self.stats.clone(),
        }
    }
}

pub struct Network {
    rpc: Rpc,
    target: u64,
    incarnation: Option<String>,
    stats: Arc<RaftStats>,
}

fn net_err(e: &(impl std::error::Error + 'static)) -> RPCError<TypeConfig> {
    RPCError::Network(NetworkError::new(e))
}

/// One Raft request to `target`, timed into `stats`.
async fn call(
    rpc: &Rpc,
    stats: &RaftStats,
    incarnation: Option<String>,
    target: u64,
    kind: Kind,
    req: RaftReq,
    timeout: Duration,
) -> Result<RaftResp, RPCError<TypeConfig>> {
    let env = Envelope { incarnation, req };
    let body = nest_rpc::encode(&env).map_err(|e| net_err(&e))?;
    // Dropped by openraft before the answer: recorded as a timeout.
    let timing = stats.time(target, kind);
    let r = rpc
        .call(NodeId(target), service::RAFT, body.into(), timeout)
        .await;
    let (outcome, r) = match r {
        Ok(b) => match nest_rpc::decode::<RaftResp>(&b) {
            Ok(resp) => (Outcome::Ok, Ok(resp)),
            Err(e) => (Outcome::Error, Err(net_err(&e))),
        },
        Err(e @ RpcError::Unreachable(..)) => (
            Outcome::Unreachable,
            Err(RPCError::Unreachable(Unreachable::new(&e))),
        ),
        Err(e @ RpcError::Timeout(_)) => (Outcome::Timeout, Err(net_err(&e))),
        Err(e) => (Outcome::Error, Err(net_err(&e))),
    };
    timing.finish(outcome);
    r
}

/// Ask `target` which Raft requests it understands. `None`: no answer, or
/// a host on 0.4.4 (it cannot decode the question).
pub(crate) async fn features(
    rpc: &Rpc,
    incarnation: Option<String>,
    target: NodeId,
    timeout: Duration,
) -> Option<Features> {
    let env = Envelope {
        incarnation,
        req: RaftReq::Features,
    };
    let body = nest_rpc::encode(&env).ok()?;
    let b = rpc
        .call(target, service::RAFT, body.into(), timeout)
        .await
        .ok()?;
    match nest_rpc::decode::<RaftResp>(&b).ok()? {
        RaftResp::Features(f) => Some(f),
        _ => None,
    }
}

impl Network {
    async fn send(
        &self,
        kind: Kind,
        req: RaftReq,
        timeout: Duration,
    ) -> Result<RaftResp, RPCError<TypeConfig>> {
        call(
            &self.rpc,
            &self.stats,
            self.incarnation.clone(),
            self.target,
            kind,
            req,
            timeout,
        )
        .await
    }

    async fn vote_like(
        &self,
        req: RaftReq,
        option: &RPCOption,
    ) -> Result<VoteResponse<TypeConfig>, RPCError<TypeConfig>> {
        match self.send(Kind::Vote, req, ttl(option)).await? {
            RaftResp::Vote(r) => r.map_err(|e| net_err(&e)),
            _ => Err(unexpected()),
        }
    }
}

fn ttl(option: &RPCOption) -> Duration {
    option.hard_ttl().max(Duration::from_millis(50))
}

fn unexpected() -> RPCError<TypeConfig> {
    net_err(&std::io::Error::other("mismatched raft response"))
}

impl RaftNetworkV2<TypeConfig> for Network {
    type SnapshotData = SnapshotFile;

    async fn append_entries(
        &mut self,
        rpc: AppendEntriesRequest<TypeConfig>,
        option: RPCOption,
    ) -> Result<AppendEntriesResponse<TypeConfig>, RPCError<TypeConfig>> {
        let kind = if rpc.entries.is_empty() {
            Kind::Heartbeat
        } else {
            Kind::Append
        };
        match self.send(kind, RaftReq::Append(rpc), ttl(&option)).await? {
            RaftResp::Append(r) => r.map_err(|e| net_err(&e)),
            _ => Err(unexpected()),
        }
    }

    async fn transfer_leader(
        &mut self,
        req: TransferLeaderRequest<TypeConfig>,
        option: RPCOption,
    ) -> Result<TransferLeaderResponse<TypeConfig>, RPCError<TypeConfig>> {
        // A host on 0.4.4 answers with a decode error: it keeps its lease
        // until the new leader's vote (marked as a transfer) arrives.
        match self
            .send(Kind::Transfer, RaftReq::TransferLeader(req), ttl(&option))
            .await?
        {
            RaftResp::TransferLeader(Ok(r)) => Ok(r),
            RaftResp::TransferLeader(Err(e)) => Err(net_err(&std::io::Error::other(e))),
            _ => Err(unexpected()),
        }
    }

    async fn vote(
        &mut self,
        rpc: VoteRequest<TypeConfig>,
        option: RPCOption,
    ) -> Result<VoteResponse<TypeConfig>, RPCError<TypeConfig>> {
        self.vote_like(RaftReq::Vote(rpc), &option).await
    }

    async fn pre_vote(
        &mut self,
        rpc: VoteRequest<TypeConfig>,
        option: RPCOption,
    ) -> Result<VoteResponse<TypeConfig>, RPCError<TypeConfig>> {
        self.vote_like(RaftReq::PreVote(rpc), &option).await
    }

    async fn full_snapshot(
        &mut self,
        vote: Vote,
        snapshot: SnapshotOf<TypeConfig, SnapshotFile>,
        cancel: impl Future<Output = ReplicationClosed> + OptionalSend + 'static,
        option: RPCOption,
    ) -> Result<SnapshotResponse<TypeConfig>, StreamingError<TypeConfig>> {
        let transfer: u64 = rand::random();
        let path = snapshot.snapshot.0.clone();
        let meta = snapshot.meta.clone();
        let send_all = async {
            let len = tokio::fs::metadata(&path)
                .await
                .map_err(|e| StreamingError::Network(NetworkError::new(&e)))?
                .len();
            let mut offset = 0u64;
            loop {
                let p = path.clone();
                let data = tokio::task::spawn_blocking(move || -> std::io::Result<Vec<u8>> {
                    use std::os::unix::fs::FileExt;
                    let f = std::fs::File::open(&p)?;
                    let n = ((len - offset) as usize).min(SNAPSHOT_CHUNK);
                    let mut buf = vec![0u8; n];
                    f.read_exact_at(&mut buf, offset)?;
                    Ok(buf)
                })
                .await
                .map_err(|e| StreamingError::Network(NetworkError::new(&e)))?
                .map_err(|e| StreamingError::Network(NetworkError::new(&e)))?;
                let n = data.len() as u64;
                let done = offset + n >= len;
                let req = RaftReq::SnapshotChunk(Chunk {
                    transfer,
                    vote,
                    meta: meta.clone(),
                    offset,
                    data,
                    done,
                });
                let resp = self
                    .send(
                        Kind::Snapshot,
                        req,
                        ttl(&option).max(Duration::from_secs(30)),
                    )
                    .await
                    .map_err(|e| match e {
                        RPCError::Unreachable(u) => StreamingError::Unreachable(u),
                        RPCError::Network(n) => StreamingError::Network(n),
                        RPCError::Timeout(t) => StreamingError::Timeout(t),
                        other => StreamingError::Network(NetworkError::new(&other)),
                    })?;
                match resp {
                    RaftResp::SnapshotChunk(Ok(Some(r))) => return Ok(r),
                    RaftResp::SnapshotChunk(Ok(None)) if !done => offset += n,
                    RaftResp::SnapshotChunk(Err(e)) => {
                        return Err(StreamingError::Network(NetworkError::new(
                            &std::io::Error::other(e),
                        )));
                    }
                    _ => {
                        return Err(StreamingError::Network(NetworkError::new(
                            &std::io::Error::other("mismatched snapshot response"),
                        )));
                    }
                }
            }
        };
        tokio::select! {
            r = send_all => r,
            closed = cancel => Err(StreamingError::Closed(closed)),
        }
    }
}

#[cfg(test)]
mod wire {
    use super::*;

    /// `RaftReq` and `Envelope` as 0.4.4 has them.
    #[derive(Serialize, Deserialize)]
    #[allow(dead_code)]
    enum RaftReq044 {
        Append(AppendEntriesRequest<TypeConfig>),
        Vote(VoteRequest<TypeConfig>),
        PreVote(VoteRequest<TypeConfig>),
        SnapshotChunk(Chunk),
    }
    #[derive(Serialize, Deserialize)]
    struct Envelope044 {
        incarnation: Option<String>,
        req: RaftReq044,
    }

    fn enc(req: RaftReq) -> Vec<u8> {
        nest_rpc::encode(&Envelope {
            incarnation: Some("r1".into()),
            req,
        })
        .unwrap()
    }

    #[test]
    fn variants_are_appended() {
        // What 0.4.4 sends still decodes here, in place.
        let v = VoteRequest::<TypeConfig>::new(Vote::new(3, 2), None);
        let old = nest_rpc::encode(&Envelope044 {
            incarnation: None,
            req: RaftReq044::PreVote(v.clone()),
        })
        .unwrap();
        let env: Envelope = nest_rpc::decode(&old).unwrap();
        assert!(matches!(env.req, RaftReq::PreVote(r) if r == v));
        let old = nest_rpc::encode(&Envelope044 {
            incarnation: Some("r1".into()),
            req: RaftReq044::Vote(v.clone()),
        })
        .unwrap();
        assert_eq!(enc(RaftReq::Vote(v)), old);
    }

    /// A host on 0.4.4 fails to decode the new requests: its handler
    /// answers with that error, which the sender treats as "cannot".
    #[test]
    fn a_0_4_4_host_cannot_decode_the_new_requests() {
        let tl = TransferLeaderRequest::<TypeConfig>::new(Vote::new_committed(4, 1), 2, None);
        for b in [
            enc(RaftReq::TransferLeader(tl.clone())),
            enc(RaftReq::Features),
        ] {
            assert!(nest_rpc::decode::<Envelope044>(&b).is_err());
        }
        let env: Envelope = nest_rpc::decode(&enc(RaftReq::TransferLeader(tl.clone()))).unwrap();
        assert!(matches!(env.req, RaftReq::TransferLeader(r) if r == tl));
        let resp = nest_rpc::encode(&RaftResp::Features(Features::HERE)).unwrap();
        assert!(matches!(
            nest_rpc::decode::<RaftResp>(&resp).unwrap(),
            RaftResp::Features(f) if f.transfer_leader
        ));
    }
}
