use nest_types::NodeId;
use parking_lot::RwLock;
use std::collections::HashSet;
use std::sync::Arc;

/// Reachability control used by tests to simulate partitions. A blocked
/// peer is unreachable in both directions from this node's point of view:
/// outgoing calls fail immediately and incoming requests are dropped.
///
/// `cut` and `mute` act on the primary (fabric) connections only, leaving
/// a standby connection (ADR-057) working: a cut primary is closed and
/// refuses to be dialled (a pulled cable, a switch that is off), a muted
/// one stays open but nothing arrives on it (a switch dropping everything).
/// `mute_lan` does the latter to the standby connections.
#[derive(Clone, Default)]
pub struct Filter {
    blocked: Arc<RwLock<HashSet<NodeId>>>,
    cut: Arc<RwLock<HashSet<NodeId>>>,
    muted: Arc<RwLock<HashSet<NodeId>>>,
    lan_muted: Arc<RwLock<HashSet<NodeId>>>,
}

impl Filter {
    pub fn block(&self, peer: NodeId) {
        self.blocked.write().insert(peer);
    }
    pub fn unblock(&self, peer: NodeId) {
        self.blocked.write().remove(&peer);
    }
    /// Clear every block, cut and mute.
    pub fn unblock_all(&self) {
        self.blocked.write().clear();
        self.cut.write().clear();
        self.muted.write().clear();
        self.lan_muted.write().clear();
    }
    pub fn is_blocked(&self, peer: NodeId) -> bool {
        self.blocked.read().contains(&peer)
    }
    /// Close the primary connections to `peer` and refuse to dial it there.
    pub fn cut(&self, peer: NodeId) {
        self.cut.write().insert(peer);
    }
    pub fn uncut(&self, peer: NodeId) {
        self.cut.write().remove(&peer);
    }
    pub fn is_cut(&self, peer: NodeId) -> bool {
        self.cut.read().contains(&peer)
    }
    /// Drop every frame `peer` sends on a primary connection.
    pub fn mute(&self, peer: NodeId) {
        self.muted.write().insert(peer);
    }
    pub fn unmute(&self, peer: NodeId) {
        self.muted.write().remove(&peer);
    }
    pub fn is_muted(&self, peer: NodeId) -> bool {
        self.muted.read().contains(&peer)
    }
    /// Drop every frame `peer` sends on a standby (LAN) connection.
    pub fn mute_lan(&self, peer: NodeId) {
        self.lan_muted.write().insert(peer);
    }
    pub fn unmute_lan(&self, peer: NodeId) {
        self.lan_muted.write().remove(&peer);
    }
    pub fn is_lan_muted(&self, peer: NodeId) -> bool {
        self.lan_muted.read().contains(&peer)
    }
}
