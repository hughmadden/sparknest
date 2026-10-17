//! Round-trip statistics for the Raft network (ADR-057): per peer and per
//! kind of request, a smoothed round trip, the slowest in the last window,
//! and counts of answers, errors, timeouts and unreachable peers. Kept
//! cheap: one short lock per request.

use nest_types::RttStats;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// The maximum is over the current and the previous window of this length.
const WINDOW: Duration = Duration::from_secs(10);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// Append-entries without entries.
    Heartbeat,
    Append,
    /// Votes and pre-votes.
    Vote,
    Snapshot,
    Transfer,
}

const KINDS: usize = 5;

impl Kind {
    fn slot(self) -> usize {
        match self {
            Kind::Heartbeat => 0,
            Kind::Append => 1,
            Kind::Vote => 2,
            Kind::Snapshot => 3,
            Kind::Transfer => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// An answer (whatever Raft said in it).
    Ok,
    /// An error answer, or one that did not decode.
    Error,
    /// No answer in time (ours, or openraft dropped the request).
    Timeout,
    /// Not sent: no connection to the peer.
    Unreachable,
}

#[derive(Default)]
struct Rtt {
    s: RttStats,
    window_start: Option<Instant>,
    cur_max: u64,
    prev_max: u64,
}

impl Rtt {
    fn roll(&mut self, now: Instant) {
        match self.window_start {
            Some(t) if now.duration_since(t) < WINDOW => {}
            Some(t) => {
                self.prev_max = if now.duration_since(t) < 2 * WINDOW {
                    self.cur_max
                } else {
                    0
                };
                self.cur_max = 0;
                self.window_start = Some(now);
            }
            None => self.window_start = Some(now),
        }
    }

    fn record(&mut self, now: Instant, outcome: Outcome, took: Duration) {
        match outcome {
            Outcome::Ok => {
                let us = took.as_micros().min(u64::MAX as u128) as u64;
                self.s.ewma_us = if self.s.ok == 0 {
                    us
                } else {
                    let e = self.s.ewma_us as i64;
                    (e + (us as i64 - e) / 8) as u64
                };
                self.s.ok += 1;
                self.s.last_us = us;
                self.roll(now);
                self.cur_max = self.cur_max.max(us);
            }
            Outcome::Error => self.s.errors += 1,
            Outcome::Timeout => self.s.timeouts += 1,
            Outcome::Unreachable => self.s.unreachable += 1,
        }
    }

    fn snapshot(&self, now: Instant) -> RttStats {
        let max = match self.window_start {
            Some(t) if now.duration_since(t) < WINDOW => self.cur_max.max(self.prev_max),
            Some(t) if now.duration_since(t) < 2 * WINDOW => self.cur_max,
            _ => 0,
        };
        RttStats {
            max_10s_us: max,
            ..self.s.clone()
        }
    }
}

/// Every peer's round trips, shared by all of openraft's network clients.
#[derive(Default)]
pub struct RaftStats {
    peers: Mutex<HashMap<u64, [Rtt; KINDS]>>,
}

/// Per-kind snapshot for one peer: heartbeat, append, vote, snapshot,
/// transfer.
pub(crate) type PeerRtts = [RttStats; KINDS];

impl RaftStats {
    pub(crate) fn record(&self, peer: u64, kind: Kind, outcome: Outcome, took: Duration) {
        let now = Instant::now();
        self.peers.lock().entry(peer).or_default()[kind.slot()].record(now, outcome, took);
    }

    pub(crate) fn peer(&self, peer: u64) -> Option<PeerRtts> {
        let now = Instant::now();
        self.peers
            .lock()
            .get(&peer)
            .map(|r| std::array::from_fn(|i| r[i].snapshot(now)))
    }

    pub(crate) fn peers(&self) -> Vec<u64> {
        self.peers.lock().keys().copied().collect()
    }

    /// Start timing one request; a request dropped before
    /// [`Timing::finish`] (openraft's own timeout) counts as a timeout.
    pub(crate) fn time(&self, peer: u64, kind: Kind) -> Timing<'_> {
        Timing {
            stats: self,
            peer,
            kind,
            start: Instant::now(),
            done: false,
        }
    }
}

pub(crate) struct Timing<'a> {
    stats: &'a RaftStats,
    peer: u64,
    kind: Kind,
    start: Instant,
    done: bool,
}

impl Timing<'_> {
    pub(crate) fn finish(mut self, outcome: Outcome) {
        self.done = true;
        self.stats
            .record(self.peer, self.kind, outcome, self.start.elapsed());
    }
}

impl Drop for Timing<'_> {
    fn drop(&mut self) {
        if !self.done {
            self.stats
                .record(self.peer, self.kind, Outcome::Timeout, self.start.elapsed());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ewma_max_and_counts() {
        let s = RaftStats::default();
        s.record(2, Kind::Heartbeat, Outcome::Ok, Duration::from_micros(800));
        for _ in 0..20 {
            s.record(2, Kind::Heartbeat, Outcome::Ok, Duration::from_micros(100));
        }
        s.record(2, Kind::Heartbeat, Outcome::Timeout, Duration::ZERO);
        s.record(2, Kind::Vote, Outcome::Unreachable, Duration::ZERO);
        s.record(2, Kind::Vote, Outcome::Error, Duration::ZERO);
        let [hb, append, vote, ..] = s.peer(2).unwrap();
        assert_eq!(hb.ok, 21);
        assert_eq!(hb.timeouts, 1);
        assert_eq!(hb.last_us, 100);
        assert_eq!(hb.max_10s_us, 800);
        assert!(hb.ewma_us < 200, "{}", hb.ewma_us);
        assert_eq!(append, RttStats::default());
        assert_eq!((vote.unreachable, vote.errors, vote.ok), (1, 1, 0));
        assert!(s.peer(3).is_none());
    }

    #[test]
    fn dropped_request_is_a_timeout() {
        let s = RaftStats::default();
        drop(s.time(4, Kind::Append));
        s.time(4, Kind::Append).finish(Outcome::Ok);
        let [_, append, ..] = s.peer(4).unwrap();
        assert_eq!((append.timeouts, append.ok), (1, 1));
    }

    #[test]
    fn max_ages_out() {
        let mut r = Rtt::default();
        let t0 = Instant::now();
        r.record(t0, Outcome::Ok, Duration::from_micros(900));
        assert_eq!(r.snapshot(t0).max_10s_us, 900);
        // Next window: the previous one's maximum still counts.
        let t1 = t0 + WINDOW + Duration::from_secs(1);
        r.record(t1, Outcome::Ok, Duration::from_micros(10));
        assert_eq!(r.snapshot(t1).max_10s_us, 900);
        // Two windows on: only recent answers.
        let t2 = t1 + WINDOW + Duration::from_secs(1);
        r.record(t2, Outcome::Ok, Duration::from_micros(20));
        assert_eq!(r.snapshot(t2).max_10s_us, 20);
        assert_eq!(r.snapshot(t2 + 3 * WINDOW).max_10s_us, 0);
    }
}
