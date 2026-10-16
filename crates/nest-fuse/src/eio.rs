//! Every EIO the mount returns is logged (ADR-056), without flooding the
//! journal when a model load fails thousands of reads at once: the first
//! failure of each (file, reason) in a window is logged in full, the rest
//! are counted and summarized when the window closes.

use parking_lot::Mutex;
use std::collections::HashSet;
use std::time::{Duration, Instant};

/// How long a (file, reason) stays quiet after it was logged, and how often
/// the suppressed ones are summarized.
const WINDOW: Duration = Duration::from_secs(10);

/// Distinct (file, reason) pairs remembered per window, so a storm over
/// many files cannot grow memory without bound.
const MAX_SEEN: usize = 4096;

pub(crate) struct EioLog {
    st: Mutex<State>,
}

struct State {
    started: Instant,
    seen: HashSet<(u64, String)>,
    suppressed: u64,
    files: HashSet<u64>,
}

/// What to do with one EIO.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    /// Log it in full; `summary` reports the previous window's suppressed
    /// failures (count, distinct files), if any.
    Log { summary: Option<(u64, usize)> },
    /// Seen in this window: counted only.
    Quiet,
}

impl EioLog {
    pub(crate) fn new() -> EioLog {
        EioLog {
            st: Mutex::new(State {
                started: Instant::now(),
                seen: HashSet::new(),
                suppressed: 0,
                files: HashSet::new(),
            }),
        }
    }

    pub(crate) fn note(&self, file: u64, reason: &str, now: Instant) -> Verdict {
        let mut st = self.st.lock();
        let mut summary = None;
        if now.duration_since(st.started) >= WINDOW {
            if st.suppressed > 0 {
                summary = Some((st.suppressed, st.files.len()));
            }
            st.started = now;
            st.seen.clear();
            st.suppressed = 0;
            st.files.clear();
        }
        let key = (file, reason.to_string());
        if st.seen.contains(&key) || st.seen.len() >= MAX_SEEN {
            st.suppressed += 1;
            if st.files.len() < MAX_SEEN {
                st.files.insert(file);
            }
            return Verdict::Quiet;
        }
        st.seen.insert(key);
        Verdict::Log { summary }
    }

    /// Suppressed failures in a window that has closed with no new EIO to
    /// report them, taken once (called periodically).
    pub(crate) fn flush(&self, now: Instant) -> Option<(u64, usize)> {
        let mut st = self.st.lock();
        if now.duration_since(st.started) < WINDOW || st.suppressed == 0 {
            return None;
        }
        let out = (st.suppressed, st.files.len());
        st.started = now;
        st.seen.clear();
        st.suppressed = 0;
        st.files.clear();
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_storm_logs_once_per_file_and_reason_then_summarizes() {
        let log = EioLog::new();
        let t0 = Instant::now();
        assert_eq!(log.note(7, "no copy", t0), Verdict::Log { summary: None });
        for _ in 0..5000 {
            assert_eq!(log.note(7, "no copy", t0), Verdict::Quiet);
        }
        // Another file, or another reason, is news.
        assert_eq!(log.note(8, "no copy", t0), Verdict::Log { summary: None });
        assert_eq!(log.note(7, "lease", t0), Verdict::Log { summary: None });
        assert_eq!(log.note(8, "no copy", t0), Verdict::Quiet);
        // The next window reports what was held back.
        let t1 = t0 + WINDOW;
        assert_eq!(
            log.note(7, "no copy", t1),
            Verdict::Log {
                summary: Some((5001, 2))
            }
        );
        assert_eq!(log.flush(t1 + WINDOW), None);
    }

    #[test]
    fn suppressed_failures_are_reported_even_if_the_storm_ends() {
        let log = EioLog::new();
        let t0 = Instant::now();
        log.note(1, "x", t0);
        log.note(1, "x", t0);
        log.note(1, "x", t0);
        assert_eq!(log.flush(t0), None, "window still open");
        assert_eq!(log.flush(t0 + WINDOW), Some((2, 1)));
        assert_eq!(log.flush(t0 + WINDOW * 2), None, "reported once");
    }

    #[test]
    fn memory_stays_bounded_over_many_files() {
        let log = EioLog::new();
        let t0 = Instant::now();
        for f in 0..(MAX_SEEN as u64 * 3) {
            log.note(f, "x", t0);
        }
        let st = log.st.lock();
        assert!(st.seen.len() <= MAX_SEEN && st.files.len() <= MAX_SEEN);
        assert_eq!(st.suppressed, MAX_SEEN as u64 * 2);
    }
}
