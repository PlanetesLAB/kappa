//! Plain-text progress lines for batch jobs.

use std::sync::atomic::Ordering;
use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::time::Instant;

/// Reporting interval in seconds.
const HEARTBEAT: u64 = 60;

/// Thread-safe progress over `total` items.
///
/// A line `label: finished/total unit (percent), eta …` is printed when items
/// are finished after [`HEARTBEAT`] seconds since the previous line.
pub struct Progress {
    label: &'static str,
    unit: &'static str,
    total: usize,
    finished: AtomicUsize,
    last: AtomicU64,
    start: Instant,
}

impl Progress {
    #[must_use]
    pub fn new(label: &'static str, unit: &'static str, total: usize) -> Self {
        Self {
            label,
            unit,
            total,
            finished: AtomicUsize::new(0),
            last: AtomicU64::new(0),
            start: Instant::now(),
        }
    }

    /// Records `finished` more items.
    pub fn add(&self, finished: usize) {
        let finished = self.finished.fetch_add(finished, Ordering::Relaxed) + finished;
        let elapsed = self.start.elapsed().as_secs_f64();
        let now = elapsed as u64;
        let last = self.last.load(Ordering::Relaxed);
        if now < last + HEARTBEAT
            || self
                .last
                .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
                .is_err()
        {
            return;
        }
        let eta = duration(self.total.saturating_sub(finished) as f64 * elapsed / finished as f64);
        println!(
            "{}: {finished}/{} {} ({:.0}%), eta {eta}",
            self.label,
            self.total,
            self.unit,
            100.0 * finished as f64 / self.total as f64,
        );
    }
}

/// Whole seconds as `42s`, `3m 05s` or `2h 07m`.
fn duration(seconds: f64) -> String {
    let s = seconds.round() as u64;
    match s {
        0..60 => format!("{s}s"),
        60..3600 => format!("{}m {:02}s", s / 60, s % 60),
        _ => format!("{}h {:02}m", s / 3600, s % 3600 / 60),
    }
}
