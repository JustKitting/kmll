use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfileDuration(Duration);

impl ProfileDuration {
    pub const ZERO: Self = Self(Duration::ZERO);

    pub const fn from_duration(duration: Duration) -> Self {
        Self(duration)
    }

    pub fn from_seconds_f64(seconds: f64) -> Option<Self> {
        if seconds.is_finite() && seconds >= 0.0 {
            Some(Self(Duration::from_secs_f64(seconds)))
        } else {
            None
        }
    }

    pub const fn as_duration(self) -> Duration {
        self.0
    }

    pub fn as_seconds_f64(self) -> f64 {
        self.0.as_secs_f64()
    }
}

#[derive(Debug, Clone)]
pub struct ProfileTimer {
    started_at: Instant,
}

impl ProfileTimer {
    pub fn start() -> Self {
        Self {
            started_at: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> ProfileDuration {
        ProfileDuration::from_duration(self.started_at.elapsed())
    }

    pub fn elapsed_seconds(&self) -> f64 {
        self.elapsed().as_seconds_f64()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GenerationTimings {
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
}

impl GenerationTimings {
    pub const fn new(prefill_seconds: f64, decode_seconds: f64) -> Self {
        Self {
            prefill_seconds,
            decode_seconds,
        }
    }

    pub fn from_durations(prefill: ProfileDuration, decode: ProfileDuration) -> Self {
        Self::new(prefill.as_seconds_f64(), decode.as_seconds_f64())
    }

    pub fn total_seconds(self) -> f64 {
        self.prefill_seconds + self.decode_seconds
    }

    pub fn decode_tokens_per_second(self, generated_token_count: usize) -> Option<f64> {
        tokens_per_second(generated_token_count, self.decode_seconds)
    }

    pub fn total_tokens_per_second(self, generated_token_count: usize) -> Option<f64> {
        tokens_per_second(generated_token_count, self.total_seconds())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleStats {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
}

impl SampleStats {
    pub fn from_finite_samples(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
            return None;
        }

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let count = sorted.len();
        let sum = sorted.iter().sum::<f64>();
        let mean = sum / count as f64;
        let median = median_sorted(&sorted);
        let min = sorted[0];
        let max = sorted[count - 1];

        Some(Self {
            count,
            mean,
            median,
            min,
            max,
        })
    }
}

pub fn rate_per_second(event_count: usize, seconds: f64) -> Option<f64> {
    if seconds.is_finite() && seconds > 0.0 {
        Some(event_count as f64 / seconds)
    } else {
        None
    }
}

pub fn tokens_per_second(token_count: usize, seconds: f64) -> Option<f64> {
    rate_per_second(token_count, seconds)
}

pub fn tokens_per_second_or_zero(token_count: usize, seconds: f64) -> f64 {
    tokens_per_second(token_count, seconds).unwrap_or(0.0)
}

pub fn median_f64(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return None;
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    Some(median_sorted(&sorted))
}

fn median_sorted(sorted: &[f64]) -> f64 {
    debug_assert!(!sorted.is_empty());
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_rejects_invalid_elapsed_time() {
        assert_eq!(tokens_per_second(10, 0.0), None);
        assert_eq!(tokens_per_second(10, -1.0), None);
        assert_eq!(tokens_per_second(10, f64::NAN), None);
        assert_eq!(tokens_per_second(10, f64::INFINITY), None);
    }

    #[test]
    fn rate_counts_zero_events_over_positive_time() {
        assert_eq!(tokens_per_second(0, 2.0), Some(0.0));
    }

    #[test]
    fn generation_timings_report_decode_and_total_rates() {
        let timings = GenerationTimings::new(1.0, 2.0);
        assert_eq!(timings.total_seconds(), 3.0);
        assert_eq!(timings.decode_tokens_per_second(8), Some(4.0));
        assert_eq!(timings.total_tokens_per_second(9), Some(3.0));
    }

    #[test]
    fn median_accepts_unsorted_finite_samples() {
        assert_eq!(median_f64(&[5.0, 1.0, 3.0]), Some(3.0));
        assert_eq!(median_f64(&[10.0, 2.0, 4.0, 8.0]), Some(6.0));
    }

    #[test]
    fn sample_stats_reject_empty_and_non_finite_samples() {
        assert_eq!(SampleStats::from_finite_samples(&[]), None);
        assert_eq!(SampleStats::from_finite_samples(&[1.0, f64::NAN]), None);
    }

    #[test]
    fn sample_stats_compute_basic_summary() {
        let stats = SampleStats::from_finite_samples(&[4.0, 1.0, 7.0]).unwrap();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.mean, 4.0);
        assert_eq!(stats.median, 4.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 7.0);
    }
}
