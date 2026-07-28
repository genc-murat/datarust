//! Distributional statistics not provided by `datarust::stats`.
//!
//! These are *interpretation* primitives — skewness, kurtosis, and histogram
//! binning — that a profiling tool needs but which don't belong in the
//! numerical-kernel layer of `datarust`. Everything here is pure Rust with no
//! dependencies, NaN-safe (callers filter non-finite values first), and built
//! on top of `mean`/`std`/sorted-slice inputs that the caller already has.

use super::column::Histogram;

/// Number of histogram bins suggested by Sturges' rule:
/// `bins = ceil(log2(n) + 1)`, with a floor of 1.
///
/// Sturges' rule assumes an approximately normal distribution and is the
/// pandas/matplotlib default bin selector. It is cheap to compute and behaves
/// well for the small-to-medium tables this crate targets; for highly skewed
/// data, Freedman–Diaconis would be tighter, but that is a future option.
pub fn sturges_bins(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let raw = (n as f64).log2() + 1.0;
    // ceil, but guard against floating-point overshoot of an integer.
    let b = raw.ceil() as usize;
    b.max(1)
}

/// Fisher–Pearson standardized third moment (sample skewness).
///
/// Uses the biased (population) third central moment divided by the cube of
/// the sample standard deviation, matching the definition used by pandas'
/// `Series.skew` with the default bias correction. Returns `0.0` when the
/// standard deviation is non-positive or fewer than three values are present
/// (skewness is undefined in both cases).
pub fn skewness(values: &[f64], mean: f64, std: f64) -> f64 {
    let n = values.len();
    if n < 3 || !std.is_finite() || std <= 0.0 {
        return 0.0;
    }
    let mut m3 = 0.0;
    for &x in values {
        let d = x - mean;
        m3 += d * d * d;
    }
    m3 /= n as f64;
    let denom = std.powi(3);
    if denom <= 0.0 {
        0.0
    } else {
        m3 / denom
    }
}

/// Excess (Fisher) kurtosis: the fourth standardized moment minus 3.
///
/// A normal distribution has excess kurtosis 0. Positive values indicate a
/// leptokurtic (heavy-tailed, peaked) distribution; negative values indicate a
/// platykurtic (light-tailed, flat) one. Returns `0.0` when the standard
/// deviation is non-positive or fewer than four values are present.
pub fn kurtosis_excess(values: &[f64], mean: f64, std: f64) -> f64 {
    let n = values.len();
    if n < 4 || !std.is_finite() || std <= 0.0 {
        return 0.0;
    }
    let mut m4 = 0.0;
    for &x in values {
        let d = x - mean;
        let d2 = d * d;
        m4 += d2 * d2;
    }
    m4 /= n as f64;
    let denom = std.powi(4);
    if denom <= 0.0 {
        0.0
    } else {
        m4 / denom - 3.0
    }
}

/// Builds an equal-width histogram over `sorted` using Sturges' rule for the
/// bin count.
///
/// `min` and `max` delimit the outer bin edges; they are normally `sorted[0]`
/// and `sorted[n-1]` (the five-number summary's extremes). When the range is
/// zero (constant column) or there is at most one value, a single bin holding
/// all values is returned.
pub fn histogram(sorted: &[f64], min: f64, max: f64) -> Histogram {
    let n = sorted.len();
    if n == 0 {
        return Histogram {
            edges: vec![],
            counts: vec![],
        };
    }
    if n == 1 || min == max {
        // Degenerate range: one bin spanning [min, max] holds everything.
        return Histogram {
            edges: vec![min, max.max(min)],
            counts: vec![n],
        };
    }

    let bins = sturges_bins(n);
    let width = (max - min) / bins as f64;
    let mut edges = Vec::with_capacity(bins + 1);
    for b in 0..=bins {
        edges.push(min + width * b as f64);
    }
    // Ensure the last edge lands exactly on max (avoid float drift excluding
    // the largest value from the final bin).
    edges[bins] = max;

    let mut counts = vec![0usize; bins];
    // `sorted` is ascending; walk it once, assigning each value to a bin.
    let mut bin = 0usize;
    for &x in sorted {
        // Advance bins until x fits. The final bin is inclusive of max.
        while bin + 1 < bins && x >= edges[bin + 1] {
            bin += 1;
        }
        counts[bin] += 1;
    }

    Histogram { edges, counts }
}

/// Counts values outside the Tukey IQR fences.
///
/// Returns `(count, fraction)` where `count` is the number of `sorted` values
/// strictly below `q1 - 1.5 * iqr` or strictly above `q3 + 1.5 * iqr`, and
/// `fraction` is `count / sorted.len()` (or `0.0` when empty). `iqr` is taken
/// as `q3 - q1`; the caller supplies the already-computed quartiles.
pub fn outlier_count(sorted: &[f64], q1: f64, q3: f64) -> (usize, f64) {
    let n = sorted.len();
    if n == 0 {
        return (0, 0.0);
    }
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;
    // sorted is ascending: count from the left while < lower, from the right
    // while > upper. A single pass would also work, but the two-pointer scan
    // avoids touching the middle bulk.
    let mut lo = 0usize;
    while lo < n && sorted[lo] < lower {
        lo += 1;
    }
    let mut hi = n;
    while hi > lo && sorted[hi - 1] > upper {
        hi -= 1;
    }
    let count = lo + (n - hi);
    (count, count as f64 / n as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sturges_bins_known_values() {
        assert_eq!(sturges_bins(0), 1);
        assert_eq!(sturges_bins(1), 1);
        // log2(8)+1 = 4 exactly.
        assert_eq!(sturges_bins(8), 4);
        // log2(1024)+1 = 11.
        assert_eq!(sturges_bins(1024), 11);
    }

    #[test]
    fn skewness_symmetric_is_near_zero() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = datarust::stats::mean(&v);
        let std = datarust::stats::std(&v, 1);
        let s = skewness(&v, mean, std);
        assert!(s.abs() < 1e-9, "symmetric skewness should be ~0, got {s}");
    }

    #[test]
    fn skewness_right_skewed_is_positive() {
        let v = vec![1.0, 1.0, 1.0, 1.0, 100.0];
        let mean = datarust::stats::mean(&v);
        let std = datarust::stats::std(&v, 1);
        let s = skewness(&v, mean, std);
        assert!(
            s > 0.0,
            "right-skewed data should have positive skewness, got {s}"
        );
    }

    #[test]
    fn skewness_too_few_values_is_zero() {
        let v = vec![1.0, 2.0];
        let s = skewness(&v, 1.5, 0.5);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn kurtosis_uniform_is_negative() {
        // A discrete uniform distribution is platykurtic (excess < 0).
        let v: Vec<f64> = (0..10).map(|x| x as f64).collect();
        let mean = datarust::stats::mean(&v);
        let std = datarust::stats::std(&v, 1);
        let k = kurtosis_excess(&v, mean, std);
        assert!(k < 0.0, "uniform should be platykurtic, got {k}");
    }

    #[test]
    fn kurtosis_peaked_is_positive() {
        // Tight central cluster with rare but moderate tails → leptokurtic.
        // (Sparse extreme tails inflate variance faster than the 4th moment,
        // which actually drives excess kurtosis *down*; a peaked shape needs
        // the mass concentrated near the mean.)
        let v = vec![
            -1.0, -0.5, -0.2, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 1.0, -3.0, 3.0,
        ];
        let mean = datarust::stats::mean(&v);
        let std = datarust::stats::std(&v, 1);
        let k = kurtosis_excess(&v, mean, std);
        assert!(k > 0.0, "peaked data should be leptokurtic, got {k}");
    }

    #[test]
    fn kurtosis_too_few_values_is_zero() {
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(kurtosis_excess(&v, 2.0, 1.0), 0.0);
    }

    #[test]
    fn histogram_six_values_uses_sturges_bins() {
        let mut sorted: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        sorted.sort_by(|a, b| a.total_cmp(b));
        let h = histogram(&sorted, 0.0, 5.0);
        // Sturges for n=6: ceil(log2(6)+1) = ceil(3.585) = 4 bins.
        assert_eq!(h.counts.len(), 4);
        assert_eq!(h.edges.len(), 5);
        // All values accounted for.
        assert_eq!(h.counts.iter().sum::<usize>(), 6);
        // Edges span [0, 5].
        assert!((h.edges.first().copied().unwrap_or(f64::NAN) - 0.0).abs() < 1e-9);
        assert!((h.edges.last().copied().unwrap_or(f64::NAN) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn histogram_constant_column_single_bin() {
        let sorted = vec![7.0, 7.0, 7.0];
        let h = histogram(&sorted, 7.0, 7.0);
        assert_eq!(h.counts, vec![3]);
    }

    #[test]
    fn histogram_empty_is_empty() {
        let h = histogram(&[], 0.0, 0.0);
        assert!(h.counts.is_empty());
        assert!(h.edges.is_empty());
    }

    #[test]
    fn outlier_count_detects_extremes() {
        // Q1 ≈ 1.5, Q3 ≈ 3.5 for [0,1,2,3,4,100] → 100 is well above the fence.
        let sorted = vec![0.0, 1.0, 2.0, 3.0, 4.0, 100.0];
        let q1 = datarust::stats::quantile(&sorted, 0.25).unwrap();
        let q3 = datarust::stats::quantile(&sorted, 0.75).unwrap();
        let (count, frac) = outlier_count(&sorted, q1, q3);
        assert!(count >= 1, "expected at least one outlier, got {count}");
        assert!((count as f64 / 6.0 - frac).abs() < 1e-9);
    }

    #[test]
    fn outlier_count_clean_data_is_zero() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let q1 = datarust::stats::quantile(&sorted, 0.25).unwrap();
        let q3 = datarust::stats::quantile(&sorted, 0.75).unwrap();
        let (count, _) = outlier_count(&sorted, q1, q3);
        assert_eq!(count, 0);
    }
}
