//! Column-wise statistics, covariance and correlation helpers.

use std::collections::HashMap;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Returns the mean of each column.
pub fn column_mean(data: &[Vec<f64>]) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }
    let cols = data[0].len();
    let n = data.len() as f64;
    #[cfg(feature = "rayon")]
    {
        (0..cols)
            .into_par_iter()
            .map(|j| {
                let s: f64 = data.iter().map(|r| r[j]).sum();
                s / n
            })
            .collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        (0..cols)
            .map(|j| {
                let s: f64 = data.iter().map(|r| r[j]).sum();
                s / n
            })
            .collect()
    }
}

/// Returns the variance of each column using the given delta degrees of freedom.
pub fn column_variance(data: &[Vec<f64>], ddof: usize) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }
    let cols = data[0].len();
    let n = data.len();
    // Guard against a non-positive denominator (ddof >= n); fall back to NaN
    // rather than producing +/-inf and propagating it through downstream transforms.
    let denom = (n - ddof) as f64;
    let means = column_mean(data);
    #[cfg(feature = "rayon")]
    {
        (0..cols)
            .into_par_iter()
            .map(|j| {
                let m = means[j];
                let s: f64 = data.iter().map(|r| (r[j] - m).powi(2)).sum();
                if denom > 0.0 {
                    s / denom
                } else {
                    f64::NAN
                }
            })
            .collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        (0..cols)
            .map(|j| {
                let m = means[j];
                let s: f64 = data.iter().map(|r| (r[j] - m).powi(2)).sum();
                if denom > 0.0 {
                    s / denom
                } else {
                    f64::NAN
                }
            })
            .collect()
    }
}

/// Returns the standard deviation of each column using the given delta degrees of freedom.
pub fn column_std(data: &[Vec<f64>], ddof: usize) -> Vec<f64> {
    column_variance(data, ddof)
        .iter()
        .map(|v| v.sqrt())
        .collect()
}

/// Returns the minimum value of each column.
pub fn column_min(data: &[Vec<f64>]) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }
    let cols = data[0].len();
    #[cfg(feature = "rayon")]
    {
        (0..cols)
            .into_par_iter()
            .map(|j| data.iter().map(|r| r[j]).fold(f64::INFINITY, f64::min))
            .collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        (0..cols)
            .map(|j| data.iter().map(|r| r[j]).fold(f64::INFINITY, f64::min))
            .collect()
    }
}

/// Returns the maximum value of each column.
pub fn column_max(data: &[Vec<f64>]) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }
    let cols = data[0].len();
    #[cfg(feature = "rayon")]
    {
        (0..cols)
            .into_par_iter()
            .map(|j| data.iter().map(|r| r[j]).fold(f64::NEG_INFINITY, f64::max))
            .collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        (0..cols)
            .map(|j| data.iter().map(|r| r[j]).fold(f64::NEG_INFINITY, f64::max))
            .collect()
    }
}

/// Median of a slice that is assumed to be sorted in non-decreasing order.
///
/// Returns `None` for an empty slice instead of panicking. Callers that can
/// guarantee a non-empty slice may safely [`Option::unwrap`] the result.
pub fn median_sorted(sorted: &[f64]) -> Option<f64> {
    let n = sorted.len();
    if n == 0 {
        return None;
    }
    if n % 2 == 1 {
        Some(sorted[n / 2])
    } else {
        Some((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0)
    }
}

/// Quantile with linear interpolation, matching numpy's default ("linear") method.
///
/// Returns `None` if the slice is empty or if `q` is outside `[0, 1]`.
pub fn quantile(sorted: &[f64], q: f64) -> Option<f64> {
    let n = sorted.len();
    if n == 0 || !(0.0..=1.0).contains(&q) {
        return None;
    }
    if n == 1 {
        return Some(sorted[0]);
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return Some(sorted[lo]);
    }
    let frac = pos - lo as f64;
    Some(sorted[lo] * (1.0 - frac) + sorted[hi] * frac)
}

/// Returns the requested quantile of each column using linear interpolation.
pub fn quantile_column(data: &[Vec<f64>], q: f64) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }
    let cols = data[0].len();
    #[cfg(feature = "rayon")]
    {
        (0..cols)
            .into_par_iter()
            .map(|j| {
                let mut col: Vec<f64> = data.iter().map(|r| r[j]).collect();
                col.sort_by(|a, b| a.total_cmp(b));
                // INVARIANT: `data` is non-empty (checked above), so each column has
                // >= 1 element and `q` validity is the caller's contract.
                quantile(&col, q).expect("non-empty column with q in [0,1]")
            })
            .collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        (0..cols)
            .map(|j| {
                let mut col: Vec<f64> = data.iter().map(|r| r[j]).collect();
                col.sort_by(|a, b| a.total_cmp(b));
                quantile(&col, q).expect("non-empty column with q in [0,1]")
            })
            .collect()
    }
}

/// Returns the median of each column.
pub fn median_column(data: &[Vec<f64>]) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }
    let cols = data[0].len();
    #[cfg(feature = "rayon")]
    {
        (0..cols)
            .into_par_iter()
            .map(|j| {
                let mut col: Vec<f64> = data.iter().map(|r| r[j]).collect();
                col.sort_by(|a, b| a.total_cmp(b));
                // INVARIANT: `data` is non-empty (checked above), so each column is non-empty.
                median_sorted(&col).expect("non-empty column")
            })
            .collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        (0..cols)
            .map(|j| {
                let mut col: Vec<f64> = data.iter().map(|r| r[j]).collect();
                col.sort_by(|a, b| a.total_cmp(b));
                median_sorted(&col).expect("non-empty column")
            })
            .collect()
    }
}

/// Most frequent value. Ties broken by smallest value (deterministic).
///
/// A column of all-equal (or empty) entries yields [`f64::NAN`] for that column.
pub fn mode_column(data: &[Vec<f64>]) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }
    let cols = data[0].len();
    #[cfg(feature = "rayon")]
    {
        (0..cols)
            .into_par_iter()
            .map(|j| {
                let mut counts: HashMap<u64, (usize, f64)> = HashMap::new();
                for r in data {
                    let key = r[j].to_bits();
                    let entry = counts.entry(key).or_insert((0, r[j]));
                    entry.0 += 1;
                    entry.1 = r[j];
                }
                let mut best: Option<(usize, f64)> = None;
                for (_, (cnt, val)) in counts {
                    match best {
                        None => best = Some((cnt, val)),
                        Some((bc, bv)) => {
                            if cnt > bc || (cnt == bc && val < bv) {
                                best = Some((cnt, val));
                            }
                        }
                    }
                }
                best.map(|(_, v)| v).unwrap_or(f64::NAN)
            })
            .collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        (0..cols)
            .map(|j| {
                let mut counts: HashMap<u64, (usize, f64)> = HashMap::new();
                for r in data {
                    let key = r[j].to_bits();
                    let entry = counts.entry(key).or_insert((0, r[j]));
                    entry.0 += 1;
                    entry.1 = r[j];
                }
                let mut best: Option<(usize, f64)> = None;
                for (_, (cnt, val)) in counts {
                    match best {
                        None => best = Some((cnt, val)),
                        Some((bc, bv)) => {
                            if cnt > bc || (cnt == bc && val < bv) {
                                best = Some((cnt, val));
                            }
                        }
                    }
                }
                best.map(|(_, v)| v).unwrap_or(f64::NAN)
            })
            .collect()
    }
}

/// Sum of each column.
pub fn column_sum(data: &[Vec<f64>]) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }
    let cols = data[0].len();
    let mut sums = vec![0.0; cols];
    for row in data {
        for (j, &v) in row.iter().enumerate() {
            sums[j] += v;
        }
    }
    sums
}

/// Covariance of already-centered data: `(1/(n-ddof)) * Xcᵀ Xc`.
///
/// This is the single canonical centered-covariance routine shared by
/// [`covariance_matrix`] (raw-data entry point), PCA and Truncated SVD.
/// `x_centered` is row-major `n × p`. A non-positive denominator (`ddof >= n`)
/// leaves the scale unchanged rather than producing infinities.
#[allow(clippy::needless_range_loop)]
pub(crate) fn covariance_centered(x_centered: &[Vec<f64>], ddof: usize) -> Vec<Vec<f64>> {
    let n = x_centered.len();
    let p = if n > 0 { x_centered[0].len() } else { 0 };
    let mut cov = vec![vec![0.0; p]; p];
    for row in x_centered {
        for i in 0..p {
            let xi = row[i];
            if xi == 0.0 {
                continue;
            }
            for j in 0..p {
                cov[i][j] += xi * row[j];
            }
        }
    }
    let denom = (n - ddof) as f64;
    if denom > 0.0 {
        let inv = 1.0 / denom;
        for i in 0..p {
            for j in 0..p {
                cov[i][j] *= inv;
            }
        }
    }
    cov
}

/// Covariance matrix (p × p) for an n × p data matrix.
///
/// `ddof=0` gives population covariance, `ddof=1` gives sample covariance.
#[allow(clippy::needless_range_loop)]
pub fn covariance_matrix(data: &[Vec<f64>], ddof: usize) -> Vec<Vec<f64>> {
    if data.is_empty() {
        return vec![];
    }
    let means = column_mean(data);
    // Center the data, then delegate to the shared centered-covariance routine.
    let centered: Vec<Vec<f64>> = data
        .iter()
        .map(|row| row.iter().enumerate().map(|(j, &v)| v - means[j]).collect())
        .collect();
    covariance_centered(&centered, ddof)
}

/// Pearson correlation matrix (p × p).
pub fn correlation_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if data.is_empty() {
        return vec![];
    }
    let p = data[0].len();
    let cov = covariance_matrix(data, 1);
    let std: Vec<f64> = (0..p).map(|j| cov[j][j].sqrt()).collect();
    let mut corr = vec![vec![0.0; p]; p];
    for i in 0..p {
        for j in i..p {
            let v = if std[i] == 0.0 || std[j] == 0.0 {
                if i == j {
                    1.0
                } else {
                    0.0
                }
            } else {
                cov[i][j] / (std[i] * std[j])
            };
            corr[i][j] = v;
            corr[j][i] = v;
        }
    }
    corr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_basic() {
        let data = vec![vec![1.0, 10.0], vec![3.0, 20.0], vec![5.0, 30.0]];
        let m = column_mean(&data);
        assert!((m[0] - 3.0).abs() < 1e-12);
        assert!((m[1] - 20.0).abs() < 1e-12);
    }

    #[test]
    fn variance_ddof() {
        let data = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let t = transpose(&data);
        let v0 = column_variance(&t, 0);
        let v1 = column_variance(&t, 1);
        // population variance = 1.25 ; sample = 1.666...
        assert!((v0[0] - 1.25).abs() < 1e-12);
        assert!((v1[0] - (5.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn quantile_linear() {
        // numpy quantile linear for [0,1,2,3,4]: q=0.5 -> 2, q=0.25 -> 1, q=0.75 -> 3
        let s = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        assert!((quantile(&s, 0.5).unwrap() - 2.0).abs() < 1e-12);
        assert!((quantile(&s, 0.25).unwrap() - 1.0).abs() < 1e-12);
        assert!((quantile(&s, 0.75).unwrap() - 3.0).abs() < 1e-12);
        // q=0.3 -> 0.3*4 = 1.2 -> interp between idx1 and idx2
        assert!((quantile(&s, 0.3).unwrap() - 1.2).abs() < 1e-12);
    }

    #[test]
    fn quantile_edge() {
        let s = [5.0_f64];
        assert!((quantile(&s, 0.5).unwrap() - 5.0).abs() < 1e-12);
        assert!((quantile(&s, 0.0).unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn quantile_none_cases() {
        assert!(quantile(&[], 0.5).is_none());
        assert!(quantile(&[1.0, 2.0], 1.5).is_none());
        assert!(quantile(&[1.0, 2.0], -0.1).is_none());
        assert!(median_sorted(&[]).is_none());
    }

    #[test]
    fn median_even_odd() {
        assert!((median_sorted(&[1.0_f64, 2.0, 3.0]).unwrap() - 2.0).abs() < 1e-12);
        assert!((median_sorted(&[1.0_f64, 2.0, 3.0, 4.0]).unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn mode_simple() {
        let data = vec![vec![1.0], vec![2.0], vec![2.0], vec![3.0]];
        let m = mode_column(&data);
        assert!((m[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn mode_tie_smallest() {
        // tie between 1.0 and 2.0 -> smallest wins
        let data = vec![vec![1.0], vec![2.0], vec![1.0], vec![2.0]];
        let m = mode_column(&data);
        assert!((m[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn min_max() {
        let data = vec![vec![3.0, -1.0], vec![5.0, 2.0], vec![1.0, 0.0]];
        let mn = column_min(&data);
        let mx = column_max(&data);
        assert!((mn[0] - 1.0).abs() < 1e-12);
        assert!((mx[0] - 5.0).abs() < 1e-12);
        assert!((mn[1] - -1.0).abs() < 1e-12);
    }

    fn transpose(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if data.is_empty() {
            return vec![];
        }
        let cols = data[0].len();
        (0..cols)
            .map(|j| data.iter().map(|r| r[j]).collect())
            .collect()
    }

    #[test]
    fn column_sum_basic() {
        let data = vec![vec![1.0, 10.0], vec![3.0, 20.0], vec![5.0, 30.0]];
        let s = column_sum(&data);
        assert!((s[0] - 9.0).abs() < 1e-12);
        assert!((s[1] - 60.0).abs() < 1e-12);
    }

    #[test]
    fn covariance_matrix_identity() {
        let data = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
        ];
        // col0: [1,0,-1,0] mean=0, var(ddof=1) = (1+0+1+0)/3 = 2/3
        // col1: [0,1,0,-1] mean=0, var(ddof=1) = 2/3
        let cov = covariance_matrix(&data, 1);
        assert!((cov[0][0] - 2.0 / 3.0).abs() < 1e-9);
        assert!((cov[1][1] - 2.0 / 3.0).abs() < 1e-9);
        assert!((cov[0][1]).abs() < 1e-9);
    }

    #[test]
    fn covariance_matrix_hand_computed() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        // mean = [3, 4]; centered = [[-2,-2],[0,0],[2,2]]
        // cov[0][0] = (4+0+4)/2 = 4; cov[1][1] = (4+0+4)/2 = 4; cov[0][1] = (4+0+4)/2 = 4
        let cov = covariance_matrix(&data, 1);
        assert!((cov[0][0] - 4.0).abs() < 1e-9);
        assert!((cov[1][1] - 4.0).abs() < 1e-9);
        assert!((cov[0][1] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn covariance_population() {
        let data = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let t = transpose(&data);
        let cov = covariance_matrix(&t, 0);
        // population variance of [1,2,3,4] = 1.25
        assert!((cov[0][0] - 1.25).abs() < 1e-12);
    }

    #[test]
    fn correlation_matrix_identity() {
        let data = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
        ];
        let corr = correlation_matrix(&data);
        assert!((corr[0][0] - 1.0).abs() < 1e-9);
        assert!((corr[1][1] - 1.0).abs() < 1e-9);
        assert!((corr[0][1]).abs() < 1e-9);
    }

    #[test]
    fn correlation_perfect_positive() {
        // col1 = 2 * col0 => perfect correlation
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let corr = correlation_matrix(&data);
        assert!((corr[0][1] - 1.0).abs() < 1e-9);
        assert!((corr[0][0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn correlation_constant_column() {
        let data = vec![vec![1.0, 10.0], vec![2.0, 10.0], vec![3.0, 10.0]];
        let corr = correlation_matrix(&data);
        // col1 is constant -> std=0 -> correlation with it is 0
        assert!((corr[0][1]).abs() < 1e-9);
        assert!((corr[1][0]).abs() < 1e-9);
        assert!((corr[1][1] - 1.0).abs() < 1e-9);
    }
}
