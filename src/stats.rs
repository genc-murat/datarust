use std::collections::HashMap;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

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

pub fn column_variance(data: &[Vec<f64>], ddof: usize) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }
    let cols = data[0].len();
    let n = data.len();
    let means = column_mean(data);
    #[cfg(feature = "rayon")]
    {
        (0..cols)
            .into_par_iter()
            .map(|j| {
                let m = means[j];
                let s: f64 = data.iter().map(|r| (r[j] - m).powi(2)).sum();
                s / (n - ddof) as f64
            })
            .collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        (0..cols)
            .map(|j| {
                let m = means[j];
                let s: f64 = data.iter().map(|r| (r[j] - m).powi(2)).sum();
                s / (n - ddof) as f64
            })
            .collect()
    }
}

pub fn column_std(data: &[Vec<f64>], ddof: usize) -> Vec<f64> {
    column_variance(data, ddof)
        .iter()
        .map(|v| v.sqrt())
        .collect()
}

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

pub fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    assert!(!sorted.is_empty(), "median of empty slice");
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Quantile with linear interpolation, matching numpy's default ("linear") method.
/// `q` must be in [0, 1].
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    assert!(n >= 1, "quantile of empty slice");
    assert!((0.0..=1.0).contains(&q), "quantile q out of [0,1]: {}", q);
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

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
                col.sort_by(|a, b| a.partial_cmp(b).unwrap());
                quantile(&col, q)
            })
            .collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        (0..cols)
            .map(|j| {
                let mut col: Vec<f64> = data.iter().map(|r| r[j]).collect();
                col.sort_by(|a, b| a.partial_cmp(b).unwrap());
                quantile(&col, q)
            })
            .collect()
    }
}

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
                col.sort_by(|a, b| a.partial_cmp(b).unwrap());
                median_sorted(&col)
            })
            .collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        (0..cols)
            .map(|j| {
                let mut col: Vec<f64> = data.iter().map(|r| r[j]).collect();
                col.sort_by(|a, b| a.partial_cmp(b).unwrap());
                median_sorted(&col)
            })
            .collect()
    }
}

/// Most frequent value. Ties broken by smallest value (deterministic).
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
                best.unwrap().1
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
                best.unwrap().1
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

/// Covariance matrix (p × p) for an n × p data matrix.
///
/// `ddof=0` gives population covariance, `ddof=1` gives sample covariance.
#[allow(clippy::needless_range_loop)]
pub fn covariance_matrix(data: &[Vec<f64>], ddof: usize) -> Vec<Vec<f64>> {
    if data.is_empty() {
        return vec![];
    }
    let p = data[0].len();
    let n = data.len() as f64;
    let means = column_mean(data);
    let mut cov = vec![vec![0.0; p]; p];
    // centered data — materialised for simpler indexing
    let centered: Vec<Vec<f64>> = data
        .iter()
        .map(|row| row.iter().enumerate().map(|(j, &v)| v - means[j]).collect())
        .collect();
    for i in 0..p {
        for j in i..p {
            let mut s = 0.0;
            for row in &centered {
                s += row[i] * row[j];
            }
            let v = s / (n - ddof as f64);
            cov[i][j] = v;
            cov[j][i] = v;
        }
    }
    cov
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
        assert!((quantile(&s, 0.5) - 2.0).abs() < 1e-12);
        assert!((quantile(&s, 0.25) - 1.0).abs() < 1e-12);
        assert!((quantile(&s, 0.75) - 3.0).abs() < 1e-12);
        // q=0.3 -> 0.3*4 = 1.2 -> interp between idx1 and idx2
        assert!((quantile(&s, 0.3) - 1.2).abs() < 1e-12);
    }

    #[test]
    fn quantile_edge() {
        let s = [5.0_f64];
        assert!((quantile(&s, 0.5) - 5.0).abs() < 1e-12);
        assert!((quantile(&s, 0.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn median_even_odd() {
        assert!((median_sorted(&[1.0_f64, 2.0, 3.0]) - 2.0).abs() < 1e-12);
        assert!((median_sorted(&[1.0_f64, 2.0, 3.0, 4.0]) - 2.5).abs() < 1e-12);
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
