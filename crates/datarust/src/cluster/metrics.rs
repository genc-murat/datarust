//! Clustering evaluation metrics.
//!
//! Provides [`silhouette_score`](crate::cluster::metrics::silhouette_score),
//! mirroring `sklearn.metrics.silhouette_score`.
//! These metrics assess clustering quality without ground-truth labels, using
//! only the feature matrix and the predicted cluster assignments.

use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;

/// Squared Euclidean distance between two equal-length rows.
#[inline]
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| {
            let d = ai - bi;
            d * d
        })
        .sum()
}

/// Mean silhouette coefficient over all samples.
///
/// For each sample `i`, the silhouette coefficient is `(b − a) / max(a, b)`,
/// where:
/// - `a` is the mean intra-cluster distance (mean distance from `i` to all
///   other points in its own cluster),
/// - `b` is the mean nearest-cluster distance (the smallest mean distance from
///   `i` to all points in any other cluster).
///
/// Returns a value in `[−1, 1]`: values near `1` indicate well-separated
/// clusters, near `0` indicate overlapping clusters, and negative values
/// indicate samples assigned to the wrong cluster.
///
/// Mirrors `sklearn.metrics.silhouette_score` with metric = Euclidean.
///
/// ```rust
/// use datarust::cluster::metrics::silhouette_score;
/// use datarust::Matrix;
///
/// // Two well-separated blobs.
/// let x = Matrix::new(vec![
///     vec![0.0, 0.0], vec![0.1, 0.1], vec![0.0, 0.1],
///     vec![10.0, 10.0], vec![10.1, 10.1], vec![10.0, 10.1],
/// ])?;
/// let labels = vec![0, 0, 0, 1, 1, 1];
/// let s = silhouette_score(&x, &labels)?;
/// assert!(s > 0.5, "well-separated clusters should have high silhouette: {s}");
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
pub fn silhouette_score(x: &Matrix, labels: &[usize]) -> Result<f64> {
    let n = x.nrows();
    if labels.len() != n {
        return Err(DatarustError::ShapeMismatch {
            expected: format!("{n} labels"),
            actual: format!("{} labels", labels.len()),
        });
    }
    if n < 2 {
        return Err(DatarustError::InvalidInput(
            "silhouette_score requires at least 2 samples".into(),
        ));
    }
    x.validate_finite()?;

    // Compact arbitrary external labels instead of allocating by the largest
    // observed value. This keeps gapped IDs such as {10, usize::MAX} safe.
    let mut classes = labels.to_vec();
    classes.sort_unstable();
    classes.dedup();
    let k = classes.len();
    if k < 2 {
        return Err(DatarustError::InvalidInput(
            "silhouette_score requires at least 2 clusters".into(),
        ));
    }
    if k >= n {
        return Err(DatarustError::InvalidInput(format!(
            "silhouette_score requires fewer clusters than samples, got {k} clusters for {n} samples"
        )));
    }

    let compact_labels: Vec<usize> = labels
        .iter()
        .map(|label| {
            classes
                .binary_search(label)
                .expect("classes was built from labels")
        })
        .collect();
    // Count members per cluster.
    let mut counts = vec![0usize; k];
    for &c in &compact_labels {
        counts[c] += 1;
    }
    // Precompute row slices for speed.
    let p = x.ncols();
    let data = x.as_slice();

    // Pre-allocate to avoid O(N) heap allocations inside the loop
    let mut cluster_sums = vec![0.0_f64; k];
    let mut cluster_counts = vec![0usize; k];

    let mut total = 0.0_f64;
    for i in 0..n {
        let row_i = &data[i * p..(i + 1) * p];
        let ci = compact_labels[i];

        // sklearn defines the silhouette coefficient of a singleton cluster
        // as zero. Skipping its accumulation preserves that contribution.
        if counts[ci] == 1 {
            continue;
        }

        // a(i): mean distance to other points in the same cluster.
        let mut sum_same = 0.0_f64;
        let mut count_same = 0usize;
        // b(i): for each other cluster, the mean distance; take the min.
        cluster_sums.fill(0.0_f64);
        cluster_counts.fill(0usize);
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = sq_dist(row_i, &data[j * p..(j + 1) * p]).sqrt();
            let cj = compact_labels[j];
            cluster_sums[cj] += d;
            cluster_counts[cj] += 1;
            if cj == ci {
                sum_same += d;
                count_same += 1;
            }
        }
        let a_i = sum_same / count_same as f64;
        // b(i): nearest other cluster's mean distance.
        let mut b_i = f64::INFINITY;
        for c in 0..k {
            if c == ci || cluster_counts[c] == 0 {
                continue;
            }
            let mean_d = cluster_sums[c] / cluster_counts[c] as f64;
            if mean_d < b_i {
                b_i = mean_d;
            }
        }
        if b_i.is_infinite() {
            // No other cluster has members; skip this sample.
            continue;
        }
        let denom = a_i.max(b_i);
        if denom > 0.0 {
            total += (b_i - a_i) / denom;
        }
    }
    Ok(total / n as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn perfect_separation_high_silhouette() {
        let x = Matrix::new(vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
        ])
        .unwrap();
        let labels = vec![0, 0, 0, 1, 1, 1];
        let s = silhouette_score(&x, &labels).unwrap();
        assert!(s > 0.5, "expected high silhouette, got {s}");
    }

    #[test]
    fn overlapping_clusters_low_silhouette() {
        // Interleaved points: distances within and across clusters are similar.
        let x = Matrix::new(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        // Alternating labels — clusters are not separated.
        let labels = vec![0, 1, 0, 1];
        let s = silhouette_score(&x, &labels).unwrap();
        assert!(
            s < 0.3,
            "expected low silhouette for interleaved clusters: {s}"
        );
    }

    #[test]
    fn single_cluster_errors() {
        let x = Matrix::new(vec![vec![0.0], vec![1.0]]).unwrap();
        let labels = vec![0, 0];
        assert!(silhouette_score(&x, &labels).is_err());
    }

    #[test]
    fn label_count_mismatch_errors() {
        let x = Matrix::new(vec![vec![0.0], vec![1.0]]).unwrap();
        assert!(silhouette_score(&x, &[0]).is_err());
    }

    #[test]
    fn three_clusters() {
        let x = Matrix::new(vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![20.0, 20.0],
            vec![20.1, 20.0],
        ])
        .unwrap();
        let labels = vec![0, 0, 1, 1, 2, 2];
        let s = silhouette_score(&x, &labels).unwrap();
        assert!(s > 0.5, "three well-separated clusters: {s}");
        assert!(approx(s, s, 1e-12)); // tautology, just exercises approx
    }

    #[test]
    fn gapped_and_maximum_labels_are_compacted() {
        let x = Matrix::new(vec![vec![0.0], vec![0.1], vec![10.0], vec![10.1]]).unwrap();
        let compact = silhouette_score(&x, &[0, 0, 1, 1]).unwrap();
        let gapped = silhouette_score(&x, &[10, 10, usize::MAX, usize::MAX]).unwrap();
        assert!(approx(compact, gapped, 1e-12));
    }

    #[test]
    fn one_cluster_per_sample_errors() {
        let x = Matrix::new(vec![vec![0.0], vec![1.0]]).unwrap();
        assert!(silhouette_score(&x, &[0, usize::MAX]).is_err());
    }

    #[test]
    fn singleton_cluster_contributes_zero() {
        let x = Matrix::new(vec![vec![0.0], vec![10.0], vec![11.0]]).unwrap();
        let score = silhouette_score(&x, &[0, 1, 1]).unwrap();
        let expected = (0.0 + 0.9 + 10.0 / 11.0) / 3.0;
        assert!(approx(score, expected, 1e-12), "score={score}");
    }

    #[test]
    fn non_finite_features_error() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let x = Matrix::new(vec![vec![0.0], vec![invalid], vec![1.0]]).unwrap();
            assert!(silhouette_score(&x, &[0, 0, 1]).is_err());
        }
    }
}
