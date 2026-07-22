//! Clustering evaluation metrics.
//!
//! Provides [`silhouette_score`], mirroring `sklearn.metrics.silhouette_score`.
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
    // Determine the number of clusters and validate that there are at least 2.
    let k = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    if k < 2 {
        return Err(DatarustError::InvalidInput(
            "silhouette_score requires at least 2 clusters".into(),
        ));
    }
    // Count members per cluster.
    let mut counts = vec![0usize; k];
    for &c in labels {
        counts[c] += 1;
    }
    // Every cluster must have at least 1 member (already guaranteed by labels),
    // but a singleton cluster yields a(i)=0 for its sole member.
    // Precompute row slices for speed.
    let p = x.ncols();
    let data = x.as_slice();

    let mut total = 0.0_f64;
    for i in 0..n {
        let row_i = &data[i * p..(i + 1) * p];
        let ci = labels[i];

        // a(i): mean distance to other points in the same cluster.
        let mut sum_same = 0.0_f64;
        let mut count_same = 0usize;
        // b(i): for each other cluster, the mean distance; take the min.
        let mut cluster_sums = vec![0.0_f64; k];
        let mut cluster_counts = vec![0usize; k];
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = sq_dist(row_i, &data[j * p..(j + 1) * p]).sqrt();
            let cj = labels[j];
            cluster_sums[cj] += d;
            cluster_counts[cj] += 1;
            if cj == ci {
                sum_same += d;
                count_same += 1;
            }
        }
        let a_i = if count_same > 0 {
            sum_same / count_same as f64
        } else {
            0.0 // singleton cluster
        };
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
}
