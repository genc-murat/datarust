//! k-Nearest-Neighbor estimators, mirroring `sklearn.neighbors`.
//!
//! Brute-force Euclidean search over the stored training matrix — the same
//! distance kernel as [`KnnImputer`](crate::imputer::KnnImputer). Tree
//! acceleration (ball/KD) is explicitly post-0.8.

pub mod classifier;
pub mod regressor;

pub use classifier::KNeighborsClassifier;
pub use regressor::KNeighborsRegressor;

use crate::matrix::Matrix;

/// Weighting scheme shared with [`KnnImputer`](crate::imputer::KnnImputer):
/// uniform votes or inverse-distance weights.
pub use crate::imputer::KnnWeights;

/// Squared Euclidean distance between two equally sized rows.
pub(crate) fn sq_euclidean(a: &[f64], b: &[f64]) -> f64 {
    let mut sq_sum = 0.0;
    for (av, bv) in a.iter().zip(b.iter()) {
        let d = av - bv;
        sq_sum += d * d;
    }
    sq_sum
}

/// Per-neighbor weights under sklearn's `weights='distance'` rule.
///
/// Plain inverse Euclidean weight `1/d`, with one crucial special case
/// (sklearn's `_get_weights`): if **any** neighbor coincides with the query
/// (zero distance), the vote is winner-takes-all — those neighbors get weight
/// 1 and every other neighbor gets 0.
pub(crate) fn distance_weights(neighbors: &[(f64, usize)]) -> Vec<f64> {
    let any_zero = neighbors.iter().any(|(d, _)| *d == 0.0);
    neighbors
        .iter()
        .map(|&(d, _)| {
            if any_zero {
                f64::from(d == 0.0)
            } else {
                1.0 / d.sqrt()
            }
        })
        .collect()
}

/// Brute-force search for the `k` nearest training rows to `query`.
///
/// Returns `(squared_distance, train_index)` pairs sorted by
/// `(distance, index)` so tie groups are deterministic (training order within
/// equal distances). `k` is assumed validated against the training size by
/// the caller.
pub(crate) fn kneighbors(train: &Matrix, query: &[f64], k: usize) -> Vec<(f64, usize)> {
    let mut distances: Vec<(f64, usize)> = (0..train.nrows())
        .map(|idx| (sq_euclidean(train.row(idx), query), idx))
        .collect();
    debug_assert!(k <= distances.len());
    if k < distances.len() {
        distances.select_nth_unstable_by(k, |a, b| a.0.total_cmp(&b.0));
    }
    distances[..k].sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    distances[..k].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix;

    fn m(rows: Vec<Vec<f64>>) -> Matrix {
        Matrix::new(rows).unwrap()
    }

    #[test]
    fn sq_euclidean_matches_manual_computation() {
        let a = [1.0, 2.0, 3.0];
        let b = [0.0, 4.0, 3.0];
        assert!((sq_euclidean(&a, &b) - (1.0 + 4.0 + 0.0)).abs() < 1e-12);
        assert_eq!(sq_euclidean(&a, &a), 0.0);
    }

    #[test]
    fn kneighbors_returns_k_sorted_by_distance_then_index() {
        let train = m(vec![vec![0.0], vec![10.0], vec![1.0], vec![1.0]]);
        let neighbors = kneighbors(&train, &[0.5], 3);
        // Distances: 0.25 (idx 0), 0.25 (idx 2), 0.25 (idx 3), 90.25 (idx 1).
        // The three-way tie is broken by training order.
        assert_eq!(neighbors, vec![(0.25, 0), (0.25, 2), (0.25, 3)]);
    }

    #[test]
    fn distance_weights_follow_sklearn_zero_distance_rule() {
        // No coincident neighbor: plain inverse Euclidean weights.
        let plain = distance_weights(&[(0.25, 0), (4.0, 1)]);
        assert!((plain[0] - 2.0).abs() < 1e-12);
        assert!((plain[1] - 0.5).abs() < 1e-12);

        // Winner-takes-all once any distance is exactly zero.
        let wta = distance_weights(&[(0.0, 0), (0.01, 1), (9.0, 2)]);
        assert_eq!(wta, vec![1.0, 0.0, 0.0]);
    }
}
