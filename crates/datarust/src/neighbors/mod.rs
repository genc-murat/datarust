//! k-Nearest-Neighbor estimators, mirroring `sklearn.neighbors`.
//!
//! Brute-force Euclidean search over the stored training matrix.

pub mod classifier;
pub mod regressor;

pub use classifier::KNeighborsClassifier;
pub use regressor::KNeighborsRegressor;

use crate::matrix::Matrix;

/// Weighting scheme shared with [`KnnImputer`](crate::imputer::KnnImputer):
/// uniform votes or inverse-distance weights.
pub use crate::imputer::KnnWeights;

/// Squared Euclidean distance in one of three ordered magnitude ranges.
/// Ordinary inputs keep their original squared distance for nearest-neighbor
/// selection. At the extremes we change units by a power of two before
/// squaring, so distinct neighbors do not collapse to zero or infinity.
#[derive(Clone, Copy, Debug)]
struct Distance {
    range: i8,
    squared: f64,
}

impl Distance {
    fn between(a: &[f64], b: &[f64]) -> Self {
        let squared: f64 = a.iter().zip(b).map(|(&a, &b)| (a - b).powi(2)).sum();
        if squared.is_normal() {
            return Self { range: 0, squared };
        }
        // 2^600 lifts even the smallest subnormal difference into the safe
        // squaring range. Its reciprocal leaves room to sum squared differences
        // across an entire row, including coordinates of opposite signs at MAX.
        if squared.is_infinite() {
            let down = 2.0_f64.powi(-600);
            let squared = a
                .iter()
                .zip(b)
                .map(|(&a, &b)| (a * down - b * down).powi(2))
                .sum();
            Self { range: 1, squared }
        } else {
            let up = 2.0_f64.powi(600);
            let squared = a.iter().zip(b).map(|(&a, &b)| ((a - b) * up).powi(2)).sum();
            Self { range: -1, squared }
        }
    }

    /// Relative inverse-distance weight for sorted, nonzero distances.
    fn ratio_to(self, farther: Self) -> f64 {
        let mut ratio = self.squared.sqrt() / farther.squared.sqrt();
        // Apply one unit conversion at a time: 2^-1200 alone underflows
        // even when the final ratio is representable.
        for _ in self.range..farther.range {
            ratio *= 2.0_f64.powi(-600);
        }
        ratio
    }
}

fn neighbor_weights(
    weights: KnnWeights,
    neighbors: &[(Distance, usize)],
) -> impl Iterator<Item = f64> + '_ {
    let nearest = neighbors[0].0;
    neighbors.iter().map(move |&(distance, _)| match weights {
        KnnWeights::Uniform => 1.0,
        KnnWeights::Distance if nearest.squared == 0.0 => f64::from(distance.squared == 0.0),
        KnnWeights::Distance => nearest.ratio_to(distance),
    })
}

/// Brute-force search, with ties broken by training order at the k boundary.
fn kneighbors(train: &Matrix, query: &[f64], k: usize) -> Vec<(Distance, usize)> {
    let mut distances: Vec<_> = (0..train.nrows())
        .map(|idx| (Distance::between(train.row(idx), query), idx))
        .collect();
    let by_distance_then_index = |a: &(Distance, usize), b: &(Distance, usize)| {
        a.0.range
            .cmp(&b.0.range)
            .then_with(|| a.0.squared.total_cmp(&b.0.squared))
            .then_with(|| a.1.cmp(&b.1))
    };
    if k < distances.len() {
        distances.select_nth_unstable_by(k, by_distance_then_index);
    }
    distances.truncate(k);
    distances.sort_unstable_by(by_distance_then_index);
    distances
}

fn validate_neighbor_count(k: usize, n_samples: usize) -> crate::error::Result<()> {
    if k == 0 || k > n_samples {
        return Err(crate::error::DatarustError::InvalidInput(format!(
            "n_neighbors ({k}) must be in 1..={n_samples}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix;

    fn m(rows: Vec<Vec<f64>>) -> Matrix {
        Matrix::new(rows).unwrap()
    }

    #[test]
    fn ordinary_squared_distance_matches_manual_computation() {
        let a = [1.0, 2.0, 3.0];
        let b = [0.0, 4.0, 3.0];
        assert!((Distance::between(&a, &b).squared - 5.0).abs() < 1e-12);
        assert_eq!(Distance::between(&a, &a).squared, 0.0);
    }

    #[test]
    fn kneighbors_returns_k_sorted_by_distance_then_index() {
        let train = m(vec![vec![0.0], vec![10.0], vec![1.0], vec![1.0]]);
        let neighbors = kneighbors(&train, &[0.5], 3);
        // Three rows have distance 0.5, and row 1 has distance 9.5.
        // The three-way tie is broken by training order.
        assert_eq!(
            neighbors.iter().map(|&(_, i)| i).collect::<Vec<_>>(),
            [0, 2, 3]
        );
    }

    #[test]
    fn boundary_ties_use_the_earliest_training_rows() {
        let train = m(vec![vec![1.0]; 80]);
        assert_eq!(
            kneighbors(&train, &[0.0], 3)
                .iter()
                .map(|&(_, i)| i)
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
    }

    #[test]
    fn extreme_distances_preserve_neighbor_order_and_relative_weights() {
        use crate::traits::{PredictProba, Predictor};

        let query = m(vec![vec![0.0]]);
        for unit in [1e200, 1e-200, f64::from_bits(1)] {
            let train = m(vec![vec![3.0 * unit], vec![unit]]);
            let mut classifier = KNeighborsClassifier::new()
                .with_n_neighbors(1)
                .with_weights(KnnWeights::Distance);
            classifier.fit(&train, &[0.0, 1.0]).unwrap();
            assert_eq!(classifier.predict(&query).unwrap(), [1.0]);
            classifier = classifier.with_n_neighbors(2);
            let proba = classifier.predict_proba(&query).unwrap();
            assert!((proba.get(0, 1) - 0.75).abs() < 1e-12);

            let mut regressor = KNeighborsRegressor::new()
                .with_n_neighbors(2)
                .with_weights(KnnWeights::Distance);
            regressor.fit(&train, &[0.0, 20.0]).unwrap();
            assert!((regressor.predict(&query).unwrap()[0] - 15.0).abs() < 1e-12);
        }
    }

    #[test]
    fn distances_beyond_f64_max_still_have_finite_relative_weights() {
        use crate::traits::{PredictProba, Predictor};

        let train = m(vec![vec![-f64::MAX], vec![0.0]]);
        let query = m(vec![vec![f64::MAX]]);
        let mut classifier = KNeighborsClassifier::new()
            .with_n_neighbors(2)
            .with_weights(KnnWeights::Distance);
        classifier.fit(&train, &[0.0, 1.0]).unwrap();
        let proba = classifier.predict_proba(&query).unwrap();
        assert!((proba.get(0, 0) - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn weights_between_the_smallest_and_largest_distance_ranges() {
        let train = m(vec![vec![1e-155], vec![1e155]]);
        let neighbors = kneighbors(&train, &[0.0], 2);
        let weights: Vec<_> = neighbor_weights(KnnWeights::Distance, &neighbors).collect();
        assert_eq!(weights[0], 1.0);
        assert!((weights[1] / 1e-310 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn neighbor_selection_preserves_distinct_nearly_equal_and_tiny_norms() {
        let tiny = f64::from_bits(1);
        for (longer, shorter) in [
            (vec![1.0, 2.0_f64.powi(-26)], vec![1.0, 0.0]),
            (vec![tiny, tiny], vec![tiny, 0.0]),
        ] {
            let train = m(vec![longer, shorter]);
            assert_eq!(kneighbors(&train, &[0.0, 0.0], 1)[0].1, 1);
        }
    }

    #[test]
    fn prediction_rejects_zero_neighbors_after_fit() {
        use crate::traits::{PredictProba, Predictor};

        let x = m(vec![vec![1.0], vec![2.0]]);
        let mut classifier = KNeighborsClassifier::new().with_n_neighbors(1);
        classifier.fit(&x, &[0.0, 1.0]).unwrap();
        assert!(classifier.with_n_neighbors(0).predict_proba(&x).is_err());
        let mut regressor = KNeighborsRegressor::new().with_n_neighbors(1);
        regressor.fit(&x, &[0.0, 1.0]).unwrap();
        assert!(regressor.with_n_neighbors(0).predict(&x).is_err());
    }

    #[test]
    fn distance_weights_follow_sklearn_zero_distance_rule() {
        let train = m(vec![vec![0.5], vec![2.0]]);
        let neighbors = kneighbors(&train, &[0.0], 2);
        let plain: Vec<_> = neighbor_weights(KnnWeights::Distance, &neighbors).collect();
        assert!((plain[0] - 1.0).abs() < 1e-12);
        assert!((plain[1] - 0.25).abs() < 1e-12);

        let train = m(vec![vec![0.0], vec![0.01], vec![9.0]]);
        let neighbors = kneighbors(&train, &[0.0], 3);
        let wta: Vec<_> = neighbor_weights(KnnWeights::Distance, &neighbors).collect();
        assert_eq!(wta, vec![1.0, 0.0, 0.0]);
    }
}
