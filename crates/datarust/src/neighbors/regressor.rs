//! k-Nearest-Neighbor regression, mirroring `sklearn.neighbors.KNeighborsRegressor`.

use crate::error::{DatarustError, Result};
use crate::imputer::KnnWeights;
use crate::matrix::Matrix;
use crate::neighbors::{kneighbors, neighbor_weights, validate_neighbor_count};
use crate::traits::{Estimator, ParamValue, Params, Predictor, Regressor};

/// k-Nearest-Neighbor regressor (sklearn's `KNeighborsRegressor`).
///
/// Brute-force Euclidean neighbor search over the fitted training matrix;
/// `weights = uniform | distance` with sklearn's aggregation rules:
///
/// - `Uniform`: the unweighted mean of the neighbors' targets.
/// - `Distance`: the weighted mean with inverse Euclidean weights `1/d`;
///   when any neighbor coincides with the query (zero distance), only those
///   neighbors contribute (winner-takes-all, sklearn's `_get_weights`).
///
/// ```rust
/// use datarust::neighbors::KNeighborsRegressor;
/// use datarust::traits::{Predictor, Regressor};
/// use datarust::Matrix;
///
/// let x = Matrix::new(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]])?;
/// let y = vec![0.0, 10.0, 20.0, 30.0];
/// let mut knn = KNeighborsRegressor::new().with_n_neighbors(2);
/// knn.fit(&x, &y)?;
/// let pred = knn.predict(&Matrix::new(vec![vec![0.5]])?)?;
/// assert!((pred[0] - 5.0).abs() < 1e-12);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct KNeighborsRegressor {
    n_neighbors: usize,
    weights: KnnWeights,
    // Fitted state
    x_train: Option<Matrix>,
    y_train: Vec<f64>,
    n_features_in_: usize,
    fitted: bool,
}

impl Default for KNeighborsRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl KNeighborsRegressor {
    /// Create a regressor with the sklearn default of 5 neighbors and
    /// uniform weights.
    pub fn new() -> Self {
        Self {
            n_neighbors: 5,
            weights: KnnWeights::Uniform,
            x_train: None,
            y_train: Vec::new(),
            n_features_in_: 0,
            fitted: false,
        }
    }

    /// Number of neighbors consulted per prediction (default: 5).
    ///
    /// Unlike most hyperparameters, `n_neighbors` and `weights` may be
    /// changed after `fit` — they only affect prediction. A zero value is
    /// rejected at `fit`/`predict` time, matching `set_params`.
    pub fn with_n_neighbors(mut self, k: usize) -> Self {
        self.n_neighbors = k;
        self
    }

    /// Neighbor weighting scheme (default: uniform).
    pub fn with_weights(mut self, weights: KnnWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Number of features seen during `fit`.
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
    }

    fn validate_prediction_input(&self, x: &Matrix) -> Result<()> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("KNeighborsRegressor".into()));
        }
        if self.n_features_in_ != x.ncols() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.n_features_in_),
                actual: format!("{} features", x.ncols()),
            });
        }
        x.validate_finite()
    }
}

impl Estimator for KNeighborsRegressor {}

impl Predictor for KNeighborsRegressor {
    fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        if y.is_empty() {
            return Err(DatarustError::EmptyInput("y is empty".into()));
        }
        if x.nrows() != y.len() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} rows in y", x.nrows()),
                actual: format!("{} rows in y", y.len()),
            });
        }
        x.validate_finite()?;
        if !y.iter().all(|v| v.is_finite()) {
            return Err(DatarustError::InvalidInput(
                "y must contain only finite values".into(),
            ));
        }
        validate_neighbor_count(self.n_neighbors, x.nrows())?;

        self.x_train = Some(x.clone());
        self.y_train = y.to_vec();
        self.n_features_in_ = x.ncols();
        self.fitted = true;
        Ok(())
    }

    fn predict(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.validate_prediction_input(x)?;
        let x_train = self
            .x_train
            .as_ref()
            .ok_or_else(|| DatarustError::NotFitted("KNeighborsRegressor".into()))?;
        validate_neighbor_count(self.n_neighbors, x_train.nrows())?;

        let mut out = Vec::with_capacity(x.nrows());
        for i in 0..x.nrows() {
            let neighbors = kneighbors(x_train, x.row(i), self.n_neighbors);
            let targets_and_weights = || {
                neighbors
                    .iter()
                    .zip(neighbor_weights(self.weights, &neighbors))
                    .map(|(&(_, idx), weight)| (self.y_train[idx], weight))
            };
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;
            for (target, weight) in targets_and_weights() {
                weighted_sum += target * weight;
                total_weight += weight;
            }
            let value = if weighted_sum.is_finite() {
                weighted_sum / total_weight
            } else {
                // The mean of finite targets is finite even if their sum overflowed.
                let scaled_sum: f64 = targets_and_weights()
                    .map(|(target, weight)| (target / f64::MAX) * weight)
                    .sum();
                (scaled_sum / total_weight).clamp(-1.0, 1.0) * f64::MAX
            };
            out.push(value);
        }
        Ok(out)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl Regressor for KNeighborsRegressor {
    fn name(&self) -> &'static str {
        "KNeighborsRegressor"
    }
}

impl Params for KNeighborsRegressor {
    fn get_params(&self) -> Vec<(std::borrow::Cow<'static, str>, ParamValue)> {
        vec![
            ("n_neighbors".into(), ParamValue::Int(self.n_neighbors)),
            (
                "weights".into(),
                ParamValue::Int(match self.weights {
                    KnnWeights::Uniform => 0,
                    KnnWeights::Distance => 1,
                }),
            ),
        ]
    }

    fn set_params(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match (name, value) {
            ("n_neighbors", ParamValue::Int(k)) if k >= 1 => {
                self.n_neighbors = k;
                // Prediction-only hyperparameter: fitted data stays valid.
                Ok(())
            }
            ("n_neighbors", ParamValue::Int(_)) => Err(DatarustError::InvalidInput(
                "n_neighbors must be >= 1".into(),
            )),
            ("weights", ParamValue::Int(0)) => {
                self.weights = KnnWeights::Uniform;
                Ok(())
            }
            ("weights", ParamValue::Int(1)) => {
                self.weights = KnnWeights::Distance;
                Ok(())
            }
            ("weights", _) => Err(DatarustError::InvalidInput(
                "weights expects Int(0)=Uniform or Int(1)=Distance".into(),
            )),
            _ => Err(DatarustError::InvalidInput(format!(
                "unknown parameter '{name}' for KNeighborsRegressor"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(rows: Vec<Vec<f64>>) -> Matrix {
        Matrix::new(rows).unwrap()
    }

    #[test]
    fn uniform_predicts_local_mean() {
        let x = m(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]);
        let y = vec![0.0, 10.0, 20.0, 30.0];
        let mut knn = KNeighborsRegressor::new().with_n_neighbors(2);
        knn.fit(&x, &y).unwrap();
        let pred = knn.predict(&m(vec![vec![0.5]])).unwrap();
        assert!((pred[0] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn finite_target_mean_survives_sum_overflow() {
        let train = m(vec![vec![0.0], vec![2.0]]);
        let query = m(vec![vec![1.0]]);
        for weights in [KnnWeights::Uniform, KnnWeights::Distance] {
            let mut model = KNeighborsRegressor::new()
                .with_n_neighbors(2)
                .with_weights(weights);
            model.fit(&train, &[f64::MAX, f64::MAX]).unwrap();
            assert_eq!(model.predict(&query).unwrap(), [f64::MAX]);
        }
    }

    #[test]
    fn distance_weights_dominant_near_neighbor() {
        // Query sits at 0.9: neighbor 1.0 (d=0.1) gets weight 10 vs weight
        // ~1.05 for 0.0 → weighted mean pulled toward 10.
        let x = m(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]);
        let y = vec![0.0, 10.0, 20.0, 30.0];
        let mut knn = KNeighborsRegressor::new()
            .with_n_neighbors(4)
            .with_weights(KnnWeights::Distance);
        knn.fit(&x, &y).unwrap();
        let pred = knn.predict(&m(vec![vec![0.9]])).unwrap();
        assert!(pred[0] > 8.0, "expected > 8, got {}", pred[0]);
    }

    #[test]
    fn coincident_point_is_winner_take_all() {
        let x = m(vec![vec![0.0], vec![1.0], vec![2.0]]);
        let y = vec![7.0, 10.0, 20.0];
        let mut knn = KNeighborsRegressor::new()
            .with_n_neighbors(3)
            .with_weights(KnnWeights::Distance);
        knn.fit(&x, &y).unwrap();
        // Query coincides with the first training point: only that neighbor
        // contributes (winner-takes-all), so the prediction is exactly 7.0.
        let pred = knn.predict(&m(vec![vec![0.0]])).unwrap();
        assert!((pred[0] - 7.0).abs() < 1e-12);
        // No coincidence → plain inverse-distance mean.
        // Query 0.5: d=0.5 → 2.0, d=0.5 → 2.0, d=1.5 → 0.667.
        // (7·2 + 10·2 + 20·0.667) / 4.667 ≈ 10.14
        let pred2 = knn.predict(&m(vec![vec![0.5]])).unwrap();
        assert!(
            (pred2[0] - 10.142857142857142).abs() < 1e-9,
            "got {}",
            pred2[0]
        );
    }

    #[test]
    fn n_neighbors_validation_and_predict_only_mutation() {
        let x = m(vec![vec![0.0], vec![1.0], vec![2.0]]);
        let y = vec![0.0, 1.0, 2.0];
        let mut knn = KNeighborsRegressor::new();
        assert!(matches!(
            knn.fit(&x, &y),
            Err(DatarustError::InvalidInput(_))
        ));
        let mut knn = KNeighborsRegressor::new().with_n_neighbors(2);
        knn.fit(&x, &y).unwrap();
        knn.set_params("n_neighbors", ParamValue::Int(3)).unwrap();
        assert!(knn.is_fitted());
        // A post-fit k beyond the training size errors, not panics.
        knn.set_params("n_neighbors", ParamValue::Int(10)).unwrap();
        assert!(matches!(
            knn.predict(&x),
            Err(DatarustError::InvalidInput(_))
        ));
        knn.set_params("n_neighbors", ParamValue::Int(2)).unwrap();
        assert!(knn.predict(&x).is_ok());
        assert!(matches!(
            knn.set_params("n_neighbors", ParamValue::Int(0)),
            Err(DatarustError::InvalidInput(_))
        ));
        assert!(matches!(
            knn.set_params("weights", ParamValue::Bool(true)),
            Err(DatarustError::InvalidInput(_))
        ));
    }

    #[test]
    fn unfitted_and_shape_errors() {
        let knn = KNeighborsRegressor::new();
        assert!(matches!(
            knn.predict(&m(vec![vec![0.0]])),
            Err(DatarustError::NotFitted(_))
        ));
        let mut knn = KNeighborsRegressor::new().with_n_neighbors(1);
        knn.fit(&m(vec![vec![0.0], vec![1.0]]), &[0.0, 1.0])
            .unwrap();
        assert!(matches!(
            knn.predict(&m(vec![vec![0.0, 1.0]])),
            Err(DatarustError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            knn.fit(&m(vec![vec![f64::NAN], vec![1.0]]), &[0.0, 1.0]),
            Err(DatarustError::InvalidInput(_))
        ));
        assert!(matches!(
            knn.fit(&m(vec![vec![0.0], vec![1.0]]), &[0.0, f64::NAN]),
            Err(DatarustError::InvalidInput(_))
        ));
    }
}
