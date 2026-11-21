//! k-Nearest-Neighbor classification, mirroring `sklearn.neighbors.KNeighborsClassifier`.

use crate::error::{DatarustError, Result};
use crate::imputer::KnnWeights;
use crate::label_space::LabelSpace;
use crate::matrix::Matrix;
use crate::neighbors::{distance_weights, kneighbors};
use crate::traits::{Classifier, Estimator, ParamValue, Params, PredictProba, Predictor};

/// k-Nearest-Neighbor classifier (sklearn's `KNeighborsClassifier`).
///
/// Brute-force Euclidean neighbor search over the fitted training matrix;
/// `weights = uniform | distance` with sklearn's aggregation rules:
///
/// - `Uniform`: majority vote; ties break to the smallest winning class
///   (sklearn's `argmax` over the sorted `classes_`).
/// - `Distance`: each neighbor votes with inverse Euclidean weight `1/d`;
///   when any neighbor coincides with the query (zero distance), the vote is
///   winner-takes-all — only those neighbors vote (sklearn's `_get_weights`).
///
/// `predict_proba` returns vote fractions (uniform) or normalized vote
/// weights (distance) per class, in `classes()` order.
///
/// ```rust
/// use datarust::neighbors::KNeighborsClassifier;
/// use datarust::traits::{Classifier, Predictor};
/// use datarust::Matrix;
///
/// let x = Matrix::new(vec![
///     vec![0.0, 0.0],
///     vec![0.5, 0.0],
///     vec![4.0, 4.0],
///     vec![4.5, 4.0],
/// ])?;
/// let y = vec![0.0, 0.0, 1.0, 1.0];
/// let mut knn = KNeighborsClassifier::new().with_n_neighbors(3);
/// knn.fit(&x, &y)?;
/// let pred = knn.predict(&x)?;
/// assert_eq!(pred, vec![0.0, 0.0, 1.0, 1.0]);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct KNeighborsClassifier {
    n_neighbors: usize,
    weights: KnnWeights,
    // Fitted state
    x_train: Option<Matrix>,
    y_encoded: Vec<usize>,
    classes_: LabelSpace,
    n_features_in_: usize,
    fitted: bool,
}

impl Default for KNeighborsClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl KNeighborsClassifier {
    /// Create a classifier with the sklearn default of 5 neighbors and
    /// uniform weights.
    pub fn new() -> Self {
        Self {
            n_neighbors: 5,
            weights: KnnWeights::Uniform,
            x_train: None,
            y_encoded: Vec::new(),
            classes_: LabelSpace::fit(&[0.0]).unwrap(),
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

    /// Sorted training classes, in `predict_proba` column order.
    pub fn classes(&self) -> &[f64] {
        self.classes_.labels()
    }

    /// Number of features seen during `fit`.
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
    }

    fn validate_prediction_input(&self, x: &Matrix) -> Result<()> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("KNeighborsClassifier".into()));
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

impl Estimator for KNeighborsClassifier {}

impl Predictor for KNeighborsClassifier {
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
        if self.n_neighbors == 0 {
            return Err(DatarustError::InvalidInput(
                "n_neighbors must be >= 1".into(),
            ));
        }
        if self.n_neighbors > x.nrows() {
            return Err(DatarustError::InvalidInput(format!(
                "n_neighbors ({}) must not exceed the training sample count ({})",
                self.n_neighbors,
                x.nrows()
            )));
        }

        let classes_ = LabelSpace::fit(y)?;
        let y_encoded = y
            .iter()
            .map(|&label| classes_.encode(label))
            .collect::<Result<_>>()?;

        self.x_train = Some(x.clone());
        self.y_encoded = y_encoded;
        self.classes_ = classes_;
        self.n_features_in_ = x.ncols();
        self.fitted = true;
        Ok(())
    }

    fn predict(&self, x: &Matrix) -> Result<Vec<f64>> {
        let proba = self.predict_proba(x)?;
        let mut out = Vec::with_capacity(proba.nrows());
        for i in 0..proba.nrows() {
            // Majority vote (uniform) / largest aggregated weight (distance),
            // ties broken by the first class in sorted order — sklearn's
            // `argmax` over `classes_`.
            let mut best_col = 0usize;
            let mut best_value = f64::NEG_INFINITY;
            for c in 0..proba.ncols() {
                if proba.get(i, c) > best_value {
                    best_value = proba.get(i, c);
                    best_col = c;
                }
            }
            out.push(self.classes_.decode(best_col)?);
        }
        Ok(out)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl Classifier for KNeighborsClassifier {}

impl PredictProba for KNeighborsClassifier {
    fn predict_proba(&self, x: &Matrix) -> Result<Matrix> {
        self.validate_prediction_input(x)?;
        let x_train = self
            .x_train
            .as_ref()
            .ok_or_else(|| DatarustError::NotFitted("KNeighborsClassifier".into()))?;
        // n_neighbors is mutable after fit; re-check it against the stored
        // training size (sklearn errors in `kneighbors`).
        if self.n_neighbors > x_train.nrows() {
            return Err(DatarustError::InvalidInput(format!(
                "n_neighbors ({}) must not exceed the training sample count ({})",
                self.n_neighbors,
                x_train.nrows()
            )));
        }
        let n_classes = self.classes_.len();
        let mut rows = Vec::with_capacity(x.nrows());
        for i in 0..x.nrows() {
            let neighbors = kneighbors(x_train, x.row(i), self.n_neighbors);
            let mut votes = vec![0.0; n_classes];
            match self.weights {
                KnnWeights::Uniform => {
                    for (_, idx) in &neighbors {
                        votes[self.y_encoded[*idx]] += 1.0;
                    }
                }
                KnnWeights::Distance => {
                    let weights = distance_weights(&neighbors);
                    for (w, (_, idx)) in weights.iter().zip(neighbors.iter()) {
                        votes[self.y_encoded[*idx]] += w;
                    }
                }
            }
            let total: f64 = votes.iter().sum();
            if total > 0.0 {
                for v in &mut votes {
                    *v /= total;
                }
            }
            rows.push(votes);
        }
        Matrix::new(rows)
    }
}

impl Params for KNeighborsClassifier {
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
                "unknown parameter '{name}' for KNeighborsClassifier"
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

    /// Two well-separated clusters; queries sit at training points.
    #[test]
    fn separates_two_clusters() {
        let x = m(vec![
            vec![0.0, 0.0],
            vec![0.5, 0.0],
            vec![4.0, 4.0],
            vec![4.5, 4.0],
        ]);
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let mut knn = KNeighborsClassifier::new().with_n_neighbors(3);
        knn.fit(&x, &y).unwrap();
        assert_eq!(knn.predict(&x).unwrap(), y);
        let proba = knn.predict_proba(&x).unwrap();
        for i in 0..x.nrows() {
            let winning = proba.get(i, 0).max(proba.get(i, 1));
            assert!(winning >= 2.0 / 3.0 - 1e-12);
        }
    }

    #[test]
    fn uniform_tie_breaks_to_smallest_class() {
        // Query at (0.5, 0.5): equidistant to all four training points.
        // k=4 → votes {A: 2, B: 2} tie → smallest label wins.
        let x = m(vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ]);
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let mut knn = KNeighborsClassifier::new().with_n_neighbors(4);
        knn.fit(&x, &y).unwrap();
        let pred = knn.predict(&m(vec![vec![0.5, 0.5]])).unwrap();
        assert_eq!(pred, vec![0.0]);
        // Swapping the label values flips the tie to 1.0 (still smallest).
        let mut knn2 = KNeighborsClassifier::new().with_n_neighbors(4);
        knn2.fit(&x, &[2.0, 2.0, 5.0, 5.0]).unwrap();
        let pred2 = knn2.predict(&m(vec![vec![0.5, 0.5]])).unwrap();
        assert_eq!(pred2, vec![2.0]);
    }

    #[test]
    fn distance_weights_handle_coincident_point() {
        // Query duplicates a training point: zero distance gets weight 1 and
        // dominates the far cluster.
        let x = m(vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
        ]);
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let mut knn = KNeighborsClassifier::new()
            .with_n_neighbors(4)
            .with_weights(KnnWeights::Distance);
        knn.fit(&x, &y).unwrap();
        let pred = knn.predict(&m(vec![vec![0.0, 0.0]])).unwrap();
        assert_eq!(pred, vec![0.0]);
        let proba = knn.predict_proba(&m(vec![vec![0.0, 0.0]])).unwrap();
        // Winner-takes-all: only the coincident point votes (weight 1).
        assert!((proba.get(0, 0) - 1.0).abs() < 1e-12);
        assert_eq!(proba.get(0, 1), 0.0);
    }

    #[test]
    fn gapped_labels_are_preserved() {
        let x = m(vec![vec![0.0], vec![1.0], vec![10.0], vec![11.0]]);
        let y = vec![2.0, 2.0, 5.0, 5.0];
        let mut knn = KNeighborsClassifier::new().with_n_neighbors(1);
        knn.fit(&x, &y).unwrap();
        assert_eq!(knn.classes(), &[2.0, 5.0]);
        assert_eq!(
            knn.predict(&m(vec![vec![0.9], vec![10.1]])).unwrap(),
            vec![2.0, 5.0]
        );
    }

    #[test]
    fn n_neighbors_validation_and_predict_only_mutation() {
        let (x, y) = (
            m(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]),
            vec![0.0, 0.0, 1.0, 1.0],
        );
        let mut knn = KNeighborsClassifier::new();
        assert!(matches!(
            knn.fit(&x, &y),
            Err(DatarustError::InvalidInput(_))
        )); // default 5 > 4 training rows
        let mut knn = KNeighborsClassifier::new().with_n_neighbors(2);
        knn.fit(&x, &y).unwrap();
        assert!(knn.predict(&x).is_ok());
        // n_neighbors is prediction-only: mutable after fit.
        knn.set_params("n_neighbors", ParamValue::Int(4)).unwrap();
        assert!(knn.is_fitted());
        assert!(knn.predict(&x).is_ok());
        // But a post-fit k beyond the training size errors, not panics.
        knn.set_params("n_neighbors", ParamValue::Int(10)).unwrap();
        assert!(matches!(
            knn.predict(&x),
            Err(DatarustError::InvalidInput(_))
        ));
        assert!(matches!(
            knn.set_params("n_neighbors", ParamValue::Int(0)),
            Err(DatarustError::InvalidInput(_))
        ));
        assert!(matches!(
            knn.set_params("weights", ParamValue::Int(7)),
            Err(DatarustError::InvalidInput(_))
        ));
    }

    #[test]
    fn unfitted_and_shape_errors() {
        let knn = KNeighborsClassifier::new();
        assert!(matches!(
            knn.predict(&m(vec![vec![0.0]])),
            Err(DatarustError::NotFitted(_))
        ));
        let mut knn = KNeighborsClassifier::new().with_n_neighbors(1);
        knn.fit(&m(vec![vec![0.0], vec![1.0]]), &[0.0, 1.0])
            .unwrap();
        assert!(matches!(
            knn.predict(&m(vec![vec![0.0, 1.0]])),
            Err(DatarustError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            knn.fit(&m(vec![vec![0.0]]), &[0.0, 1.0]),
            Err(DatarustError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            knn.fit(&m(vec![vec![f64::NAN], vec![1.0]]), &[0.0, 1.0]),
            Err(DatarustError::InvalidInput(_))
        ));
    }
}
