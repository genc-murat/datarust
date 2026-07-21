//! Logistic regression for binary classification via IRLS.
//!
//! Mirrors `sklearn.linear_model.LogisticRegression` (binary, no regularization).
//! Estimates `P(y = 1 | x) = σ(x·β + b)` by maximising the log-likelihood via
//! Iteratively Reweighted Least Squares (IRLS, a.k.a. Newton-Raphson on the
//! logistic loss). Each iteration solves a weighted least-squares problem
//!
//! `β_{t+1} = (Xᵀ W X)⁻¹ Xᵀ W z`
//!
//! where `p_i = σ(x_i·β)`, `W_i = p_i (1 − p_i)` and
//! `z_i = x_i·β + (y_i − p_i) / W_i`. The weighted system is solved with the
//! shared Cholesky (default) or eigen-pseudo-inverse (SVD) solver.

use crate::error::{DatarustError, Result};
use crate::linalg::cholesky;
use crate::matrix::Matrix;
use crate::stats;
use crate::traits::{Classifier, Estimator, PredictProba, Predictor};

/// Solver strategy for [`LogisticRegression`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LogisticSolver {
    /// IRLS with Cholesky decomposition of `Xᵀ W X` each iteration (default).
    /// Fast; requires full column rank (quasi-separation may make `W` singular).
    #[default]
    Cholesky,
    /// IRLS with eigendecomposition pseudo-inverse. Robust to rank-deficiency.
    Svd,
}

/// Binary logistic regression solved by IRLS.
///
/// `predict` returns hard `{0.0, 1.0}` labels, while
/// [`predict_proba`](Self::predict_proba) returns an `(n_samples, 2)` matrix
/// with columns `P(class=0)` and `P(class=1)`.
///
/// ```rust
/// use datarust::linear_model::LogisticRegression;
/// use datarust::traits::{Classifier, Predictor};
/// use datarust::Matrix;
///
/// // Linearly separable: y = 1 when x0 > 0.
/// let x = Matrix::new(vec![
///     vec![-3.0], vec![-2.0], vec![-1.0],
///     vec![ 1.0], vec![ 2.0], vec![ 3.0],
/// ])?;
/// let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
///
/// let mut model = LogisticRegression::new();
/// model.fit(&x, &y)?;
/// let classes = model.predict_class(&x)?;
/// assert_eq!(classes, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LogisticRegression {
    fit_intercept: bool,
    solver: LogisticSolver,
    max_iter: usize,
    tol: f64,
    // Fitted state.
    coef_: Vec<f64>,
    intercept_: f64,
    n_features_in_: usize,
    n_iter_: usize,
    fitted: bool,
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Numerically stable logistic sigmoid: `σ(t) = 1 / (1 + e^{−t})`.
///
/// Clamps `t` to `±500` to avoid overflow; beyond that the sigmoid saturates.
#[inline]
fn sigmoid(t: f64) -> f64 {
    if t >= 0.0 {
        let z = (-t.min(500.0)).exp();
        1.0 / (1.0 + z)
    } else {
        let z = t.max(-500.0).exp();
        z / (1.0 + z)
    }
}

impl LogisticRegression {
    /// Creates a new estimator: `fit_intercept = true`, `solver = Cholesky`,
    /// `max_iter = 100`, `tol = 1e-4`.
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
            solver: LogisticSolver::Cholesky,
            max_iter: 100,
            tol: 1e-4,
            coef_: Vec::new(),
            intercept_: 0.0,
            n_features_in_: 0,
            n_iter_: 0,
            fitted: false,
        }
    }

    /// Builder: whether to fit an intercept term (default `true`).
    pub fn with_fit_intercept(mut self, b: bool) -> Self {
        self.fit_intercept = b;
        self
    }

    /// Builder: choose the solver (default [`LogisticSolver::Cholesky`]).
    pub fn with_solver(mut self, s: LogisticSolver) -> Self {
        self.solver = s;
        self
    }

    /// Builder: maximum IRLS iterations (default `100`).
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Builder: convergence tolerance (default `1e-4`). Stops when the maximum
    /// coefficient change in a sweep drops below this value.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Fitted coefficients `β`.
    pub fn coef(&self) -> &[f64] {
        &self.coef_
    }

    /// Fitted intercept term.
    pub fn intercept(&self) -> f64 {
        self.intercept_
    }

    /// Number of features seen during `fit`.
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
    }

    /// Number of IRLS iterations actually run.
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// Per-class probability estimates in sklearn-compatible column order:
    /// `P(class=0)`, then `P(class=1)`.
    pub fn predict_proba(&self, x: &Matrix) -> Result<Matrix> {
        <Self as PredictProba>::predict_proba(self, x)
    }

    /// Positive-class probability `P(y = 1 | x)` for each row.
    ///
    /// This is a convenience for binary workflows. Use
    /// [`predict_proba`](Self::predict_proba) when sklearn-compatible
    /// two-column probability output is needed.
    pub fn predict_positive_proba(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.positive_probabilities(x)
    }

    /// Backward-compatible alias for [`predict`](Predictor::predict).
    pub fn predict_class(&self, x: &Matrix) -> Result<Vec<f64>> {
        <Self as Predictor>::predict(self, x)
    }

    /// Mean accuracy of the prediction, mirroring `estimator.score` in sklearn.
    pub fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64> {
        let pred = self.predict_class(x)?;
        crate::metrics::classification::accuracy_score(y, &pred)
    }

    fn positive_probabilities(&self, x: &Matrix) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("LogisticRegression".into()));
        }
        if x.ncols() != self.n_features_in_ {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.n_features_in_),
                actual: format!("{} features", x.ncols()),
            });
        }
        let p = self.n_features_in_;
        let beta = &self.coef_;
        let intercept = self.intercept_;
        let src = x.as_slice();
        let n = x.nrows();
        let mut out = vec![0.0; n];
        for i in 0..n {
            let row = &src[i * p..(i + 1) * p];
            let mut eta = intercept;
            for j in 0..p {
                eta += beta[j] * row[j];
            }
            out[i] = sigmoid(eta);
        }
        Ok(out)
    }
}

impl Estimator for LogisticRegression {}

impl Predictor for LogisticRegression {
    fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        let n = x.nrows();
        let p = x.ncols();
        if n == 0 {
            return Err(DatarustError::EmptyInput("X has no rows".into()));
        }
        if p == 0 {
            return Err(DatarustError::EmptyInput("X has no columns".into()));
        }
        if y.len() != n {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} targets", n),
                actual: format!("{} targets", y.len()),
            });
        }
        // Validate binary labels in {0, 1}.
        for (i, &v) in y.iter().enumerate() {
            if v != 0.0 && v != 1.0 {
                return Err(DatarustError::InvalidInput(format!(
                    "LogisticRegression requires binary targets in {{0.0, 1.0}}, found {v} at index {i}"
                )));
            }
        }
        if self.max_iter == 0 {
            return Err(DatarustError::InvalidConfig("max_iter must be > 0".into()));
        }

        let x_slice = x.as_slice();
        // Center X by its (unweighted) column mean when fitting an intercept.
        let (design, x_mean) = if self.fit_intercept {
            let x_mean = stats::column_mean_flat(x_slice, n, p);
            let mut xc = vec![0.0; n * p];
            for i in 0..n {
                for j in 0..p {
                    xc[i * p + j] = x_slice[i * p + j] - x_mean[j];
                }
            }
            (xc, x_mean)
        } else {
            (x_slice.to_vec(), Vec::new())
        };
        let y_mean = y.iter().sum::<f64>() / n as f64;

        // IRLS loop on centered features.
        let mut beta = vec![0.0_f64; p];
        let mut n_iter = 0;
        for _ in 0..self.max_iter {
            n_iter += 1;
            // Compute p_i, W_i and adjusted response z_i for each sample.
            let mut w = vec![0.0_f64; n];
            let mut z = vec![0.0_f64; n];
            for i in 0..n {
                let row = &design[i * p..(i + 1) * p];
                let eta: f64 = (0..p).map(|j| beta[j] * row[j]).sum();
                let p_i = sigmoid(eta);
                // Clamp weight away from 0/1 to avoid degeneracy under separation.
                let wi = (p_i * (1.0 - p_i)).clamp(1e-12, 1.0);
                w[i] = wi;
                z[i] = eta + (y[i] - p_i) / wi;
            }
            // Build weighted design X_w (row i scaled by sqrt(w_i)) and solve
            // (X_wᵀ X_w) β = X_wᵀ (sqrt(w) ⊙ z)  ==  (Xᵀ W X) β = Xᵀ W z.
            let mut xw_flat = vec![0.0; n * p];
            for i in 0..n {
                let sw = w[i].sqrt();
                for j in 0..p {
                    xw_flat[i * p + j] = design[i * p + j] * sw;
                }
            }
            let xw = Matrix::from_flat(n, p, xw_flat)?;
            let xtw = xw.transpose();
            let xtwx = xtw.matmul(&xw)?; // p×p
            let wz: Vec<f64> = (0..n).map(|i| w[i].sqrt() * z[i]).collect();
            let wz_col = Matrix::from_flat(n, 1, wz)?;
            let xtwz = xtw.matmul(&wz_col)?; // p×1
            let rhs = xtwz.as_slice().to_vec();

            let new_beta = match self.solver {
                LogisticSolver::Cholesky => cholesky::solve_spd_system(xtwx.as_slice(), p, &rhs),
                LogisticSolver::Svd => {
                    super::linear_regression::solve_via_eig_pinv(xtwx.as_slice(), &rhs, p)
                }
            }?;
            // Check convergence by max coordinate change.
            let max_delta = beta
                .iter()
                .zip(new_beta.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            beta = new_beta;
            if max_delta < self.tol {
                break;
            }
        }

        // Recover intercept: b = logit(ȳ) − x̄ · β.
        let intercept = if self.fit_intercept {
            let base = y_mean.clamp(1e-12, 1.0 - 1e-12);
            let logit_base = (base / (1.0 - base)).ln();
            logit_base
                - x_mean
                    .iter()
                    .zip(beta.iter())
                    .map(|(m, &bj)| m * bj)
                    .sum::<f64>()
        } else {
            0.0
        };

        self.coef_ = beta;
        self.intercept_ = intercept;
        self.n_features_in_ = p;
        self.n_iter_ = n_iter;
        self.fitted = true;
        Ok(())
    }

    fn predict(&self, x: &Matrix) -> Result<Vec<f64>> {
        Ok(self
            .positive_probabilities(x)?
            .into_iter()
            .map(|p| if p >= 0.5 { 1.0 } else { 0.0 })
            .collect())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl Classifier for LogisticRegression {}

impl PredictProba for LogisticRegression {
    fn predict_proba(&self, x: &Matrix) -> Result<Matrix> {
        let positive = self.positive_probabilities(x)?;
        let mut probabilities = Vec::with_capacity(positive.len() * 2);
        for p in positive {
            probabilities.push(1.0 - p);
            probabilities.push(p);
        }
        Matrix::from_flat(x.nrows(), 2, probabilities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// Linearly separable dataset: y = 1 when x0 > 0.
    fn separable() -> (Matrix, Vec<f64>) {
        let rows: Vec<Vec<f64>> = (-5..=5)
            .map(|i| vec![i as f64 * 0.5])
            .filter(|r| r[0] != 0.0)
            .collect();
        let x = Matrix::new(rows.clone()).unwrap();
        let y: Vec<f64> = rows
            .iter()
            .map(|r| if r[0] > 0.0 { 1.0 } else { 0.0 })
            .collect();
        (x, y)
    }

    #[test]
    fn classifies_separable_data() {
        let (x, y) = separable();
        let mut model = LogisticRegression::new().with_max_iter(100);
        model.fit(&x, &y).unwrap();
        let classes = model.predict_class(&x).unwrap();
        assert_eq!(classes, y);
    }

    #[test]
    fn predict_returns_hard_labels() {
        let (x, y) = separable();
        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();
        assert_eq!(model.predict(&x).unwrap(), y);
    }

    #[test]
    fn predict_proba_has_two_normalized_columns() {
        let (x, y) = separable();
        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();
        let probabilities = model.predict_proba(&x).unwrap();
        assert_eq!(probabilities.ncols(), 2);
        for (i, &label) in y.iter().enumerate() {
            let p0 = probabilities.get(i, 0);
            let p1 = probabilities.get(i, 1);
            assert!(approx(p0 + p1, 1.0, 1e-12));
            assert_eq!(if p1 >= 0.5 { 1.0 } else { 0.0 }, label);
        }
    }

    #[test]
    fn n_iter_recorded() {
        let (x, y) = separable();
        let mut model = LogisticRegression::new().with_max_iter(50);
        model.fit(&x, &y).unwrap();
        assert!(model.n_iter() > 0 && model.n_iter() <= 50);
    }

    #[test]
    fn cholesky_and_svd_agree() {
        // On perfectly separable data logistic coefficients diverge, so compare
        // the two solvers on overlapping (non-separable) data where the MLE is
        // finite and well-defined.
        let rows: Vec<Vec<f64>> = vec![
            vec![-2.0, 0.5],
            vec![-1.5, 0.3],
            vec![-1.0, -0.2],
            vec![-0.5, 0.1],
            vec![0.0, 0.4],
            vec![0.5, -0.1],
            vec![1.0, 0.2],
            vec![1.5, -0.3],
            vec![2.0, 0.0],
            vec![-0.8, 0.6], // overlap: x0<0 but labeled 1
            vec![0.8, -0.4], // overlap: x0>0 but labeled 0
            vec![1.2, 0.5],
        ];
        let x = Matrix::new(rows.clone()).unwrap();
        let y: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0];

        let mut chol = LogisticRegression::new().with_solver(LogisticSolver::Cholesky);
        chol.fit(&x, &y).unwrap();
        let mut svd = LogisticRegression::new().with_solver(LogisticSolver::Svd);
        svd.fit(&x, &y).unwrap();
        // With overlapping classes the MLE is finite; both solvers must agree.
        for i in 0..2 {
            assert!(
                (chol.coef()[i] - svd.coef()[i]).abs() < 1.0,
                "solver disagreement at {i}: chol={} svd={}",
                chol.coef()[i],
                svd.coef()[i]
            );
        }
        // Predictions should agree closely.
        let pc = chol.predict_positive_proba(&x).unwrap();
        let ps = svd.predict_positive_proba(&x).unwrap();
        for (a, b) in pc.iter().zip(ps.iter()) {
            assert!((a - b).abs() < 0.05, "disagreement: {a} vs {b}");
        }
    }

    #[test]
    fn non_binary_label_rejected() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let mut model = LogisticRegression::new();
        let err = model.fit(&x, &[0.0, 1.0, 2.0]).unwrap_err();
        assert!(matches!(err, DatarustError::InvalidInput(_)));
    }

    #[test]
    fn predict_before_fit_errors() {
        let model = LogisticRegression::new();
        let x = Matrix::new(vec![vec![1.0]]).unwrap();
        assert!(matches!(
            model.predict(&x).unwrap_err(),
            DatarustError::NotFitted(_)
        ));
    }

    #[test]
    fn predict_shape_mismatch() {
        let (x, y) = separable();
        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();
        let bad = Matrix::new(vec![vec![1.0, 2.0]]).unwrap();
        assert!(matches!(
            model.predict(&bad).unwrap_err(),
            DatarustError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn score_returns_accuracy() {
        let (x, y) = separable();
        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();
        let acc = model.score(&x, &y).unwrap();
        assert!(approx(acc, 1.0, 1e-9), "accuracy={acc}");
    }

    #[test]
    fn intercept_recovered() {
        let (x, y) = separable();
        let mut model = LogisticRegression::new().with_max_iter(200);
        model.fit(&x, &y).unwrap();
        // For balanced separable data, intercept should be near 0.
        assert!(
            model.intercept().abs() < 1.0,
            "intercept={}",
            model.intercept()
        );
    }

    #[test]
    fn fit_intercept_false() {
        let (x, y) = separable();
        let mut model = LogisticRegression::new()
            .with_fit_intercept(false)
            .with_max_iter(100);
        model.fit(&x, &y).unwrap();
        assert!(model.intercept().abs() < 1e-12);
        // Should still classify correctly when data is separable.
        let classes = model.predict_class(&x).unwrap();
        assert_eq!(classes, y);
    }

    #[test]
    fn multivariate_separable() {
        // 2D with non-collinear features: y = 1 when x0 > 0 (feature 1 is noise).
        // Skip x0 == 0 so the decision boundary is unambiguous.
        let rows: Vec<Vec<f64>> = (0..40)
            .map(|i| {
                let i = i as f64 - 19.5; // shifts so no sample sits exactly at 0
                vec![i, (i * 0.3).sin()]
            })
            .collect();
        let x = Matrix::new(rows.clone()).unwrap();
        let y: Vec<f64> = rows
            .iter()
            .map(|r| if r[0] > 0.0 { 1.0 } else { 0.0 })
            .collect();
        let mut model = LogisticRegression::new().with_max_iter(100);
        model.fit(&x, &y).unwrap();
        let classes = model.predict_class(&x).unwrap();
        assert_eq!(classes, y);
    }
}
