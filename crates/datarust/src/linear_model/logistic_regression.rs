//! Logistic regression for binary and multiclass classification.
//!
//! Mirrors `sklearn.linear_model.LogisticRegression` (no regularization).
//!
//! - **Binary** targets are fit via Iteratively Reweighted Least
//!   Squares (IRLS / Newton-Raphson on the logistic loss). `predict` returns
//!   the original hard labels and `predict_proba` returns an `(n, 2)` matrix.
//! - **Multiclass** targets are fit via multinomial (softmax)
//!   logistic regression with Newton-Raphson on the cross-entropy loss. The
//!   reference class is the last one; coefficients for the remaining classes are
//!   estimated jointly. `predict` returns the highest-probability class label
//!   and `predict_proba` returns an `(n, k)` matrix.
//!
//! Both solvers use the shared Cholesky (default) or eigen-pseudo-inverse (SVD)
//! linear system solver from [`crate::linalg`].

use crate::error::{DatarustError, Result};
use crate::label_space::LabelSpace;
use crate::linalg::cholesky;
use crate::matrix::Matrix;
use crate::stats;
use crate::traits::{Classifier, Estimator, PredictProba, Predictor};

/// Solver strategy for [`LogisticRegression`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LogisticSolver {
    /// Newton step via Cholesky decomposition of the Hessian each iteration
    /// (default). Fast; requires full column rank (quasi-separation may make
    /// the Hessian singular).
    #[default]
    Cholesky,
    /// Newton step via eigendecomposition pseudo-inverse. Robust to
    /// rank-deficiency.
    Svd,
}

/// Logistic regression for binary and multiclass classification.
///
/// `fit` accepts arbitrary non-negative integer-valued labels, compacts them
/// internally, and dispatches to the binary IRLS or multinomial softmax solver.
/// Predictions always use the original labels supplied during fitting.
///
/// `predict` returns hard class labels (one per row); `predict_proba` returns a
/// probability matrix with one column per class in sorted label order.
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
    /// One coefficient row per class. For binary (`k = 2`) this holds a single
    /// row (the positive-class coefficients); the negative-class coefficients
    /// are implicitly zero.
    coef_: Vec<Vec<f64>>,
    /// One intercept per class (binary: single entry).
    intercept_: Vec<f64>,
    /// Sorted unique class labels seen during `fit`.
    classes_: Vec<f64>,
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

/// Numerically stable softmax over a slice of logits, computed in place into
/// `out`. Subtracts the max logit before exponentiating.
#[inline]
fn softmax(logits: &[f64], out: &mut [f64]) {
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for (o, &l) in out.iter_mut().zip(logits.iter()) {
        let e = (l - m).exp();
        *o = e;
        sum += e;
    }
    for o in out.iter_mut() {
        *o /= sum;
    }
}

impl LogisticRegression {
    fn validate_fitted_state(&self) -> Result<()> {
        let k = self.classes_.len();
        let expected_rows = k.saturating_sub(1);
        if k < 2
            || self.coef_.len() != expected_rows
            || self.intercept_.len() != expected_rows
            || self
                .coef_
                .iter()
                .any(|row| row.len() != self.n_features_in_)
            || self.classes_.iter().any(|v| !v.is_finite())
            || self.intercept_.iter().any(|v| !v.is_finite())
            || self.coef_.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(DatarustError::InvalidInput(
                "LogisticRegression has inconsistent fitted state".into(),
            ));
        }
        Ok(())
    }

    /// Creates a new estimator: `fit_intercept = true`, `solver = Cholesky`,
    /// `max_iter = 100`, `tol = 1e-4`.
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
            solver: LogisticSolver::Cholesky,
            max_iter: 100,
            tol: 1e-4,
            coef_: Vec::new(),
            intercept_: Vec::new(),
            classes_: Vec::new(),
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

    /// Builder: maximum Newton iterations (default `100`). Must be at least
    /// `1`.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Builder: convergence tolerance (default `1e-4`). Stops when the maximum
    /// coefficient change in an iteration drops below this value. Must be
    /// finite and `>= 0`.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Fitted coefficients: one row per class.
    ///
    /// For binary classification (`k = 2`) this returns a single row — the
    /// positive-class coefficients; the negative-class coefficients are
    /// implicitly zero. For multiclass (`k > 2`) it returns `k − 1` rows, one
    /// per non-reference class.
    pub fn coef(&self) -> &[Vec<f64>] {
        &self.coef_
    }

    /// Fitted intercept terms, one per class (binary: single entry).
    pub fn intercept(&self) -> &[f64] {
        &self.intercept_
    }

    /// Sorted unique class labels seen during `fit`.
    pub fn classes(&self) -> &[f64] {
        &self.classes_
    }

    /// Number of features seen during `fit`.
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
    }

    /// Number of Newton iterations actually run.
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// Per-class probability estimates in sorted-label column order.
    ///
    /// For binary input the result is `(n, 2)`; for multiclass it is `(n, k)`.
    /// In both cases column `i` is the probability of `classes()[i]`.
    pub fn predict_proba(&self, x: &Matrix) -> Result<Matrix> {
        <Self as PredictProba>::predict_proba(self, x)
    }

    /// Probability of the second sorted class for each row.
    ///
    /// For the conventional `{0, 1}` target this is `P(y = 1 | x)`. For an
    /// arbitrary binary target such as `{2, 5}`, it is `P(y = 5 | x)`. Use
    /// [`predict_proba_for_class`](Self::predict_proba_for_class) when the
    /// desired original label should be explicit.
    pub fn predict_positive_proba(&self, x: &Matrix) -> Result<Vec<f64>> {
        if self.classes_.len() != 2 {
            return Err(DatarustError::InvalidInput(format!(
                "predict_positive_proba requires a binary model (2 classes), this model has {} classes; use predict_proba",
                self.classes_.len()
            )));
        }
        self.positive_probabilities_binary(x)
    }

    /// Probability assigned to one fitted class label for every row.
    ///
    /// This is the unambiguous alternative to
    /// [`predict_positive_proba`](Self::predict_positive_proba) when binary
    /// labels are not `{0, 1}` or when a multiclass probability column is
    /// needed.
    pub fn predict_proba_for_class(&self, x: &Matrix, class_label: f64) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("LogisticRegression".into()));
        }
        let class_index = self
            .classes_
            .iter()
            .position(|&label| label == class_label)
            .ok_or_else(|| {
                DatarustError::UnknownLabel(format!(
                    "class {class_label}; fitted classes are {:?}",
                    self.classes_
                ))
            })?;
        let probabilities = self.probabilities(x)?;
        Ok((0..probabilities.nrows())
            .map(|row| probabilities.get(row, class_index))
            .collect())
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

    /// Binary-mode probabilities for the second sorted fitted class.
    fn positive_probabilities_binary(&self, x: &Matrix) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("LogisticRegression".into()));
        }
        self.validate_fitted_state()?;
        if x.ncols() != self.n_features_in_ {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.n_features_in_),
                actual: format!("{} features", x.ncols()),
            });
        }
        x.validate_finite()?;
        let p = self.n_features_in_;
        let beta = &self.coef_[0];
        let intercept = self.intercept_[0];
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

    /// Compute the per-class probability matrix for arbitrary k.
    fn probabilities(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("LogisticRegression".into()));
        }
        self.validate_fitted_state()?;
        if x.ncols() != self.n_features_in_ {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.n_features_in_),
                actual: format!("{} features", x.ncols()),
            });
        }
        x.validate_finite()?;
        let n = x.nrows();
        let p = self.n_features_in_;
        let k = self.classes_.len();
        if k == 2 {
            // Binary: single positive-class row, negative class is the complement.
            let pos = self.positive_probabilities_binary(x)?;
            let mut data = Vec::with_capacity(n * 2);
            for q in pos {
                data.push(1.0 - q);
                data.push(q);
            }
            return Matrix::from_flat(n, 2, data);
        }
        // Multiclass: k−1 coefficient rows, reference class = last.
        let mut data = vec![0.0; n * k];
        let src = x.as_slice();
        let mut logits = vec![0.0; k];
        let mut probs = vec![0.0; k];
        for i in 0..n {
            let row = &src[i * p..(i + 1) * p];
            // Non-reference classes 0..k-2 use fitted coefficients.
            for (c, slot) in logits.iter_mut().enumerate().take(k - 1) {
                let mut eta = self.intercept_[c];
                let beta = &self.coef_[c];
                for j in 0..p {
                    eta += beta[j] * row[j];
                }
                *slot = eta;
            }
            // Reference class (k−1) has zero logit.
            logits[k - 1] = 0.0;
            softmax(&logits, &mut probs);
            for c in 0..k {
                data[i * k + c] = probs[c];
            }
        }
        Matrix::from_flat(n, k, data)
    }

    // ── Binary IRLS fit (existing algorithm, refactored to store per-class) ──

    fn fit_binary(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        let n = x.nrows();
        let p = x.ncols();
        let x_slice = x.as_slice();
        // Center X by its (unweighted) column mean when fitting an intercept.
        let (design, x_mean) = if self.fit_intercept {
            let x_mean = stats::column_mean_flat(x_slice, n, p);
            let mut xc = vec![0.0; n * p];
            for i in 0..n {
                for (j, m) in x_mean.iter().enumerate() {
                    xc[i * p + j] = x_slice[i * p + j] - m;
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
            // Solve (Xᵀ W X) β = Xᵀ W z. Build the weighted Gram matrix and
            // right-hand side directly by rank-1 accumulation over the rows —
            // one cache-friendly pass instead of materialising the weighted
            // design matrix, transposing it, and running the general matmul.
            // Only the lower triangle of G is accumulated (half the work) and
            // mirrored afterwards so the eigen pseudo-inverse sees a full
            // symmetric matrix.
            let mut g = vec![0.0_f64; p * p];
            let mut rhs = vec![0.0_f64; p];
            for i in 0..n {
                let wi = w[i];
                let row = &design[i * p..(i + 1) * p];
                let wz = wi * z[i];
                // Xᵀ W z
                for (r, &xj) in rhs.iter_mut().zip(row.iter()) {
                    *r += wz * xj;
                }
                // Xᵀ W X, lower triangle only.
                for j in 0..p {
                    let wxj = wi * row[j];
                    let gj = &mut g[j * p..(j + 1) * p];
                    for (slot, &xjp) in gj.iter_mut().take(j + 1).zip(row.iter()) {
                        *slot += wxj * xjp;
                    }
                }
            }
            for j in 0..p {
                for jp in (j + 1)..p {
                    g[j * p + jp] = g[jp * p + j];
                }
            }

            let new_beta = match self.solver {
                LogisticSolver::Cholesky => cholesky::solve_spd_system(&g, p, &rhs),
                LogisticSolver::Svd => super::linear_regression::solve_via_eig_pinv(&g, &rhs, p),
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

        self.coef_ = vec![beta];
        self.intercept_ = vec![intercept];
        self.classes_ = vec![0.0, 1.0];
        self.n_features_in_ = p;
        self.n_iter_ = n_iter;
        self.fitted = true;
        Ok(())
    }

    // ── Multinomial (softmax) Newton-Raphson fit ────────────────────────
    //
    // Parameterization: the last class is the reference (coefficients zero).
    // We estimate `B`, an `(k-1) × p` matrix flattened row-major into a vector
    // of length `(k-1)*p`. Intercepts are absorbed via column-mean centering
    // and recovered at the end.
    fn fit_multiclass(&mut self, x: &Matrix, y_idx: &[usize], classes: &[f64]) -> Result<()> {
        let n = x.nrows();
        let p = x.ncols();
        let x_slice = x.as_slice();
        let k = classes.len();
        let km1 = k - 1;

        let (design, x_mean) = if self.fit_intercept {
            let x_mean = stats::column_mean_flat(x_slice, n, p);
            let mut xc = vec![0.0; n * p];
            for i in 0..n {
                for (j, m) in x_mean.iter().enumerate() {
                    xc[i * p + j] = x_slice[i * p + j] - m;
                }
            }
            (xc, x_mean)
        } else {
            (x_slice.to_vec(), Vec::new())
        };

        // One-hot indicator matrix Y (n × km1) for the non-reference classes.
        let mut y_onehot = vec![0.0_f64; n * km1];
        for i in 0..n {
            if y_idx[i] < km1 {
                y_onehot[i * km1 + y_idx[i]] = 1.0;
            }
        }

        // Parameter vector (km1 * p), initialized to zeros.
        let mut beta = vec![0.0_f64; km1 * p];
        let mut intercept_vec = vec![0.0_f64; km1];
        let mut n_iter = 0;

        for _ in 0..self.max_iter {
            n_iter += 1;
            // Compute current probabilities P (n × k) and log-likelihood gradient.
            let mut probs = vec![0.0_f64; n * k]; // full k columns
            for i in 0..n {
                let row = &design[i * p..(i + 1) * p];
                let mut logits = vec![0.0_f64; k];
                for c in 0..km1 {
                    let mut eta = 0.0;
                    let bc = &beta[c * p..(c + 1) * p];
                    for j in 0..p {
                        eta += bc[j] * row[j];
                    }
                    logits[c] = eta;
                }
                logits[km1] = 0.0; // reference
                let p_row = &mut probs[i * k..(i + 1) * k];
                softmax(&logits, p_row);
            }

            // Gradient of the *negative* log-likelihood (minimization framing):
            // g = Xᵀ (P[:, :km1] − Y). The Newton step then subtracts the
            // search direction from β.
            let mut grad = vec![0.0_f64; km1 * p];
            for i in 0..n {
                let row = &design[i * p..(i + 1) * p];
                for c in 0..km1 {
                    let resid = probs[i * k + c] - y_onehot[i * km1 + c];
                    for j in 0..p {
                        grad[c * p + j] += row[j] * resid;
                    }
                }
            }

            // Hessian H (km1*p × km1*p), block structure:
            //   H[(c,p),(c',p')] = Σ_i x_ip * x_ip' * P_ic * (δ_{cc'} − P_ic')
            // Each (c, c') block is a symmetric rank-1-sum; accumulate only the
            // lower triangle of every block and mirror the blocks afterwards.
            let d = km1 * p;
            let mut hess = vec![0.0_f64; d * d];
            for i in 0..n {
                let row = &design[i * p..(i + 1) * p];
                for c in 0..km1 {
                    let pc = probs[i * k + c];
                    for cp in 0..km1 {
                        let coef = pc * ((if c == cp { 1.0 } else { 0.0 }) - probs[i * k + cp]);
                        if coef.abs() < 1e-15 {
                            continue;
                        }
                        let block_row = c * p;
                        let block_col = cp * p;
                        for j in 0..p {
                            let wxj = row[j] * coef;
                            for jp in 0..=j {
                                hess[(block_row + j) * d + (block_col + jp)] += wxj * row[jp];
                            }
                        }
                    }
                }
            }
            // Mirror the lower triangle of each symmetric block.
            for c in 0..km1 {
                for cp in 0..km1 {
                    let block_row = c * p;
                    let block_col = cp * p;
                    for j in 0..p {
                        for jp in (j + 1)..p {
                            hess[(block_row + jp) * d + (block_col + j)] =
                                hess[(block_row + j) * d + (block_col + jp)];
                        }
                    }
                }
            }

            // Newton step for the minimization framing: H Δ = g, β -= Δ. H is
            // the Hessian of the negative log-likelihood, which is symmetric
            // positive-semidefinite. Add a tiny ridge (Levenberg–Marquardt
            // damping) to keep the matrix positive-definite when probabilities
            // saturate (e.g. under near-perfect separation). If the Cholesky
            // solver still fails, fall back to the SVD pseudo-inverse.
            let mut regularized = hess.clone();
            let ridge = 1e-10;
            for i in 0..d {
                regularized[i * d + i] += ridge;
            }
            let delta = match self.solver {
                LogisticSolver::Cholesky => cholesky::solve_spd_system(&regularized, d, &grad)
                    .or_else(|_| {
                        super::linear_regression::solve_via_eig_pinv(&regularized, &grad, d)
                    }),
                LogisticSolver::Svd => {
                    super::linear_regression::solve_via_eig_pinv(&regularized, &grad, d)
                }
            }?;

            let max_delta = delta.iter().cloned().fold(0.0_f64, f64::max).abs();
            for (b, &dd) in beta.iter_mut().zip(delta.iter()) {
                *b -= dd;
            }
            if max_delta < self.tol {
                break;
            }
        }

        // Recover intercepts from centering.
        if self.fit_intercept {
            for c in 0..km1 {
                let bc = &beta[c * p..(c + 1) * p];
                let dot: f64 = x_mean.iter().zip(bc.iter()).map(|(m, &bj)| m * bj).sum();
                intercept_vec[c] = -dot;
            }
        }

        // Pack coefficient rows and intercepts.
        let coef_rows: Vec<Vec<f64>> = (0..km1)
            .map(|c| beta[c * p..(c + 1) * p].to_vec())
            .collect();
        self.coef_ = coef_rows;
        self.intercept_ = intercept_vec;
        self.classes_ = classes.to_vec();
        self.n_features_in_ = p;
        self.n_iter_ = n_iter;
        self.fitted = true;
        Ok(())
    }
}

impl Estimator for LogisticRegression {}

impl crate::traits::Params for LogisticRegression {
    fn get_params(&self) -> Vec<(&'static str, crate::traits::ParamValue)> {
        use crate::traits::ParamValue;
        vec![
            ("max_iter", ParamValue::Int(self.max_iter)),
            ("tol", ParamValue::Float(self.tol)),
            ("fit_intercept", ParamValue::Bool(self.fit_intercept)),
        ]
    }

    fn set_params(&mut self, name: &str, value: crate::traits::ParamValue) -> Result<()> {
        use crate::traits::ParamValue;
        match (name, value) {
            ("max_iter", ParamValue::Int(0)) => {
                return Err(DatarustError::InvalidConfig("max_iter must be > 0".into()));
            }
            ("max_iter", ParamValue::Int(v)) => self.max_iter = v,
            ("tol", ParamValue::Float(v)) if !v.is_finite() || v < 0.0 => {
                return Err(DatarustError::InvalidConfig(format!(
                    "tol must be finite and >= 0, got {v}"
                )));
            }
            ("tol", ParamValue::Float(v)) => self.tol = v,
            ("fit_intercept", ParamValue::Bool(v)) => self.fit_intercept = v,
            (other, _) => {
                return Err(DatarustError::InvalidInput(format!(
                    "LogisticRegression has no tunable parameter '{other}'"
                )));
            }
        }
        self.fitted = false;
        Ok(())
    }
}

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
        x.validate_finite()?;
        if self.max_iter == 0 {
            return Err(DatarustError::InvalidConfig("max_iter must be > 0".into()));
        }
        if !self.tol.is_finite() || self.tol < 0.0 {
            return Err(DatarustError::InvalidConfig(format!(
                "tol must be finite and >= 0, got {}",
                self.tol
            )));
        }

        // Validate labels once and share the same compact vocabulary used by
        // classification metrics.
        let label_space = LabelSpace::fit(y)?;
        let classes = label_space.labels().to_vec();
        if classes.len() < 2 {
            return Err(DatarustError::InvalidInput(
                "LogisticRegression requires at least 2 distinct classes".into(),
            ));
        }
        let k = classes.len();

        if k == 2 {
            // Binary path: relabel to {0, 1} and use the faster IRLS solver.
            let c0 = classes[0];
            let y_bin: Vec<f64> = y.iter().map(|&v| if v == c0 { 0.0 } else { 1.0 }).collect();
            // fit_binary sets classes_ to {0, 1}; override with the original
            // labels so non-canonical labels (e.g. {2, 5}) are preserved.
            self.fit_binary(x, &y_bin)?;
            self.classes_ = classes;
            return Ok(());
        }

        // For multiclass, sklearn uses the classes in sorted order; remap y.
        let y_idx: Vec<usize> = y
            .iter()
            .map(|&label| label_space.encode(label))
            .collect::<Result<_>>()?;
        // The fit_multiclass routine uses compact indices internally while
        // preserving these original sorted labels for prediction and
        // probability-column metadata.
        self.fit_multiclass(x, &y_idx, &classes)
    }

    fn predict(&self, x: &Matrix) -> Result<Vec<f64>> {
        let probs = self.probabilities(x)?;
        let n = probs.nrows();
        let k = probs.ncols();
        let mut out = vec![0.0; n];
        let data = probs.as_slice();
        for i in 0..n {
            let row = &data[i * k..(i + 1) * k];
            let best = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            out[i] = self.classes_[best];
        }
        Ok(out)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl Classifier for LogisticRegression {}

impl PredictProba for LogisticRegression {
    fn predict_proba(&self, x: &Matrix) -> Result<Matrix> {
        self.probabilities(x)
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
                (chol.coef()[0][i] - svd.coef()[0][i]).abs() < 1.0,
                "solver disagreement at {i}: chol={} svd={}",
                chol.coef()[0][i],
                svd.coef()[0][i]
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
    fn non_integer_label_rejected() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let mut model = LogisticRegression::new();
        let err = model.fit(&x, &[0.0, 1.0, 2.5]).unwrap_err();
        assert!(matches!(err, DatarustError::InvalidInput(_)));
    }

    #[test]
    fn invalid_solver_configuration_is_rejected() {
        let (x, y) = separable();
        let mut zero_iterations = LogisticRegression::new().with_max_iter(0);
        assert!(matches!(
            zero_iterations.fit(&x, &y),
            Err(DatarustError::InvalidConfig(_))
        ));
        for tol in [-1.0, f64::NAN, f64::INFINITY] {
            let mut model = LogisticRegression::new().with_tol(tol);
            assert!(matches!(
                model.fit(&x, &y),
                Err(DatarustError::InvalidConfig(_))
            ));
        }
    }

    #[test]
    fn params_reject_invalid_solver_values_without_mutating_state() {
        use crate::traits::{ParamValue, Params};

        let mut model = LogisticRegression::new();
        assert!(matches!(
            model.set_params("max_iter", ParamValue::Int(0)),
            Err(DatarustError::InvalidConfig(_))
        ));
        assert!(matches!(
            model.set_params("tol", ParamValue::Float(f64::NAN)),
            Err(DatarustError::InvalidConfig(_))
        ));
        assert_eq!(
            model.get_params(),
            vec![
                ("max_iter", ParamValue::Int(100)),
                ("tol", ParamValue::Float(1e-4)),
                ("fit_intercept", ParamValue::Bool(true)),
            ]
        );
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
    fn fit_intercept_false() {
        let (x, y) = separable();
        let mut model = LogisticRegression::new()
            .with_fit_intercept(false)
            .with_max_iter(100);
        model.fit(&x, &y).unwrap();
        assert!(model.intercept()[0].abs() < 1e-12);
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

    // ── Multiclass tests ──────────────────────────────────────────────

    /// Three linearly separable clusters in 2-D space, placed at the vertices
    /// of a triangle so a softmax classifier can separate them with linear
    /// decision boundaries. Returns (x, y) with labels {0, 1, 2}.
    fn separable_3class_2d() -> (Matrix, Vec<f64>) {
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut y = Vec::new();
        // Class 0: bottom-left corner.
        for _ in 0..12 {
            rows.push(vec![-5.0, -5.0]);
            y.push(0.0);
        }
        // Class 1: top-center.
        for _ in 0..12 {
            rows.push(vec![0.0, 5.0]);
            y.push(1.0);
        }
        // Class 2: bottom-right.
        for _ in 0..12 {
            rows.push(vec![5.0, -5.0]);
            y.push(2.0);
        }
        (Matrix::new(rows).unwrap(), y)
    }

    #[test]
    fn multiclass_classifies_separable() {
        let (x, y) = separable_3class_2d();
        let mut model = LogisticRegression::new().with_max_iter(200);
        model.fit(&x, &y).unwrap();
        let pred = model.predict_class(&x).unwrap();
        assert_eq!(pred.len(), y.len());
        let acc = model.score(&x, &y).unwrap();
        assert!(approx(acc, 1.0, 1e-9), "multiclass accuracy={acc}");
    }

    #[test]
    fn multiclass_predict_proba_normalizes() {
        let (x, y) = separable_3class_2d();
        let mut model = LogisticRegression::new().with_max_iter(200);
        model.fit(&x, &y).unwrap();
        let probs = model.predict_proba(&x).unwrap();
        assert_eq!(probs.ncols(), 3);
        for i in 0..x.nrows() {
            let sum = (0..3).map(|c| probs.get(i, c)).sum::<f64>();
            assert!(approx(sum, 1.0, 1e-9), "row {i} sums to {sum}");
        }
    }

    #[test]
    fn multiclass_classes_sorted() {
        let (x, y) = separable_3class_2d();
        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();
        assert_eq!(model.classes(), &[0.0, 1.0, 2.0]);
    }

    #[test]
    fn multiclass_preserves_non_contiguous_labels() {
        let (x, y) = separable_3class_2d();
        let remapped: Vec<f64> = y
            .iter()
            .map(|label| match *label as usize {
                0 => 2.0,
                1 => 5.0,
                _ => 9.0,
            })
            .collect();
        let mut model = LogisticRegression::new().with_max_iter(200);
        model.fit(&x, &remapped).unwrap();

        assert_eq!(model.classes(), &[2.0, 5.0, 9.0]);
        assert_eq!(model.predict(&x).unwrap(), remapped);

        let all_probabilities = model.predict_proba(&x).unwrap();
        let class_five = model.predict_proba_for_class(&x, 5.0).unwrap();
        for (row, &probability) in class_five.iter().enumerate() {
            assert!(approx(probability, all_probabilities.get(row, 1), 1e-12));
        }
        assert!(model.predict_proba_for_class(&x, 7.0).is_err());
    }

    #[test]
    fn multiclass_coef_shape() {
        let (x, y) = separable_3class_2d();
        let mut model = LogisticRegression::new().with_max_iter(100);
        model.fit(&x, &y).unwrap();
        // k-1 = 2 coefficient rows, each of length n_features = 2.
        assert_eq!(model.coef().len(), 2);
        assert_eq!(model.coef()[0].len(), 2);
    }

    #[test]
    fn predict_positive_proba_rejects_multiclass() {
        let (x, y) = separable_3class_2d();
        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();
        let err = model.predict_positive_proba(&x).unwrap_err();
        assert!(matches!(err, DatarustError::InvalidInput(_)));
    }

    #[test]
    fn multiclass_two_features() {
        // 2-D, 3 well-separated Gaussian-ish clusters.
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut y = Vec::new();
        for _ in 0..15 {
            rows.push(vec![-5.0, -5.0]);
            y.push(0.0);
        }
        for _ in 0..15 {
            rows.push(vec![0.0, 5.0]);
            y.push(1.0);
        }
        for _ in 0..15 {
            rows.push(vec![5.0, -5.0]);
            y.push(2.0);
        }
        let x = Matrix::new(rows).unwrap();
        let mut model = LogisticRegression::new().with_max_iter(200);
        model.fit(&x, &y).unwrap();
        let acc = model.score(&x, &y).unwrap();
        assert!(approx(acc, 1.0, 1e-9), "2-D multiclass accuracy={acc}");
    }
}
