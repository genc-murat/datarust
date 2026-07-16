//! Ordinary least-squares linear regression.
//!
//! Mirrors `sklearn.linear_model.LinearRegression`. Estimates coefficients `β`
//! minimising `||Xβ - y||²` via either:
//!
//! - **Cholesky** (default): form the normal equations `XᵀX β = Xᵀy` and solve
//!   with a Cholesky decomposition. Fast and dependency-free; requires `XᵀX`
//!   to be positive-definite (i.e. `X` has full column rank).
//! - **SVD**: compute the pseudo-inverse of `XᵀX` via Jacobi eigendecomposition.
//!   Numerically stable for rank-deficient / collinear inputs, at higher cost.
//!
//! When `fit_intercept` is enabled (the default), the data is centered before
//! fitting and the intercept is recovered as `ȳ − x̄ · β`.

use crate::error::{DatarustError, Result};
use crate::linalg::cholesky;
use crate::matrix::Matrix;
use crate::stats;
use crate::traits::Regressor;

/// Solver strategy for [`LinearRegression`].
///
/// Mirrors the `solver` concept used across `sklearn.linear_model` estimators
/// (e.g. `Ridge(solver="cholesky" | "svd" | ...)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LinearSolver {
    /// Solve the normal equations `XᵀX β = Xᵀy` via Cholesky decomposition.
    /// Fast and dependency-free; requires `X` to have full column rank.
    #[default]
    Cholesky,
    /// Compute the pseudo-inverse of `XᵀX` via eigendecomposition. Stable for
    /// rank-deficient / collinear inputs, at higher cost.
    Svd,
}

/// Ordinary least-squares linear regression.
///
/// Estimates `y ≈ Xβ + b` by minimising the residual sum of squares.
///
/// ```rust
/// use datarust::linear_model::LinearRegression;
/// use datarust::traits::Regressor;
/// use datarust::Matrix;
///
/// let x = Matrix::new(vec![
///     vec![1.0, 2.0],
///     vec![2.0, 1.0],
///     vec![3.0, 4.0],
///     vec![4.0, 3.0],
/// ])?;
/// let y = vec![5.0, 5.0, 11.0, 11.0];
///
/// let mut model = LinearRegression::new();
/// model.fit(&x, &y)?;
/// let pred = model.predict(&x)?;
/// assert!(pred.iter().zip(y.iter()).all(|(p, &t)| (p - t).abs() < 1e-9));
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinearRegression {
    fit_intercept: bool,
    solver: LinearSolver,
    // Fitted state.
    coef_: Vec<f64>,
    intercept_: f64,
    n_features_in_: usize,
    fitted: bool,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearRegression {
    /// Creates a new estimator with defaults: `fit_intercept = true`,
    /// `solver = Cholesky`.
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
            solver: LinearSolver::Cholesky,
            coef_: Vec::new(),
            intercept_: 0.0,
            n_features_in_: 0,
            fitted: false,
        }
    }

    /// Builder: whether to fit an intercept term (default `true`).
    pub fn with_fit_intercept(mut self, b: bool) -> Self {
        self.fit_intercept = b;
        self
    }

    /// Builder: choose the solver (default [`LinearSolver::Cholesky`]).
    pub fn with_solver(mut self, s: LinearSolver) -> Self {
        self.solver = s;
        self
    }

    /// Fitted coefficients `β`, one entry per feature. Empty if not fitted.
    pub fn coef(&self) -> &[f64] {
        &self.coef_
    }

    /// Fitted intercept term `b` (0.0 if `fit_intercept = false`).
    pub fn intercept(&self) -> f64 {
        self.intercept_
    }

    /// Number of features seen during `fit`.
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
    }

    /// R² (coefficient of determination) of the prediction, mirroring
    /// `estimator.score(X, y)` in scikit-learn.
    ///
    /// Returns the [`r2_score`](crate::metrics::regression::r2_score) of `self.predict(X)`
    /// against `y`.
    pub fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64> {
        let pred = self.predict(x)?;
        crate::metrics::regression::r2_score(y, &pred)
    }

    /// Solve the normal equations `A β = c` where `A = XᵀX`, `c = Xᵀy`.
    fn solve_normal(
        a_flat: Vec<f64>,
        c: Vec<f64>,
        p: usize,
        solver: LinearSolver,
    ) -> Result<Vec<f64>> {
        match solver {
            LinearSolver::Cholesky => cholesky::solve_spd_system(&a_flat, p, &c),
            LinearSolver::Svd => solve_via_eig_pinv(&a_flat, &c, p),
        }
    }
}

/// Solve `A x = b` for symmetric PSD `A` via eigendecomposition-based
/// pseudo-inverse: `A⁺ = V Λ⁺ Vᵀ` where `Λ⁺` reciprocates eigenvalues above a
/// tolerance and zeroes the rest. Robust to rank-deficiency.
pub(crate) fn solve_via_eig_pinv(a: &[f64], b: &[f64], p: usize) -> Result<Vec<f64>> {
    let mut a_buf = a.to_vec();
    let (vals, vecs) = crate::decomposition::jacobi::eigh_flat(&mut a_buf, p)
        .ok_or_else(|| DatarustError::Singular("eigendecomposition failed".into()))?;
    // Tolerance scaled by the largest eigenvalue, mirroring numpy.linalg.pinv
    // (rcond * max_singular_value). Here `A`'s eigenvalues are the squares of
    // the singular values of `X`; use rcond = 1e-10 (tighter than pinv's default
    // because A is already squared).
    let max_val = vals.iter().cloned().fold(0.0_f64, f64::max);
    let tol = (max_val.max(1.0) * 1e-10).max(f64::MIN_POSITIVE);
    // x = sum_k (v_k · b / lambda_k) v_k, skipping tiny eigenvalues.
    let mut x = vec![0.0_f64; p];
    for k in 0..p {
        let lambda = vals[k];
        if lambda.abs() <= tol {
            continue;
        }
        // v_k · b
        let vk = &vecs[k * p..(k + 1) * p];
        let mut dot = 0.0;
        for i in 0..p {
            dot += vk[i] * b[i];
        }
        let scale = dot / lambda;
        for i in 0..p {
            x[i] += scale * vk[i];
        }
    }
    Ok(x)
}

impl Regressor for LinearRegression {
    fn name(&self) -> &'static str {
        "LinearRegression"
    }

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

        // Build the design matrix used in the normal equations. When
        // fit_intercept is set, center X and y (equivalent to adding a bias
        // column, but keeps XᵀX p×p and lets the intercept be recovered from
        // the column means afterwards).
        let x_slice = x.as_slice();
        let (design, y_work, x_mean, y_mean) = if self.fit_intercept {
            let x_mean = stats::column_mean_flat(x_slice, n, p);
            let y_mean = y.iter().sum::<f64>() / n as f64;
            let mut xc = vec![0.0; n * p];
            for i in 0..n {
                for j in 0..p {
                    xc[i * p + j] = x_slice[i * p + j] - x_mean[j];
                }
            }
            let yc: Vec<f64> = y.iter().map(|&v| v - y_mean).collect();
            (xc, yc, x_mean, y_mean)
        } else {
            (x_slice.to_vec(), y.to_vec(), Vec::new(), 0.0)
        };

        // Form the normal equations A = XᵀX (p×p) and c = Xᵀy (p) using the
        // already-optimised Matrix::matmul path (auto-vectorising scalar GEMM,
        // or tuned pure-Rust GEMM under the `matrixmultiply` feature).
        let design_mat = Matrix::from_flat(n, p, design)?;
        let xt = design_mat.transpose();
        let xtx = xt.matmul(&design_mat)?; // p×p
                                           // Xᵀy: treat y as an n×1 column.
        let y_col = Matrix::from_flat(n, 1, y_work)?;
        let xty_mat = xt.matmul(&y_col)?; // p×1
        let xty = xty_mat.as_slice().to_vec();

        let beta = Self::solve_normal(xtx.as_slice().to_vec(), xty, p, self.solver)?;

        // Intercept: b = ȳ − x̄ · β (0 if fit_intercept = false).
        let intercept = if self.fit_intercept {
            y_mean
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
        self.fitted = true;
        Ok(())
    }

    fn predict(&self, x: &Matrix) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("LinearRegression".into()));
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
        let n = x.nrows();
        let src = x.as_slice();
        let mut out = vec![intercept; n];
        for i in 0..n {
            let row = &src[i * p..(i + 1) * p];
            let mut s = intercept;
            for j in 0..p {
                s += beta[j] * row[j];
            }
            out[i] = s;
        }
        Ok(out)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn fit_perfect_line_with_intercept() {
        // y = 2x + 1
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]).unwrap();
        let y = vec![3.0, 5.0, 7.0, 9.0];
        let mut m = LinearRegression::new();
        m.fit(&x, &y).unwrap();
        assert!((m.coef()[0] - 2.0).abs() < 1e-9, "coef={}", m.coef()[0]);
        assert!(
            (m.intercept() - 1.0).abs() < 1e-9,
            "intercept={}",
            m.intercept()
        );
        let pred = m.predict(&x).unwrap();
        assert!(approx(&pred, &y, 1e-9));
    }

    #[test]
    fn fit_multivariate_known_coef() {
        // y = 2*x0 - 3.5*x1 + 5*x2 + 7 with non-collinear features.
        let rows: Vec<Vec<f64>> = (0..50)
            .map(|i| {
                let i = i as f64;
                // Distinct, non-collinear feature functions of the index.
                vec![i.sin(), i.cos(), (i + 1.0).ln()]
            })
            .collect();
        let x = Matrix::new(rows.clone()).unwrap();
        let y: Vec<f64> = rows
            .iter()
            .map(|r| 2.0 * r[0] - 3.5 * r[1] + 5.0 * r[2] + 7.0)
            .collect();
        let mut m = LinearRegression::new();
        m.fit(&x, &y).unwrap();
        assert!((m.coef()[0] - 2.0).abs() < 1e-6, "coef0={}", m.coef()[0]);
        assert!((m.coef()[1] - (-3.5)).abs() < 1e-6, "coef1={}", m.coef()[1]);
        assert!((m.coef()[2] - 5.0).abs() < 1e-6, "coef2={}", m.coef()[2]);
        assert!(
            (m.intercept() - 7.0).abs() < 1e-6,
            "intercept={}",
            m.intercept()
        );
    }

    #[test]
    fn fit_intercept_false() {
        // y = 3x + 0 (no intercept term).
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![3.0, 6.0, 9.0];
        let mut m = LinearRegression::new().with_fit_intercept(false);
        m.fit(&x, &y).unwrap();
        assert!((m.coef()[0] - 3.0).abs() < 1e-9);
        assert!(m.intercept().abs() < 1e-12);
    }

    #[test]
    fn cholesky_and_svd_agree_full_rank() {
        let rows: Vec<Vec<f64>> = (0..30)
            .map(|i| {
                let i = i as f64;
                vec![i.sin(), (i + 7.0).ln(), (i * 0.3).exp()]
            })
            .collect();
        let x = Matrix::new(rows.clone()).unwrap();
        // Deterministic y with a noise-free linear signal.
        let y: Vec<f64> = rows
            .iter()
            .map(|r| 1.5 * r[0] - 2.0 * r[1] + 0.3 * r[2] + 4.0)
            .collect();

        let mut m_chol = LinearRegression::new();
        m_chol.fit(&x, &y).unwrap();
        let mut m_svd = LinearRegression::new().with_solver(LinearSolver::Svd);
        m_svd.fit(&x, &y).unwrap();

        for i in 0..3 {
            assert!(
                (m_chol.coef()[i] - m_svd.coef()[i]).abs() < 1e-6,
                "solver disagreement at {i}: chol={} svd={}",
                m_chol.coef()[i],
                m_svd.coef()[i]
            );
        }
        assert!((m_chol.intercept() - m_svd.intercept()).abs() < 1e-6);
    }

    #[test]
    fn svd_handles_rank_deficiency() {
        // Duplicate column → XᵀX is rank-deficient; Cholesky must fail, SVD
        // must succeed (minimum-norm solution).
        let x = Matrix::new(vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]]).unwrap();
        let y = vec![2.0, 4.0, 6.0];

        // Cholesky path must fail with Singular.
        let mut m_chol = LinearRegression::new();
        let chol_res = m_chol.fit(&x, &y);
        assert!(matches!(chol_res, Err(DatarustError::Singular(_))));

        // SVD path succeeds.
        let mut m_svd = LinearRegression::new().with_solver(LinearSolver::Svd);
        m_svd.fit(&x, &y).unwrap();
        let pred = m_svd.predict(&x).unwrap();
        // Predictions should still recover y well.
        assert!(approx(&pred, &y, 1e-6));
    }

    #[test]
    fn predict_before_fit_errors() {
        let m = LinearRegression::new();
        let x = Matrix::new(vec![vec![1.0]]).unwrap();
        let err = m.predict(&x).unwrap_err();
        assert!(matches!(err, DatarustError::NotFitted(_)));
    }

    #[test]
    fn predict_shape_mismatch() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
        let mut m = LinearRegression::new();
        m.fit(&x, &[3.0, 5.0]).unwrap();
        let bad = Matrix::new(vec![vec![1.0, 2.0]]).unwrap();
        let err = m.predict(&bad).unwrap_err();
        assert!(matches!(err, DatarustError::ShapeMismatch { .. }));
    }

    #[test]
    fn fit_shape_mismatch_y() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
        let mut m = LinearRegression::new();
        let err = m.fit(&x, &[1.0]).unwrap_err(); // y too short
        assert!(matches!(err, DatarustError::ShapeMismatch { .. }));
    }

    #[test]
    fn fit_predict_convenience() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![2.0, 4.0, 6.0];
        let mut m = LinearRegression::new();
        let pred = m.fit_predict(&x, &y).unwrap();
        assert!(approx(&pred, &y, 1e-9));
    }

    #[test]
    fn n_features_in() {
        let rows: Vec<Vec<f64>> = (0..10)
            .map(|i| {
                let i = i as f64;
                vec![i.sin(), i.cos(), (i + 1.0).ln()]
            })
            .collect();
        let x = Matrix::new(rows).unwrap();
        let y: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let mut m = LinearRegression::new();
        m.fit(&x, &y).unwrap();
        assert_eq!(m.n_features_in(), 3);
    }

    #[test]
    fn fit_rejects_y_length_mismatch() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let mut m = LinearRegression::new();
        let err = m.fit(&x, &[1.0, 2.0]).unwrap_err(); // y too short
        assert!(matches!(err, DatarustError::ShapeMismatch { .. }));
    }

    #[test]
    fn predict_returns_n_rows() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]).unwrap();
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let mut m = LinearRegression::new().with_fit_intercept(false);
        m.fit(&x, &y).unwrap();
        let pred = m.predict(&x).unwrap();
        assert_eq!(pred.len(), x.nrows());
    }

    #[test]
    fn constant_target() {
        // All targets equal → intercept should be that constant, coef zero.
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![5.0]]).unwrap();
        let y = vec![7.0, 7.0, 7.0];
        let mut m = LinearRegression::new();
        m.fit(&x, &y).unwrap();
        assert!(
            (m.intercept() - 7.0).abs() < 1e-9,
            "intercept={}",
            m.intercept()
        );
        assert!(m.coef()[0].abs() < 1e-9);
    }

    #[test]
    fn fit_new_data_predicts_correctly() {
        // Fit on one dataset, predict on a held-out point.
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]).unwrap();
        let y = vec![2.0, 4.0, 6.0, 8.0]; // y = 2x
        let mut m = LinearRegression::new().with_fit_intercept(false);
        m.fit(&x, &y).unwrap();
        let new = Matrix::new(vec![vec![10.0]]).unwrap();
        let pred = m.predict(&new).unwrap();
        assert!((pred[0] - 20.0).abs() < 1e-9);
    }
}
