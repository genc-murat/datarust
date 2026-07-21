//! Ridge regression (L2-regularized least squares).
//!
//! Mirrors `sklearn.linear_model.Ridge`. Estimates coefficients `β` minimising
//!
//! `||Xβ − y||² + α ||β||²`
//!
//! Unlike [`LinearRegression`](super::linear_regression::LinearRegression), the
//! `α‖β‖²` penalty guarantees the system matrix `XᵀX + αI` is positive-definite
//! even when `X` is rank-deficient or collinear, so the Cholesky solver always
//! succeeds for `α > 0`.

use crate::error::{DatarustError, Result};
use crate::linalg::cholesky;
use crate::matrix::Matrix;
use crate::stats;
use crate::traits::{Estimator, Predictor, Regressor};

/// Solver strategy for [`Ridge`].
///
/// Mirrors `Ridge(solver="cholesky" | "svd")` in scikit-learn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RidgeSolver {
    /// Solve the penalized normal equations `(XᵀX + αI) β = Xᵀy` via Cholesky.
    /// Fast and dependency-free. Always succeeds for `α > 0`.
    #[default]
    Cholesky,
    /// Eigendecomposition-based pseudo-inverse path. Stable for severely
    /// ill-conditioned inputs, at higher cost.
    Svd,
}

/// Ridge regression with L2 regularization.
///
/// ```rust
/// use datarust::linear_model::Ridge;
/// use datarust::traits::Predictor;
/// use datarust::Matrix;
///
/// let x = Matrix::new(vec![
///     vec![0.0, 0.0],
///     vec![1.0, 1.0],
///     vec![2.0, 2.0],
///     vec![3.0, 3.0],
/// ])?;
/// // Collinear features: LinearRegression would be singular; Ridge handles it.
/// let y = vec![1.0, 2.0, 3.0, 4.0];
/// let mut model = Ridge::new().with_alpha(1.0);
/// model.fit(&x, &y)?;
/// let pred = model.predict(&x)?;
/// assert_eq!(pred.len(), 4);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Ridge {
    alpha: f64,
    fit_intercept: bool,
    solver: RidgeSolver,
    // Fitted state.
    coef_: Vec<f64>,
    intercept_: f64,
    n_features_in_: usize,
    fitted: bool,
}

impl Default for Ridge {
    fn default() -> Self {
        Self::new()
    }
}

impl Ridge {
    /// Creates a new Ridge with `alpha = 1.0`, `fit_intercept = true`,
    /// `solver = Cholesky`.
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
            solver: RidgeSolver::Cholesky,
            coef_: Vec::new(),
            intercept_: 0.0,
            n_features_in_: 0,
            fitted: false,
        }
    }

    /// Builder: regularization strength `alpha` (default `1.0`). Larger values
    /// shrink coefficients more aggressively. Must be `>= 0`.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Builder: whether to fit an intercept term (default `true`).
    pub fn with_fit_intercept(mut self, b: bool) -> Self {
        self.fit_intercept = b;
        self
    }

    /// Builder: choose the solver (default [`RidgeSolver::Cholesky`]).
    pub fn with_solver(mut self, s: RidgeSolver) -> Self {
        self.solver = s;
        self
    }

    /// Fitted coefficients `β`.
    pub fn coef(&self) -> &[f64] {
        &self.coef_
    }

    /// Fitted intercept term (0.0 if `fit_intercept = false`).
    pub fn intercept(&self) -> f64 {
        self.intercept_
    }

    /// Number of features seen during `fit`.
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
    }

    /// R² of the prediction against `y`, mirroring `estimator.score` in sklearn.
    pub fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64> {
        let pred = Predictor::predict(self, x)?;
        crate::metrics::regression::r2_score(y, &pred)
    }
}

impl Estimator for Ridge {}

impl Predictor for Ridge {
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
        if self.alpha < 0.0 {
            return Err(DatarustError::InvalidConfig(format!(
                "alpha must be >= 0, got {}",
                self.alpha
            )));
        }

        let x_slice = x.as_slice();
        // Center X and y when fitting an intercept; equivalent to adding a bias
        // column but keeps the system p×p.
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

        // Form XᵀX (p×p) and Xᵀy (p) via the optimised matmul path.
        let design_mat = Matrix::from_flat(n, p, design)?;
        let xt = design_mat.transpose();
        let mut xtx = xt.matmul(&design_mat)?; // p×p, flat
        let y_col = Matrix::from_flat(n, 1, y_work)?;
        let xty_mat = xt.matmul(&y_col)?; // p×1
        let xty = xty_mat.as_slice().to_vec();

        // Add the L2 penalty α on the diagonal: (XᵀX + αI).
        let alpha = self.alpha;
        for i in 0..p {
            xtx.as_mut_slice()[i * p + i] += alpha;
        }

        let beta = match self.solver {
            RidgeSolver::Cholesky => cholesky::solve_spd_system(xtx.as_slice(), p, &xty),
            RidgeSolver::Svd => {
                super::linear_regression::solve_via_eig_pinv(xtx.as_slice(), &xty, p)
            }
        }?;

        // Intercept: b = ȳ − x̄ · β.
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
            return Err(DatarustError::NotFitted("Ridge".into()));
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

impl Regressor for Ridge {
    fn name(&self) -> &'static str {
        "Ridge"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_model::LinearRegression;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn sample_xy() -> (Matrix, Vec<f64>) {
        // Non-collinear features.
        let rows: Vec<Vec<f64>> = (0..30)
            .map(|i| {
                let i = i as f64;
                vec![i.sin(), (i + 7.0).ln(), (i * 0.3 + 1.0).exp() * 0.01]
            })
            .collect();
        let y: Vec<f64> = rows
            .iter()
            .map(|r| 2.0 * r[0] - 1.0 * r[1] + 0.5 * r[2] + 3.0)
            .collect();
        (Matrix::new(rows).unwrap(), y)
    }

    #[test]
    fn alpha_zero_matches_linear_regression() {
        let (x, y) = sample_xy();
        let mut lr = LinearRegression::new();
        lr.fit(&x, &y).unwrap();
        let mut ridge = Ridge::new().with_alpha(0.0);
        ridge.fit(&x, &y).unwrap();
        for i in 0..3 {
            assert!(
                approx(ridge.coef()[i], lr.coef()[i], 1e-6),
                "coef[{i}]: ridge={} lr={}",
                ridge.coef()[i],
                lr.coef()[i]
            );
        }
        assert!(approx(ridge.intercept(), lr.intercept(), 1e-6));
    }

    #[test]
    fn shrinkage_with_larger_alpha() {
        let (x, y) = sample_xy();
        let mut small = Ridge::new().with_alpha(0.01);
        small.fit(&x, &y).unwrap();
        let mut large = Ridge::new().with_alpha(1000.0);
        large.fit(&x, &y).unwrap();
        // L2 norm of coefficients should shrink.
        let norm_small: f64 = small.coef().iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_large: f64 = large.coef().iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            norm_large < norm_small,
            "larger alpha should shrink coefficients: small={} large={}",
            norm_small,
            norm_large
        );
    }

    #[test]
    fn handles_rank_deficient_input() {
        // Duplicate columns — LinearRegression (Cholesky) would fail, Ridge succeeds.
        let x = Matrix::new(vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
        ])
        .unwrap();
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let mut model = Ridge::new().with_alpha(1.0);
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        // With strong signal and mild penalty, predictions should still be good.
        assert!(pred.iter().zip(y.iter()).all(|(p, &t)| (p - t).abs() < 1.0));
    }

    #[test]
    fn cholesky_and_svd_agree() {
        let (x, y) = sample_xy();
        let mut chol = Ridge::new().with_alpha(1.0);
        chol.fit(&x, &y).unwrap();
        let mut svd = Ridge::new().with_alpha(1.0).with_solver(RidgeSolver::Svd);
        svd.fit(&x, &y).unwrap();
        for i in 0..3 {
            assert!(
                approx(chol.coef()[i], svd.coef()[i], 1e-6),
                "solver disagreement at {i}"
            );
        }
    }

    #[test]
    fn predict_before_fit_errors() {
        let model = Ridge::new();
        let x = Matrix::new(vec![vec![1.0]]).unwrap();
        let err = model.predict(&x).unwrap_err();
        assert!(matches!(err, DatarustError::NotFitted(_)));
    }

    #[test]
    fn negative_alpha_rejected() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
        let mut model = Ridge::new().with_alpha(-1.0);
        let err = model.fit(&x, &[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, DatarustError::InvalidConfig(_)));
    }

    #[test]
    fn shape_mismatch_y_rejected() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
        let mut model = Ridge::new();
        let err = model.fit(&x, &[1.0]).unwrap_err();
        assert!(matches!(err, DatarustError::ShapeMismatch { .. }));
    }

    #[test]
    fn predict_shape_mismatch() {
        let x = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]).unwrap();
        let mut model = Ridge::new();
        model.fit(&x, &[1.0, 2.0, 3.0]).unwrap();
        let bad = Matrix::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();
        let err = model.predict(&bad).unwrap_err();
        assert!(matches!(err, DatarustError::ShapeMismatch { .. }));
    }

    #[test]
    fn fit_intercept_false() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![2.0, 4.0, 6.0]; // y = 2x, no intercept
        let mut model = Ridge::new().with_alpha(0.001).with_fit_intercept(false);
        model.fit(&x, &y).unwrap();
        assert!(approx(model.coef()[0], 2.0, 1e-3));
        assert!(model.intercept().abs() < 1e-12);
    }

    #[test]
    fn score_returns_r2() {
        let (x, y) = sample_xy();
        let mut model = Ridge::new().with_alpha(0.01);
        model.fit(&x, &y).unwrap();
        let r2 = model.score(&x, &y).unwrap();
        // Near-zero penalty on a clean linear signal → R² ≈ 1.
        assert!(r2 > 0.99, "r2 too low: {r2}");
    }
}
