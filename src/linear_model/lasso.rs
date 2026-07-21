//! Lasso regression (L1-regularized least squares).
//!
//! Mirrors `sklearn.linear_model.Lasso`. Estimates coefficients `β` minimising
//!
//! `(1 / (2n)) ||Xβ − y||² + α ||β||₁`
//!
//! The L1 penalty drives some coefficients to **exactly zero**, producing a
//! sparse model that performs implicit feature selection. This is the key
//! difference from [`Ridge`](super::ridge::Ridge) (L2), which only shrinks.
//!
//! Solved by **coordinate descent** with soft-thresholding: each coefficient is
//! updated in turn while the others are held fixed, using the precomputed Gram
//! matrix `XᵀX` and `Xᵀy` for `O(p)` per-coordinate cost.

use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::stats;
use crate::traits::{Estimator, Predictor, Regressor};

/// Lasso regression with L1 regularization, solved via coordinate descent.
///
/// ```rust
/// use datarust::linear_model::Lasso;
/// use datarust::traits::Predictor;
/// use datarust::Matrix;
///
/// let x = Matrix::new(vec![
///     vec![1.0, 0.0],
///     vec![0.0, 1.0],
///     vec![1.0, 1.0],
///     vec![2.0, 0.0],
/// ])?;
/// let y = vec![2.0, 0.0, 2.0, 4.0]; // y depends mostly on feature 0
/// let mut model = Lasso::new().with_alpha(0.1).with_max_iter(1000);
/// model.fit(&x, &y)?;
/// // With sufficient penalty, the irrelevant coefficient shrinks toward 0.
/// let pred = model.predict(&x)?;
/// assert_eq!(pred.len(), 4);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Lasso {
    alpha: f64,
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
    // Fitted state.
    coef_: Vec<f64>,
    intercept_: f64,
    n_features_in_: usize,
    n_iter_: usize,
    fitted: bool,
}

impl Default for Lasso {
    fn default() -> Self {
        Self::new()
    }
}

impl Lasso {
    /// Creates a new Lasso with `alpha = 1.0`, `fit_intercept = true`,
    /// `max_iter = 1000`, `tol = 1e-4`.
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
            coef_: Vec::new(),
            intercept_: 0.0,
            n_features_in_: 0,
            n_iter_: 0,
            fitted: false,
        }
    }

    /// Builder: regularization strength `alpha` (default `1.0`). Larger values
    /// produce more sparse coefficients. Must be `>= 0`.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Builder: whether to fit an intercept term (default `true`).
    pub fn with_fit_intercept(mut self, b: bool) -> Self {
        self.fit_intercept = b;
        self
    }

    /// Builder: maximum coordinate-descent iterations (default `1000`).
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Builder: convergence tolerance (default `1e-4`). The solver stops when
    /// the maximum coefficient change in a sweep drops below this value.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Fitted coefficients `β`. Some entries may be exactly zero (sparsity).
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

    /// Number of coordinate-descent iterations actually run.
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// R² of the prediction against `y`.
    pub fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64> {
        let pred = Predictor::predict(self, x)?;
        crate::metrics::regression::r2_score(y, &pred)
    }
}

/// Soft-thresholding operator: `S(x, α) = sign(x) · max(|x| − α, 0)`.
///
/// This is the proximal operator of the L1 norm and the core of each
/// coordinate-descent update.
#[inline]
fn soft_threshold(x: f64, alpha: f64) -> f64 {
    if x > alpha {
        x - alpha
    } else if x < -alpha {
        x + alpha
    } else {
        0.0
    }
}

impl Estimator for Lasso {}

impl Predictor for Lasso {
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
        // Center X and y when fitting an intercept.
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

        // Precompute Gram matrix G = XᵀX (p×p) and q = Xᵀy (p).
        let design_mat = Matrix::from_flat(n, p, design)?;
        let xt = design_mat.transpose();
        let gram = xt.matmul(&design_mat)?; // p×p
        let y_col = Matrix::from_flat(n, 1, y_work)?;
        let xty_mat = xt.matmul(&y_col)?; // p×1
        let q = xty_mat.as_slice().to_vec();
        let g = gram.as_slice();

        // Coordinate descent. Objective (per sklearn):
        //   (1/(2n)) ||y - Xβ||² + α ||β||₁
        // The coordinate-wise optimum for β_j (others fixed) is:
        //   β_j = S(q_j - Σ_{k≠j} G[j,k] β_k, n·α) / G[j,j]
        // where S is the soft-thresholding operator.
        let mut beta = vec![0.0_f64; p];
        let n_f = n as f64;
        let threshold = n_f * self.alpha;
        let mut n_iter = 0;
        for _ in 0..self.max_iter {
            n_iter += 1;
            let mut max_delta = 0.0_f64;
            for j in 0..p {
                let g_jj = g[j * p + j];
                if g_jj == 0.0 {
                    continue;
                }
                // ρ_j = q_j − Σ_{k≠j} G[j,k] β_k.
                let mut rho = q[j];
                for k in 0..p {
                    if k != j {
                        rho -= g[j * p + k] * beta[k];
                    }
                }
                let old = beta[j];
                let new = soft_threshold(rho, threshold) / g_jj;
                beta[j] = new;
                let delta = (new - old).abs();
                if delta > max_delta {
                    max_delta = delta;
                }
            }
            if max_delta < self.tol {
                break;
            }
        }

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
        self.n_iter_ = n_iter;
        self.fitted = true;
        Ok(())
    }

    fn predict(&self, x: &Matrix) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("Lasso".into()));
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

impl Regressor for Lasso {
    fn name(&self) -> &'static str {
        "Lasso"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn sample_xy() -> (Matrix, Vec<f64>) {
        let rows: Vec<Vec<f64>> = (0..40)
            .map(|i| {
                let i = i as f64;
                vec![i.sin(), (i + 7.0).ln(), (i * 0.3 + 1.0).exp() * 0.01]
            })
            .collect();
        let y: Vec<f64> = rows
            .iter()
            .map(|r| 3.0 * r[0] - 2.0 * r[1] + 0.0 * r[2] + 1.0)
            .collect();
        (Matrix::new(rows).unwrap(), y)
    }

    #[test]
    fn near_zero_alpha_recovers_signal() {
        let (x, y) = sample_xy();
        let mut model = Lasso::new()
            .with_alpha(0.001)
            .with_max_iter(5000)
            .with_tol(1e-8);
        model.fit(&x, &y).unwrap();
        // Feature 2 has zero true weight; with tiny penalty it stays near zero.
        assert!(
            model.coef()[2].abs() < 0.1,
            "coef[2] should be near zero: {}",
            model.coef()[2]
        );
        // Features 0 and 1 should be recovered approximately.
        assert!(approx(model.coef()[0], 3.0, 0.2));
        assert!(approx(model.coef()[1], -2.0, 0.2));
    }

    #[test]
    fn sparsity_with_large_alpha() {
        let (x, y) = sample_xy();
        let mut model = Lasso::new().with_alpha(1.0).with_max_iter(2000);
        model.fit(&x, &y).unwrap();
        // At least one coefficient should be exactly zero (sparsity).
        let zeros = model.coef().iter().filter(|c| c.abs() < 1e-10).count();
        assert!(
            zeros >= 1,
            "expected at least one zero coefficient, got coefs {:?}",
            model.coef()
        );
    }

    #[test]
    fn very_large_alpha_zeros_all() {
        let (x, y) = sample_xy();
        let mut model = Lasso::new().with_alpha(1e6).with_max_iter(100);
        model.fit(&x, &y).unwrap();
        for (i, &c) in model.coef().iter().enumerate() {
            assert!(c.abs() < 1e-6, "coef[{i}] should be ~0, got {c}");
        }
    }

    #[test]
    fn n_iter_recorded() {
        let (x, y) = sample_xy();
        let mut model = Lasso::new().with_alpha(0.1).with_max_iter(1000);
        model.fit(&x, &y).unwrap();
        assert!(model.n_iter() > 0 && model.n_iter() <= 1000);
    }

    #[test]
    fn predict_before_fit_errors() {
        let model = Lasso::new();
        let x = Matrix::new(vec![vec![1.0]]).unwrap();
        let err = model.predict(&x).unwrap_err();
        assert!(matches!(err, DatarustError::NotFitted(_)));
    }

    #[test]
    fn negative_alpha_rejected() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
        let mut model = Lasso::new().with_alpha(-1.0);
        let err = model.fit(&x, &[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, DatarustError::InvalidConfig(_)));
    }

    #[test]
    fn shape_mismatch_y_rejected() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
        let mut model = Lasso::new();
        let err = model.fit(&x, &[1.0]).unwrap_err();
        assert!(matches!(err, DatarustError::ShapeMismatch { .. }));
    }

    #[test]
    fn predict_shape_mismatch() {
        let x = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]).unwrap();
        let mut model = Lasso::new().with_alpha(0.1);
        model.fit(&x, &[1.0, 2.0, 3.0]).unwrap();
        let bad = Matrix::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();
        let err = model.predict(&bad).unwrap_err();
        assert!(matches!(err, DatarustError::ShapeMismatch { .. }));
    }

    #[test]
    fn perfect_line_fit_intercept_false() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]).unwrap();
        let y = vec![3.0, 6.0, 9.0, 12.0]; // y = 3x
        let mut model = Lasso::new()
            .with_alpha(0.001)
            .with_fit_intercept(false)
            .with_max_iter(5000)
            .with_tol(1e-10);
        model.fit(&x, &y).unwrap();
        assert!(approx(model.coef()[0], 3.0, 1e-3));
        assert!(model.intercept().abs() < 1e-12);
    }

    #[test]
    fn zero_alpha_runs() {
        // alpha=0 is allowed (reduces to OLS via coordinate descent).
        let (x, y) = sample_xy();
        let mut model = Lasso::new()
            .with_alpha(0.0)
            .with_max_iter(5000)
            .with_tol(1e-10);
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        // Predictions should be reasonable.
        assert!(pred.iter().zip(y.iter()).all(|(p, &t)| (p - t).abs() < 1.0));
    }

    #[test]
    fn intercept_recovered() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![12.0, 14.0, 16.0]; // y = 2x + 10
        let mut model = Lasso::new()
            .with_alpha(0.001)
            .with_max_iter(5000)
            .with_tol(1e-10);
        model.fit(&x, &y).unwrap();
        assert!(approx(model.coef()[0], 2.0, 1e-2));
        assert!(approx(model.intercept(), 10.0, 1e-1));
    }
}
