use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::traits::{default_input_names, FeatureNames};
use crate::Transformer;

/// Transformation method for [`PowerTransformer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PowerMethod {
    /// Yeo-Johnson transform; works for positive and negative data.
    #[default]
    YeoJohnson,
    /// Box-Cox transform; requires strictly positive data.
    BoxCox,
}

/// Apply a power transform to make data more Gaussian-like, mirroring
/// `sklearn.preprocessing.PowerTransformer`.
///
/// After the power transform, the data is standardized (zero mean, unit
/// variance) per column when `standardize = true` (default).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PowerTransformer {
    method: PowerMethod,
    standardize: bool,
    /// Fitted lambda per column.
    lambdas: Vec<f64>,
    /// Mean per column (for standardization after transform).
    means: Vec<f64>,
    /// Std per column (for standardization after transform).
    stds: Vec<f64>,
    n_features: usize,
    fitted: bool,
}

impl PowerTransformer {
    /// Creates a new transformer with default settings.
    pub fn new() -> Self {
        Self {
            method: PowerMethod::YeoJohnson,
            standardize: true,
            lambdas: vec![],
            means: vec![],
            stds: vec![],
            n_features: 0,
            fitted: false,
        }
    }

    /// Builder: set the power transform method.
    pub fn method(mut self, m: PowerMethod) -> Self {
        self.method = m;
        self
    }

    /// Builder: enable or disable post-transform standardization (default true).
    pub fn standardize(mut self, s: bool) -> Self {
        self.standardize = s;
        self
    }

    /// Returns the fitted lambda per column.
    pub fn lambdas(&self) -> &[f64] {
        &self.lambdas
    }

    /// Apply the power transform to a single value.
    fn transform_one(x: f64, lam: f64, method: PowerMethod) -> f64 {
        match method {
            PowerMethod::BoxCox => {
                if lam.abs() < 1e-8 {
                    x.ln()
                } else {
                    (x.powf(lam) - 1.0) / lam
                }
            }
            PowerMethod::YeoJohnson => {
                if x >= 0.0 {
                    if lam.abs() < 1e-8 {
                        (x + 1.0).ln()
                    } else {
                        ((x + 1.0).powf(lam) - 1.0) / lam
                    }
                } else if (lam - 2.0).abs() < 1e-8 {
                    -((-x + 1.0).ln())
                } else {
                    -(((-x + 1.0).powf(2.0 - lam) - 1.0) / (2.0 - lam))
                }
            }
        }
    }

    /// Inverse of the power transform for a single value.
    fn inverse_one(x: f64, lam: f64, method: PowerMethod) -> f64 {
        match method {
            PowerMethod::BoxCox => {
                if lam.abs() < 1e-8 {
                    x.exp()
                } else {
                    (x * lam + 1.0).powf(1.0 / lam)
                }
            }
            PowerMethod::YeoJohnson => {
                if x >= 0.0 {
                    if lam.abs() < 1e-8 {
                        x.exp() - 1.0
                    } else {
                        (x * lam + 1.0).powf(1.0 / lam) - 1.0
                    }
                } else if (lam - 2.0).abs() < 1e-8 {
                    1.0 - (-x).exp()
                } else {
                    1.0 - (-(x * (2.0 - lam)) + 1.0).powf(1.0 / (2.0 - lam))
                }
            }
        }
    }

    /// Log-likelihood for a column given a candidate lambda.
    /// Returns the value to be maximized.
    fn neg_log_likelihood(col: &[f64], lam: f64, method: PowerMethod) -> f64 {
        let n = col.len() as f64;
        let transformed: Vec<f64> = col
            .iter()
            .map(|&v| Self::transform_one(v, lam, method))
            .collect();
        let mean: f64 = transformed.iter().sum::<f64>() / n;
        let variance: f64 = transformed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        // Log-likelihood (ignoring constants):
        // L = -(n/2) * ln(sigma^2) + (lambda - 1) * sum(log(|x| + 1)) [Yeo-Johnson]
        // or (lambda - 1) * sum(log(x)) [Box-Cox]
        let jacobian = match method {
            PowerMethod::BoxCox => col.iter().map(|&v| v.abs().ln()).sum::<f64>(),
            PowerMethod::YeoJohnson => col
                .iter()
                .map(|&v| {
                    if v >= 0.0 {
                        (v + 1.0).ln()
                    } else {
                        (-v + 1.0).ln()
                    }
                })
                .sum::<f64>(),
        };
        -(n / 2.0) * variance.ln() + (lam - 1.0) * jacobian
    }

    /// Find optimal lambda via golden-section search in [-5, 5].
    fn optimal_lambda(col: &[f64], method: PowerMethod) -> f64 {
        let neg_lik = |lam: f64| -Self::neg_log_likelihood(col, lam, method);

        // Coarse grid search first.
        let grid: Vec<f64> = (-50..=50).map(|i| i as f64 / 10.0).collect();
        let mut best_lam = 0.0;
        let mut best_val = f64::INFINITY;
        for &lam in &grid {
            let val = neg_lik(lam);
            if val < best_val {
                best_val = val;
                best_lam = lam;
            }
        }

        // Refine with golden-section search around the best.
        let mut lo = best_lam - 0.2;
        let mut hi = best_lam + 0.2;
        let phi = (5.0_f64.sqrt() - 1.0) / 2.0;
        for _ in 0..60 {
            let mid1 = hi - phi * (hi - lo);
            let mid2 = lo + phi * (hi - lo);
            if neg_lik(mid1) < neg_lik(mid2) {
                hi = mid2;
            } else {
                lo = mid1;
            }
        }
        (lo + hi) / 2.0
    }
}

impl Default for PowerTransformer {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for PowerTransformer {
    fn name(&self) -> &'static str {
        "PowerTransformer"
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        let ncols = x.ncols();
        if matches!(self.method, PowerMethod::BoxCox) {
            for j in 0..ncols {
                for i in 0..x.nrows() {
                    if x.get(i, j) <= 0.0 {
                        return Err(DatarustError::InvalidInput(format!(
                            "Box-Cox requires strictly positive data; non-positive at col {} row {}",
                            j, i
                        )));
                    }
                }
            }
        }
        let mut lambdas = Vec::with_capacity(ncols);
        let mut means = Vec::with_capacity(ncols);
        let mut stds = Vec::with_capacity(ncols);
        for j in 0..ncols {
            let col = x.col(j);
            let lam = Self::optimal_lambda(&col, self.method);
            let transformed: Vec<f64> = col
                .iter()
                .map(|&v| Self::transform_one(v, lam, self.method))
                .collect();
            let mean: f64 = transformed.iter().sum::<f64>() / transformed.len() as f64;
            let variance: f64 = transformed.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / transformed.len() as f64;
            let std = variance.sqrt();
            lambdas.push(lam);
            means.push(mean);
            stds.push(if std < 1e-12 { 1.0 } else { std });
        }
        self.lambdas = lambdas;
        self.means = means;
        self.stds = stds;
        self.n_features = ncols;
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("PowerTransformer".into()));
        }
        if x.ncols() != self.n_features {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.n_features),
                actual: format!("{} features", x.ncols()),
            });
        }
        x.validate_no_nan()?;
        // Box-Cox requires strictly positive input; validate on new data
        // since fit-time validation does not cover unseen values.
        if matches!(self.method, PowerMethod::BoxCox) {
            for i in 0..x.nrows() {
                for j in 0..x.ncols() {
                    let v = x.get(i, j);
                    if v <= 0.0 {
                        return Err(DatarustError::InvalidInput(format!(
                            "Box-Cox transform requires positive finite values; got {} at ({}, {})",
                            v, i, j
                        )));
                    }
                }
            }
        }
        let mut out = vec![vec![0.0; x.ncols()]; x.nrows()];
        for (i, out_row) in out.iter_mut().enumerate() {
            for (j, cell) in out_row.iter_mut().enumerate() {
                let val = Self::transform_one(x.get(i, j), self.lambdas[j], self.method);
                *cell = if self.standardize {
                    (val - self.means[j]) / self.stds[j]
                } else {
                    val
                };
            }
        }
        Matrix::new(out)
    }

    fn inverse_transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("PowerTransformer".into()));
        }
        if x.ncols() != self.n_features {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.n_features),
                actual: format!("{} features", x.ncols()),
            });
        }
        let mut out = vec![vec![0.0; x.ncols()]; x.nrows()];
        for (i, out_row) in out.iter_mut().enumerate() {
            for (j, cell) in out_row.iter_mut().enumerate() {
                // Undo optional standardization, then undo the power transform.
                let val = if self.standardize {
                    x.get(i, j) * self.stds[j] + self.means[j]
                } else {
                    x.get(i, j)
                };
                *cell = Self::inverse_one(val, self.lambdas[j], self.method);
            }
        }
        Matrix::new(out)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl FeatureNames for PowerTransformer {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        match input_features {
            Some(fs) => fs.to_vec(),
            None => default_input_names(self.n_features),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn yeo_johnson_lambda_zero_is_log() {
        // For x>=0, lambda≈0: transform = ln(x+1)
        let val = PowerTransformer::transform_one(3.0, 0.0, PowerMethod::YeoJohnson);
        assert!(approx(val, (4.0f64).ln(), 1e-12));
    }

    #[test]
    fn yeo_johnson_lambda_one_is_identity_for_positive() {
        // For x>=0, lambda=1: transform = x (since ((x+1)-1)/1 = x)
        let val = PowerTransformer::transform_one(5.0, 1.0, PowerMethod::YeoJohnson);
        assert!(approx(val, 5.0, 1e-12));
    }

    #[test]
    fn box_cox_lambda_zero_is_log() {
        let val = PowerTransformer::transform_one(3.0, 0.0, PowerMethod::BoxCox);
        assert!(approx(val, (3.0f64).ln(), 1e-12));
    }

    #[test]
    fn box_cox_lambda_one_is_identity() {
        // (x^1 - 1)/1 = x - 1
        let val = PowerTransformer::transform_one(5.0, 1.0, PowerMethod::BoxCox);
        assert!(approx(val, 4.0, 1e-12));
    }

    #[test]
    fn yeo_johnson_roundtrip() {
        for &lam in &[0.0, 0.5, 1.0, 1.5, 2.0] {
            for &x in &[0.5, 1.0, 5.0, -0.5, -3.0] {
                let t = PowerTransformer::transform_one(x, lam, PowerMethod::YeoJohnson);
                let inv = PowerTransformer::inverse_one(t, lam, PowerMethod::YeoJohnson);
                assert!(approx(x, inv, 1e-6), "lam={} x={} inv={}", lam, x, inv);
            }
        }
    }

    #[test]
    fn box_cox_roundtrip() {
        for &lam in &[0.0, 0.5, 1.0, 1.5] {
            for &x in &[0.5, 1.0, 5.0, 10.0] {
                let t = PowerTransformer::transform_one(x, lam, PowerMethod::BoxCox);
                let inv = PowerTransformer::inverse_one(t, lam, PowerMethod::BoxCox);
                assert!(approx(x, inv, 1e-6), "lam={} x={} inv={}", lam, x, inv);
            }
        }
    }

    #[test]
    fn yeo_johnson_fits_and_standardizes() {
        let x = Matrix::new(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![10.0],
            vec![50.0],
        ])
        .unwrap();
        let mut pt = PowerTransformer::new();
        let out = pt.fit_transform(&x).unwrap();
        // After standardization: mean≈0, std≈1
        let mean: f64 = (0..5).map(|i| out.get(i, 0)).sum::<f64>() / 5.0;
        assert!(approx(mean, 0.0, 1e-6));
        let variance: f64 = (0..5).map(|i| (out.get(i, 0) - mean).powi(2)).sum::<f64>() / 5.0;
        assert!(approx(variance, 1.0, 1e-4));
    }

    #[test]
    fn inverse_transform_yeo_johnson() {
        let x = Matrix::new(vec![
            vec![1.0, 4.0],
            vec![2.0, 8.0],
            vec![3.0, 15.0],
            vec![10.0, 30.0],
            vec![50.0, 90.0],
        ])
        .unwrap();
        let mut pt = PowerTransformer::new();
        let transformed = pt.fit_transform(&x).unwrap();
        let restored = pt.inverse_transform(&transformed).unwrap();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!(
                    approx(x.get(i, j), restored.get(i, j), 1e-4),
                    "mismatch at ({},{}): {} vs {}",
                    i,
                    j,
                    x.get(i, j),
                    restored.get(i, j)
                );
            }
        }
    }

    #[test]
    fn inverse_transform_box_cox() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]]).unwrap();
        let mut pt = PowerTransformer::new().method(PowerMethod::BoxCox);
        let transformed = pt.fit_transform(&x).unwrap();
        let restored = pt.inverse_transform(&transformed).unwrap();
        for i in 0..x.nrows() {
            assert!(
                approx(x.get(i, 0), restored.get(i, 0), 1e-4),
                "mismatch at row {}: {} vs {}",
                i,
                x.get(i, 0),
                restored.get(i, 0)
            );
        }
    }

    #[test]
    fn box_cox_positive_data_only() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]]).unwrap();
        let mut pt = PowerTransformer::new().method(PowerMethod::BoxCox);
        let out = pt.fit_transform(&x).unwrap();
        let mean: f64 = (0..5).map(|i| out.get(i, 0)).sum::<f64>() / 5.0;
        assert!(approx(mean, 0.0, 1e-6));
    }

    #[test]
    fn box_cox_rejects_non_positive() {
        let x = Matrix::new(vec![vec![1.0], vec![0.0], vec![-1.0]]).unwrap();
        let mut pt = PowerTransformer::new().method(PowerMethod::BoxCox);
        assert!(pt.fit(&x).is_err());
    }

    #[test]
    fn no_standardize_option() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]]).unwrap();
        let mut pt = PowerTransformer::new()
            .method(PowerMethod::BoxCox)
            .standardize(false);
        let out = pt.fit_transform(&x).unwrap();
        // Without standardization, values won't have mean 0
        let mean: f64 = (0..5).map(|i| out.get(i, 0)).sum::<f64>() / 5.0;
        assert!(!approx(mean, 0.0, 0.01));
    }

    #[test]
    fn transform_before_fit_errors() {
        let pt = PowerTransformer::new();
        let x = Matrix::new(vec![vec![1.0]]).unwrap();
        assert!(matches!(pt.transform(&x), Err(DatarustError::NotFitted(_))));
    }

    #[test]
    fn feature_names_preserved() {
        let x = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let mut pt = PowerTransformer::new();
        pt.fit(&x).unwrap();
        let names = pt.feature_names_out(Some(&["a".into(), "b".into()]));
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn lambda_estimated_reasonable() {
        // For data [1,2,3,4,5], the optimal Yeo-Johnson lambda should be
        // positive (data is roughly linear, not needing much transform).
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]]).unwrap();
        let mut pt = PowerTransformer::new().standardize(false);
        pt.fit(&x).unwrap();
        let lam = pt.lambdas()[0];
        assert!(lam > 0.0 && lam < 2.0, "lambda = {}", lam);
    }
}
