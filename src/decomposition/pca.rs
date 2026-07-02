use crate::decomposition::jacobi;
use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::stats;
use crate::Transformer;

/// How to specify the number of principal components, mirroring sklearn's
/// `n_components` parameter.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PCAComponents {
    /// Keep exactly this many components.
    Count(usize),
    /// Keep the smallest number of components such that the cumulative
    /// explained variance ratio is at least the given value in (0, 1).
    Variance(f64),
    /// Keep all components (`min(n_samples, n_features)`).
    All,
}

/// Principal Component Analysis (PCA), mirroring `sklearn.decomposition.PCA`.
///
/// Centers the data, computes the covariance matrix, and projects onto the
/// top eigenvectors found via Jacobi eigenvalue decomposition.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PCA {
    n_components: PCAComponents,
    whiten: bool,
    mean: Vec<f64>,
    components: Vec<Vec<f64>>, // k x p, rows are principal axes
    explained_variance: Vec<f64>,
    explained_variance_ratio: Vec<f64>,
    n_components_: usize,
    n_samples_: usize,
    total_variance_: f64,
    fitted: bool,
}

impl PCA {
    /// Creates a new PCA with the given component selection.
    pub fn new(n_components: PCAComponents) -> Self {
        Self {
            n_components,
            whiten: false,
            mean: vec![],
            components: vec![],
            explained_variance: vec![],
            explained_variance_ratio: vec![],
            n_components_: 0,
            n_samples_: 0,
            total_variance_: 0.0,
            fitted: false,
        }
    }

    /// Sets whether to whiten the projected components.
    pub fn whiten(mut self, b: bool) -> Self {
        self.whiten = b;
        self
    }

    /// Returns the principal axes (one row per component).
    pub fn components(&self) -> &[Vec<f64>] {
        &self.components
    }

    /// Returns the variance explained by each kept component.
    pub fn explained_variance(&self) -> &[f64] {
        &self.explained_variance
    }

    /// Returns the fraction of total variance explained by each kept component.
    pub fn explained_variance_ratio(&self) -> &[f64] {
        &self.explained_variance_ratio
    }

    /// Estimated noise variance, computed as the mean of discarded
    /// eigenvalues.  Returns 0 when all components are kept.
    pub fn noise_variance(&self) -> f64 {
        let p = self.mean.len();
        let max_k = self.n_samples_.min(p);
        if self.n_components_ < max_k {
            let kept: f64 = self.explained_variance.iter().sum();
            (self.total_variance_ - kept) / (max_k - self.n_components_) as f64
        } else {
            0.0
        }
    }

    /// Returns the per-feature mean estimated during fit.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Returns the number of components kept after fit.
    pub fn n_components(&self) -> usize {
        self.n_components_
    }

    fn select_k(&self, vals: &[f64], max_k: usize) -> Result<usize> {
        match self.n_components {
            PCAComponents::Count(k) => {
                if k == 0 {
                    return Err(DatarustError::InvalidConfig(
                        "n_components must be > 0".into(),
                    ));
                }
                Ok(k.min(max_k))
            }
            PCAComponents::All => Ok(max_k),
            PCAComponents::Variance(target) => {
                if !(0.0..=1.0).contains(&target) || target <= 0.0 {
                    return Err(DatarustError::InvalidConfig(format!(
                        "variance target must be in (0,1], got {}",
                        target
                    )));
                }
                let total: f64 = vals.iter().sum();
                if total <= 0.0 {
                    return Ok(max_k);
                }
                let mut acc = 0.0;
                for (k, &v) in vals.iter().enumerate() {
                    acc += v / total;
                    if acc >= target {
                        return Ok(k + 1);
                    }
                }
                Ok(max_k)
            }
        }
    }
}

/// Default: keep 95% of variance, no whitening.
impl Default for PCA {
    fn default() -> Self {
        Self {
            n_components: PCAComponents::Variance(0.95),
            whiten: false,
            mean: vec![],
            components: vec![],
            explained_variance: vec![],
            explained_variance_ratio: vec![],
            n_components_: 0,
            n_samples_: 0,
            total_variance_: 0.0,
            fitted: false,
        }
    }
}

impl Transformer for PCA {
    fn name(&self) -> &'static str {
        "PCA"
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        let n = x.nrows();
        let p = x.ncols();
        self.n_samples_ = n;
        self.mean = stats::column_mean(x.rows_ref());
        // Center
        let xc = self.centered(x);
        let cov = jacobi::covariance(&xc, 1);
        let (mut vals, vecs) = jacobi::eigh(&cov).ok_or_else(|| {
            DatarustError::Singular("covariance matrix is empty or non-square".into())
        })?;
        // Clip tiny negative eigenvalues from numerical noise
        for v in vals.iter_mut() {
            if *v < 0.0 && v.abs() < 1e-10 {
                *v = 0.0;
            }
        }
        let total_var: f64 = vals.iter().sum();
        let max_k = n.min(p);
        let k = self.select_k(&vals, max_k)?;
        self.n_components_ = k;
        self.total_variance_ = total_var;
        self.components = vecs.into_iter().take(k).collect();
        self.explained_variance = vals.iter().take(k).copied().collect();
        self.explained_variance_ratio = if total_var > 0.0 {
            vals.iter().take(k).map(|v| v / total_var).collect()
        } else {
            vec![0.0; k]
        };
        self.fitted = true;
        Ok(())
    }

    #[allow(clippy::needless_range_loop)]
    fn transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("PCA".into()));
        }
        if x.ncols() != self.mean.len() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.mean.len()),
                actual: format!("{} features", x.ncols()),
            });
        }
        let k = self.n_components_;
        let mut out = vec![vec![0.0; k]; x.nrows()];
        for (i, row) in x.rows_ref().iter().enumerate() {
            // center then project: (row - mean) . component[j]
            let centered: Vec<f64> = (0..row.len()).map(|c| row[c] - self.mean[c]).collect();
            for j in 0..k {
                let comp = &self.components[j];
                let mut s = 0.0;
                for c in 0..centered.len() {
                    s += centered[c] * comp[c];
                }
                if self.whiten {
                    let var = self.explained_variance[j];
                    let scale = if var > 0.0 { (var).sqrt() } else { 1.0 };
                    out[i][j] = s / scale;
                } else {
                    out[i][j] = s;
                }
            }
        }
        Matrix::new(out)
    }

    #[allow(clippy::needless_range_loop)]
    fn inverse_transform(&self, projected: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("PCA".into()));
        }
        if projected.ncols() != self.n_components_ {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} components", self.n_components_),
                actual: format!("{} columns", projected.ncols()),
            });
        }
        let p = self.mean.len();
        let mut out = vec![vec![0.0; p]; projected.nrows()];
        for (i, row) in projected.rows_ref().iter().enumerate() {
            for j in 0..self.n_components_ {
                let comp = &self.components[j];
                let mut val = row[j];
                if self.whiten {
                    let var = self.explained_variance[j];
                    let scale = if var > 0.0 { var.sqrt() } else { 1.0 };
                    val *= scale;
                }
                for c in 0..p {
                    out[i][c] += val * comp[c];
                }
            }
            // add mean back
            for c in 0..p {
                out[i][c] += self.mean[c];
            }
        }
        Matrix::new(out)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl PCA {
    fn centered(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let mut out = vec![vec![0.0; x.ncols()]; x.nrows()];
        for (i, row) in x.rows_ref().iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                out[i][j] = v - self.mean[j];
            }
        }
        out
    }
}

impl crate::traits::FeatureNames for PCA {
    fn feature_names_out(&self, _input_features: Option<&[String]>) -> Vec<String> {
        (0..self.n_components_)
            .map(|i| format!("pca{}", i))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn explained_variance_ratio_sums_to_one_when_all() {
        // Data with clear principal structure
        let x = Matrix::new(vec![
            vec![2.0, 0.0],
            vec![0.0, 2.0],
            vec![-2.0, 0.0],
            vec![0.0, -2.0],
        ])
        .unwrap();
        let mut pca = PCA::new(PCAComponents::All);
        pca.fit(&x).unwrap();
        let sum: f64 = pca.explained_variance_ratio().iter().sum();
        assert!(approx(sum, 1.0, 1e-9));
    }

    #[test]
    fn perfect_linear_data_one_component() {
        // y = 2x perfectly correlated; one component explains ~100% variance.
        let x = Matrix::new(vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
        ])
        .unwrap();
        let mut pca = PCA::new(PCAComponents::Count(1));
        pca.fit(&x).unwrap();
        let r0 = pca.explained_variance_ratio()[0];
        assert!(r0 > 0.999, "first component ratio = {}", r0);
    }

    #[test]
    fn reconstruction_zero_error_all_components() {
        let x = Matrix::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 10.0],
            vec![2.0, 1.0, 0.0],
        ])
        .unwrap();
        let mut pca = PCA::new(PCAComponents::All);
        let proj = pca.fit_transform(&x).unwrap();
        let recon = pca.inverse_transform(&proj).unwrap();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!(
                    approx(recon.get(i, j), x.get(i, j), 1e-6),
                    "recon {},{} = {} vs {}",
                    i,
                    j,
                    recon.get(i, j),
                    x.get(i, j)
                );
            }
        }
    }

    #[test]
    fn n_components_count_limits_output() {
        let x = Matrix::new(vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4.0, 3.0, 2.0, 1.0],
            vec![1.0, 3.0, 2.0, 4.0],
            vec![4.0, 2.0, 3.0, 1.0],
            vec![2.0, 2.0, 2.0, 2.0],
        ])
        .unwrap();
        let mut pca = PCA::new(PCAComponents::Count(2));
        let out = pca.fit_transform(&x).unwrap();
        assert_eq!(out.ncols(), 2);
        assert_eq!(pca.n_components(), 2);
    }

    #[test]
    fn variance_target_selects_k() {
        let x = Matrix::new(vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
        ])
        .unwrap();
        let mut pca = PCA::new(PCAComponents::Variance(0.95));
        pca.fit(&x).unwrap();
        assert_eq!(pca.n_components(), 1);
        assert!(pca.explained_variance_ratio()[0] >= 0.95);
    }

    #[test]
    fn variance_target_invalid() {
        let x = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let mut pca = PCA::new(PCAComponents::Variance(0.0));
        assert!(pca.fit(&x).is_err());
        let mut pca2 = PCA::new(PCAComponents::Variance(1.5));
        assert!(pca2.fit(&x).is_err());
    }

    #[test]
    fn whiten_produces_unit_variance() {
        let x = Matrix::new(vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
            vec![5.0, 10.0],
        ])
        .unwrap();
        let mut pca = PCA::new(PCAComponents::All).whiten(true);
        let out = pca.fit_transform(&x).unwrap();
        // whitened components should have variance ~1 when measured with the
        // same ddof (=1) the covariance/eigenvalues use.
        let means = crate::stats::column_mean(out.rows_ref());
        let vars = crate::stats::column_variance(out.rows_ref(), 1);
        for (k, &v) in vars.iter().enumerate() {
            // skip near-zero-variance components (degenerate eigenvalue)
            if pca.explained_variance()[k] > 1e-6 {
                assert!((v - 1.0).abs() < 1e-6, "comp {} var {}", k, v);
            }
        }
        for m in &means {
            assert!(m.abs() < 1e-9);
        }
    }

    #[test]
    fn transform_before_fit_errors() {
        let pca = PCA::new(PCAComponents::All);
        let x = Matrix::new(vec![vec![1.0, 2.0]]).unwrap();
        assert!(matches!(
            pca.transform(&x),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn transform_shape_mismatch() {
        let x = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let mut pca = PCA::new(PCAComponents::All);
        pca.fit(&x).unwrap();
        let bad = Matrix::new(vec![vec![1.0, 2.0]]).unwrap();
        assert!(pca.transform(&bad).is_err());
    }

    #[test]
    fn n_components_capped_by_min_dim() {
        // 3 samples x 5 features -> max_k = 3
        let x = Matrix::new(vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 1.0, 4.0, 3.0, 6.0],
            vec![5.0, 4.0, 3.0, 2.0, 1.0],
        ])
        .unwrap();
        let mut pca = PCA::new(PCAComponents::Count(10));
        pca.fit(&x).unwrap();
        assert_eq!(pca.n_components(), 3);
    }

    #[test]
    fn components_are_orthonormal() {
        let x = Matrix::new(vec![
            vec![2.5, 2.4],
            vec![0.5, 0.7],
            vec![2.2, 2.9],
            vec![1.9, 2.2],
            vec![3.1, 3.0],
        ])
        .unwrap();
        let mut pca = PCA::new(PCAComponents::All);
        pca.fit(&x).unwrap();
        let comps = pca.components();
        for c in comps {
            let nrm: f64 = c.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((nrm - 1.0).abs() < 1e-8);
        }
        // orthogonality (only meaningful if 2 components)
        if comps.len() == 2 {
            let dot: f64 = comps[0]
                .iter()
                .zip(comps[1].iter())
                .map(|(a, b)| a * b)
                .sum();
            assert!(dot.abs() < 1e-8);
        }
    }

    #[test]
    fn constant_data_zero_variance() {
        let x = Matrix::new(vec![vec![5.0, 5.0], vec![5.0, 5.0], vec![5.0, 5.0]]).unwrap();
        let mut pca = PCA::new(PCAComponents::All);
        let out = pca.fit_transform(&x).unwrap();
        // no variance -> all projections zero
        for i in 0..out.nrows() {
            for j in 0..out.ncols() {
                assert!(out.get(i, j).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn noise_variance_all_components_zero() {
        let x = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let mut pca = PCA::new(PCAComponents::All);
        pca.fit(&x).unwrap();
        // all components kept -> noise = 0
        assert!((pca.noise_variance() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn noise_variance_few_components_positive() {
        let x = Matrix::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![2.0, 4.0, 6.0],
        ])
        .unwrap();
        let mut pca = PCA::new(PCAComponents::Count(1));
        pca.fit(&x).unwrap();
        // noise_variance > 0 because we discarded components
        assert!(pca.noise_variance() > 0.0);
        // noise_variance < first eigenvalue (dominant component)
        assert!(pca.noise_variance() < pca.explained_variance()[0]);
    }

    #[test]
    fn noise_variance_constant_data() {
        let x = Matrix::new(vec![vec![5.0, 5.0], vec![5.0, 5.0], vec![5.0, 5.0]]).unwrap();
        let mut pca = PCA::new(PCAComponents::Count(1));
        pca.fit(&x).unwrap();
        // all eigenvalues are 0 (no variance), so noise = 0
        assert!((pca.noise_variance() - 0.0).abs() < 1e-12);
    }
}
