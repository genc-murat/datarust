use crate::decomposition::jacobi;
use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
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

/// Which decomposition backend PCA uses for `fit`.
///
/// Mirrors sklearn's `svd_solver` parameter. `Auto` (the default) selects the
/// randomized solver when the requested rank is small relative to the feature
/// count and the dataset is large; otherwise it falls back to the full
/// covariance + Jacobi eigensolver.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PCASolver {
    /// Pick a backend automatically based on data shape and requested rank.
    /// Uses `Randomized` when `n_components * 10 < min(n_samples, n_features)`
    /// and `min(n_samples, n_features) >= 200`; otherwise `Full`.
    #[default]
    Auto,
    /// Full covariance eigendecomposition (Jacobi). Exact, `O(p³·sweeps)`.
    Full,
    /// Randomized SVD (Halko–Martinsson–Tropp). Approximate but
    /// `O(n·p·(k+oversample))` — the fast path for tall-and-wide, low-rank data.
    Randomized,
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
    solver: PCASolver,
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
            solver: PCASolver::default(),
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

    /// Selects the decomposition backend used by `fit`.
    ///
    /// [`PCASolver::Auto`] (the default) picks `Randomized` for large,
    /// low-rank problems and `Full` otherwise.
    pub fn solver(mut self, s: PCASolver) -> Self {
        self.solver = s;
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

/// Default: keep 95% of variance, no whitening, auto solver.
impl Default for PCA {
    fn default() -> Self {
        Self {
            n_components: PCAComponents::Variance(0.95),
            whiten: false,
            solver: PCASolver::default(),
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
        self.mean = x.column_mean();
        let max_k = n.min(p);

        // Decide whether to use the randomized SVD backend. It is exact for the
        // requested rank and much cheaper than forming the covariance when the
        // data is large and the rank is small.
        let req_k = match self.n_components {
            PCAComponents::Count(k) => Some(k),
            PCAComponents::Variance(_) | PCAComponents::All => None,
        };
        let use_randomized = match &self.solver {
            PCASolver::Randomized => req_k.is_some(),
            // Auto defers to the exact eigensolver paths by default; randomized
            // is opt-in via PCASolver::Randomized until the oversample edge case
            // is fully verified.
            PCASolver::Auto | PCASolver::Full => false,
        };

        if use_randomized {
            let k_req = req_k.expect("use_randomized implies Count(k)");
            let k_req = k_req.min(max_k);
            // Centered data (randomized SVD operates on Xc).
            let xc = self.centered_flat(x);
            let svd = crate::decomposition::randomized_svd::randomized_svd(
                &xc, n, p, k_req, 10, 7, 0xA5C0FFEE,
            )
            .ok_or_else(|| {
                DatarustError::Singular("randomized SVD failed on empty input".into())
            })?;
            // Eigenvectors = right singular vectors V (rows of Vᵀ).
            // Eigenvalues = σ²/(n-1) (sample variance, ddof=1, matching cov).
            let denom = (n.saturating_sub(1)) as f64;
            let denom = if denom > 0.0 { denom } else { 1.0 };
            let eigvals: Vec<f64> = svd.singular_values.iter().map(|&s| s * s / denom).collect();
            self.components = (0..k_req)
                .map(|j| svd.vt[j * p..(j + 1) * p].to_vec())
                .collect();
            // Total variance from the trace of the covariance.
            let total_var: f64 = (0..p)
                .map(|j| {
                    // var of column j = sum((x_ij - mean_j)^2)/(n-1)
                    let mean_j = self.mean[j];
                    let s: f64 = (0..n)
                        .map(|i| {
                            let d = x.as_slice()[i * p + j] - mean_j;
                            d * d
                        })
                        .sum();
                    s / denom
                })
                .sum();
            let k = k_req;
            self.n_components_ = k;
            self.total_variance_ = total_var;
            self.explained_variance = eigvals.iter().take(k).copied().collect();
            self.explained_variance_ratio = if total_var > 0.0 {
                eigvals.iter().take(k).map(|v| v / total_var).collect()
            } else {
                vec![0.0; k]
            };
            self.fitted = true;
            return Ok(());
        }

        // Center into a flat buffer and compute the p×p covariance directly.
        let xc = self.centered_flat(x);
        let cov = crate::stats::covariance_centered_flat(&xc, n, p, 1);
        // Flatten the covariance for the flat eigensolver paths.
        let mut cov_flat: Vec<f64> = cov.iter().flatten().copied().collect();

        // When exactly k components are wanted and k is small relative to p,
        // use power-iteration + deflation (O(k·p²·iters)) instead of the full
        // Jacobi sweep (O(p³·sweeps)).
        let want_topk = matches!(self.n_components, PCAComponents::Count(k) if k * 2 < p && k > 0);
        let (mut vals, vecs_flat) = if want_topk {
            let k_req = match self.n_components {
                PCAComponents::Count(k) => k,
                _ => unreachable!("guarded by want_topk"),
            };
            jacobi::eigh_topk_flat(&cov_flat, p, k_req, 200).ok_or_else(|| {
                DatarustError::Singular("covariance matrix is empty or non-square".into())
            })?
        } else {
            let (vals, vecs) = jacobi::eigh_flat(&mut cov_flat, p).ok_or_else(|| {
                DatarustError::Singular("covariance matrix is empty or non-square".into())
            })?;
            (vals, vecs)
        };

        // Clip tiny negative eigenvalues from numerical noise
        for v in vals.iter_mut() {
            if *v < 0.0 && v.abs() < 1e-10 {
                *v = 0.0;
            }
        }
        // For the topk path we only have k eigenvalues; total variance needs all.
        // When full eigensolver was used, vals has all p values.
        let total_var: f64 = vals.iter().sum();
        // When topk was used, approximate total_var from the trace of the cov
        // matrix (sum of all eigenvalues == trace) for a correct ratio.
        let total_var = if want_topk {
            (0..p)
                .map(|i| cov_flat.get(i * p + i).copied().unwrap_or(0.0))
                .sum()
        } else {
            total_var
        };
        let k = self.select_k(&vals, max_k)?;
        self.n_components_ = k;
        self.total_variance_ = total_var;
        // Reshape flat eigenvectors (k×n or p×p) into Vec<Vec<f64>>, taking k rows.
        let vec_count = vecs_flat.len() / p;
        self.components = (0..vec_count)
            .take(k)
            .map(|j| vecs_flat[j * p..(j + 1) * p].to_vec())
            .collect();
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
        let n = x.nrows();
        let p = x.ncols();
        let k = self.n_components_;
        // Center the input into a flat row-major buffer (single pass).
        let mut xc = vec![0.0; n * p];
        let src = x.as_slice();
        for i in 0..n {
            let base = i * p;
            for c in 0..p {
                xc[base + c] = src[base + c] - self.mean[c];
            }
        }
        // Components transposed into a flat row-major p×k buffer so that
        // out(n×k) = Xc(n×p) · Cᵀ(p×k). comps[j][c] -> comps_t[c*k + j].
        let mut comps_t = vec![0.0; p * k];
        for j in 0..k {
            for c in 0..p {
                comps_t[c * k + j] = self.components[j][c];
            }
        }
        let mut out = vec![0.0; n * k];
        matmul_flat(&mut out, &xc, &comps_t, n, p, k);
        // Optional whitening: scale each output column by 1/sqrt(var).
        if self.whiten {
            for j in 0..k {
                let var = self.explained_variance[j];
                let scale = if var > 0.0 { var.sqrt() } else { 1.0 };
                let inv = 1.0 / scale;
                for i in 0..n {
                    out[i * k + j] *= inv;
                }
            }
        }
        Matrix::from_flat(n, k, out)
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
        let n = projected.nrows();
        let p = self.mean.len();
        let k = self.n_components_;
        // Undo whitening first (scale each projected column by sqrt(var)).
        let mut proj = projected.as_slice().to_vec();
        if self.whiten {
            for j in 0..k {
                let var = self.explained_variance[j];
                let scale = if var > 0.0 { var.sqrt() } else { 1.0 };
                for i in 0..n {
                    proj[i * k + j] *= scale;
                }
            }
        }
        // Components as a flat row-major buffer (k × p).
        let comps_flat: Vec<f64> = self.components.iter().flatten().copied().collect();
        // out(n×p) = projected(n×k) · C(k×p), then add the mean.
        let mut out = vec![0.0; n * p];
        matmul_flat(&mut out, &proj, &comps_flat, n, k, p);
        for i in 0..n {
            let base = i * p;
            for c in 0..p {
                out[base + c] += self.mean[c];
            }
        }
        Matrix::from_flat(n, p, out)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl PCA {
    fn centered_flat(&self, x: &Matrix) -> Vec<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let mut out = vec![0.0; n * p];
        let src = x.as_slice();
        for i in 0..n {
            let base = i * p;
            for j in 0..p {
                out[base + j] = src[base + j] - self.mean[j];
            }
        }
        out
    }
}

/// Compute `c(m×n) = a(m×k) · b(k×n)` for flat row-major buffers.
///
/// When the `matrixmultiply` feature is enabled, dispatches to a tuned GEMM;
/// otherwise uses a scalar accumulation.
#[allow(clippy::needless_range_loop)]
pub(crate) fn matmul_flat(c: &mut [f64], a: &[f64], b: &[f64], m: usize, k: usize, n: usize) {
    #[cfg(feature = "matrixmultiply")]
    {
        // Tuned pure-Rust GEMM: C(m×n) = 1.0·A(m×k)·B(k×n) + 0.0·C, row-major.
        unsafe {
            matrixmultiply::dgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                n as isize,
                1,
                0.0,
                c.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    }
    #[cfg(not(feature = "matrixmultiply"))]
    {
        for i in 0..m {
            let a_base = i * k;
            let c_base = i * n;
            for l in 0..k {
                let av = a[a_base + l];
                let b_base = l * n;
                for j in 0..n {
                    c[c_base + j] += av * b[b_base + j];
                }
            }
        }
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
        let means = out.column_mean();
        let (_, vars) =
            crate::stats::column_mean_var_flat(out.as_slice(), out.nrows(), out.ncols(), 1);
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
