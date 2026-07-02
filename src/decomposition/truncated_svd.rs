use crate::decomposition::jacobi;
use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::Transformer;

/// How to specify the number of components for [`TruncatedSVD`],
/// mirroring sklearn's `n_components` parameter in `TruncatedSVD`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SVDComponents {
    /// Keep exactly this many components.
    Count(usize),
    /// Keep the smallest number of components such that the cumulative
    /// explained variance ratio is at least the given value in (0, 1).
    Variance(f64),
    /// Keep all components (`min(n_samples, n_features)`).
    All,
}

impl From<usize> for SVDComponents {
    fn from(n: usize) -> Self {
        SVDComponents::Count(n)
    }
}

impl From<f64> for SVDComponents {
    fn from(v: f64) -> Self {
        SVDComponents::Variance(v)
    }
}

/// Dimensionality reduction via truncated SVD (aka LSA), mirroring
/// `sklearn.decomposition.TruncatedSVD`.
///
/// Unlike PCA, this does **not** center the data, which makes it suitable for
/// sparse inputs like TF-IDF matrices. The right singular vectors are obtained
/// as the eigenvectors of X^T X via Jacobi eigenvalue decomposition.
///
/// Supports flexible component selection via [`SVDComponents`]:
/// ```rust
/// use datarust::decomposition::TruncatedSVD;
///
/// // By exact count (backward-compatible):
/// let svd = TruncatedSVD::new(5).unwrap();
///
/// // By variance threshold:
/// let svd = TruncatedSVD::new(0.95).unwrap();
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TruncatedSVD {
    components_spec: SVDComponents,
    components: Vec<Vec<f64>>, // k x p, rows are right singular vectors
    singular_values: Vec<f64>,
    explained_variance: Vec<f64>,
    explained_variance_ratio: Vec<f64>,
    n_components_: usize,
    n_samples_: usize,
    fitted: bool,
}

impl TruncatedSVD {
    /// Creates a new TruncatedSVD with the given component selection.
    pub fn new<C: Into<SVDComponents>>(components: C) -> Result<Self> {
        let spec = components.into();
        match &spec {
            SVDComponents::Count(n) if *n == 0 => {
                return Err(DatarustError::InvalidConfig(
                    "n_components must be > 0".into(),
                ));
            }
            SVDComponents::Variance(v) if *v <= 0.0 || *v >= 1.0 => {
                return Err(DatarustError::InvalidConfig(
                    "variance threshold must be in (0, 1)".into(),
                ));
            }
            _ => {}
        }
        Ok(Self {
            components_spec: spec,
            components: vec![],
            singular_values: vec![],
            explained_variance: vec![],
            explained_variance_ratio: vec![],
            n_components_: 0,
            n_samples_: 0,
            fitted: false,
        })
    }

    /// Returns the right singular vectors (one row per component).
    pub fn components(&self) -> &[Vec<f64>] {
        &self.components
    }

    /// Returns the singular values of the kept components.
    pub fn singular_values(&self) -> &[f64] {
        &self.singular_values
    }

    /// Returns the variance explained by each kept component.
    pub fn explained_variance(&self) -> &[f64] {
        &self.explained_variance
    }

    /// Returns the fraction of total variance explained by each kept component.
    pub fn explained_variance_ratio(&self) -> &[f64] {
        &self.explained_variance_ratio
    }

    /// Number of components determined during fit.
    pub fn n_components(&self) -> usize {
        self.n_components_
    }

    fn xtx(x: &Matrix) -> Vec<Vec<f64>> {
        let p = x.ncols();
        let mut m = vec![vec![0.0; p]; p];
        for row in x.rows_ref() {
            for i in 0..p {
                if row[i] == 0.0 {
                    continue;
                }
                for j in 0..p {
                    m[i][j] += row[i] * row[j];
                }
            }
        }
        m
    }

    fn resolve_components(&self, vals: &[f64]) -> Result<usize> {
        let total = vals.len();
        match &self.components_spec {
            SVDComponents::Count(n) => {
                if *n == 0 {
                    return Ok(total);
                }
                if *n > total {
                    return Err(DatarustError::InvalidConfig(format!(
                        "n_components={} must be <= n_features={}",
                        n, total
                    )));
                }
                Ok(*n)
            }
            SVDComponents::Variance(threshold) => {
                let var_sum: f64 = vals.iter().sum();
                if var_sum <= 0.0 {
                    return Ok(1);
                }
                let mut cum = 0.0;
                for (i, &v) in vals.iter().enumerate() {
                    cum += v / var_sum;
                    if cum >= *threshold {
                        return Ok((i + 1).max(1));
                    }
                }
                Ok(total)
            }
            SVDComponents::All => Ok(total),
        }
    }
}

impl Transformer for TruncatedSVD {
    fn name(&self) -> &'static str {
        "TruncatedSVD"
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        let n = x.nrows();
        self.n_samples_ = n;
        let m = Self::xtx(x);
        let (mut vals, vecs) = jacobi::eigh(&m)
            .ok_or_else(|| DatarustError::Singular("XᵀX matrix is empty or non-square".into()))?;
        for v in vals.iter_mut() {
            if *v < 0.0 && v.abs() < 1e-10 {
                *v = 0.0;
            }
        }
        // eigenvalues from jacobi are descending
        let k = self.resolve_components(&vals)?;
        self.n_components_ = k;
        let total_var: f64 = vals.iter().sum();
        self.components = vecs.into_iter().take(k).collect();
        self.singular_values = vals.iter().take(k).map(|v| v.max(0.0).sqrt()).collect();
        let denom = (n.saturating_sub(1)) as f64;
        let denom = if denom > 0.0 { denom } else { 1.0 };
        self.explained_variance = vals.iter().take(k).map(|v| v / denom).collect();
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
            return Err(DatarustError::NotFitted("TruncatedSVD".into()));
        }
        if x.ncols() != self.components[0].len() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.components[0].len()),
                actual: format!("{} features", x.ncols()),
            });
        }
        let k = self.n_components_;
        let mut out = vec![vec![0.0; k]; x.nrows()];
        for (i, row) in x.rows_ref().iter().enumerate() {
            for j in 0..k {
                let comp = &self.components[j];
                let mut s = 0.0;
                for c in 0..row.len() {
                    s += row[c] * comp[c];
                }
                out[i][j] = s;
            }
        }
        Matrix::new(out)
    }

    fn inverse_transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("TruncatedSVD".into()));
        }
        if x.ncols() != self.n_components_ {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} components", self.n_components_),
                actual: format!("{} columns", x.ncols()),
            });
        }
        let p = self.components[0].len();
        let k = self.n_components_;
        let mut out = vec![vec![0.0; p]; x.nrows()];
        for (i, row) in x.rows_ref().iter().enumerate() {
            for (j, comp) in self.components.iter().enumerate().take(k) {
                for c in 0..p {
                    out[i][c] += row[j] * comp[c];
                }
            }
        }
        Matrix::new(out)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl crate::traits::FeatureNames for TruncatedSVD {
    fn feature_names_out(&self, _input_features: Option<&[String]>) -> Vec<String> {
        if self.fitted {
            (0..self.n_components_)
                .map(|i| format!("svd{}", i))
                .collect()
        } else {
            match &self.components_spec {
                SVDComponents::Count(n) => (0..*n).map(|i| format!("svd{}", i)).collect(),
                SVDComponents::Variance(_) | SVDComponents::All => {
                    vec!["svd*".to_string()]
                }
            }
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
    fn basic_transform_shape() {
        let x = Matrix::new(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ])
        .unwrap();
        let mut svd = TruncatedSVD::new(2).unwrap();
        let out = svd.fit_transform(&x).unwrap();
        assert_eq!(out.ncols(), 2);
        assert_eq!(out.nrows(), 4);
        assert_eq!(svd.singular_values().len(), 2);
    }

    #[test]
    fn singular_values_match_eigenvalues() {
        // For an orthogonal-ish matrix, singular values relate to X^T X eigenvalues
        let x = Matrix::new(vec![vec![3.0, 0.0], vec![0.0, 4.0], vec![0.0, 0.0]]).unwrap();
        let mut svd = TruncatedSVD::new(2).unwrap();
        svd.fit(&x).unwrap();
        // X^T X = diag(9, 16) -> eigenvalues 16, 9 -> singular values 4, 3
        assert!(approx(svd.singular_values()[0], 4.0, 1e-8));
        assert!(approx(svd.singular_values()[1], 3.0, 1e-8));
    }

    #[test]
    fn explained_variance_ratio_descending() {
        let x = Matrix::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
            vec![1.0, 1.0, 1.0],
        ])
        .unwrap();
        let mut svd = TruncatedSVD::new(2).unwrap();
        svd.fit(&x).unwrap();
        let r = svd.explained_variance_ratio();
        assert!(r[0] >= r[1]);
        assert!(r[0] > 0.0);
    }

    #[test]
    fn reconstruction_via_inverse_transform() {
        // TruncatedSVD does not center; inverse_transform should recover
        // original when all components are kept.
        let x = Matrix::new(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]).unwrap();
        let mut svd = TruncatedSVD::new(2).unwrap();
        let proj = svd.fit_transform(&x).unwrap();
        let recovered = svd.inverse_transform(&proj).unwrap();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!(approx(recovered.get(i, j), x.get(i, j), 1e-7));
            }
        }
    }

    #[test]
    fn inverse_transform_reduced_rank_approx() {
        // With k < n_features, inverse_transform gives a low-rank approximation.
        let x = Matrix::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
            vec![1.0, 2.0, 3.0],
        ])
        .unwrap();
        let mut svd = TruncatedSVD::new(1).unwrap();
        let proj = svd.fit_transform(&x).unwrap();
        let recovered = svd.inverse_transform(&proj).unwrap();
        // Should have the right shape
        assert_eq!(recovered.nrows(), x.nrows());
        assert_eq!(recovered.ncols(), x.ncols());
        // The dominant rank-1 component should capture most of the variance
        // (this data is essentially rank-1: col1 = 2*col0, col2 = 3*col0)
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!(approx(recovered.get(i, j), x.get(i, j), 1e-5));
            }
        }
    }

    #[test]
    fn inverse_transform_before_fit_errors() {
        let svd = TruncatedSVD::new(1).unwrap();
        let x = Matrix::new(vec![vec![1.0]]).unwrap();
        assert!(matches!(
            svd.inverse_transform(&x),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn inverse_transform_shape_mismatch() {
        let x = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let mut svd = TruncatedSVD::new(2).unwrap();
        svd.fit(&x).unwrap();
        let bad = Matrix::new(vec![vec![1.0]]).unwrap();
        assert!(svd.inverse_transform(&bad).is_err());
    }

    #[test]
    fn n_components_too_large_errors() {
        let x = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let mut svd = TruncatedSVD::new(5).unwrap();
        assert!(svd.fit(&x).is_err());
    }

    #[test]
    fn zero_n_components_errors() {
        assert!(TruncatedSVD::new(0).is_err());
    }

    #[test]
    fn variance_threshold_invalid() {
        assert!(TruncatedSVD::new(SVDComponents::Variance(0.0)).is_err());
        assert!(TruncatedSVD::new(SVDComponents::Variance(1.0)).is_err());
        assert!(TruncatedSVD::new(SVDComponents::Variance(0.5)).is_ok());
    }

    #[test]
    fn variance_threshold_selects_k() {
        let x = Matrix::new(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ])
        .unwrap();
        let mut svd = TruncatedSVD::new(SVDComponents::Variance(0.95)).unwrap();
        svd.fit(&x).unwrap();
        assert!(svd.n_components() >= 1);
        assert!(svd.n_components() <= 3);
    }

    #[test]
    fn all_components_kept() {
        let x = Matrix::new(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]).unwrap();
        let mut svd = TruncatedSVD::new(SVDComponents::All).unwrap();
        svd.fit(&x).unwrap();
        assert_eq!(svd.n_components(), 2);
        let proj = svd.transform(&x).unwrap();
        assert_eq!(proj.ncols(), 2);
    }

    #[test]
    fn transform_before_fit_errors() {
        let svd = TruncatedSVD::new(1).unwrap();
        let x = Matrix::new(vec![vec![1.0, 2.0]]).unwrap();
        assert!(matches!(
            svd.transform(&x),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn transform_shape_mismatch() {
        let x = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let mut svd = TruncatedSVD::new(2).unwrap();
        svd.fit(&x).unwrap();
        let bad = Matrix::new(vec![vec![1.0, 2.0]]).unwrap();
        assert!(svd.transform(&bad).is_err());
    }

    #[test]
    fn components_orthonormal() {
        let x = Matrix::new(vec![
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![2.0, 3.0, 1.0],
        ])
        .unwrap();
        let mut svd = TruncatedSVD::new(2).unwrap();
        svd.fit(&x).unwrap();
        for c in svd.components() {
            let nrm: f64 = c.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((nrm - 1.0).abs() < 1e-8);
        }
        let dot: f64 = svd.components()[0]
            .iter()
            .zip(svd.components()[1].iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(dot.abs() < 1e-8);
    }
}
