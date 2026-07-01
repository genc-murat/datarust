use crate::decomposition::jacobi;
use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::Transformer;

/// Dimensionality reduction via truncated SVD (aka LSA), mirroring
/// `sklearn.decomposition.TruncatedSVD`.
///
/// Unlike PCA, this does **not** center the data, which makes it suitable for
/// sparse inputs like TF-IDF matrices. The right singular vectors are obtained
/// as the eigenvectors of X^T X via Jacobi eigenvalue decomposition.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TruncatedSVD {
    n_components: usize,
    components: Vec<Vec<f64>>, // k x p, rows are right singular vectors
    singular_values: Vec<f64>,
    explained_variance: Vec<f64>,
    explained_variance_ratio: Vec<f64>,
    n_samples_: usize,
    fitted: bool,
}

impl TruncatedSVD {
    pub fn new(n_components: usize) -> Result<Self> {
        if n_components == 0 {
            return Err(DatarustError::InvalidConfig(
                "n_components must be > 0".into(),
            ));
        }
        Ok(Self {
            n_components,
            components: vec![],
            singular_values: vec![],
            explained_variance: vec![],
            explained_variance_ratio: vec![],
            n_samples_: 0,
            fitted: false,
        })
    }

    pub fn components(&self) -> &[Vec<f64>] {
        &self.components
    }

    pub fn singular_values(&self) -> &[f64] {
        &self.singular_values
    }

    pub fn explained_variance(&self) -> &[f64] {
        &self.explained_variance
    }

    pub fn explained_variance_ratio(&self) -> &[f64] {
        &self.explained_variance_ratio
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
}

impl Transformer for TruncatedSVD {
    fn name(&self) -> &'static str {
        "TruncatedSVD"
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        let n = x.nrows();
        let p = x.ncols();
        self.n_samples_ = n;
        if self.n_components > p {
            return Err(DatarustError::InvalidConfig(format!(
                "n_components={} must be <= n_features={}",
                self.n_components, p
            )));
        }
        let m = Self::xtx(x);
        let (mut vals, vecs) = jacobi::eigh(&m);
        for v in vals.iter_mut() {
            if *v < 0.0 && v.abs() < 1e-10 {
                *v = 0.0;
            }
        }
        let total_var: f64 = vals.iter().sum();
        let k = self.n_components;
        self.components = vecs.into_iter().take(k).collect();
        self.singular_values = vals.iter().take(k).map(|v| v.max(0.0).sqrt()).collect();
        // sklearn: explained_variance = (X . V_k)^2 / (n - 1)
        // but also equals eigenvalue / (n-1) since X^T X eigenvalue = sigma^2.
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
        let k = self.n_components;
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

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl crate::traits::FeatureNames for TruncatedSVD {
    fn feature_names_out(&self, _input_features: Option<&[String]>) -> Vec<String> {
        (0..self.n_components)
            .map(|i| format!("svd{}", i))
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
    #[allow(clippy::needless_range_loop)]
    fn reconstruction_uses_no_centering() {
        // TruncatedSVD does not center; projecting then reconstructing should
        // recover the part of X lying in the component subspace.
        let x = Matrix::new(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]).unwrap();
        let k = 2;
        let mut svd = TruncatedSVD::new(k).unwrap();
        let proj = svd.fit_transform(&x).unwrap();
        // reconstruct: proj (n x k) @ components (k x p)
        let comps = svd.components();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let mut s = 0.0;
                for c in 0..k {
                    s += proj.get(i, c) * comps[c][j];
                }
                assert!(approx(s, x.get(i, j), 1e-7));
            }
        }
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
