use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::traits::{default_input_names, FeatureNames};
use crate::Transformer;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Norm kind for row-wise normalization, mirroring
/// `sklearn.preprocessing.Normalizer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Norm {
    L1,
    #[default]
    L2,
    Max,
}

/// Normalize samples individually to unit norm (row-wise).
///
/// `Normalizer` is stateless — `fit` does nothing meaningful — matching
/// sklearn behavior.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Normalizer {
    norm: Norm,
    n_features: usize,
    fitted: bool,
}

impl Normalizer {
    pub fn new(norm: Norm) -> Self {
        Self {
            norm,
            n_features: 0,
            fitted: false,
        }
    }

    pub fn norm(&self) -> Norm {
        self.norm
    }

    fn scale_row(&self, row: &[f64]) -> Vec<f64> {
        match self.norm {
            Norm::L1 => {
                let s: f64 = row.iter().map(|v| v.abs()).sum();
                if s == 0.0 {
                    row.to_vec()
                } else {
                    row.iter().map(|v| v / s).collect()
                }
            }
            Norm::L2 => {
                let s: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
                if s == 0.0 {
                    row.to_vec()
                } else {
                    row.iter().map(|v| v / s).collect()
                }
            }
            Norm::Max => {
                let m = row.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
                if m == 0.0 {
                    row.to_vec()
                } else {
                    row.iter().map(|v| v / m).collect()
                }
            }
        }
    }
}

impl Default for Normalizer {
    fn default() -> Self {
        Self::new(Norm::default())
    }
}

impl FeatureNames for Normalizer {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        match input_features {
            Some(fs) => fs.to_vec(),
            None => default_input_names(self.n_features),
        }
    }
}

impl Transformer for Normalizer {
    fn name(&self) -> &'static str {
        "Normalizer"
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        self.n_features = x.ncols();
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("Normalizer".into()));
        }
        #[cfg(feature = "rayon")]
        {
            let out: Vec<Vec<f64>> = x.rows_ref().par_iter().map(|r| self.scale_row(r)).collect();
            Matrix::new(out)
        }
        #[cfg(not(feature = "rayon"))]
        {
            let out: Vec<Vec<f64>> = x.rows_ref().iter().map(|r| self.scale_row(r)).collect();
            Matrix::new(out)
        }
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m1() -> Matrix {
        Matrix::new(vec![vec![3.0, 4.0], vec![1.0, 2.0], vec![0.0, 0.0]]).unwrap()
    }

    #[test]
    fn l2_row_norm_is_one() {
        let mut n = Normalizer::new(Norm::L2);
        let out = n.fit_transform(&m1()).unwrap();
        // 3,4 -> 0.6, 0.8 ; norm = 1
        assert!((out.get(0, 0) - 0.6).abs() < 1e-12);
        assert!((out.get(0, 1) - 0.8).abs() < 1e-12);
        for i in 0..2 {
            let r = out.row(i);
            let nrm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((nrm - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn l1_sum_abs_one() {
        let mut n = Normalizer::new(Norm::L1);
        let out = n.fit_transform(&m1()).unwrap();
        // 3,4 -> 3/7, 4/7
        assert!((out.get(0, 0) - 3.0 / 7.0).abs() < 1e-12);
        assert!((out.get(0, 1) - 4.0 / 7.0).abs() < 1e-12);
        for i in 0..2 {
            let s: f64 = out.row(i).iter().map(|v| v.abs()).sum();
            assert!((s - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn max_norm() {
        let mut n = Normalizer::new(Norm::Max);
        let out = n.fit_transform(&m1()).unwrap();
        // 3,4 -> 0.75, 1.0
        assert!((out.get(0, 0) - 0.75).abs() < 1e-12);
        assert!((out.get(0, 1) - 1.0).abs() < 1e-12);
        for i in 0..2 {
            let m = out.row(i).iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            assert!((m - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn zero_row_passthrough() {
        let mut n = Normalizer::new(Norm::L2);
        let out = n.fit_transform(&m1()).unwrap();
        // zero row stays zero
        assert!((out.get(2, 0) - 0.0).abs() < 1e-12);
        assert!((out.get(2, 1) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn transform_before_fit_errors() {
        let n = Normalizer::new(Norm::L2);
        assert!(matches!(
            n.transform(&m1()),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn negative_values_l2() {
        let x = Matrix::new(vec![vec![-3.0, 4.0]]).unwrap();
        let mut n = Normalizer::new(Norm::L2);
        let out = n.fit_transform(&x).unwrap();
        assert!((out.get(0, 0) - (-0.6)).abs() < 1e-12);
        assert!((out.get(0, 1) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn independent_of_other_rows() {
        // Adding an extreme row does not change other rows' normalization.
        let x = Matrix::new(vec![vec![3.0, 4.0], vec![1000.0, 2000.0]]).unwrap();
        let mut n = Normalizer::new(Norm::L2);
        let out = n.fit_transform(&x).unwrap();
        assert!((out.get(0, 0) - 0.6).abs() < 1e-12);
        assert!((out.get(0, 1) - 0.8).abs() < 1e-12);
    }
}
