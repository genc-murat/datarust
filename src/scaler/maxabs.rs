use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::traits::{default_input_names, FeatureNames};
use crate::Transformer;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Scale each feature by its maximum absolute value.
///
/// Mirrors `sklearn.preprocessing.MaxAbsScaler`.  The output is in the range
/// `[-1, 1]` for every column where the maximum absolute value is non-zero.
/// Unlike `StandardScaler` it does **not** center the data, which preserves
/// sparsity — making it a good fit for sparse matrices.
///
/// ```text
/// scaled = x / max_abs
/// ```
///
/// When the column contains only zeros the scaled values remain zero
/// (division by zero is guarded).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MaxAbsScaler {
    max_abs: Vec<f64>,
    fitted: bool,
}

impl MaxAbsScaler {
    pub fn new() -> Self {
        Self {
            max_abs: vec![],
            fitted: false,
        }
    }

    /// Fitted per-column maximum absolute values.
    pub fn max_abs(&self) -> &[f64] {
        &self.max_abs
    }
}

impl Default for MaxAbsScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureNames for MaxAbsScaler {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        match input_features {
            Some(fs) => fs.to_vec(),
            None => default_input_names(self.max_abs.len()),
        }
    }
}

impl Transformer for MaxAbsScaler {
    fn name(&self) -> &'static str {
        "MaxAbsScaler"
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        let ncols = x.ncols();
        let mut max_abs = vec![0.0f64; ncols];
        for row in x.rows_ref() {
            for (j, &v) in row.iter().enumerate() {
                let a = v.abs();
                if a > max_abs[j] {
                    max_abs[j] = a;
                }
            }
        }
        self.max_abs = max_abs;
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("MaxAbsScaler".into()));
        }
        if self.max_abs.len() != x.ncols() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.max_abs.len()),
                actual: format!("{} features", x.ncols()),
            });
        }
        #[cfg(feature = "rayon")]
        {
            let max_abs = &self.max_abs;
            let rows: Vec<Vec<f64>> = x
                .rows_ref()
                .par_iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(j, &v)| {
                            if max_abs[j] == 0.0 {
                                0.0
                            } else {
                                v / max_abs[j]
                            }
                        })
                        .collect()
                })
                .collect();
            Matrix::new(rows)
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut out = vec![vec![0.0; x.ncols()]; x.nrows()];
            for (i, row) in x.rows_ref().iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    out[i][j] = if self.max_abs[j] == 0.0 {
                        0.0
                    } else {
                        v / self.max_abs[j]
                    };
                }
            }
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
        Matrix::new(vec![
            vec![0.0, 10.0],
            vec![0.0, 10.0],
            vec![1.0, 100.0],
            vec![1.0, 100.0],
        ])
        .unwrap()
    }

    #[test]
    fn fit_transform_basic() {
        let mut s = MaxAbsScaler::new();
        let out = s.fit_transform(&m1()).unwrap();
        assert!((s.max_abs()[0] - 1.0).abs() < 1e-12);
        assert!((s.max_abs()[1] - 100.0).abs() < 1e-12);
        // row 0: 0/1=0, 10/100=0.1
        assert!((out.get(0, 0) - 0.0).abs() < 1e-12);
        assert!((out.get(0, 1) - 0.1).abs() < 1e-12);
        // row 2: 1/1=1, 100/100=1
        assert!((out.get(2, 0) - 1.0).abs() < 1e-12);
        assert!((out.get(2, 1) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn output_range_minus1_to_1() {
        let mut s = MaxAbsScaler::new();
        let out = s.fit_transform(&m1()).unwrap();
        for i in 0..out.nrows() {
            for j in 0..out.ncols() {
                let v = out.get(i, j);
                assert!(v.abs() <= 1.0 + 1e-12, "out of range: {}", v);
            }
        }
    }

    #[test]
    fn constant_column_zero() {
        let x = Matrix::new(vec![vec![5.0, 0.0], vec![5.0, 0.0]]).unwrap();
        let mut s = MaxAbsScaler::new();
        let out = s.fit_transform(&x).unwrap();
        // col0: 5/5=1, col1: max_abs=0 -> keep as 0
        assert!((out.get(0, 0) - 1.0).abs() < 1e-12);
        assert!((out.get(0, 1) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn negative_values() {
        let x = Matrix::new(vec![vec![-3.0, 10.0], vec![3.0, -10.0]]).unwrap();
        let mut s = MaxAbsScaler::new();
        let out = s.fit_transform(&x).unwrap();
        assert!((s.max_abs()[0] - 3.0).abs() < 1e-12);
        assert!((s.max_abs()[1] - 10.0).abs() < 1e-12);
        assert!((out.get(0, 0) - (-1.0)).abs() < 1e-12);
        assert!((out.get(1, 1) - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn transform_before_fit_errors() {
        let s = MaxAbsScaler::new();
        assert!(matches!(
            s.transform(&m1()),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn shape_mismatch_on_transform() {
        let mut s = MaxAbsScaler::new();
        s.fit(&m1()).unwrap();
        let bad = Matrix::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();
        assert!(s.transform(&bad).is_err());
    }

    #[test]
    fn inverse_round_trip() {
        let mut s = MaxAbsScaler::new();
        let out = s.fit_transform(&m1()).unwrap();
        let original = m1();
        for i in 0..original.nrows() {
            for j in 0..original.ncols() {
                let recovered = out.get(i, j) * s.max_abs()[j];
                assert!((recovered - original.get(i, j)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn feature_names_passthrough() {
        let mut s = MaxAbsScaler::new();
        s.fit(&m1()).unwrap();
        let names = s.feature_names_out(Some(&["a".to_string(), "b".to_string()]));
        assert_eq!(names, vec!["a", "b"]);
        let default = s.feature_names_out(None);
        assert_eq!(default, vec!["x0", "x1"]);
    }
}
