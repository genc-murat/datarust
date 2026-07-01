use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::stats;
use crate::traits::{default_input_names, FeatureNames};
use crate::Transformer;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Scale features to a given range, mirroring `sklearn.preprocessing.MinMaxScaler`.
///
/// Default range is `[0, 1]`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MinMaxScaler {
    feature_range: (f64, f64),
    min: Vec<f64>,
    data_range: Vec<f64>,
    fitted: bool,
}

impl MinMaxScaler {
    pub fn new() -> Self {
        Self {
            feature_range: (0.0, 1.0),
            min: vec![],
            data_range: vec![],
            fitted: false,
        }
    }

    /// Builder: set the output feature range. `lo` must be strictly less than `hi`.
    pub fn feature_range(mut self, lo: f64, hi: f64) -> Self {
        self.feature_range = (lo, hi);
        self
    }

    pub fn min(&self) -> &[f64] {
        &self.min
    }

    pub fn data_range(&self) -> &[f64] {
        &self.data_range
    }

    pub fn feature_range_value(&self) -> (f64, f64) {
        self.feature_range
    }
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureNames for MinMaxScaler {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        match input_features {
            Some(fs) => fs.to_vec(),
            None => default_input_names(self.min.len()),
        }
    }
}

impl Transformer for MinMaxScaler {
    fn name(&self) -> &'static str {
        "MinMaxScaler"
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        let (lo, hi) = self.feature_range;
        if lo >= hi {
            return Err(DatarustError::InvalidConfig(format!(
                "feature_range lo={} must be < hi={}",
                lo, hi
            )));
        }
        let data = x.rows_ref();
        let min = stats::column_min(data);
        let max = stats::column_max(data);
        let data_range: Vec<f64> = (0..x.ncols()).map(|j| max[j] - min[j]).collect();
        self.min = min;
        self.data_range = data_range;
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("MinMaxScaler".into()));
        }
        if self.min.len() != x.ncols() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.min.len()),
                actual: format!("{} features", x.ncols()),
            });
        }
        let (lo, hi) = self.feature_range;
        let span = hi - lo;
        #[cfg(feature = "rayon")]
        {
            let min = &self.min;
            let data_range = &self.data_range;
            let rows: Vec<Vec<f64>> = x
                .rows_ref()
                .par_iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(j, &v)| {
                            let dr = data_range[j];
                            if dr == 0.0 {
                                lo
                            } else {
                                lo + (v - min[j]) * span / dr
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
                    let dr = self.data_range[j];
                    if dr == 0.0 {
                        out[i][j] = lo;
                    } else {
                        out[i][j] = lo + (v - self.min[j]) * span / dr;
                    }
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
        Matrix::new(vec![vec![-1.0, 10.0], vec![0.0, 20.0], vec![1.0, 30.0]]).unwrap()
    }

    #[test]
    fn fit_transform_default_01() {
        let mut s = MinMaxScaler::new();
        let out = s.fit_transform(&m1()).unwrap();
        assert!((s.min()[0] - (-1.0)).abs() < 1e-12);
        assert!((s.data_range()[0] - 2.0).abs() < 1e-12);
        assert!((s.data_range()[1] - 20.0).abs() < 1e-12);
        // col0 row0 -> 0
        assert!((out.get(0, 0) - 0.0).abs() < 1e-12);
        // col0 row1 -> (0-(-1))/2 = 0.5
        assert!((out.get(1, 0) - 0.5).abs() < 1e-12);
        // col0 row2 -> 1
        assert!((out.get(2, 0) - 1.0).abs() < 1e-12);
        // col1 row0 -> 0, row1 -> 0.5, row2 -> 1
        assert!((out.get(0, 1) - 0.0).abs() < 1e-12);
        assert!((out.get(2, 1) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn custom_range_minus1_1() {
        let mut s = MinMaxScaler::new().feature_range(-1.0, 1.0);
        let out = s.fit_transform(&m1()).unwrap();
        assert!((out.get(0, 0) - (-1.0)).abs() < 1e-12);
        assert!((out.get(1, 0) - 0.0).abs() < 1e-12);
        assert!((out.get(2, 0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn constant_column_mapped_to_lo() {
        let x = Matrix::new(vec![vec![7.0], vec![7.0], vec![7.0]]).unwrap();
        let mut s = MinMaxScaler::new();
        let out = s.fit_transform(&x).unwrap();
        for i in 0..3 {
            assert!((out.get(i, 0) - 0.0).abs() < 1e-12);
        }
    }

    #[test]
    fn constant_column_custom_range() {
        let x = Matrix::new(vec![vec![7.0], vec![7.0]]).unwrap();
        let mut s = MinMaxScaler::new().feature_range(5.0, 15.0);
        let out = s.fit_transform(&x).unwrap();
        // zero range -> mapped to lo = 5
        assert!((out.get(0, 0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn extrapolation_on_new_data() {
        // sklearn does NOT clamp; values beyond range extrapolate.
        let mut s = MinMaxScaler::new();
        s.fit(&m1()).unwrap();
        let new = Matrix::new(vec![vec![2.0, 40.0]]).unwrap();
        let out = s.transform(&new).unwrap();
        assert!((out.get(0, 0) - 1.5).abs() < 1e-12);
        assert!((out.get(0, 1) - 1.5).abs() < 1e-12);
    }

    #[test]
    fn invalid_range_rejected() {
        let mut s = MinMaxScaler::new().feature_range(1.0, 1.0);
        assert!(s.fit(&m1()).is_err());
        let mut s2 = MinMaxScaler::new().feature_range(5.0, 3.0);
        assert!(s2.fit(&m1()).is_err());
    }

    #[test]
    fn transform_before_fit_errors() {
        let s = MinMaxScaler::new();
        assert!(matches!(
            s.transform(&m1()),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn shape_mismatch() {
        let mut s = MinMaxScaler::new();
        s.fit(&m1()).unwrap();
        let bad = Matrix::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();
        assert!(s.transform(&bad).is_err());
    }

    #[test]
    fn inverse_round_trip() {
        let mut s = MinMaxScaler::new().feature_range(-2.0, 8.0);
        let out = s.fit_transform(&m1()).unwrap();
        let original = m1();
        for i in 0..original.nrows() {
            for j in 0..original.ncols() {
                let z = out.get(i, j);
                let (lo, hi) = (-2.0_f64, 8.0_f64);
                let dr = s.data_range()[j];
                let recovered = if dr == 0.0 {
                    s.min()[j]
                } else {
                    s.min()[j] + (z - lo) * dr / (hi - lo)
                };
                assert!((recovered - original.get(i, j)).abs() < 1e-9);
            }
        }
    }
}
