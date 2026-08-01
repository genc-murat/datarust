use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::stats;
use crate::traits::{default_input_names, FeatureNames};
use crate::Transformer;

/// Scale features to a given range, mirroring `sklearn.preprocessing.MinMaxScaler`.
///
/// Default range is `[0, 1]`.
///
/// # Examples
///
/// ```rust,no_run
/// use datarust::matrix::Matrix;
/// use datarust::scaler::MinMaxScaler;
/// use datarust::Transformer;
///
/// let x = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]).unwrap();
/// let mut scaler = MinMaxScaler::new();
/// let scaled = scaler.fit_transform(&x).unwrap();
/// // scaled values are in [0, 1] range
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MinMaxScaler {
    feature_range: (f64, f64),
    min: Vec<f64>,
    data_range: Vec<f64>,
    fitted: bool,
}

impl MinMaxScaler {
    /// Creates a new scaler with the default `[0, 1]` range.
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

    /// Fitted per-column minimum values.
    pub fn min(&self) -> &[f64] {
        &self.min
    }

    /// Fitted per-column data range (max - min).
    pub fn data_range(&self) -> &[f64] {
        &self.data_range
    }

    /// Returns the configured output feature range.
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
        if !lo.is_finite() || !hi.is_finite() || lo >= hi {
            return Err(DatarustError::InvalidConfig(format!(
                "feature_range values must be finite and satisfy lo < hi, got lo={} hi={}",
                lo, hi
            )));
        }
        x.validate_finite()?;
        // Single fused min+max pass over flat storage.
        let (min, max) = stats::column_min_max_flat(x.as_slice(), x.nrows(), x.ncols());
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
        if self.min.len() != self.data_range.len()
            || !self.feature_range.0.is_finite()
            || !self.feature_range.1.is_finite()
            || self.feature_range.0 >= self.feature_range.1
            || self
                .min
                .iter()
                .chain(&self.data_range)
                .any(|v| !v.is_finite())
        {
            return Err(DatarustError::InvalidInput(
                "MinMaxScaler has inconsistent fitted state".into(),
            ));
        }
        if self.min.len() != x.ncols() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.min.len()),
                actual: format!("{} features", x.ncols()),
            });
        }
        x.validate_finite()?;
        // Flat-storage transform with fused NaN check.
        let (lo, hi) = self.feature_range;
        let span = hi - lo;
        let nrows = x.nrows();
        let ncols = x.ncols();
        let min = &self.min;
        let data_range = &self.data_range;
        let src = x.as_slice();
        let mut out = vec![0.0; nrows * ncols];
        #[cfg(feature = "rayon")]
        if nrows >= 4096 {
            use rayon::prelude::*;
            out.par_chunks_mut(ncols)
                .zip(src.par_chunks(ncols))
                .for_each(|(out_row, in_row)| {
                    for (j, &v) in in_row.iter().enumerate() {
                        let dr = data_range[j];
                        out_row[j] = if dr == 0.0 {
                            lo
                        } else {
                            lo + (v - min[j]) * span / dr
                        };
                    }
                });
            if out.par_iter().any(|v| v.is_nan()) {
                for i in 0..nrows {
                    for j in 0..ncols {
                        if src[i * ncols + j].is_nan() {
                            return Err(DatarustError::InvalidInput(format!(
                                "NaN value at position ({i}, {j})"
                            )));
                        }
                    }
                }
            }
            return Matrix::from_flat(nrows, ncols, out);
        }
        for i in 0..nrows {
            let base = i * ncols;
            for j in 0..ncols {
                let v = src[base + j];
                if v.is_nan() {
                    return Err(DatarustError::InvalidInput(format!(
                        "NaN value at position ({i}, {j})"
                    )));
                }
                let dr = data_range[j];
                out[base + j] = if dr == 0.0 {
                    lo
                } else {
                    lo + (v - min[j]) * span / dr
                };
            }
        }
        Matrix::from_flat(nrows, ncols, out)
    }

    fn inverse_transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("MinMaxScaler".into()));
        }
        if self.min.len() != self.data_range.len()
            || !self.feature_range.0.is_finite()
            || !self.feature_range.1.is_finite()
            || self.feature_range.0 >= self.feature_range.1
            || self
                .min
                .iter()
                .chain(&self.data_range)
                .any(|v| !v.is_finite())
        {
            return Err(DatarustError::InvalidInput(
                "MinMaxScaler has inconsistent fitted state".into(),
            ));
        }
        if self.min.len() != x.ncols() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.min.len()),
                actual: format!("{} features", x.ncols()),
            });
        }
        x.validate_finite()?;
        let (lo, hi) = self.feature_range;
        let span = hi - lo;
        let nrows = x.nrows();
        let ncols = x.ncols();
        let min = &self.min;
        let data_range = &self.data_range;
        let src = x.as_slice();
        let mut out = vec![0.0; nrows * ncols];
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            out.par_chunks_mut(ncols)
                .zip(src.par_chunks(ncols))
                .for_each(|(out_row, in_row)| {
                    for (j, &z) in in_row.iter().enumerate() {
                        let dr = data_range[j];
                        out_row[j] = if dr == 0.0 {
                            min[j]
                        } else {
                            min[j] + (z - lo) * dr / span
                        };
                    }
                });
        }
        #[cfg(not(feature = "rayon"))]
        {
            for i in 0..nrows {
                let base = i * ncols;
                for j in 0..ncols {
                    let z = src[base + j];
                    let dr = data_range[j];
                    out[base + j] = if dr == 0.0 {
                        min[j]
                    } else {
                        min[j] + (z - lo) * dr / span
                    };
                }
            }
        }
        Matrix::from_flat(nrows, ncols, out)
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
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(MinMaxScaler::new()
                .feature_range(invalid, 1.0)
                .fit(&m1())
                .is_err());
            assert!(MinMaxScaler::new()
                .feature_range(0.0, invalid)
                .fit(&m1())
                .is_err());
        }
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
    fn inverse_transform_round_trip() {
        let mut s = MinMaxScaler::new().feature_range(-2.0, 8.0);
        let x = m1();
        let out = s.fit_transform(&x).unwrap();
        let recovered = s.inverse_transform(&out).unwrap();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!((recovered.get(i, j) - x.get(i, j)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn inverse_transform_default_range() {
        let mut s = MinMaxScaler::new();
        let x = m1();
        let out = s.fit_transform(&x).unwrap();
        let recovered = s.inverse_transform(&out).unwrap();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!((recovered.get(i, j) - x.get(i, j)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn inverse_transform_constant_column() {
        let x = Matrix::new(vec![vec![7.0], vec![7.0]]).unwrap();
        let mut s = MinMaxScaler::new();
        let out = s.fit_transform(&x).unwrap();
        let recovered = s.inverse_transform(&out).unwrap();
        for i in 0..2 {
            assert!((recovered.get(i, 0) - 7.0).abs() < 1e-9);
        }
    }

    #[test]
    fn inverse_transform_before_fit_errors() {
        let s = MinMaxScaler::new();
        assert!(s.inverse_transform(&m1()).is_err());
    }

    #[test]
    fn inverse_transform_shape_mismatch() {
        let mut s = MinMaxScaler::new();
        s.fit(&m1()).unwrap();
        let bad = Matrix::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();
        assert!(s.inverse_transform(&bad).is_err());
    }

    #[test]
    fn default_accessors_and_generated_feature_names() {
        let mut s = MinMaxScaler::default().feature_range(-3.0, 7.0);
        assert_eq!(s.feature_range_value(), (-3.0, 7.0));
        s.fit(&m1()).unwrap();
        assert_eq!(s.feature_names_out(None), vec!["x0", "x1"]);
    }

    #[test]
    fn transform_rejects_nan_with_its_position() {
        let mut s = MinMaxScaler::new();
        s.fit(&m1()).unwrap();
        let x = Matrix::new(vec![vec![0.0, f64::NAN]]).unwrap();
        let err = s.transform(&x).unwrap_err();
        assert!(matches!(
            err,
            DatarustError::InvalidInput(message) if message.contains("(0, 1)")
        ));
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn parallel_transform_handles_variable_constant_and_nan_columns() {
        let fit = Matrix::new(vec![vec![0.0, 5.0], vec![10.0, 5.0]]).unwrap();
        let mut s = MinMaxScaler::new();
        s.fit(&fit).unwrap();

        let mut values = Vec::with_capacity(4096 * 2);
        for i in 0..4096 {
            values.extend_from_slice(&[(i % 11) as f64, 5.0]);
        }
        let x = Matrix::from_flat(4096, 2, values.clone()).unwrap();
        let out = s.transform(&x).unwrap();
        assert_eq!((out.nrows(), out.ncols()), (4096, 2));
        assert_eq!(out.get(0, 1), 0.0);
        assert!((out.get(10, 0) - 1.0).abs() < 1e-12);

        values[17 * 2] = f64::NAN;
        let with_nan = Matrix::from_flat(4096, 2, values).unwrap();
        let err = s.transform(&with_nan).unwrap_err();
        assert!(matches!(
            err,
            DatarustError::InvalidInput(message) if message.contains("(17, 0)")
        ));
    }
}
