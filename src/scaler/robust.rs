use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::stats;
use crate::traits::{default_input_names, FeatureNames};
use crate::Transformer;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Scale features using statistics that are robust to outliers, mirroring
/// `sklearn.preprocessing.RobustScaler`.
///
/// Centers to the median and scales according to the interquartile range
/// (default Q3 - Q1) using numpy-default linear quantile interpolation.
/// The quantile range can be customized via [`quantile_range`](RobustScaler::quantile_range).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RobustScaler {
    with_centering: bool,
    with_scaling: bool,
    quantile_range: (f64, f64),
    center: Vec<f64>,
    scale: Vec<f64>,
    fitted: bool,
}

impl RobustScaler {
    /// Creates a new scaler with default settings.
    pub fn new() -> Self {
        Self {
            with_centering: true,
            with_scaling: true,
            quantile_range: (0.25, 0.75),
            center: vec![],
            scale: vec![],
            fitted: false,
        }
    }

    /// Builder: center by median (default true).
    pub fn with_centering(mut self, b: bool) -> Self {
        self.with_centering = b;
        self
    }

    /// Builder: scale by IQR (default true).
    pub fn with_scaling(mut self, b: bool) -> Self {
        self.with_scaling = b;
        self
    }

    /// Builder: set the quantile range used to compute the IQR scale.
    /// Must satisfy `0 < lo < hi < 1`.  Default is `(0.25, 0.75)`.
    pub fn quantile_range(mut self, lo: f64, hi: f64) -> Self {
        self.quantile_range = (lo, hi);
        self
    }

    /// Fitted per-column centers (medians).
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    /// Fitted per-column scales (IQR).
    pub fn scale(&self) -> &[f64] {
        &self.scale
    }

    /// Returns the configured quantile range.
    pub fn quantile_range_value(&self) -> (f64, f64) {
        self.quantile_range
    }
}

impl Default for RobustScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureNames for RobustScaler {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        match input_features {
            Some(fs) => fs.to_vec(),
            None => default_input_names(self.center.len()),
        }
    }
}

impl Transformer for RobustScaler {
    fn name(&self) -> &'static str {
        "RobustScaler"
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        let (q_lo, q_hi) = self.quantile_range;
        if q_lo <= 0.0 || q_hi >= 1.0 || q_lo >= q_hi {
            return Err(DatarustError::InvalidConfig(format!(
                "quantile_range ({}, {}) must satisfy 0 < lo < hi < 1",
                q_lo, q_hi
            )));
        }
        let data = x.rows_ref();
        let q1 = stats::quantile_column(data, q_lo)?;
        let q3 = stats::quantile_column(data, q_hi)?;
        let median = stats::median_column(data);
        let scale: Vec<f64> = (0..x.ncols())
            .map(|j| {
                if self.with_scaling {
                    q3[j] - q1[j]
                } else {
                    1.0
                }
            })
            .collect();
        let center: Vec<f64> = (0..x.ncols())
            .map(|j| if self.with_centering { median[j] } else { 0.0 })
            .collect();
        self.center = center;
        self.scale = scale;
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("RobustScaler".into()));
        }
        if self.center.len() != x.ncols() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.center.len()),
                actual: format!("{} features", x.ncols()),
            });
        }
        x.validate_no_nan()?;
        #[cfg(feature = "rayon")]
        {
            let center = &self.center;
            let scale = &self.scale;
            let rows: Vec<Vec<f64>> = x
                .rows_ref()
                .par_iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(j, &v)| {
                            let s = scale[j];
                            if s == 0.0 {
                                (v - center[j]) * 0.0
                            } else {
                                (v - center[j]) / s
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
                    let s = self.scale[j];
                    if s == 0.0 {
                        out[i][j] = (v - self.center[j]) * 0.0;
                    } else {
                        out[i][j] = (v - self.center[j]) / s;
                    }
                }
            }
            Matrix::new(out)
        }
    }

    fn inverse_transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("RobustScaler".into()));
        }
        if self.center.len() != x.ncols() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.center.len()),
                actual: format!("{} features", x.ncols()),
            });
        }
        #[cfg(feature = "rayon")]
        {
            let center = &self.center;
            let scale = &self.scale;
            let rows: Vec<Vec<f64>> = x
                .rows_ref()
                .par_iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(j, &z)| z * scale[j] + center[j])
                        .collect()
                })
                .collect();
            Matrix::new(rows)
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut out = vec![vec![0.0; x.ncols()]; x.nrows()];
            for (i, row) in x.rows_ref().iter().enumerate() {
                for (j, &z) in row.iter().enumerate() {
                    out[i][j] = z * self.scale[j] + self.center[j];
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
        // col0: 0..10 ; col1: with an outlier
        Matrix::new(vec![
            vec![0.0, 1.0],
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![4.0, 5.0],
            vec![5.0, 6.0],
            vec![6.0, 7.0],
            vec![7.0, 8.0],
            vec![8.0, 9.0],
            vec![9.0, 10.0],
        ])
        .unwrap()
    }

    #[test]
    fn median_and_iqr_computed() {
        let mut s = RobustScaler::new();
        s.fit(&m1()).unwrap();
        // median of 0..9 is 4.5
        assert!((s.center()[0] - 4.5).abs() < 1e-12);
        // Q1 = 2.25, Q3 = 6.75 -> IQR = 4.5
        assert!((s.scale()[0] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn fit_transform_value() {
        let mut s = RobustScaler::new();
        let out = s.fit_transform(&m1()).unwrap();
        // value 9 -> (9 - 4.5)/4.5 = 1
        assert!((out.get(9, 0) - 1.0).abs() < 1e-9);
        // value 0 -> -1
        assert!((out.get(0, 0) - (-1.0)).abs() < 1e-9);
        // median is 4.5; row 5 has value 5 -> (5 - 4.5)/4.5 = 0.1111
        assert!((out.get(5, 0) - (0.5 / 4.5)).abs() < 1e-9);
        // average of symmetric rows 4 and 5 centers to 0 (mean of -0.111 and +0.111)
        let midpair = (out.get(4, 0) + out.get(5, 0)) / 2.0;
        assert!(midpair.abs() < 1e-9);
    }

    #[test]
    fn with_centering_false() {
        let mut s = RobustScaler::new().with_centering(false);
        let out = s.fit_transform(&m1()).unwrap();
        // no centering: 9 / 4.5 = 2
        assert!((out.get(9, 0) - 2.0).abs() < 1e-9);
        assert!((out.get(0, 0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn with_scaling_false() {
        let mut s = RobustScaler::new().with_scaling(false);
        let out = s.fit_transform(&m1()).unwrap();
        // only centered: 9 - 4.5 = 4.5
        assert!((out.get(9, 0) - 4.5).abs() < 1e-9);
    }

    #[test]
    fn robust_to_outlier() {
        let with_outlier = Matrix::new(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![1000.0],
        ])
        .unwrap();
        // RobustScaler median of value 3 should stay small; std-based scaler would explode.
        let mut s = RobustScaler::new();
        s.fit(&with_outlier).unwrap();
        assert!((s.center()[0] - 3.5).abs() < 1e-9);
        // IQR should be robust
        assert!(s.scale()[0] < 10.0);
    }

    #[test]
    fn zero_iqr_constant_column() {
        let x = Matrix::new(vec![vec![2.0], vec![2.0], vec![2.0]]).unwrap();
        let mut s = RobustScaler::new();
        let out = s.fit_transform(&x).unwrap();
        assert!((s.scale()[0] - 0.0).abs() < 1e-12);
        for i in 0..3 {
            assert!((out.get(i, 0) - 0.0).abs() < 1e-12);
        }
    }

    #[test]
    fn transform_before_fit_errors() {
        let s = RobustScaler::new();
        assert!(matches!(
            s.transform(&m1()),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn shape_mismatch() {
        let mut s = RobustScaler::new();
        s.fit(&m1()).unwrap();
        let bad = Matrix::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();
        assert!(s.transform(&bad).is_err());
    }

    #[test]
    fn quantile_linear_consistency() {
        // For evenly spaced 0..9, q1 should be 2.25 (numpy linear) not 2.0
        let x = Matrix::new((0..10).map(|i| vec![i as f64, 0.0]).collect()).unwrap();
        let mut s = RobustScaler::new();
        s.fit(&x).unwrap();
        assert!((s.center()[0] - 4.5).abs() < 1e-12);
        assert!((s.scale()[0] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn inverse_transform_round_trip() {
        let mut s = RobustScaler::new();
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
    fn inverse_transform_with_centering_false() {
        let mut s = RobustScaler::new().with_centering(false);
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
    fn inverse_transform_with_scaling_false() {
        let mut s = RobustScaler::new().with_scaling(false);
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
    fn inverse_transform_zero_iqr() {
        let x = Matrix::new(vec![vec![2.0], vec![2.0], vec![2.0]]).unwrap();
        let mut s = RobustScaler::new();
        let out = s.fit_transform(&x).unwrap();
        let recovered = s.inverse_transform(&out).unwrap();
        // scale=0 -> all transformed values 0.0; inverse gives center (median=2.0)
        for i in 0..3 {
            assert!((recovered.get(i, 0) - 2.0).abs() < 1e-9);
        }
    }

    #[test]
    fn inverse_transform_before_fit_errors() {
        let s = RobustScaler::new();
        assert!(s.inverse_transform(&m1()).is_err());
    }

    #[test]
    fn inverse_transform_shape_mismatch() {
        let mut s = RobustScaler::new();
        s.fit(&m1()).unwrap();
        let bad = Matrix::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();
        assert!(s.inverse_transform(&bad).is_err());
    }

    #[test]
    fn custom_quantile_range_wider_scale() {
        // With a wider quantile range, the IQR is larger -> scale is larger
        // -> transformed values are smaller in magnitude
        let mut s_default = RobustScaler::new();
        let mut s_wide = RobustScaler::new().quantile_range(0.1, 0.9);
        let x = m1();
        s_default.fit(&x).unwrap();
        s_wide.fit(&x).unwrap();
        // Wide range should have larger scale (IQR) -> abs values are smaller
        assert!(s_wide.scale()[0] > s_default.scale()[0]);
    }

    #[test]
    fn custom_quantile_range_narrower_scale() {
        let mut s = RobustScaler::new().quantile_range(0.4, 0.6);
        let x = m1();
        s.fit(&x).unwrap();
        // With q=(0.4,0.6), IQR is smaller -> scale is smaller -> values larger
        assert!(s.scale()[0] < 4.5); // default scale is 4.5
    }

    #[test]
    fn invalid_quantile_range_rejected() {
        let mut s = RobustScaler::new().quantile_range(0.5, 0.5);
        let x = m1();
        assert!(s.fit(&x).is_err());
        let mut s2 = RobustScaler::new().quantile_range(0.0, 0.75);
        assert!(s2.fit(&x).is_err());
        let mut s3 = RobustScaler::new().quantile_range(0.25, 1.0);
        assert!(s3.fit(&x).is_err());
    }
}
