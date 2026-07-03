use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::stats;
use crate::traits::{default_input_names, FeatureNames};
use crate::Transformer;

/// Standardize features by removing the mean and scaling to unit variance.
///
/// Mirrors `sklearn.preprocessing.StandardScaler`. Uses population
/// standard deviation (ddof = 0) by default, matching sklearn.
///
/// ```rust,ignore
/// let mut s = StandardScaler::new();
/// let out = s.fit_transform(&x)?;
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StandardScaler {
    with_mean: bool,
    with_std: bool,
    mean: Vec<f64>,
    std: Vec<f64>,
    fitted: bool,
}

impl StandardScaler {
    /// Creates a new scaler with default settings.
    pub fn new() -> Self {
        Self {
            with_mean: true,
            with_std: true,
            mean: vec![],
            std: vec![],
            fitted: false,
        }
    }

    /// Builder: center the data by the mean (default true).
    pub fn with_mean(mut self, b: bool) -> Self {
        self.with_mean = b;
        self
    }

    /// Builder: scale to unit variance (default true).
    pub fn with_std(mut self, b: bool) -> Self {
        self.with_std = b;
        self
    }

    /// Fitted per-column means (empty if not fitted or `with_mean=false`).
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Fitted per-column standard deviations.
    pub fn std(&self) -> &[f64] {
        &self.std
    }

    fn compute(x: &Matrix, with_mean: bool, with_std: bool) -> (Vec<f64>, Vec<f64>) {
        let ncols = x.ncols();
        // Single fused Welford pass over flat storage (1 pass, stride-1 reads).
        let (mean, var) = stats::column_mean_var_flat(x.as_slice(), x.nrows(), x.ncols(), 0);
        let mean = if with_mean { mean } else { vec![0.0; ncols] };
        let std = if with_std {
            var.iter().map(|v| v.sqrt()).collect()
        } else {
            vec![1.0; ncols]
        };
        (mean, std)
    }

    fn scale(value: f64, mean: f64, std: f64) -> f64 {
        if std == 0.0 {
            // sklearn: when std is 0, the feature is set to [0.0 - mean]/1 then *0
            // effectively centers to 0 (or unchanged if with_mean=false).
            (value - mean) * 0.0
        } else {
            (value - mean) / std
        }
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureNames for StandardScaler {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        match input_features {
            Some(fs) => fs.to_vec(),
            None => default_input_names(self.mean.len()),
        }
    }
}

impl Transformer for StandardScaler {
    fn name(&self) -> &'static str {
        "StandardScaler"
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        let (mean, std) = Self::compute(x, self.with_mean, self.with_std);
        self.mean = mean;
        self.std = std;
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("StandardScaler".into()));
        }
        if self.mean.len() != x.ncols() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.mean.len()),
                actual: format!("{} features", x.ncols()),
            });
        }
        // Flat-storage transform: write directly into a contiguous output buffer
        // with stride-1 reads from the input. NaN check is fused into the loop.
        let nrows = x.nrows();
        let ncols = x.ncols();
        let mean = &self.mean;
        let std = &self.std;
        let src = x.as_slice();
        let mut out = vec![0.0; nrows * ncols];
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            // Paralel chunks: disjoint mutable row slices, stride-1 reads/writes.
            out.par_chunks_mut(ncols)
                .zip(src.par_chunks(ncols))
                .for_each(|(out_row, in_row)| {
                    for (j, &v) in in_row.iter().enumerate() {
                        out_row[j] = Self::scale(v, mean[j], std[j]);
                    }
                });
            // NaN check after transform: any NaN in input produces NaN in output
            // (arithmetic propagates it), so a single par_any pass suffices.
            if out.par_iter().any(|v| v.is_nan()) {
                // Locate the offending position for a helpful error message.
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
        }
        #[cfg(not(feature = "rayon"))]
        {
            for i in 0..nrows {
                let base = i * ncols;
                for j in 0..ncols {
                    let v = src[base + j];
                    if v.is_nan() {
                        return Err(DatarustError::InvalidInput(format!(
                            "NaN value at position ({i}, {j})"
                        )));
                    }
                    out[base + j] = Self::scale(v, mean[j], std[j]);
                }
            }
        }
        Matrix::from_flat(nrows, ncols, out)
    }

    fn inverse_transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("StandardScaler".into()));
        }
        if self.mean.len() != x.ncols() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.mean.len()),
                actual: format!("{} features", x.ncols()),
            });
        }
        let nrows = x.nrows();
        let ncols = x.ncols();
        let mean = &self.mean;
        let std = &self.std;
        let src = x.as_slice();
        let mut out = vec![0.0; nrows * ncols];
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            out.par_chunks_mut(ncols)
                .zip(src.par_chunks(ncols))
                .for_each(|(out_row, in_row)| {
                    for (j, &z) in in_row.iter().enumerate() {
                        out_row[j] = if std[j] == 0.0 {
                            mean[j]
                        } else {
                            z * std[j] + mean[j]
                        };
                    }
                });
        }
        #[cfg(not(feature = "rayon"))]
        {
            for i in 0..nrows {
                let base = i * ncols;
                for j in 0..ncols {
                    out[base + j] = if std[j] == 0.0 {
                        mean[j]
                    } else {
                        src[base + j] * std[j] + mean[j]
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
        // 2 features, 4 samples
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
        let mut s = StandardScaler::new();
        let out = s.fit_transform(&m1()).unwrap();
        // mean of col 0 = 0.5, std = 0.5 ; col 1 mean = 55, std = 45
        assert!((s.mean()[0] - 0.5).abs() < 1e-12);
        assert!((s.mean()[1] - 55.0).abs() < 1e-12);
        assert!((s.std()[0] - 0.5).abs() < 1e-12);
        assert!((s.std()[1] - 45.0).abs() < 1e-12);
        // First row col0: (0 - 0.5)/0.5 = -1
        assert!((out.get(0, 0) - (-1.0)).abs() < 1e-12);
        // col1 first row: (10 - 55)/45 = -1
        assert!((out.get(0, 1) - (-1.0)).abs() < 1e-12);
        // last row: +1, +1
        assert!((out.get(3, 0) - 1.0).abs() < 1e-12);
        assert!((out.get(3, 1) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn property_zero_mean_unit_std() {
        let mut s = StandardScaler::new();
        let out = s.fit_transform(&m1()).unwrap();
        let means = out.column_mean();
        let (_, vars) = stats::column_mean_var_flat(out.as_slice(), out.nrows(), out.ncols(), 0);
        let stds: Vec<f64> = vars.iter().map(|v| v.sqrt()).collect();
        for m in &means {
            assert!(m.abs() < 1e-9, "mean not zero: {}", m);
        }
        for sd in &stds {
            assert!((sd - 1.0).abs() < 1e-9, "std not one: {}", sd);
        }
    }

    #[test]
    fn with_mean_false() {
        let mut s = StandardScaler::new().with_mean(false);
        let out = s.fit_transform(&m1()).unwrap();
        // not centered: only scaled. col0 row0: 0/0.5 = 0
        assert!((out.get(0, 0) - 0.0).abs() < 1e-12);
        assert!((out.get(2, 0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn with_std_false() {
        let mut s = StandardScaler::new().with_std(false);
        let out = s.fit_transform(&m1()).unwrap();
        // centered only. col0 row0: 0 - 0.5 = -0.5
        assert!((out.get(0, 0) - (-0.5)).abs() < 1e-12);
        assert!((out.get(2, 0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn constant_column() {
        let x = Matrix::new(vec![vec![5.0], vec![5.0], vec![5.0]]).unwrap();
        let mut s = StandardScaler::new();
        let out = s.fit_transform(&x).unwrap();
        assert!((s.std()[0] - 0.0).abs() < 1e-12);
        // std=0 -> sklearn sets result to 0
        for i in 0..3 {
            assert!((out.get(i, 0) - 0.0).abs() < 1e-12);
        }
    }

    #[test]
    fn transform_before_fit_errors() {
        let s = StandardScaler::new();
        let err = s.transform(&m1()).unwrap_err();
        assert!(matches!(err, DatarustError::NotFitted(_)));
    }

    #[test]
    fn transform_new_data_uses_fitted_params() {
        let mut s = StandardScaler::new();
        s.fit(&m1()).unwrap();
        let new = Matrix::new(vec![vec![1.0, 100.0]]).unwrap();
        let out = s.transform(&new).unwrap();
        // same as last training row
        assert!((out.get(0, 0) - 1.0).abs() < 1e-12);
        assert!((out.get(0, 1) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn shape_mismatch_on_transform() {
        let mut s = StandardScaler::new();
        s.fit(&m1()).unwrap();
        let bad = Matrix::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();
        assert!(s.transform(&bad).is_err());
    }

    #[test]
    fn inverse_transform_round_trip() {
        let mut s = StandardScaler::new();
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
    fn inverse_transform_with_mean_false() {
        let mut s = StandardScaler::new().with_mean(false);
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
    fn inverse_transform_with_std_false() {
        let mut s = StandardScaler::new().with_std(false);
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
        let x = Matrix::new(vec![vec![5.0], vec![5.0], vec![5.0]]).unwrap();
        let mut s = StandardScaler::new();
        let out = s.fit_transform(&x).unwrap();
        let recovered = s.inverse_transform(&out).unwrap();
        for i in 0..3 {
            assert!((recovered.get(i, 0) - 5.0).abs() < 1e-9);
        }
    }

    #[test]
    fn inverse_transform_before_fit_errors() {
        let s = StandardScaler::new();
        let x = m1();
        assert!(s.inverse_transform(&x).is_err());
    }

    #[test]
    fn inverse_transform_shape_mismatch() {
        let mut s = StandardScaler::new();
        s.fit(&m1()).unwrap();
        let bad = Matrix::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();
        assert!(s.inverse_transform(&bad).is_err());
    }

    #[test]
    fn feature_names_passthrough() {
        let mut s = StandardScaler::new();
        s.fit(&m1()).unwrap();
        let names = s.feature_names_out(Some(&["age".to_string(), "sal".to_string()]));
        assert_eq!(names, vec!["age", "sal"]);
        let default_names = s.feature_names_out(None);
        assert_eq!(default_names, vec!["x0", "x1"]);
    }
}
