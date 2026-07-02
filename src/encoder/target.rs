use std::collections::HashMap;

use crate::error::{DatarustError, Result};
use crate::matrix::{Matrix, StrMatrix};
use crate::traits::{default_input_names, FeatureNames, TargetTransformer};

/// Strategy for categories unseen during `fit` when transforming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UnknownTarget {
    /// Use the global target mean for unknown categories (default, sklearn-like).
    #[default]
    GlobalMean,
    /// Return NaN for unknown categories.
    NaN,
    /// Raise an error on unknown categories.
    Error,
}

/// Replace each category with the (smoothed) mean of the target variable,
/// mirroring `sklearn.preprocessing.TargetEncoder`.
///
/// Smoothed encoding for category `c`:
/// ```text
///   (n_c * mean_c + smoothing * global_mean) / (n_c + smoothing)
/// ```
/// Operates on a 2-D [`StrMatrix`] and a 1-D target vector.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TargetEncoder {
    smoothing: f64,
    unknown: UnknownTarget,
    /// Per-column mapping: category -> encoded value.
    mappings: Vec<HashMap<String, f64>>,
    global_means: Vec<f64>,
    fitted: bool,
}

impl TargetEncoder {
    /// Creates a new target encoder with the given smoothing.
    pub fn new(smoothing: f64) -> Result<Self> {
        if smoothing < 0.0 {
            return Err(DatarustError::InvalidConfig(format!(
                "smoothing must be >= 0, got {}",
                smoothing
            )));
        }
        Ok(Self {
            smoothing,
            unknown: UnknownTarget::default(),
            mappings: vec![],
            global_means: vec![],
            fitted: false,
        })
    }

    /// Sets how unknown categories are handled during transform.
    pub fn unknown(mut self, u: UnknownTarget) -> Self {
        self.unknown = u;
        self
    }

    /// Returns the smoothing factor.
    pub fn smoothing(&self) -> f64 {
        self.smoothing
    }

    /// Learns the smoothed per-category target means.
    pub fn fit(&mut self, x: &StrMatrix, y: &[f64]) -> Result<()> {
        if y.len() != x.nrows() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} targets", x.nrows()),
                actual: format!("{} targets", y.len()),
            });
        }
        let ncols = x.ncols();
        let global_mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
        let mut mappings = Vec::with_capacity(ncols);
        let mut global_means = Vec::with_capacity(ncols);
        for j in 0..ncols {
            let col = x.column(j);
            let mut sums: HashMap<String, (f64, f64)> = HashMap::new();
            for (cat, &target) in col.iter().zip(y.iter()) {
                let e = sums.entry(cat.clone()).or_insert((0.0, 0.0));
                e.0 += target;
                e.1 += 1.0;
            }
            let mut map: HashMap<String, f64> = HashMap::new();
            for (cat, (sum, count)) in sums {
                let mean_c = sum / count;
                let smoothed =
                    (count * mean_c + self.smoothing * global_mean) / (count + self.smoothing);
                map.insert(cat, smoothed);
            }
            mappings.push(map);
            global_means.push(global_mean);
        }
        self.mappings = mappings;
        self.global_means = global_means;
        self.fitted = true;
        Ok(())
    }

    /// Replaces each category with its encoded target mean.
    pub fn transform(&self, x: &StrMatrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("TargetEncoder".into()));
        }
        if x.ncols() != self.mappings.len() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} categorical columns", self.mappings.len()),
                actual: format!("{} columns", x.ncols()),
            });
        }
        let mut out = vec![vec![0.0; x.ncols()]; x.nrows()];

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let mappings = &self.mappings;
            let global_means = &self.global_means;
            let unknown = self.unknown;
            let x_data = &x.data;
            out.par_iter_mut()
                .enumerate()
                .try_for_each(|(i, out_row)| {
                    for (j, cell) in out_row.iter_mut().enumerate() {
                        let val = &x_data[i][j];
                        *cell = match mappings[j].get(val) {
                            Some(&v) => v,
                            None => match unknown {
                                UnknownTarget::GlobalMean => global_means[j],
                                UnknownTarget::NaN => f64::NAN,
                                UnknownTarget::Error => {
                                    return Err(DatarustError::UnknownCategory(format!(
                                        "column {} value '{}'",
                                        j, val
                                    )))
                                }
                            },
                        };
                    }
                    Ok(())
                })?;
        }

        #[cfg(not(feature = "rayon"))]
        {
            for (i, out_row) in out.iter_mut().enumerate() {
                for (j, cell) in out_row.iter_mut().enumerate() {
                    let val = x.get(i, j);
                    *cell = match self.mappings[j].get(val) {
                        Some(&v) => v,
                        None => match self.unknown {
                            UnknownTarget::GlobalMean => self.global_means[j],
                            UnknownTarget::NaN => f64::NAN,
                            UnknownTarget::Error => {
                                return Err(DatarustError::UnknownCategory(format!(
                                    "column {} value '{}'",
                                    j, val
                                )))
                            }
                        },
                    };
                }
            }
        }

        Matrix::new(out)
    }

    /// Fits the encoder and transforms the input in one step.
    pub fn fit_transform(&mut self, x: &StrMatrix, y: &[f64]) -> Result<Matrix> {
        self.fit(x, y)?;
        self.transform(x)
    }
}

impl TargetTransformer for TargetEncoder {
    fn name(&self) -> &'static str {
        "TargetEncoder"
    }

    fn fit(&mut self, x: &StrMatrix, y: &[f64]) -> Result<()> {
        self.fit(x, y)
    }

    fn transform(&self, x: &StrMatrix) -> Result<Matrix> {
        self.transform(x)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl FeatureNames for TargetEncoder {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        let n = self.mappings.len();
        match input_features {
            Some(fs) => (0..n)
                .map(|i| fs.get(i).cloned().unwrap_or_else(|| format!("x{}", i)))
                .collect(),
            None => default_input_names(n),
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
    fn basic_no_smoothing() {
        // city -> target mean
        let x = StrMatrix::from_column(["Istanbul", "Ankara", "Izmir", "Istanbul"]).unwrap();
        let y = vec![1.0, 0.0, 1.0, 1.0];
        let mut te = TargetEncoder::new(0.0).unwrap();
        let out = te.fit_transform(&x, &y).unwrap();
        // Istanbul mean = (1+1)/2 = 1 ; Ankara = 0 ; Izmir = 1
        assert!(approx(out.get(0, 0), 1.0, 1e-12));
        assert!(approx(out.get(1, 0), 0.0, 1e-12));
        assert!(approx(out.get(2, 0), 1.0, 1e-12));
        assert!(approx(out.get(3, 0), 1.0, 1e-12));
    }

    #[test]
    fn smoothing_pulls_toward_global() {
        let x = StrMatrix::from_column(["a", "a", "b"]).unwrap();
        let y = vec![1.0, 1.0, 0.0];
        // global mean = 2/3
        let mut te = TargetEncoder::new(1.0).unwrap();
        te.fit(&x, &y).unwrap();
        // 'a': count=2, mean=1 -> (2*1 + 1*(2/3))/(2+1) = (2 + 0.666)/3 = 0.888
        let val_a = te.mappings[0].get("a").copied().unwrap();
        assert!(approx(val_a, 8.0 / 9.0, 1e-9));
        // 'b': count=1, mean=0 -> (0 + 2/3)/2 = 1/3
        let val_b = te.mappings[0].get("b").copied().unwrap();
        assert!(approx(val_b, 1.0 / 3.0, 1e-9));
    }

    #[test]
    fn unknown_uses_global_mean_by_default() {
        let x = StrMatrix::from_column(["a", "b"]).unwrap();
        let y = vec![1.0, 0.0];
        let mut te = TargetEncoder::new(0.0).unwrap();
        te.fit(&x, &y).unwrap();
        let x2 = StrMatrix::from_column(["a", "z"]).unwrap();
        let out = te.transform(&x2).unwrap();
        assert!(approx(out.get(0, 0), 1.0, 1e-12));
        // 'z' unknown -> global mean 0.5
        assert!(approx(out.get(1, 0), 0.5, 1e-12));
    }

    #[test]
    fn unknown_error_mode() {
        let x = StrMatrix::from_column(["a", "b"]).unwrap();
        let y = vec![1.0, 0.0];
        let mut te = TargetEncoder::new(0.0)
            .unwrap()
            .unknown(UnknownTarget::Error);
        te.fit(&x, &y).unwrap();
        let x2 = StrMatrix::from_column(["z"]).unwrap();
        assert!(te.transform(&x2).is_err());
    }

    #[test]
    fn unknown_nan_mode() {
        let x = StrMatrix::from_column(["a", "b"]).unwrap();
        let y = vec![1.0, 0.0];
        let mut te = TargetEncoder::new(0.0).unwrap().unknown(UnknownTarget::NaN);
        te.fit(&x, &y).unwrap();
        let x2 = StrMatrix::from_column(["a", "z"]).unwrap();
        let out = te.transform(&x2).unwrap();
        assert!(out.get(1, 0).is_nan());
    }

    #[test]
    fn multi_column() {
        let x =
            StrMatrix::from_strings(vec![vec!["a", "x"], vec!["a", "y"], vec!["b", "x"]]).unwrap();
        let y = vec![1.0, 0.0, 1.0];
        let mut te = TargetEncoder::new(0.0).unwrap();
        let out = te.fit_transform(&x, &y).unwrap();
        assert_eq!(out.ncols(), 2);
        // col0 'a' mean = 0.5, 'b' = 1.0 ; col1 'x' mean = 1.0, 'y' = 0.0
        assert!(approx(out.get(0, 0), 0.5, 1e-12));
        assert!(approx(out.get(2, 0), 1.0, 1e-12));
        assert!(approx(out.get(0, 1), 1.0, 1e-12));
        assert!(approx(out.get(1, 1), 0.0, 1e-12));
    }

    #[test]
    fn negative_smoothing_rejected() {
        assert!(TargetEncoder::new(-1.0).is_err());
    }

    #[test]
    fn target_count_mismatch() {
        let x = StrMatrix::from_column(["a", "b"]).unwrap();
        let mut te = TargetEncoder::new(0.0).unwrap();
        assert!(te.fit(&x, &[1.0]).is_err());
    }

    #[test]
    fn transform_before_fit_errors() {
        let te = TargetEncoder::new(0.0).unwrap();
        let x = StrMatrix::from_column(["a"]).unwrap();
        assert!(matches!(te.transform(&x), Err(DatarustError::NotFitted(_))));
    }

    #[test]
    fn transform_large_multi_column() {
        let x = StrMatrix::from_strings(vec![
            vec!["a", "x"],
            vec!["b", "y"],
            vec!["a", "x"],
            vec!["b", "y"],
            vec!["a", "x"],
            vec!["b", "y"],
        ])
        .unwrap();
        let y = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let mut te = TargetEncoder::new(0.0).unwrap();
        let out = te.fit_transform(&x, &y).unwrap();
        assert_eq!(out.nrows(), 6);
        assert_eq!(out.ncols(), 2);
        assert!(approx(out.get(0, 0), 1.0, 1e-12));
        assert!(approx(out.get(1, 0), 0.0, 1e-12));
    }

    #[test]
    fn feature_names_short_input_pads_with_synthetic() {
        let x = StrMatrix::from_strings(vec![vec!["a", "x"], vec!["b", "y"]]).unwrap();
        let y = vec![1.0, 0.0];
        let mut te = TargetEncoder::new(0.0).unwrap();
        te.fit(&x, &y).unwrap();
        // 2 columns but only 1 name provided
        let names = te.feature_names_out(Some(&["city".into()]));
        assert_eq!(names, vec!["city", "x1"]);
    }
}
