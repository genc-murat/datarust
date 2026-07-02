use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::stats;
use crate::traits::{default_input_names, FeatureNames};
use crate::Transformer;

/// Feature selector that removes low-variance features, mirroring
/// `sklearn.feature_selection.VarianceThreshold`.
///
/// Keeps features whose variance exceeds `threshold`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VarianceThreshold {
    threshold: f64,
    variances: Vec<f64>,
    support_mask: Vec<bool>,
    fitted: bool,
}

impl VarianceThreshold {
    /// Creates a new selector that drops features with variance at or below `threshold`.
    pub fn new(threshold: f64) -> Result<Self> {
        if threshold < 0.0 {
            return Err(DatarustError::InvalidConfig(format!(
                "threshold must be >= 0, got {}",
                threshold
            )));
        }
        Ok(Self {
            threshold,
            variances: vec![],
            support_mask: vec![],
            fitted: false,
        })
    }

    /// Returns the per-feature variance computed during fit.
    pub fn variances(&self) -> &[f64] {
        &self.variances
    }

    /// Returns the boolean mask of kept features.
    pub fn get_support(&self) -> &[bool] {
        &self.support_mask
    }

    /// Returns the configured variance threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

impl Default for VarianceThreshold {
    fn default() -> Self {
        Self::new(0.0).expect("default threshold valid")
    }
}

impl FeatureNames for VarianceThreshold {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        let names: Vec<String> = match input_features {
            Some(fs) => fs.to_vec(),
            None => default_input_names(self.support_mask.len()),
        };
        self.support_mask
            .iter()
            .enumerate()
            .filter(|(_, &keep)| keep)
            .map(|(j, _)| names[j].clone())
            .collect()
    }
}

impl Transformer for VarianceThreshold {
    fn name(&self) -> &'static str {
        "VarianceThreshold"
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        // sklearn uses population variance (ddof=0)
        self.variances = stats::column_variance(x.rows_ref(), 0);
        self.support_mask = self.variances.iter().map(|&v| v > self.threshold).collect();
        if !self.support_mask.iter().any(|&b| b) {
            return Err(DatarustError::InvalidConfig(format!(
                "no feature exceeds threshold {}",
                self.threshold
            )));
        }
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("VarianceThreshold".into()));
        }
        if self.support_mask.len() != x.ncols() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.support_mask.len()),
                actual: format!("{} features", x.ncols()),
            });
        }
        let kept: Vec<usize> = self
            .support_mask
            .iter()
            .enumerate()
            .filter(|(_, &keep)| keep)
            .map(|(j, _)| j)
            .collect();
        let mut out = vec![vec![0.0; kept.len()]; x.nrows()];
        for (i, out_row) in out.iter_mut().enumerate() {
            let x_row = x.row(i);
            for (k, &j) in kept.iter().enumerate() {
                out_row[k] = x_row[j];
            }
        }
        Matrix::new(out)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m1() -> Matrix {
        // col0 constant (var 0), col1 varied, col2 slightly varied
        Matrix::new(vec![
            vec![5.0, 1.0, 10.0],
            vec![5.0, 2.0, 11.0],
            vec![5.0, 3.0, 10.0],
            vec![5.0, 4.0, 11.0],
        ])
        .unwrap()
    }

    #[test]
    fn default_removes_constant_column() {
        let mut vt = VarianceThreshold::default();
        let out = vt.fit_transform(&m1()).unwrap();
        // col0 (constant) dropped
        assert_eq!(vt.get_support(), &[false, true, true]);
        assert_eq!(out.ncols(), 2);
        // remaining cols keep their values
        assert_eq!(out.get(0, 0), 1.0);
        assert_eq!(out.get(0, 1), 10.0);
    }

    #[test]
    fn variances_computed() {
        let mut vt = VarianceThreshold::default();
        vt.fit(&m1()).unwrap();
        assert!((vt.variances()[0] - 0.0).abs() < 1e-12);
        // col1 values 1,2,3,4 -> variance = 1.25
        assert!((vt.variances()[1] - 1.25).abs() < 1e-12);
        // col2 values 10,11,10,11 -> variance = 0.25
        assert!((vt.variances()[2] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn threshold_filters_more() {
        // threshold = 0.5 keeps only col1 (var 1.25); col2 (0.25) dropped
        let mut vt = VarianceThreshold::new(0.5).unwrap();
        let out = vt.fit_transform(&m1()).unwrap();
        assert_eq!(vt.get_support(), &[false, true, false]);
        assert_eq!(out.ncols(), 1);
    }

    #[test]
    fn boundary_threshold_strict_greater() {
        // threshold exactly 0.25 -> col2 var == 0.25 is NOT > 0.25, dropped
        let mut vt = VarianceThreshold::new(0.25).unwrap();
        let out = vt.fit_transform(&m1()).unwrap();
        assert_eq!(vt.get_support(), &[false, true, false]);
        assert_eq!(out.ncols(), 1);
    }

    #[test]
    fn all_features_dropped_errors() {
        let x = Matrix::new(vec![vec![5.0, 9.0], vec![5.0, 9.0]]).unwrap(); // all constant
        let mut vt = VarianceThreshold::default();
        assert!(vt.fit(&x).is_err());
    }

    #[test]
    fn negative_threshold_rejected() {
        assert!(VarianceThreshold::new(-1.0).is_err());
    }

    #[test]
    fn transform_before_fit_errors() {
        let vt = VarianceThreshold::default();
        assert!(matches!(
            vt.transform(&m1()),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn transform_new_data_same_columns() {
        let mut vt = VarianceThreshold::default();
        vt.fit(&m1()).unwrap();
        let new = Matrix::new(vec![vec![99.0, 7.0, 88.0]]).unwrap();
        let out = vt.transform(&new).unwrap();
        assert_eq!(out.row(0), [7.0, 88.0]);
    }

    #[test]
    fn shape_mismatch() {
        let mut vt = VarianceThreshold::default();
        vt.fit(&m1()).unwrap();
        let bad = Matrix::new(vec![vec![1.0, 2.0]]).unwrap();
        assert!(vt.transform(&bad).is_err());
    }

    #[test]
    fn binary_features() {
        // Bernoulli with p: var = p(1-p)
        let x = Matrix::new(vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0]]).unwrap();
        let mut vt = VarianceThreshold::new(0.0).unwrap();
        vt.fit(&x).unwrap();
        // var = 0.25
        assert!((vt.variances()[0] - 0.25).abs() < 1e-12);
    }
}
