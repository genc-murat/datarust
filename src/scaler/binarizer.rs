use crate::error::Result;
use crate::matrix::Matrix;
use crate::traits::{default_input_names, FeatureNames};
use crate::Transformer;

/// Threshold features to binary (0.0 / 1.0) values, mirroring
/// `sklearn.preprocessing.Binarizer`.
///
/// Values strictly greater than `threshold` become 1.0; all others become 0.0.
/// The transformer is stateless (no statistics learned during fit).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Binarizer {
    threshold: f64,
    n_features: usize,
    fitted: bool,
}

impl Binarizer {
    /// Creates a new binarizer with a default threshold of 0.0.
    pub fn new() -> Self {
        Self {
            threshold: 0.0,
            n_features: 0,
            fitted: false,
        }
    }

    /// Set the decision threshold (default 0.0).
    pub fn threshold(mut self, t: f64) -> Self {
        self.threshold = t;
        self
    }

    /// Returns the configured decision threshold.
    pub fn threshold_value(&self) -> f64 {
        self.threshold
    }
}

impl Default for Binarizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for Binarizer {
    fn name(&self) -> &'static str {
        "Binarizer"
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        self.n_features = x.ncols();
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, x: &Matrix) -> Result<Matrix> {
        let out: Vec<Vec<f64>> = x
            .rows_ref()
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&v| if v > self.threshold { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        Matrix::new(out)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl FeatureNames for Binarizer {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        match input_features {
            Some(fs) => fs.to_vec(),
            None => default_input_names(self.n_features),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_threshold_zero() {
        let x = Matrix::new(vec![vec![-1.0, 0.0, 1.0, 2.0], vec![0.5, -0.5, 3.0, -3.0]]).unwrap();
        let mut b = Binarizer::new();
        let out = b.fit_transform(&x).unwrap();
        assert_eq!(out.row(0), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(out.row(1), [1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn custom_threshold() {
        let x = Matrix::new(vec![vec![1.0, 2.0, 3.0, 4.0]]).unwrap();
        let mut b = Binarizer::new().threshold(2.5);
        let out = b.fit_transform(&x).unwrap();
        assert_eq!(out.row(0), [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn threshold_boundary_is_zero() {
        let x = Matrix::new(vec![vec![5.0, 5.0]]).unwrap();
        let mut b = Binarizer::new().threshold(5.0);
        let out = b.fit_transform(&x).unwrap();
        // 5.0 > 5.0 is false -> 0
        assert_eq!(out.row(0), [0.0, 0.0]);
    }

    #[test]
    fn is_fitted_after_fit() {
        let x = Matrix::new(vec![vec![1.0]]).unwrap();
        let mut b = Binarizer::new();
        assert!(!b.is_fitted());
        b.fit(&x).unwrap();
        assert!(b.is_fitted());
    }

    #[test]
    fn feature_names_preserved() {
        let b = Binarizer::new();
        let names = b.feature_names_out(Some(&["a".into(), "b".into()]));
        assert_eq!(names, vec!["a", "b"]);
    }
}
