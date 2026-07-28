//! Enum wrapper for target (supervised) categorical transformers, enabling
//! serialization of [`ColumnTransformer`] under the `serde` feature.
//!
//! [`ColumnTransformer`]: crate::compose::ColumnTransformer

use crate::encoder::TargetEncoder;
use crate::error::Result;
use crate::matrix::{Matrix, StrMatrix};
use crate::traits::{FeatureNames, TargetTransformer};

/// Concrete target transformer variant, wrapping a [`TargetEncoder`] for use
/// in [`crate::ColumnTransformer`].
///
/// Each variant wraps a single supervised encoder that implements
/// [`TargetTransformer`] (i.e. operates on `(StrMatrix, &[f64]) -> Matrix`).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TargetTransformerKind {
    /// Wraps a `TargetEncoder`.
    TargetEncoder(TargetEncoder),
}

impl TargetTransformer for TargetTransformerKind {
    fn name(&self) -> &'static str {
        match self {
            Self::TargetEncoder(t) => t.name(),
        }
    }

    fn fit(&mut self, x: &StrMatrix, y: &[f64]) -> Result<()> {
        match self {
            Self::TargetEncoder(t) => t.fit(x, y),
        }
    }

    fn transform(&self, x: &StrMatrix) -> Result<Matrix> {
        match self {
            Self::TargetEncoder(t) => t.transform(x),
        }
    }

    fn is_fitted(&self) -> bool {
        match self {
            Self::TargetEncoder(t) => t.is_fitted(),
        }
    }

    fn inverse_transform(&self, y: &Matrix) -> Result<StrMatrix> {
        match self {
            Self::TargetEncoder(t) => t.inverse_transform(y),
        }
    }
}

impl FeatureNames for TargetTransformerKind {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        match self {
            Self::TargetEncoder(t) => t.feature_names_out(input_features),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_str() -> StrMatrix {
        StrMatrix::from_column(["A", "A", "B", "B"]).unwrap()
    }

    #[test]
    fn name_delegates_to_inner() {
        let kind = TargetTransformerKind::TargetEncoder(TargetEncoder::new(0.0).unwrap());
        assert_eq!(kind.name(), TargetEncoder::new(0.0).unwrap().name());
    }

    #[test]
    fn fit_transform_dispatches_to_target_encoder() {
        let s = sample_str();
        // Targets: A→0.0, B→1.0. With smoothing=0 the encoding is the category mean.
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let mut kind = TargetTransformerKind::TargetEncoder(TargetEncoder::new(0.0).unwrap());
        assert!(!kind.is_fitted());
        let out = kind.fit_transform(&s, &y).unwrap();
        assert_eq!(out.nrows(), 4);
        assert_eq!(out.ncols(), 1);
        // A maps to mean 0.0, B to mean 1.0.
        assert!((out.get(0, 0) - 0.0).abs() < 1e-9);
        assert!((out.get(2, 0) - 1.0).abs() < 1e-9);
        assert!(kind.is_fitted());
    }

    #[test]
    fn transform_before_fit_errors() {
        let s = sample_str();
        let kind = TargetTransformerKind::TargetEncoder(TargetEncoder::new(0.0).unwrap());
        let err = kind.transform(&s).unwrap_err();
        assert!(matches!(err, crate::error::DatarustError::NotFitted(_)));
    }

    #[test]
    fn smoothing_pulls_toward_global_mean() {
        // With heavy smoothing, every category encoding approaches the global mean.
        let s = sample_str();
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let mut kind = TargetTransformerKind::TargetEncoder(TargetEncoder::new(1000.0).unwrap());
        let out = kind.fit_transform(&s, &y).unwrap();
        // Global mean = 0.5; with smoothing=1000 both categories are pulled close to it.
        let global = 0.5;
        for i in 0..s.nrows() {
            assert!(
                (out.get(i, 0) - global).abs() < 1e-3,
                "row {} got {} expected ~{}",
                i,
                out.get(i, 0),
                global
            );
        }
    }

    #[test]
    fn feature_names_delegates_to_inner() {
        // The wrapper delegates to the inner encoder; mappings are populated
        // only after fit, so we fit first.
        let s = sample_str();
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let mut kind = TargetTransformerKind::TargetEncoder(TargetEncoder::new(0.0).unwrap());
        kind.fit(&s, &y).unwrap();
        let names = kind.feature_names_out(Some(&["cat".into()]));
        assert_eq!(names, vec!["cat".to_string()]);
    }
}
