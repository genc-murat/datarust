//! Enum wrapper for target (supervised) categorical transformers, enabling
//! serialization of [`ColumnTransformer`] under the `serde` feature.
//!
//! [`ColumnTransformer`]: crate::compose::ColumnTransformer

use crate::encoder::TargetEncoder;
use crate::error::Result;
use crate::matrix::{Matrix, StrMatrix};
use crate::traits::{FeatureNames, TargetTransformer};

/// Concrete target transformer variant, wrapping a [`TargetEncoder`] for use
/// in [`ColumnTransformer`].
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
