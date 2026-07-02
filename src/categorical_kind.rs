//! Enum wrapper for categorical transformer types, enabling serialization of
//! [`ColumnTransformer`] under the `serde` feature.
//!
//! [`ColumnTransformer`]: crate::compose::ColumnTransformer

use crate::encoder::FrequencyEncoder;
use crate::encoder::OneHotEncoder;
use crate::encoder::OrdinalEncoder;
use crate::error::Result;
use crate::matrix::{Matrix, StrMatrix};
use crate::traits::{CategoricalTransformer, FeatureNames};

/// Concrete categorical transformer variant, used to allow
/// [`ColumnTransformer`] to apply any categorical encoder to a column block.
///
/// Each variant wraps a single encoder type that implements
/// [`CategoricalTransformer`] (i.e. operates on `StrMatrix -> Matrix`).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CategoricalTransformerKind {
    /// Wraps a `OneHotEncoder`.
    OneHotEncoder(OneHotEncoder),
    /// Wraps an `OrdinalEncoder`.
    OrdinalEncoder(OrdinalEncoder),
    /// Wraps a `FrequencyEncoder`.
    FrequencyEncoder(FrequencyEncoder),
}

impl CategoricalTransformer for CategoricalTransformerKind {
    fn name(&self) -> &'static str {
        match self {
            Self::OneHotEncoder(t) => t.name(),
            Self::OrdinalEncoder(t) => t.name(),
            Self::FrequencyEncoder(t) => t.name(),
        }
    }

    fn fit(&mut self, x: &StrMatrix) -> Result<()> {
        match self {
            Self::OneHotEncoder(t) => t.fit(x),
            Self::OrdinalEncoder(t) => t.fit(x),
            Self::FrequencyEncoder(t) => t.fit(x),
        }
    }

    fn transform(&self, x: &StrMatrix) -> Result<Matrix> {
        match self {
            Self::OneHotEncoder(t) => t.transform(x),
            Self::OrdinalEncoder(t) => t.transform(x),
            Self::FrequencyEncoder(t) => t.transform(x),
        }
    }

    fn is_fitted(&self) -> bool {
        match self {
            Self::OneHotEncoder(t) => t.is_fitted(),
            Self::OrdinalEncoder(t) => t.is_fitted(),
            Self::FrequencyEncoder(t) => t.is_fitted(),
        }
    }

    fn inverse_transform(&self, y: &Matrix) -> Result<StrMatrix> {
        match self {
            Self::OneHotEncoder(t) => t.inverse_transform(y),
            Self::OrdinalEncoder(t) => t.inverse_transform(y),
            Self::FrequencyEncoder(t) => t.inverse_transform(y),
        }
    }
}

impl FeatureNames for CategoricalTransformerKind {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        match self {
            Self::OneHotEncoder(t) => t.feature_names_out(input_features),
            Self::OrdinalEncoder(t) => t.feature_names_out(input_features),
            Self::FrequencyEncoder(t) => t.feature_names_out(input_features),
        }
    }
}
