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
/// [`crate::ColumnTransformer`] to apply any categorical encoder to a column block.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_str() -> StrMatrix {
        StrMatrix::from_column(["Red", "Blue", "Green", "Red"]).unwrap()
    }

    #[test]
    fn name_delegates_to_inner() {
        let kind = CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new());
        assert_eq!(kind.name(), OneHotEncoder::new().name());
    }

    #[test]
    fn fit_transform_dispatches_to_onehot() {
        let s = sample_str();
        let mut kind = CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new());
        assert!(!kind.is_fitted());
        let out = kind.fit_transform(&s).unwrap();
        // 3 sorted categories: Blue, Green, Red -> 3 columns.
        assert_eq!(out.ncols(), 3);
        assert_eq!(out.nrows(), 4);
        assert!(kind.is_fitted());
        // Red -> [0,0,1]
        assert_eq!(out.row(0), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn fit_transform_dispatches_to_ordinal() {
        use crate::encoder::OrdinalCategories;
        let s = sample_str();
        let mut kind = CategoricalTransformerKind::OrdinalEncoder(OrdinalEncoder::new(
            OrdinalCategories::Auto,
        ));
        let out = kind.fit_transform(&s).unwrap();
        assert_eq!(out.nrows(), 4);
        assert_eq!(out.ncols(), 1);
        // Sorted cats: Blue=0, Green=1, Red=2
        assert_eq!(out.row(0), [2.0]); // Red
        assert_eq!(out.row(1), [0.0]); // Blue
    }

    #[test]
    fn fit_transform_dispatches_to_frequency() {
        let s = sample_str();
        let mut kind = CategoricalTransformerKind::FrequencyEncoder(FrequencyEncoder::new(false));
        let out = kind.fit_transform(&s).unwrap();
        // Red appears twice -> count 2; Blue and Green once.
        assert_eq!(out.row(0), [2.0]); // Red
        assert_eq!(out.row(1), [1.0]); // Blue
        assert_eq!(out.row(2), [1.0]); // Green
    }

    #[test]
    fn transform_before_fit_errors() {
        let s = sample_str();
        let kind = CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new());
        let err = kind.transform(&s).unwrap_err();
        assert!(matches!(err, crate::error::DatarustError::NotFitted(_)));
    }

    #[test]
    fn inverse_transform_round_trips_onehot() {
        let s = sample_str();
        let mut kind = CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new());
        let encoded = kind.fit_transform(&s).unwrap();
        let decoded = kind.inverse_transform(&encoded).unwrap();
        // Each decoded row should reproduce the original category string.
        for i in 0..s.nrows() {
            assert_eq!(decoded.get(i, 0), s.get(i, 0));
        }
    }

    #[test]
    fn feature_names_delegates_to_inner() {
        // The wrapper delegates to the inner encoder; feature names depend on
        // the fitted state (category_lists is populated only after fit).
        let s = sample_str();
        let mut kind = CategoricalTransformerKind::OrdinalEncoder(OrdinalEncoder::new(
            crate::encoder::OrdinalCategories::Auto,
        ));
        kind.fit(&s).unwrap();
        let names = kind.feature_names_out(Some(&["color".into()]));
        assert_eq!(names, vec!["color".to_string()]);
    }
}
