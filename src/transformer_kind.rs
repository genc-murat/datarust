//! Enum wrapper for all transformer types, enabling serialization of
//! [`Pipeline`] and [`ColumnTransformer`] under the `serde` feature.
//!
//! [`Pipeline`]: crate::pipeline::Pipeline
//! [`ColumnTransformer`]: crate::compose::ColumnTransformer

use crate::error::{DatarustError, Result};
use crate::function_transformer::FunctionTransformer;
use crate::imputer::{KnnImputer, SimpleImputer};
use crate::matrix::Matrix;
use crate::polynomial::PolynomialFeatures;
use crate::scaler::{
    Binarizer, KBinsDiscretizer, MaxAbsScaler, MinMaxScaler, Normalizer, PowerTransformer,
    QuantileTransformer, RobustScaler, StandardScaler,
};
use crate::selection::{SelectKBest, VarianceThreshold};
use crate::traits::FeatureNames;
use crate::Transformer;

/// Concrete transformer variant, used to serialize / deserialize pipeline
/// steps.  Each variant wraps a single transformer type that implements
/// [`Transformer`] (i.e. operates on `Matrix`).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TransformerKind {
    /// Wraps a `StandardScaler`.
    StandardScaler(StandardScaler),
    /// Wraps a `MinMaxScaler`.
    MinMaxScaler(MinMaxScaler),
    /// Wraps a `MaxAbsScaler`.
    MaxAbsScaler(MaxAbsScaler),
    /// Wraps a `RobustScaler`.
    RobustScaler(RobustScaler),
    /// Wraps a `Normalizer`.
    Normalizer(Normalizer),
    /// Wraps a `Binarizer`.
    Binarizer(Binarizer),
    /// Wraps a `KBinsDiscretizer`.
    KBinsDiscretizer(KBinsDiscretizer),
    /// Wraps a `QuantileTransformer`.
    QuantileTransformer(QuantileTransformer),
    /// Wraps a `PowerTransformer`.
    PowerTransformer(PowerTransformer),
    /// Wraps a `PolynomialFeatures` transformer.
    PolynomialFeatures(PolynomialFeatures),
    /// Wraps a `VarianceThreshold` selector.
    VarianceThreshold(VarianceThreshold),
    /// Wraps a `PCA` decomposition.
    PCA(crate::decomposition::PCA),
    /// Wraps a `TruncatedSVD` decomposition.
    TruncatedSVD(crate::decomposition::TruncatedSVD),
    /// Wraps a `SimpleImputer`.
    SimpleImputer(SimpleImputer),
    /// Wraps a `KnnImputer`.
    KnnImputer(KnnImputer),
    /// Wraps a `SelectKBest` selector.
    SelectKBest(SelectKBest),
    /// Wraps a `FunctionTransformer`.
    FunctionTransformer(FunctionTransformer),
}

impl TransformerKind {
    /// Return a human-readable tag for the variant.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::StandardScaler(_) => "StandardScaler",
            Self::MinMaxScaler(_) => "MinMaxScaler",
            Self::MaxAbsScaler(_) => "MaxAbsScaler",
            Self::RobustScaler(_) => "RobustScaler",
            Self::Normalizer(_) => "Normalizer",
            Self::Binarizer(_) => "Binarizer",
            Self::KBinsDiscretizer(_) => "KBinsDiscretizer",
            Self::QuantileTransformer(_) => "QuantileTransformer",
            Self::PowerTransformer(_) => "PowerTransformer",
            Self::PolynomialFeatures(_) => "PolynomialFeatures",
            Self::VarianceThreshold(_) => "VarianceThreshold",
            Self::PCA(_) => "PCA",
            Self::TruncatedSVD(_) => "TruncatedSVD",
            Self::SimpleImputer(_) => "SimpleImputer",
            Self::KnnImputer(_) => "KnnImputer",
            Self::SelectKBest(_) => "SelectKBest",
            Self::FunctionTransformer(_) => "FunctionTransformer",
        }
    }
}

impl Transformer for TransformerKind {
    fn name(&self) -> &'static str {
        match self {
            Self::StandardScaler(t) => t.name(),
            Self::MinMaxScaler(t) => t.name(),
            Self::MaxAbsScaler(t) => t.name(),
            Self::RobustScaler(t) => t.name(),
            Self::Normalizer(t) => t.name(),
            Self::Binarizer(t) => t.name(),
            Self::KBinsDiscretizer(t) => t.name(),
            Self::QuantileTransformer(t) => t.name(),
            Self::PowerTransformer(t) => t.name(),
            Self::PolynomialFeatures(t) => t.name(),
            Self::VarianceThreshold(t) => t.name(),
            Self::PCA(t) => t.name(),
            Self::TruncatedSVD(t) => t.name(),
            Self::SimpleImputer(t) => t.name(),
            Self::KnnImputer(t) => t.name(),
            Self::SelectKBest(t) => t.name(),
            Self::FunctionTransformer(t) => t.name(),
        }
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        match self {
            Self::StandardScaler(t) => t.fit(x),
            Self::MinMaxScaler(t) => t.fit(x),
            Self::MaxAbsScaler(t) => t.fit(x),
            Self::RobustScaler(t) => t.fit(x),
            Self::Normalizer(t) => t.fit(x),
            Self::Binarizer(t) => t.fit(x),
            Self::KBinsDiscretizer(t) => t.fit(x),
            Self::QuantileTransformer(t) => t.fit(x),
            Self::PowerTransformer(t) => t.fit(x),
            Self::PolynomialFeatures(t) => t.fit(x),
            Self::VarianceThreshold(t) => t.fit(x),
            Self::PCA(t) => t.fit(x),
            Self::TruncatedSVD(t) => t.fit(x),
            Self::SimpleImputer(t) => t.fit(x),
            Self::KnnImputer(t) => t.fit(x),
            Self::SelectKBest(_) => Err(DatarustError::InvalidInput(
                "SelectKBest requires labels; use fit_with_labels".into(),
            )),
            Self::FunctionTransformer(t) => t.fit(x),
        }
    }

    fn fit_with_target(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        match self {
            Self::SelectKBest(t) => t.fit_with_numeric_labels(x, y),
            _ => self.fit(x),
        }
    }

    fn transform(&self, x: &Matrix) -> Result<Matrix> {
        match self {
            Self::StandardScaler(t) => t.transform(x),
            Self::MinMaxScaler(t) => t.transform(x),
            Self::MaxAbsScaler(t) => t.transform(x),
            Self::RobustScaler(t) => t.transform(x),
            Self::Normalizer(t) => t.transform(x),
            Self::Binarizer(t) => t.transform(x),
            Self::KBinsDiscretizer(t) => t.transform(x),
            Self::QuantileTransformer(t) => t.transform(x),
            Self::PowerTransformer(t) => t.transform(x),
            Self::PolynomialFeatures(t) => t.transform(x),
            Self::VarianceThreshold(t) => t.transform(x),
            Self::PCA(t) => t.transform(x),
            Self::TruncatedSVD(t) => t.transform(x),
            Self::SimpleImputer(t) => t.transform(x),
            Self::KnnImputer(t) => t.transform(x),
            Self::SelectKBest(t) => t.transform(x),
            Self::FunctionTransformer(t) => t.transform(x),
        }
    }

    fn inverse_transform(&self, x: &Matrix) -> Result<Matrix> {
        match self {
            Self::StandardScaler(t) => t.inverse_transform(x),
            Self::MinMaxScaler(t) => t.inverse_transform(x),
            Self::MaxAbsScaler(t) => t.inverse_transform(x),
            Self::RobustScaler(t) => t.inverse_transform(x),
            Self::Normalizer(t) => t.inverse_transform(x),
            Self::Binarizer(t) => t.inverse_transform(x),
            Self::KBinsDiscretizer(t) => t.inverse_transform(x),
            Self::QuantileTransformer(t) => t.inverse_transform(x),
            Self::PowerTransformer(t) => t.inverse_transform(x),
            Self::PolynomialFeatures(t) => t.inverse_transform(x),
            Self::VarianceThreshold(t) => t.inverse_transform(x),
            Self::PCA(t) => t.inverse_transform(x),
            Self::TruncatedSVD(t) => t.inverse_transform(x),
            Self::SimpleImputer(t) => t.inverse_transform(x),
            Self::KnnImputer(t) => t.inverse_transform(x),
            Self::SelectKBest(t) => t.inverse_transform(x),
            Self::FunctionTransformer(t) => t.inverse_transform(x),
        }
    }

    fn is_fitted(&self) -> bool {
        match self {
            Self::StandardScaler(t) => t.is_fitted(),
            Self::MinMaxScaler(t) => t.is_fitted(),
            Self::MaxAbsScaler(t) => t.is_fitted(),
            Self::RobustScaler(t) => t.is_fitted(),
            Self::Normalizer(t) => t.is_fitted(),
            Self::Binarizer(t) => t.is_fitted(),
            Self::KBinsDiscretizer(t) => t.is_fitted(),
            Self::QuantileTransformer(t) => t.is_fitted(),
            Self::PowerTransformer(t) => t.is_fitted(),
            Self::PolynomialFeatures(t) => t.is_fitted(),
            Self::VarianceThreshold(t) => t.is_fitted(),
            Self::PCA(t) => t.is_fitted(),
            Self::TruncatedSVD(t) => t.is_fitted(),
            Self::SimpleImputer(t) => t.is_fitted(),
            Self::KnnImputer(t) => t.is_fitted(),
            Self::SelectKBest(t) => t.is_fitted(),
            Self::FunctionTransformer(t) => t.is_fitted(),
        }
    }
}

impl FeatureNames for TransformerKind {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        match self {
            Self::StandardScaler(t) => t.feature_names_out(input_features),
            Self::MinMaxScaler(t) => t.feature_names_out(input_features),
            Self::MaxAbsScaler(t) => t.feature_names_out(input_features),
            Self::RobustScaler(t) => t.feature_names_out(input_features),
            Self::Normalizer(t) => t.feature_names_out(input_features),
            Self::Binarizer(t) => t.feature_names_out(input_features),
            Self::KBinsDiscretizer(t) => t.feature_names_out(input_features),
            Self::QuantileTransformer(t) => t.feature_names_out(input_features),
            Self::PowerTransformer(t) => t.feature_names_out(input_features),
            Self::PolynomialFeatures(t) => t.feature_names_out(input_features),
            Self::VarianceThreshold(t) => t.feature_names_out(input_features),
            Self::PCA(t) => t.feature_names_out(input_features),
            Self::TruncatedSVD(t) => t.feature_names_out(input_features),
            Self::SimpleImputer(t) => t.feature_names_out(input_features),
            Self::KnnImputer(t) => t.feature_names_out(input_features),
            Self::SelectKBest(t) => t.feature_names_out(input_features),
            Self::FunctionTransformer(t) => t.feature_names_out(input_features),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scaler::{Binarizer, MinMaxScaler, StandardScaler};

    fn sample_matrix() -> Matrix {
        Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]).unwrap()
    }

    #[test]
    fn tag_returns_expected_strings() {
        assert_eq!(
            TransformerKind::StandardScaler(StandardScaler::new()).tag(),
            "StandardScaler"
        );
        assert_eq!(
            TransformerKind::MinMaxScaler(MinMaxScaler::new()).tag(),
            "MinMaxScaler"
        );
        assert_eq!(
            TransformerKind::Binarizer(Binarizer::new()).tag(),
            "Binarizer"
        );
    }

    #[test]
    fn name_delegates_to_inner() {
        let kind = TransformerKind::StandardScaler(StandardScaler::new());
        assert_eq!(kind.name(), StandardScaler::new().name());
    }

    #[test]
    fn dispatches_fit_and_transform() {
        let x = sample_matrix();
        // StandardScaler via the wrapper.
        let mut kind = TransformerKind::StandardScaler(StandardScaler::new());
        assert!(!kind.is_fitted());
        let out = kind.fit_transform(&x).unwrap();
        // Column 0 mean = 3, std = sqrt(((−2)²+0²+2²)/3); (1−3)/std should be negative.
        assert!(out.get(0, 0) < 0.0);
        assert!(kind.is_fitted());

        // Binarizer via the wrapper (threshold 0.0 default -> all positive values map to 1).
        let mut bin = TransformerKind::Binarizer(Binarizer::new());
        let bin_out = bin.fit_transform(&x).unwrap();
        assert_eq!(bin_out.row(0), [1.0, 1.0]);
        assert_eq!(bin_out.row(2), [1.0, 1.0]);
    }

    #[test]
    fn select_k_best_fit_errors_without_labels() {
        use crate::selection::{ScoreFunc, SelectKBest};
        let x = sample_matrix();
        let mut kind =
            TransformerKind::SelectKBest(SelectKBest::new(ScoreFunc::FClassif, 1).unwrap());
        let err = kind.fit(&x).unwrap_err();
        assert!(matches!(err, DatarustError::InvalidInput(_)));
    }

    #[test]
    fn is_fitted_reflects_inner_state() {
        let x = sample_matrix();
        let mut kind = TransformerKind::MinMaxScaler(MinMaxScaler::new());
        assert!(!kind.is_fitted());
        kind.fit(&x).unwrap();
        assert!(kind.is_fitted());
    }

    #[test]
    fn transform_before_fit_errors() {
        let x = sample_matrix();
        let kind = TransformerKind::StandardScaler(StandardScaler::new());
        let err = kind.transform(&x).unwrap_err();
        assert!(matches!(err, DatarustError::NotFitted(_)));
    }

    #[test]
    fn feature_names_delegates_to_inner() {
        let kind = TransformerKind::StandardScaler(StandardScaler::new());
        let names = kind.feature_names_out(Some(&["alpha".into(), "beta".into()]));
        assert_eq!(names, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[test]
    fn inverse_transform_round_trips_through_wrapper() {
        let x = sample_matrix();
        let mut kind = TransformerKind::MinMaxScaler(MinMaxScaler::new());
        let t = kind.fit_transform(&x).unwrap();
        let r = kind.inverse_transform(&t).unwrap();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!((r.get(i, j) - x.get(i, j)).abs() < 1e-9, "i={i} j={j}");
            }
        }
    }
}
