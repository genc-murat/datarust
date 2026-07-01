//! Enum wrapper for all transformer types, enabling serialization of
//! [`Pipeline`] and [`ColumnTransformer`] under the `serde` feature.
//!
//! [`Pipeline`]: crate::pipeline::Pipeline
//! [`ColumnTransformer`]: crate::compose::ColumnTransformer

use crate::error::{DatarustError, Result};
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
    StandardScaler(StandardScaler),
    MinMaxScaler(MinMaxScaler),
    MaxAbsScaler(MaxAbsScaler),
    RobustScaler(RobustScaler),
    Normalizer(Normalizer),
    Binarizer(Binarizer),
    KBinsDiscretizer(KBinsDiscretizer),
    QuantileTransformer(QuantileTransformer),
    PowerTransformer(PowerTransformer),
    PolynomialFeatures(PolynomialFeatures),
    VarianceThreshold(VarianceThreshold),
    PCA(crate::decomposition::PCA),
    TruncatedSVD(crate::decomposition::TruncatedSVD),
    SimpleImputer(SimpleImputer),
    KnnImputer(KnnImputer),
    SelectKBest(SelectKBest),
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
        }
    }
}
