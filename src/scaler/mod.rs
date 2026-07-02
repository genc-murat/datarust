//! Feature scaling and discretization transformers.

/// Threshold-based binarization transformer.
pub mod binarizer;
/// Continuous-to-discrete bin discretization.
pub mod kbins;
/// Maximum-absolute-value scaler.
pub mod maxabs;
/// Range-based feature scaler.
pub mod minmax;
/// Row-wise normalization transformer.
pub mod normalizer;
/// Power-transform Gaussianizer.
pub mod power;
/// Quantile-based feature transformer.
pub mod quantile;
/// Outlier-robust median/IQR scaler.
pub mod robust;
/// Mean/variance standardizing scaler.
pub mod standard;

pub use binarizer::Binarizer;
pub use kbins::{BinStrategy, KBinsDiscretizer, KBinsEncode};
pub use maxabs::MaxAbsScaler;
pub use minmax::MinMaxScaler;
pub use normalizer::{Norm, Normalizer};
pub use power::{PowerMethod, PowerTransformer};
pub use quantile::{OutputDistribution, QuantileTransformer};
pub use robust::RobustScaler;
pub use standard::StandardScaler;
