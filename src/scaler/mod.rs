//! Feature scaling and discretization transformers.

pub mod binarizer;
pub mod kbins;
pub mod maxabs;
pub mod minmax;
pub mod normalizer;
pub mod power;
pub mod quantile;
pub mod robust;
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
