//! # datarust
//!
//! Scikit-Learn Preprocessing in Rust. A modular, dependency-free
//! data-preprocessing library built on a lightweight `Matrix` type
//! backed by `Vec<Vec<f64>>`.
//!
//! ## Modules
//!
//! - [`scaler`] — StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
//! - [`encoder`] — LabelEncoder, OneHotEncoder, OrdinalEncoder, TargetEncoder, FrequencyEncoder
//! - [`imputer`] — SimpleImputer (mean / median / most_frequent / constant) and KnnImputer
//! - [`polynomial`] — PolynomialFeatures
//! - [`selection`] — VarianceThreshold, SelectKBest
//! - [`decomposition`] — PCA, TruncatedSVD
//! - [`pipeline`] — sequential Transformer pipelines
//! - [`compose`] — ColumnTransformer
//!
//! All numeric transformers implement the [`Transformer`] trait.

pub mod compose;
pub mod decomposition;
pub mod encoder;
pub mod error;
pub mod function_transformer;
pub mod imputer;
pub mod matrix;
pub mod pipeline;
pub mod polynomial;
pub mod scaler;
pub mod selection;
#[cfg(feature = "serde")]
pub mod serialize;
pub mod stats;
pub mod traits;
pub mod transformer_kind;

pub use error::{DatarustError, Result};
pub use matrix::{Matrix, SparseMatrix, StrMatrix};
pub use traits::{default_input_names, FeatureNames, Transformer};
