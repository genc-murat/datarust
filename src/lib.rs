//! # datarust
//!
//! Scikit-Learn Preprocessing in Rust. A modular, dependency-free
//! data-preprocessing library built on a lightweight `Matrix` type
//! backed by `Vec<Vec<f64>>`.
//!
//! ## Modules
//!
//! - [`scaler`] — StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer,
//!   Binarizer, KBinsDiscretizer, QuantileTransformer, PowerTransformer
//! - [`encoder`] — LabelEncoder, OneHotEncoder, OrdinalEncoder, TargetEncoder, FrequencyEncoder
//! - [`imputer`] — SimpleImputer (mean / median / most_frequent / constant) and KnnImputer
//! - [`polynomial`] — PolynomialFeatures
//! - [`selection`] — VarianceThreshold, SelectKBest
//! - [`decomposition`] — PCA, TruncatedSVD
//! - [`pipeline`] — sequential Transformer pipelines
//! - [`compose`] — ColumnTransformer
//! - [`function_transformer`] — wrap arbitrary functions as a Transformer
//! - [`stats`] — column statistics, covariance and correlation matrices
//! - [`matrix`] — `Matrix`, `StrMatrix` and `SparseMatrix` data containers
//! - [`serialize`] — JSON save/load (requires the `serde` feature)
//! - [`transformer_kind`] — type-erased `TransformerKind` enum wrapper
//! - [`categorical_kind`] — type-erased `CategoricalTransformerKind` enum wrapper for encoders
//! - [`target_kind`] — type-erased `TargetTransformerKind` enum wrapper for supervised encoders
//!
//! All numeric transformers implement the [`Transformer`] trait.
//! Categorical encoders (OneHot, Ordinal, Frequency) implement the
//! [`CategoricalTransformer`] trait.
//! The [`TargetEncoder`] implements the [`TargetTransformer`] trait (requires
//! target values during `fit`).
//! The [`LabelEncoder`] implements the [`LabelTransformer`] trait (1-D
//! string ↔ int mapping).
//!
//! [`TargetEncoder`]: encoder::TargetEncoder
//! [`LabelEncoder`]: encoder::LabelEncoder
//!
//! # Features
//!
//! - `serde` — enables JSON serialization via [`serialize`].
//! - `rayon` — enables parallel column operations for large datasets.
//!
//! The default build has **zero external dependencies**.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unreachable_patterns)]

pub mod categorical_kind;
pub mod compose;
pub mod decomposition;
pub mod encoder;
/// Error types returned by fallible operations.
pub mod error;
/// Wrap arbitrary functions as a [`Transformer`].
pub mod function_transformer;
pub mod imputer;
pub mod matrix;
/// Sequential transformer pipelines.
pub mod pipeline;
/// Generate polynomial feature combinations.
pub mod polynomial;
pub mod scaler;
pub mod selection;
#[cfg(feature = "serde")]
pub mod serialize;
pub mod stats;
pub mod target_kind;
pub mod traits;
pub mod transformer_kind;

pub use categorical_kind::CategoricalTransformerKind;
pub use compose::{ColumnSpec, ColumnTransformer, Output, Remainder, Table};
pub use encoder::{
    DropStrategy, FrequencyEncoder, HandleUnknown, LabelEncoder, OneHotEncoder, OrdinalCategories,
    OrdinalEncoder, OrdinalHandleUnknown, TargetEncoder, UnknownFrequency, UnknownTarget,
};
pub use error::{DatarustError, Result};
pub use matrix::{Matrix, SparseMatrix, StrMatrix};
pub use pipeline::Pipeline;
pub use target_kind::TargetTransformerKind;
pub use traits::{
    default_input_names, CategoricalTransformer, FeatureNames, LabelTransformer, TargetTransformer,
    Transformer,
};
pub use transformer_kind::TransformerKind;
