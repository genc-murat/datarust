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
//! - [`linear_model`] — LinearRegression, Ridge, Lasso, LogisticRegression
//! - [`metrics`] — regression metrics (MSE, MAE, R², ...) and classification metrics (accuracy, F1, ...)
//! - [`model_selection`] — train_test_split, KFold, StratifiedKFold, cross_val_score
//! - [`pipeline`] — sequential Transformer pipelines
//! - [`compose`] — ColumnTransformer
//! - [`cluster`] — KMeans (Lloyd's algorithm, k-means++ initialization)
//! - [`function_transformer`] — wrap arbitrary functions as a Transformer
//! - [`stats`] — column and 1-D statistics, covariance and correlation matrices
//! - [`matrix`] — `Matrix`, `StrMatrix` and `SparseMatrix` data containers
//! - [`serialize`] — JSON save/load (requires the `serde` feature)
//! - [`transformer_kind`] — type-erased `TransformerKind` enum wrapper
//! - [`categorical_kind`] — type-erased `CategoricalTransformerKind` enum wrapper for encoders
//! - [`target_kind`] — type-erased `TargetTransformerKind` enum wrapper for supervised encoders
//!
//! All numeric transformers implement the [`Transformer`] trait. Supervised
//! estimators implement [`Predictor`] (`fit` with features + target, then
//! `predict`); regressors additionally implement [`Regressor`] and classifiers
//! implement [`Classifier`] / [`PredictProba`] where appropriate.
//! Categorical encoders (OneHot, Ordinal, Frequency) implement the
//! [`CategoricalTransformer`] trait.
//! The [`TargetEncoder`] implements the [`TargetTransformer`] trait (requires
//! target values during `fit`).
//! The [`LabelEncoder`] implements the [`LabelTransformer`] trait (1-D
//! string ↔ int mapping).
//! Clustering estimators (KMeans) implement the [`Clusterer`] trait (`fit` on
//! `X` only, then `predict` returning cluster indices).
//!
//! [`TargetEncoder`]: encoder::TargetEncoder
//! [`LabelEncoder`]: encoder::LabelEncoder
//!
//! # Features
//!
//! - `serde` — enables JSON serialization via [`serialize`].
//! - `rayon` — enables parallel column/row operations for large datasets.
//! - `matrixmultiply` — enables a tuned pure-Rust GEMM (no system BLAS) for
//!   `Matrix::matmul` and covariance computation, speeding up PCA and
//!   TruncatedSVD on large dense inputs.
//! - `datasets` — embeds classic toy datasets (Iris, Breast Cancer, Wine,
//!   Diabetes) as `const` arrays for examples, tests, and onboarding.
//!
//! The default build has **zero external dependencies**.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unreachable_patterns)]

pub mod categorical_kind;
pub mod cluster;
pub mod compose;
#[cfg(feature = "datasets")]
pub mod datasets;
pub mod decomposition;
pub mod encoder;
/// Error types returned by fallible operations.
pub mod error;
/// Wrap arbitrary functions as a [`Transformer`].
pub mod function_transformer;
pub mod imputer;
/// Shared linear-algebra primitives (Cholesky solver, etc.).
pub mod linalg;
/// Regression & classification estimators: LinearRegression, Ridge, Lasso, LogisticRegression.
pub mod linear_model;
pub mod matrix;
/// Model-evaluation metrics: regression (MSE, R², ...) and classification (accuracy, F1, ...).
pub mod metrics;
/// Model selection: train_test_split, KFold, cross_val_score.
pub mod model_selection;
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
pub use cluster::{KMeans, KMeansInit};
pub use compose::{ColumnSpec, ColumnTransformer, Output, Remainder, Table};
pub use encoder::{
    DropStrategy, FrequencyEncoder, HandleUnknown, LabelEncoder, OneHotEncoder, OrdinalCategories,
    OrdinalEncoder, OrdinalHandleUnknown, TargetEncoder, UnknownFrequency, UnknownTarget,
};
pub use error::{DatarustError, Result};
pub use linear_model::{
    Lasso, LinearRegression, LinearSolver, LogisticRegression, LogisticSolver, Ridge, RidgeSolver,
};
pub use matrix::{Matrix, SparseMatrix, StrMatrix};
pub use model_selection::{
    cross_val_score, train_test_split, KFold, StratifiedKFold, TrainTestSplit,
};
pub use pipeline::{Pipeline, SupervisedPipeline};
pub use target_kind::TargetTransformerKind;
pub use traits::{
    default_input_names, CategoricalTransformer, Classifier, Clusterer, Estimator, FeatureNames,
    LabelTransformer, ParamValue, Params, PredictProba, Predictor, Regressor, TargetTransformer,
    Transformer,
};
pub use transformer_kind::TransformerKind;
