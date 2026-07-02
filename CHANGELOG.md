# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-03

### Added
- `CategoricalTransformer` trait (one-hot, ordinal, frequency encoders).
- `TargetTransformer` trait (target encoder).
- `LabelTransformer` trait for 1-D label encoding.
- `CategoricalTransformerKind`, `TargetTransformerKind`, `LabelTransformerKind` enum wrappers.
- `ColumnSpec::Categorical` and `ColumnSpec::Target` variants in `ColumnTransformer`.
- `Output { numeric, categorical }` struct with row-count validation + serde.
- `ColumnTransformer::add_categorical()`, `add_target()`, `fit_with_target()`, `fit_transform_with_target()`, `transform_to_table()`, `fit_transform_to_table()`.
- `FeatureNames` trait implementation for all categorical encoders.
- Rayon parallelism in `OneHotEncoder`, `OrdinalEncoder`, `FrequencyEncoder`, `TargetEncoder` (gated on `rayon` feature).
- `inverse_transform` on `CategoricalTransformer`/`TargetTransformer` with default error fallback.
- `OneHotEncoder::encode_one()` helper extracted; rayon path now calls it (eliminating inline duplicate).
- `FrequencyEncoder::UnknownFrequency` enum for out-of-category handling.
- `LabelHandleUnknown` `Eq` derive.
- `OrdinalCategories` with `#[default] Auto`.
- `OneHotOutput` with `Debug + Clone + PartialEq` derive.
- `Default` impls for `OrdinalEncoder`, `FrequencyEncoder`, `QuantileTransformer`, `KBinsDiscretizer`, `PCA`, `TruncatedSVD`, `SelectKBest`, `FunctionTransformer`.
- `Matrix::validate_no_nan()` — single-pass NaN check returning `Result`.
- `Matrix::checked_get(i, j) -> Option<f64>`, `StrMatrix::checked_get -> Option<&str>`, `SparseMatrix::checked_get -> Option<f64>` (panic-safe bounds-checked access).
- `debug_assert!` bounds checks on existing `get` methods.
- `KnnImputer::fit()` validates `n_neighbors >= 1`.
- `[package.metadata.docs.rs]` in `Cargo.toml`.
- GitHub Actions `doc` job (`cargo doc --all-features --no-deps`).
- Benchmarks for `OneHotEncoder`, `PowerTransformer`, `ColumnTransformer`.
- `DatarustError::source()` returns the original `std::io::Error` / `serde_json::Error`.
- `examples/basic_preprocessing.rs`, `examples/pipeline_workflow.rs`.
- Lib.rs re-exports: `ColumnTransformer`, `Pipeline`, `TransformerKind`, `ColumnSpec`, `Remainder`, `Table`, `Output`, `UnknownFrequency`.

### Changed
- `DatarustError::Io(String)` → `Io(std::io::Error)`, `DatarustError::Serde(String)` → `Serde(serde_json::Error)` (preserves original error).
- `DatarustError` no longer derives `PartialEq` (inner error types are not `PartialEq`).
- `Matrix`/`StrMatrix` serde `Deserialize` now validates through `::new()` constructor (rejects malformed JSON with error instead of panic).
- `QuantileTransformer::transform_value` returns `Result<f64>` and checks for NaN.
- `PowerTransformer::transform` validates Box-Cox positivity on new data (not just fit-time).
- `OrdinalEncoder::inverse_transform` rejects NaN indices.
- `KBinsDiscretizer::value_to_bin` returns `Result<usize>` and rejects NaN.
- `stats::quantile_column` returns `Result<Vec<f64>>` and validates `q ∈ [0,1]`.
- `Matrix::from_flat` uses `checked_mul` for overflow-safe size calculation.
- `SparseMatrix::density` uses `saturating_mul` for overflow safety.
- All scalers (`StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer`, `PowerTransformer`) now reject NaN input during `transform`.
- `ColumnTransformer::fit()` errors when `Target` specs present (use `fit_with_target()`); `fit_inner()` extracted.
- `ColumnTransformer::feature_names_out(None)` uses stored `max_col_index` (no longer recomputes inline).
- All encoders pad short `input_features` with `"x{i}"` instead of truncating.

### Fixed
- `QuantileTransformer` no longer panics on NaN input.
- `PowerTransformer` Box-Cox no longer silently accepts non-positive values on new data.
- `Matrix`/`StrMatrix` serde deserialization no longer panics on bad JSON.
- `OrdinalEncoder` sentinel (`-1`) decodes to empty string instead of out-of-range panic.
- `LabelEncoder` sentinel (`usize::MAX`) decodes to empty string.
- `OneHotEncoder` rayon path had ~30 lines of duplicated code (now calls `encode_one()`).
- `transform_to_table` duplicated `total_cat_cols` computation (now computed once).
- `KnnImputer` no longer accepts `n_neighbors == 0` silently.
- `PCAComponents::Variance(0.95)` now works as intended for PCA default.
- 6 broken doc links fixed (private items, out-of-scope references, redundant targets).

## [0.1.0] - 2026-06-??
