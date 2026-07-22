# Architecture

## Overview

**datarust** is a scikit-learn-inspired data preprocessing library for Rust. It is organized around a small set of core traits, a lightweight matrix type, and a flat module hierarchy. The design prioritizes **zero external dependencies by default**, **type safety**, and **composability**.

```
src/
├── lib.rs                  # Crate root: re-exports, module declarations
├── error.rs                # DatarustError enum + Result alias
├── traits.rs               # Estimator, Predictor, Regressor, Classifier,
│                           #   Clusterer, Transformer, FeatureNames,
│                           #   categorical traits
├── matrix.rs               # Matrix (f64), StrMatrix (String), SparseMatrix (CSR)
├── stats.rs                # Column-wise statistics (mean, var, quantile, covariance…)
├── pipeline.rs             # Sequential and supervised pipelines
├── transformer_kind.rs     # Type-erased enum over all Transformer impls
├── categorical_kind.rs     # Type-erased enum over CategoricalTransformer impls
├── target_kind.rs          # Type-erased enum over TargetTransformer impls
├── function_transformer.rs # Closure-based Transformer wrapper
├── polynomial.rs           # PolynomialFeatures
├── serialize.rs            # JSON save/load (serde feature gate)
├── datasets/               # Embedded toy datasets (datasets feature gate)
│   ├── mod.rs              #   Dataset struct + loaders
│   ├── iris.rs             #   Iris (150×4, 3 classes)
│   ├── breast_cancer.rs    #   Breast Cancer (569×30, binary)
│   ├── wine.rs             #   Wine (178×13, 3 classes)
│   └── diabetes.rs         #   Diabetes (442×10, regression)
│
├── scaler/                 # Numeric scalers (Standard, MinMax, Robust, …)
│   ├── mod.rs
│   ├── standard.rs
│   ├── minmax.rs
│   ├── robust.rs
│   ├── maxabs.rs
│   ├── normalizer.rs
│   ├── binarizer.rs
│   ├── kbins.rs
│   ├── quantile.rs
│   └── power.rs
│
├── encoder/                # Categorical encoders (OneHot, Ordinal, …)
│   ├── mod.rs
│   ├── onehot.rs
│   ├── ordinal.rs
│   ├── frequency.rs
│   ├── label.rs
│   └── target.rs
│
├── imputer/                # Missing value imputation
│   ├── mod.rs
│   ├── simple.rs
│   └── knn.rs
│
├── selection/              # Feature selection
│   ├── mod.rs
│   ├── variance_threshold.rs
│   └── select_k_best.rs
│
├── decomposition/          # Dimensionality reduction
│   ├── mod.rs
│   ├── pca.rs
│   ├── truncated_svd.rs
│   └── jacobi.rs           # Jacobi eigenvalue decomposition (internal)
│
├── cluster/                # Clustering estimators + metrics
│   ├── mod.rs
│   ├── kmeans.rs           # KMeans (Lloyd's algorithm, k-means++)
│   └── metrics.rs          # silhouette_score
│
└── compose/                # Composition utilities
    ├── mod.rs
    ├── column_transformer.rs  # ColumnTransformer + Table
    └── output.rs              # Output { numeric, categorical }
```

## Core Traits

The library defines transformer and supervised-estimator traits in `traits.rs`,
each targeting a different data modality:

### `Predictor`, `Regressor`, `Classifier` (supervised)
```
fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()>
fn predict(&self, x: &Matrix) -> Result<Vec<f64>>
fn is_fitted(&self) -> bool
```

Every supervised model implements `Predictor`. Regression models additionally
implement `Regressor`; classifiers implement `Classifier` and may implement
`PredictProba` for `(n_samples, n_classes)` probability matrices. A
`SupervisedPipeline<E>` fits transformer steps and a final `E: Predictor`
together, passing targets to supervised selectors before fitting the estimator.

### `Clusterer` (unsupervised clustering)
```
fn fit(&mut self, x: &Matrix) -> Result<()>
fn predict(&self, x: &Matrix) -> Result<Vec<usize>>      // cluster indices
fn fit_predict(&mut self, x: &Matrix) -> Result<Vec<usize>>  // default: fit + predict
fn fit_transform(&mut self, x: &Matrix) -> Result<Matrix>    // default: one-hot labels
fn n_clusters(&self) -> usize
fn is_fitted(&self) -> bool
```

The unsupervised counterpart to `Predictor`. `fit` takes only `X` (no target
`y`), and `predict` returns cluster indices as `Vec<usize>` rather than the
regression targets / class labels returned by supervised predictors. The
default `fit_transform` emits a one-hot encoding of the cluster assignments.
Implemented by: KMeans.

### `Transformer` (numeric → numeric)
```
fn fit(&mut self, x: &Matrix) -> Result<()>
fn transform(&self, x: &Matrix) -> Result<Matrix>
fn fit_transform(&mut self, x: &Matrix) -> Result<Matrix>  // default: fit + transform
fn inverse_transform(&self, x: &Matrix) -> Result<Matrix>  // default: Err
fn is_fitted(&self) -> bool
```

Implemented by: all scalers, PCA, TruncatedSVD, PolynomialFeatures, VarianceThreshold, SelectKBest, SimpleImputer, KnnImputer, FunctionTransformer, Binarizer, and Pipeline.

### `CategoricalTransformer` (categorical → numeric)
```
fn fit(&mut self, x: &StrMatrix) -> Result<()>
fn transform(&self, x: &StrMatrix) -> Result<Matrix>
fn fit_transform(&mut self, x: &StrMatrix) -> Result<Matrix>  // default
fn inverse_transform(&self, y: &Matrix) -> Result<StrMatrix>  // default: Err
fn is_fitted(&self) -> bool
```

Implemented by: OneHotEncoder, OrdinalEncoder, FrequencyEncoder.

### `TargetTransformer` (categorical + targets → numeric)
```
fn fit(&mut self, x: &StrMatrix, y: &[f64]) -> Result<()>
fn transform(&self, x: &StrMatrix) -> Result<Matrix>
fn fit_transform(&mut self, x: &StrMatrix, y: &[f64]) -> Result<Matrix>  // default
fn inverse_transform(&self, y: &Matrix) -> Result<StrMatrix>  // default: Err
fn is_fitted(&self) -> bool
```

Implemented by: TargetEncoder.

### `LabelTransformer` (1-D labels → indices)
```
fn fit(&mut self, x: &[String]) -> Result<()>
fn transform(&self, x: &[String]) -> Result<Vec<usize>>
fn fit_transform(&mut self, x: &[String]) -> Result<Vec<usize>>  // default
fn inverse_transform(&self, x: &[usize]) -> Result<Vec<String>>
fn is_fitted(&self) -> bool
```

Implemented by: LabelEncoder.

### `FeatureNames` (output column names)
```
fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String>
```

Implemented by every transformer that produces named output columns.

### `Params` (hyperparameter introspection)
```
fn get_params(&self) -> Vec<(&'static str, ParamValue)>
fn set_params(&mut self, name: &str, value: ParamValue) -> Result<()>
```

An opt-in trait for estimators whose hyperparameters should be searchable (the
foundation for future `GridSearchCV`). `ParamValue` is a typed enum (`Float`,
`Int`, `Bool`). Not every estimator needs `Params` — only those with tunable
hyperparameters.

Implemented by: KMeans, LogisticRegression.

## Type Erasure

Heterogeneous pipelines require type erasure. There are three erasure enums:

- **`TransformerKind`** — wraps any `Transformer` impl; used in `Pipeline` and `ColumnTransformer::add_numeric`.
- **`CategoricalTransformerKind`** — wraps any `CategoricalTransformer` impl; used in `ColumnTransformer::add_categorical`.
- **`TargetTransformerKind`** — wraps any `TargetTransformer` impl; used in `ColumnTransformer::add_target`.

Each erasure enum delegates all trait methods to the inner wrapper via a `match`, implements `serde::Serialize`/`Deserialize` (under the `serde` feature), and supports `Debug`/`Clone`.

## Matrix Types

### `Matrix`
Row-major `Vec<Vec<f64>>`. Construction via `Matrix::new()` validates:
- Non-empty (≥1 row, ≥1 column)
- All rows have the same length

Operations: `get`, `set`, `row`, `col`, `transpose`, `matmul`, `select_columns`, `select_rows`, `rows_ref`, `validate_no_nan`.

### `StrMatrix`
Row-major `Vec<Vec<String>>`. Same validation and operations as `Matrix` (minus arithmetic).

### `SparseMatrix`
CSR (Compressed Sparse Row) format: `indptr: Vec<usize>`, `indices: Vec<usize>`, `data: Vec<f64>`. Used primarily by `OneHotEncoder::transform_sparse` for memory-efficient one-hot encoding of high-cardinality features.

## Design Decisions

### Zero dependencies by default
All linear algebra (eigendecomposition via Jacobi, SVD via one-sided Jacobi, covariance) is implemented in pure Rust — no BLAS/LAPACK required. Optional features (`serde`, `rayon`) pull in dependencies only when needed.

### sklearn-inspired but Rust-native
The API mirrors scikit-learn conventions (`fit`, `transform`, `fit_transform`, `inverse_transform`, `feature_names_out`) but uses Rust idioms:
- `Result` for all fallible operations (no panicking except for programming errors caught by `debug_assert!`)
- Builder-pattern configuration (`with_mean(true)`, `.threshold(0.5)`)
- Enum-based configuration instead of string parameters (`Norm::L2`, `ImputeStrategy::Median`)
- Trait-based dispatch rather than inheritance

### Separation of numeric, categorical, and target traits
Four traits (Transformer, CategoricalTransformer, TargetTransformer, LabelTransformer) keep type signatures precise. This prevents calling `fit` without target values on a TargetEncoder, and makes the `ColumnTransformer`'s three spec types (Numeric, Categorical, Target) a natural fit.

### Panic-free error handling
Every fallible public API returns `Result<T, DatarustError>`. The error enum covers:
- **NotFitted** — transform before fit
- **InvalidInput** — NaN values, bad configuration
- **ShapeMismatch** — wrong number of columns
- **EmptyInput** — zero-row or zero-column data
- **UnknownCategory/Label** — unseen categories at transform time
- **InvalidConfig** — illegal parameter combination
- **Singular** — numerical breakdown (division by zero, singular matrix)
- **Io** — filesystem errors
- **Serde** — serialization errors (serde feature only)

## Parallelism (rayon feature)

When `--features rayon` is enabled, column-wise statistics and per-column transformations are parallelized via `rayon::par_iter()`. The affected operations include:
- All column statistics in `stats.rs`
- Column-wise scaling loops in scalers
- Transform and inverse_transform in encoders
- Distance computation in KNN imputer

## Serialization (serde feature)

When `--features serde` is enabled, all transformers derive `Serialize`/`Deserialize`. `Pipeline` and `ColumnTransformer` are serializable via their `TransformerKind`/`CategoricalTransformerKind`/`TargetTransformerKind` enum wrappers. Function pointers in `FunctionTransformer` are skipped during serialization and must be restored after deserialization via `set_func()`.

## Error Handling

The canonical type alias is `datarust::error::Result<T>` = `Result<T, DatarustError>`.

`DatarustError` implements `std::error::Error` with `source()` returning the original error for `Io` and `Serde` variants. All variants are `Display`-formatted with descriptive messages.

```rust
pub enum DatarustError {
    NotFitted(String),
    InvalidInput(String),
    ShapeMismatch { expected: String, actual: String },
    EmptyInput(String),
    AllMissing(String),
    UnknownCategory(String),
    UnknownLabel(String),
    InvalidConfig(String),
    Singular(String),
    Io(std::io::Error),
    Serde(serde_json::Error),  // serde feature only
}
```
