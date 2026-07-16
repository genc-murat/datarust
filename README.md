# datarust

[![crates.io](https://img.shields.io/crates/v/datarust.svg)](https://crates.io/crates/datarust)
[![docs.rs](https://docs.rs/datarust/badge.svg)](https://docs.rs/datarust)
[![Documentation](https://img.shields.io/badge/docs-book-blue.svg)](https://genc-murat.github.io/datarust/)
[![CI](https://github.com/genc-murat/datarust/actions/workflows/ci.yml/badge.svg)](https://github.com/genc-murat/datarust/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Scikit-Learn Preprocessing in Rust** — a modular, dependency-free data preprocessing library built on a lightweight `Matrix` type.

📖 **[Read the documentation book →](https://genc-murat.github.io/datarust/)**

```rust,ignore
let mut scaler = StandardScaler::new();
let normalized = scaler.fit_transform(&data)?;
```

## Features

| Category | Transformers |
|---|---|
| **Scalers** | StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer (L1/L2/Max) |
| **Discretizers** | KBinsDiscretizer (Uniform / Quantile / KMeans), Binarizer |
| **Distribution Transformers** | QuantileTransformer (Uniform / Normal output), PowerTransformer (Yeo-Johnson / Box-Cox) |
| **Encoders** | LabelEncoder (+ handle_unknown), OneHotEncoder (+ CSR sparse output), OrdinalEncoder, TargetEncoder, FrequencyEncoder |
| **Imputers** | SimpleImputer (mean / median / most_frequent / constant), KnnImputer (uniform / distance) |
| **Polynomial** | PolynomialFeatures (degree, interaction_only, include_bias) |
| **Selection** | VarianceThreshold, SelectKBest (ANOVA F / Chi2 / Mutual Information) |
| **Decomposition** | PCA (with whiten, inverse_transform), TruncatedSVD (SVDComponents: Count/Variance/All) |
| **Linear Models** | LinearRegression (Cholesky & SVD), Ridge (L2), Lasso (L1, coordinate descent, sparse) |
| **Classification** | LogisticRegression (binary, IRLS solver, Cholesky & SVD) |
| **Metrics** | Regression: MSE/RMSE, MAE, R², max_error, explained_variance. Classification: accuracy, precision, recall, F1, confusion_matrix, log_loss |
| **Model Selection** | train_test_split, KFold, StratifiedKFold, cross_val_score |
| **Pipeline** | Sequential Pipeline (serde-serializable), ColumnTransformer (numeric + categorical) |
| **Feature Names** | `FeatureNames` trait on all transformers for output column names |
| **Serialization** | JSON save/load via optional `serde` feature |
| **Parallelism** | Rayon-backed column operations via optional `rayon` feature |
| **Sparse** | CSR `SparseMatrix` type for memory-efficient one-hot output |

**Default build has zero external dependencies.** All linear algebra (eigenvalue decomposition, covariance) is implemented in pure Rust using the Jacobi algorithm.

## Quick Start

```rust
use datarust::scaler::*;
use datarust::Matrix;

// Create a 4×2 matrix
let x = Matrix::new(vec![
    vec![1.0, 10.0],
    vec![2.0, 20.0],
    vec![3.0, 30.0],
    vec![4.0, 40.0],
])?;

// Standardize: (x - mean) / std (population, ddof=0)
let mut scaler = StandardScaler::new();
let standardized = scaler.fit_transform(&x)?;

// Scale to [0, 1]
let mut minmax = MinMaxScaler::new();
let scaled = minmax.fit_transform(&x)?;
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
datarust = "0.3"
```

### Optional features

```toml
[dependencies]
datarust = { version = "0.3", features = ["serde", "rayon"] }
```

- **`serde`** — enables JSON serialization/deserialization of fitted transformers via `datarust::serialize::{save_json, load_json, to_json, from_json}`.
- **`rayon`** — enables parallel column statistics and transforms for large datasets.
- **`matrixmultiply`** — enables a tuned pure-Rust GEMM (no system BLAS) for matrix products and covariance computation, speeding up PCA and TruncatedSVD on large dense inputs. The default build remains zero-external-dependency.

## Core Concepts

### Matrix

The fundamental data container is [`Matrix`](https://docs.rs/datarust/latest/datarust/struct.Matrix.html), a row-major dense matrix backed by a single contiguous `Vec<f64>` buffer with validation:

```rust
let m = Matrix::new(vec![
    vec![1.0, 2.0, 3.0],
    vec![4.0, 5.0, 6.0],
])?;
assert_eq!(m.nrows(), 2);
assert_eq!(m.ncols(), 3);
assert_eq!(m.get(0, 1), 2.0);
```

Categorical data uses [`StrMatrix`](https://docs.rs/datarust/latest/datarust/struct.StrMatrix.html) (`Vec<Vec<String>>`), and sparse output is available as [`SparseMatrix`](https://docs.rs/datarust/latest/datarust/struct.SparseMatrix.html) (CSR).

### Transformer Trait

All numeric transformers implement the [`Transformer`](https://docs.rs/datarust/latest/datarust/trait.Transformer.html) trait:

```rust
pub trait Transformer {
    fn name(&self) -> &'static str;
    fn fit(&mut self, x: &Matrix) -> Result<()>;
    fn transform(&self, x: &Matrix) -> Result<Matrix>;
    fn fit_transform(&mut self, x: &Matrix) -> Result<Matrix> { ... }
    fn is_fitted(&self) -> bool;
}
```

### Regressor Trait

Regression estimators (currently [`LinearRegression`](#linearregression)) implement the [`Regressor`](https://docs.rs/datarust/latest/datarust/trait.Regressor.html) trait — the supervised counterpart of `Transformer`, with `predict` instead of `transform`:

```rust
pub trait Regressor {
    fn name(&self) -> &'static str;
    fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()>;
    fn predict(&self, x: &Matrix) -> Result<Vec<f64>>;
    fn fit_predict(&mut self, x: &Matrix, y: &[f64]) -> Result<Vec<f64>> { ... }
    fn is_fitted(&self) -> bool;
}
```

### CategoricalTransformer Trait

Categorical encoders (OneHot, Ordinal, Frequency) implement the [`CategoricalTransformer`](https://docs.rs/datarust/latest/datarust/trait.CategoricalTransformer.html) trait (`StrMatrix → Matrix`):

```rust
pub trait CategoricalTransformer {
    fn name(&self) -> &'static str;
    fn fit(&mut self, x: &StrMatrix) -> Result<()>;
    fn transform(&self, x: &StrMatrix) -> Result<Matrix>;
    fn fit_transform(&mut self, x: &StrMatrix) -> Result<Matrix> { ... }
    fn inverse_transform(&self, _y: &Matrix) -> Result<StrMatrix> { ... }
    fn is_fitted(&self) -> bool;
}
```

OneHotEncoder and OrdinalEncoder provide real `inverse_transform`; FrequencyEncoder returns an error (non-injective).

### TargetTransformer Trait

The [`TargetTransformer`](https://docs.rs/datarust/latest/datarust/trait.TargetTransformer.html) trait extends categorical encoding to supervised transformers that require target values during `fit`:

```rust
pub trait TargetTransformer {
    fn name(&self) -> &'static str;
    fn fit(&mut self, x: &StrMatrix, y: &[f64]) -> Result<()>;
    fn transform(&self, x: &StrMatrix) -> Result<Matrix>;
    fn fit_transform(&mut self, x: &StrMatrix, y: &[f64]) -> Result<Matrix> { ... }
    fn inverse_transform(&self, _y: &Matrix) -> Result<StrMatrix> { ... }
    fn is_fitted(&self) -> bool;
}
```

All target transformers (currently only `TargetEncoder`) support `fit_transform` with target values and a default `inverse_transform` that returns an error.

### LabelTransformer Trait

The [`LabelTransformer`](https://docs.rs/datarust/latest/datarust/trait.LabelTransformer.html) trait maps 1-D string labels to integer indices (`&[String] → &[usize]`), used by `LabelEncoder`:

```rust
pub trait LabelTransformer {
    fn name(&self) -> &'static str;
    fn fit(&mut self, x: &[String]) -> Result<()>;
    fn transform(&self, x: &[String]) -> Result<Vec<usize>>;
    fn inverse_transform(&self, x: &[usize]) -> Result<Vec<String>>;
    fn fit_transform(&mut self, x: &[String]) -> Result<Vec<usize>> { ... }
    fn is_fitted(&self) -> bool;
}
```

### Errors

Operations return `Result<T, DatarustError>` with variants for `NotFitted`, `InvalidInput`, `ShapeMismatch`, `EmptyInput`, `AllMissing`, `UnknownCategory`, `UnknownLabel`, `InvalidConfig`, and `Singular`.

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for a deep dive into the crate's module layout, trait hierarchy, type erasure, design decisions, and error handling philosophy.

Key architectural highlights:

| Layer | Description |
|---|---|
| **Matrix types** | `Matrix` (f64), `StrMatrix` (String), `SparseMatrix` (CSR) — all with validation |
| **Core traits** | `Transformer`, `CategoricalTransformer`, `TargetTransformer`, `LabelTransformer`, `FeatureNames` |
| **Type erasure** | `TransformerKind`, `CategoricalTransformerKind`, `TargetTransformerKind` — enable heterogeneous `Pipeline` and `ColumnTransformer` |
| **Features** | `serde` (JSON save/load), `rayon` (parallel iterators) — both optional, zero deps by default |

## API Reference

### Scalers

#### StandardScaler

Standardize features by removing the mean and scaling to unit variance.
Uses population standard deviation (`ddof = 0`), matching sklearn.

```rust
use datarust::scaler::StandardScaler;

let mut s = StandardScaler::new()
    .with_mean(true)
    .with_std(true);
let out = s.fit_transform(&x)?;
// out[i][j] = (x[i][j] - mean[j]) / std[j]
```

#### MinMaxScaler

Scale each feature to a given range (default `[0, 1]`).

```rust
use datarust::scaler::MinMaxScaler;

let mut s = MinMaxScaler::new()
    .feature_range(-1.0, 1.0);
let out = s.fit_transform(&x)?;
// out[i][j] = (x[i][j] - min[j]) / (max[j] - min[j]) * range + lo
```

#### RobustScaler

Scale using median and IQR (outlier-resistant).

```rust
use datarust::scaler::RobustScaler;

let mut s = RobustScaler::new();
let out = s.fit_transform(&x)?;
// out[i][j] = (x[i][j] - median[j]) / (q75[j] - q25[j])
```

#### MaxAbsScaler

Scale by dividing by the maximum absolute value per feature.
Preserves sparsity structure.

```rust
use datarust::scaler::MaxAbsScaler;

let mut s = MaxAbsScaler::new();
let out = s.fit_transform(&x)?;
// out[i][j] = x[i][j] / max(abs(col_j))
```

#### Normalizer

Normalize samples individually to unit norm (row-wise).

```rust
use datarust::scaler::{Normalizer, Norm};

let mut n = Normalizer::new(Norm::L2);  // L1, L2, or Max
let out = n.fit_transform(&x)?;
// row := row / norm(row)
```

#### Binarizer

Binarize features (thresholding at a given value).

```rust
use datarust::scaler::Binarizer;

let mut b = Binarizer::new().threshold(0.5);
let out = b.fit_transform(&x)?;
// out[i][j] = 1.0 if x[i][j] > 0.5 else 0.0
```

#### KBinsDiscretizer

Bin continuous data into intervals.

```rust
use datarust::scaler::{KBinsDiscretizer, BinStrategy, KBinsEncode};

let mut kb = KBinsDiscretizer::new(5)?
    .strategy(BinStrategy::Uniform)    // Uniform, Quantile, or KMeans
    .encode(KBinsEncode::OneHotDense); // Ordinal or OneHotDense
let out = kb.fit_transform(&x)?;
```

#### QuantileTransformer

Transform features using quantile information to follow a uniform or normal distribution. Robust to outliers.

```rust
use datarust::scaler::{QuantileTransformer, OutputDistribution};

let mut qt = QuantileTransformer::new(1000)?
    .output_distribution(OutputDistribution::Normal); // or Uniform
let out = qt.fit_transform(&x)?;
```

#### PowerTransformer

Apply a power transform (Yeo-Johnson or Box-Cox) to make data more Gaussian-like. Lambda is estimated via MLE with golden-section search.

```rust
use datarust::scaler::{PowerTransformer, PowerMethod};

let mut pt = PowerTransformer::new()
    .method(PowerMethod::YeoJohnson) // or BoxCox (requires positive data)
    .standardize(true);              // zero-mean, unit-variance after transform
let out = pt.fit_transform(&x)?;
```

### Encoders

#### LabelEncoder

Encode string labels as integer values 0..n_classes-1 (sorted lexicographically).

```rust
use datarust::encoder::{LabelEncoder, LabelHandleUnknown};

let mut encoder = LabelEncoder::new();
encoder.fit(&["dog", "cat", "bird"])?;
let encoded = encoder.transform(&["dog", "bird"])?;
// encoded = [1, 2]

// Handle unknown labels gracefully (returns usize::MAX):
let mut encoder = LabelEncoder::new()
    .handle_unknown(LabelHandleUnknown::Ignore);
encoder.fit(&["a", "b"])?;
let out = encoder.transform(&["a", "z", "b"])?;
// out = [0, usize::MAX, 1]
```

#### OneHotEncoder

Encode categorical features as a one-hot numeric matrix.

```rust
use datarust::encoder::{OneHotEncoder, DropStrategy, HandleUnknown};
use datarust::StrMatrix;

let s = StrMatrix::from_column(["Red", "Blue", "Green", "Red"])?;
let mut ohe = OneHotEncoder::new()
    .drop(DropStrategy::First)
    .handle_unknown(HandleUnknown::Ignore);
let dense = ohe.fit_transform(&s)?;  // Matrix
let sparse = ohe.fit_transform_sparse(&s)?;  // SparseMatrix (CSR)

// Inverse transform reconstructs categories from one-hot codes
let decoded = ohe.inverse_transform(&dense)?;
assert_eq!(decoded.get(0, 0), "Red");

// Sparse inverse via conversion
let decoded_sparse = ohe.inverse_transform_sparse(&sparse)?;
```

The CSR [`SparseMatrix`](#sparsematrix-csr) output stores only the `1.0` positions, saving significant memory for high-cardinality columns. `transform_sparse` and `inverse_transform` are both parallelized under the `rayon` feature.

#### OrdinalEncoder

Encode categorical features as integer codes with optional user-defined ordering.

```rust
use datarust::encoder::{OrdinalEncoder, OrdinalCategories, OrdinalHandleUnknown};

// Auto: sorted lexicographically
let mut enc = OrdinalEncoder::new(OrdinalCategories::Auto);
let out = enc.fit_transform(&s)?;

// Manual: custom order (categories per column)
let mut enc = OrdinalEncoder::new(OrdinalCategories::Manual(vec![
    vec!["small".into(), "medium".into(), "large".into()],
]));
let out = enc.fit_transform(&s)?;

// Handle unknown categories with -1.0 sentinel
let mut enc = OrdinalEncoder::new(OrdinalCategories::Auto)
    .handle_unknown(OrdinalHandleUnknown::UseNegOne);
enc.fit(&s)?;
let out = enc.transform(&unknown_data)?;
// Unknown category → -1.0; inverse_transform → empty string ""
```

#### TargetEncoder

Replace categories with the smoothed mean of the target variable. Implements [`TargetTransformer`](#targettransformer-trait) (requires `y` during `fit`).

```rust
use datarust::encoder::TargetEncoder;

let mut te = TargetEncoder::new(5.0); // smoothing factor
te.fit(&categorical, &target)?;
let out = te.transform(&categorical)?;
```

Controlled via [`UnknownTarget`](https://docs.rs/datarust/latest/datarust/encoder/enum.UnknownTarget.html): `GlobalMean` (default), `NaN`, or `Error` for unseen categories.

#### FrequencyEncoder

Replace categories with their frequency (count or proportion). Implements [`CategoricalTransformer`](#categoricaltransformer-trait) with configurable unknown handling.

```rust
use datarust::encoder::{FrequencyEncoder, UnknownFrequency};

// Raw counts
let mut fe = FrequencyEncoder::new(false);
let out = fe.fit_transform(&s)?;

// Normalized proportions with error on unknown categories
let mut fe = FrequencyEncoder::new(true)
    .handle_unknown(UnknownFrequency::Error);
let out = fe.fit_transform(&s)?;
```

### Imputers

#### SimpleImputer

Impute missing values (`f64::NAN`) using a column-wise strategy.

```rust
use datarust::imputer::{SimpleImputer, ImputeStrategy};

let mut imp = SimpleImputer::new(ImputeStrategy::Mean); // Median, MostFrequent, or Constant(val)
let out = imp.fit_transform(&x)?;
```

#### KnnImputer

Impute missing values using k-Nearest Neighbors. Distance is computed over co-observed features only.

```rust
use datarust::imputer::{KnnImputer, KnnWeights};

let mut knn = KnnImputer::new(5, KnnWeights::Uniform); // or Distance
let out = knn.fit_transform(&x)?;
```

### Polynomial Features

```rust
use datarust::polynomial::PolynomialFeatures;

let mut poly = PolynomialFeatures::new(2)  // degree
    .include_bias(true)                     // include intercept column
    .interaction_only(false);               // only cross-terms
let out = poly.fit_transform(&x)?;
```

### Selection

#### VarianceThreshold

Remove features with variance below a threshold.

```rust
use datarust::selection::VarianceThreshold;

let mut vt = VarianceThreshold::new(0.01)?;
let out = vt.fit_transform(&x)?;
```

#### SelectKBest

Keep the k highest-scoring features according to a univariate statistical test.

```rust
use datarust::selection::{SelectKBest, ScoreFunc};

let mut skb = SelectKBest::new(ScoreFunc::FClassif, 2)?; // Chi2 or MutualInformation
skb.fit_with_labels(&x, &labels)?;
let out = skb.transform(&x)?;
```

### Decomposition

#### PCA

Principal Component Analysis via Jacobi eigenvalue decomposition.

```rust
use datarust::decomposition::{PCA, PCAComponents};

let mut pca = PCA::new(PCAComponents::Variance(0.95)) // Count(2) or All
    .whiten(true);
let projected = pca.fit_transform(&x)?;

// Components: pca.components()
// Explained variance: pca.explained_variance_ratio()
// Reconstruct: pca.inverse_transform(&projected)?
```

#### TruncatedSVD

Dimensionality reduction via truncated SVD (suitable for sparse or TF-IDF data).
Does **not** center the data. Supports flexible component selection via [`SVDComponents`](https://docs.rs/datarust/latest/datarust/decomposition/enum.SVDComponents.html).

```rust
use datarust::decomposition::{TruncatedSVD, SVDComponents};

// By exact count:
let mut svd = TruncatedSVD::new(5).unwrap();
let out = svd.fit_transform(&x)?;

// By variance threshold (keeps enough components to explain 95% variance):
let mut svd = TruncatedSVD::new(0.95).unwrap();
let out = svd.fit_transform(&x)?;

// Keep all components:
let mut svd = TruncatedSVD::new(SVDComponents::All).unwrap();
let out = svd.fit_transform(&x)?;
```

### Linear Models

#### LinearRegression

Ordinary least-squares regression — the crate's first `predict`-capable estimator. Estimates `y ≈ Xβ + b` by minimising the residual sum of squares. Mirrors `sklearn.linear_model.LinearRegression`.

Two solvers are available:
- **Cholesky** (default) — solves `XᵀX β = Xᵀy` via a pure-Rust Cholesky decomposition. Fast and dependency-free; requires `X` to have full column rank.
- **SVD** — eigen-decomposition-based pseudo-inverse. Numerically stable for rank-deficient / collinear inputs, at higher cost.

```rust
use datarust::linear_model::{LinearRegression, LinearSolver};
use datarust::traits::Regressor;

let mut model = LinearRegression::new()
    .with_fit_intercept(true)           // default true
    .with_solver(LinearSolver::Cholesky); // or LinearSolver::Svd

model.fit(&x, &y)?;
let pred = model.predict(&new_x)?;

// Fitted parameters
model.coef();          // &[f64] — coefficients β
model.intercept();     // f64    — intercept b
model.n_features_in(); // usize

// R² of the prediction (mirrors estimator.score in sklearn)
let r2 = model.score(&x, &y)?;
```

#### Ridge

L2-regularized regression. Minimises `‖Xβ − y‖² + α‖β‖²`. Mirrors `sklearn.linear_model.Ridge`.

The `α` penalty shrinks coefficients toward zero (reducing variance at the cost of bias) and guarantees the system matrix `XᵀX + αI` is positive-definite — so Ridge **succeeds on rank-deficient / collinear inputs** where `LinearRegression` would fail.

```rust
use datarust::linear_model::{Ridge, RidgeSolver};
use datarust::traits::Regressor;

let mut model = Ridge::new()
    .with_alpha(1.0)                      // regularization strength
    .with_solver(RidgeSolver::Cholesky);  // or RidgeSolver::Svd

model.fit(&x, &y)?;
let pred = model.predict(&new_x)?;
```

#### Lasso

L1-regularized regression. Minimises `(1/(2n))‖Xβ − y‖² + α‖β‖₁`. Mirrors `sklearn.linear_model.Lasso`.

The L1 penalty drives irrelevant coefficients to **exactly zero**, producing a sparse model that performs implicit feature selection — the key difference from Ridge. Solved by coordinate descent with soft-thresholding.

```rust
use datarust::linear_model::Lasso;
use datarust::traits::Regressor;

let mut model = Lasso::new()
    .with_alpha(0.1)          // larger alpha → more sparsity
    .with_max_iter(1000)      // default 1000
    .with_tol(1e-4);          // convergence tolerance

model.fit(&x, &y)?;
let pred = model.predict(&new_x)?;

model.coef();   // some entries may be exactly 0.0 (sparsity)
model.n_iter(); // iterations actually run
```

#### LogisticRegression

Binary classification via IRLS (Iteratively Reweighted Least Squares). Mirrors `sklearn.linear_model.LogisticRegression`.

Estimates `P(y = 1 | x) = σ(x·β + b)` by maximising the log-likelihood via Newton-Raphson. Each iteration solves a weighted least-squares system using the shared Cholesky (default) or SVD solver. Targets must be `0.0` or `1.0`.

```rust
use datarust::linear_model::{LogisticRegression, LogisticSolver};
use datarust::traits::Regressor;

let mut model = LogisticRegression::new()
    .with_solver(LogisticSolver::Cholesky) // or LogisticSolver::Svd
    .with_max_iter(100)                    // default 100
    .with_tol(1e-4);                       // convergence tolerance

model.fit(&x, &y)?;          // y must be 0.0 / 1.0
let probs = model.predict(&new_x)?;        // Vec<f64> of P(y=1|x) in [0,1]
let classes = model.predict_class(&new_x)?; // Vec<f64> of 0.0 / 1.0 (threshold 0.5)
let acc = model.score(&x, &y)?;            // mean accuracy (f64)
```

### Metrics

Regression metrics mirroring `sklearn.metrics`. Each takes `y_true` and `y_pred` as `&[f64]`.

```rust
use datarust::metrics::regression::*;

let mse  = mean_squared_error(&y_true, &y_pred, true)?;   // squared=true → MSE
let rmse = mean_squared_error(&y_true, &y_pred, false)?;  // squared=false → RMSE
let mae  = mean_absolute_error(&y_true, &y_pred)?;
let r2   = r2_score(&y_true, &y_pred)?;
let me   = max_error(&y_true, &y_pred)?;
let ev   = explained_variance_score(&y_true, &y_pred)?;
```

Classification metrics for binary labels (`0.0` / `1.0`):

```rust
use datarust::metrics::classification::*;

let acc  = accuracy_score(&y_true, &y_pred)?;
let prec = precision_score(&y_true, &y_pred)?;
let rec  = recall_score(&y_true, &y_pred)?;
let f1   = f1_score(&y_true, &y_pred)?;
let cm   = confusion_matrix(&y_true, &y_pred)?; // [[tn, fp], [fn, tp]]
let ll   = log_loss(&y_true, &y_proba, 1e-15)?;  // cross-entropy
```

### Model Selection

Train/test splitting and cross-validation, mirroring `sklearn.model_selection`.

#### train_test_split

```rust
use datarust::model_selection::{train_test_split, TrainTestSplit};

// Quick split with defaults (25% test, shuffled):
let (x_tr, x_te, y_tr, y_te) = train_test_split(&x, &y)?;

// Or configure via the builder:
let (x_tr, x_te, y_tr, y_te) = TrainTestSplit::new()
    .with_test_size(0.2)
    .with_shuffle(true)
    .with_random_state(42)
    .split(&x, &y)?;
```

#### KFold and StratifiedKFold

```rust
use datarust::model_selection::{KFold, StratifiedKFold};

// K-fold: each sample is in the test set exactly once.
let cv = KFold::new().with_n_splits(5).with_shuffle(true).with_random_state(42);
for (train_idx, test_idx) in cv.split(n_samples)? {
    // ...
}

// Stratified: preserves class balance in each fold (pass y).
let scv = StratifiedKFold::new().with_n_splits(5);
for (train_idx, test_idx) in scv.split(&y)? {
    // ...
}
```

#### cross_val_score

Evaluate any `Regressor + Clone` estimator with a user-supplied scorer:

```rust
use datarust::model_selection::{cross_val_score, KFold};
use datarust::linear_model::LinearRegression;
use datarust::metrics::regression::r2_score;

let cv = KFold::new().with_n_splits(5);
let scores = cross_val_score(&LinearRegression::new(), &x, &y, &cv, r2_score)?;
// scores.len() == 5; one R² per fold.
```

For classification, pass `accuracy_score` from `metrics::classification` instead.

### Pipeline

Chain multiple transformers sequentially. Fits and transforms each step on the output of the previous one. Serializable under the `serde` feature.

```rust
use datarust::pipeline::Pipeline;
use datarust::transformer_kind::TransformerKind;

let mut pipe = Pipeline::new()
    .push("scale", TransformerKind::StandardScaler(StandardScaler::new()))
    .push("pca", TransformerKind::PCA(PCA::new(PCAComponents::Count(5))))
    .push("clip", TransformerKind::Binarizer(Binarizer::new().threshold(0.0)));
let out = pipe.fit_transform(&x)?;

// Inspect step names
assert_eq!(pipe.names(), vec!["scale", "pca", "clip"]);
```

All 17 transformer types are available as `TransformerKind` variants, enabling type-safe heterogeneous pipelines.

### ColumnTransformer

Apply different transformers to different columns of a mixed numeric/categorical dataset. Returns a combined numeric matrix or an [`Output`](#output) preserving the numeric/categorical split.

```rust
use datarust::compose::{ColumnTransformer, Remainder, Table, Output};
use datarust::encoder::OneHotEncoder;
use datarust::scaler::StandardScaler;
use datarust::transformer_kind::TransformerKind;

let table = Table::new(numeric, categorical)?;

let mut ct = ColumnTransformer::new()
    .remainder(Remainder::Passthrough)  // retain unselected columns
    .add_numeric("scale", vec![0, 1], TransformerKind::StandardScaler(StandardScaler::new()))
    .add_categorical("city", vec![0], OneHotEncoder::new());
let out = ct.fit_transform(&table)?;

// Preserve the numeric/categorical split
let output: Output = ct.fit_transform_to_table(&table)?;
// output.numeric → Matrix, output.categorical → StrMatrix

// Target specs require fit_with_target
let mut ct = ColumnTransformer::new()
    .add_target("te", vec![0], TargetEncoder::new(5.0)?);
ct.fit_with_target(&table, &y)?;  // fit() would error — use fit_with_target()

// Feature names compose from all sub-transformers
let names = ct.feature_names_out(Some(&["age", "salary", "city"]));
assert_eq!(names, vec!["age", "salary", "city_Istanbul", "city_Ankara", "city_Izmir"]);
```

### Output

The [`Output`](https://docs.rs/datarust/latest/datarust/compose/struct.Output.html) struct returned by `transform_to_table` preserves numeric and categorical columns in separate matrices. Validates row-count consistency at construction:

```rust
let output = Output::new(numeric_matrix, categorical_matrix)?;
assert_eq!(output.numeric.nrows(), output.categorical.nrows());
```

### Feature Names

All output-producing transformers implement the [`FeatureNames`](https://docs.rs/datarust/latest/datarust/trait.FeatureNames.html) trait:

```rust
pub trait FeatureNames {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String>;
}
```

```rust
let scaler = StandardScaler::new();
// (assuming fitted)
let names = scaler.feature_names_out(Some(&["age", "salary"]));
assert_eq!(names, vec!["age", "salary"]);

let names = scaler.feature_names_out(None);
assert_eq!(names, vec!["x0", "x1"]);
```

Pipeline chains names through all steps; OneHotEncoder appends `_category` suffixes; PCA/TruncatedSVD generate `pca0`/`svd0` names; VarianceThreshold and SelectKBest filter names by the selected mask; ColumnTransformer composes names from all sub-transformers.

### Inverse Transform

Several transformers support reversing the transformation via `inverse_transform`, returning an approximation of the original input:

| Transformer | Trait | Notes |
|---|---|---|
| StandardScaler | [`Transformer`](https://docs.rs/datarust/latest/datarust/trait.Transformer.html) | `x = z * std + mean` |
| MinMaxScaler | `Transformer` | `x = z * (max - min) + min` |
| RobustScaler | `Transformer` | `x = z * iqr + median` |
| MaxAbsScaler | `Transformer` | `x = z * max_abs` |
| PowerTransformer | `Transformer` | `x = inverse_power(z)`, un-standardizes first |
| PCA | `Transformer` | via `components_` matrix multiply |
| TruncatedSVD | `Transformer` | via `components_` matrix multiply |
| OneHotEncoder | [`CategoricalTransformer`](#categoricaltransformer-trait) | `Matrix` → `StrMatrix` (dense + sparse via `inverse_transform_sparse`) |
| OrdinalEncoder | `CategoricalTransformer` | `-1.0` sentinel → empty string |
| LabelEncoder | [`LabelTransformer`](#labeltransformer-trait) | `usize::MAX` sentinel → empty string |

```rust
let mut s = StandardScaler::new();
let transformed = s.fit_transform(&x)?;
let reconstructed = s.inverse_transform(&transformed)?;
// reconstructed ≈ x (within floating-point precision)

// Categorical inverse_transform via trait
let mut ohe = OneHotEncoder::new();
let encoded = ohe.fit_transform(&cats)?;
let decoded: StrMatrix = ohe.inverse_transform(&encoded)?;

// Label inverse via LabelTransformer
let mut le = LabelEncoder::new();
let indices = le.fit_transform(&labels)?;
let back: Vec<String> = le.inverse_transform(&indices)?;
```

Transformers that do not support inverse return an error (e.g. Binarizer, Normalizer, FrequencyEncoder, TargetEncoder).

### FunctionTransformer

Wrap arbitrary functions as a [`Transformer`](https://docs.rs/datarust/latest/datarust/trait.Transformer.html), mirroring `sklearn.preprocessing.FunctionTransformer`.

```rust
use datarust::function_transformer::FunctionTransformer;

fn times_two(x: &Matrix) -> Result<Matrix> {
    let out: Vec<Vec<f64>> = x.rows_ref()
        .iter()
        .map(|row| row.iter().map(|&v| v * 2.0).collect())
        .collect();
    Matrix::new(out)
}

let mut ft = FunctionTransformer::new(times_two);
let out = ft.fit_transform(&x)?;
// out[i][j] = x[i][j] * 2
```

An inverse function can be set via `.with_inverse(func)`. At deserialization (serde feature), function pointers are skipped — call `set_func()` to restore.

### Pipeline Ergonomics

[`Pipeline`](https://docs.rs/datarust/latest/datarust/struct.Pipeline.html) provides runtime access to individual steps without consuming or destructuring the pipeline:

| Method | Description |
|---|---|
| `get_step(name)` | Borrow a step by name |
| `get_step_mut(name)` | Mutably borrow a step by name |
| `step(index)` | Borrow a step and its name by index |
| `step_mut(index)` | Mutably borrow a step and its name by index |
| `remove_step(index)` | Remove and return a step |
| `insert_step(index, name, t)` | Insert a step at a position |
| `set_step(name, t)` | Replace a step by name |

```rust
let mut pipe = Pipeline::new()
    .push("scale", TransformerKind::StandardScaler(StandardScaler::new()))
    .push("reduce", TransformerKind::PCA(PCA::new(PCAComponents::Count(5))));

// Replace the scaler
pipe.set_step("scale", TransformerKind::RobustScaler(RobustScaler::new()));

// Access the PCA step's explained variance
if let TransformerKind::PCA(pca) = pipe.get_step("reduce").unwrap() {
    println!("explained variance: {:?}", pca.explained_variance_ratio());
}
```

### Matrix Slicing

[`Matrix`](https://docs.rs/datarust/latest/datarust/struct.Matrix.html) supports column and row slicing with bounds checking:

```rust
let m = Matrix::new(vec![
    vec![1.0, 2.0, 3.0],
    vec![4.0, 5.0, 6.0],
])?;

let cols = m.select_columns(&[0, 2])?;  // columns 0 and 2
assert_eq!(cols.get(0, 0), 1.0);
assert_eq!(cols.get(0, 1), 3.0);

let rows = m.select_rows(&[1])?;  // only row 1
assert_eq!(rows.nrows(), 1);
```

### Covariance & Correlation

The [`stats`](https://docs.rs/datarust/latest/datarust/stats/index.html) module provides matrix-level statistical operations:

```rust
use datarust::stats::{covariance_matrix, correlation_matrix};

let data = Matrix::new(vec![
    vec![1.0, 2.0],
    vec![3.0, 4.0],
    vec![5.0, 6.0],
])?;
let cov = covariance_matrix(&data, 0);  // ddof=0 (population)
let corr = correlation_matrix(&data);   // Pearson (ddof=1)
```

PCA also exposes [`noise_variance()`](https://docs.rs/datarust/latest/datarust/decomposition/struct.PCA.html#method.noise_variance) — the average eigenvalue of discarded components, matching sklearn's `PCA.noise_variance_`.

## Serialization

Enable the `serde` feature for JSON save/load of fitted transformers.

```toml
datarust = { version = "0.3", features = ["serde"] }
```

```rust
use datarust::serialize::{save_json, load_json, to_json, from_json};
use datarust::scaler::StandardScaler;

// String round-trip
let mut scaler = StandardScaler::new();
scaler.fit(&x)?;
let json = to_json(&scaler)?;
let restored: StandardScaler = from_json(&json)?;

// File round-trip
save_json(&scaler, "scaler.json")?;
let reloaded: StandardScaler = load_json("scaler.json")?;
```

All leaf transformers, `Pipeline` (via `TransformerKind`), and `ColumnTransformer` are serializable.

## Parallelism

Enable the `rayon` feature for parallel column operations on large datasets.

```toml
datarust = { version = "0.3", features = ["rayon"] }
```

When enabled, the following use parallel iterators:

- **Statistics:** `column_mean`, `column_variance`, `column_min`, `column_max`, `column_median`, `column_mode`, `column_quantile`
- **Scalers:** StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
- **Encoders:** OneHotEncoder (dense + sparse transform, inverse_transform), OrdinalEncoder (transform), FrequencyEncoder (transform), TargetEncoder (transform)
- **Imputation:** KNN Imputer distance computation

## Feature Comparison: datarust vs sklearn

| Transformer | datarust | sklearn |
|---|---|---|
| StandardScaler | ✓ (ddof=0) | ✓ (ddof=0) |
| MinMaxScaler | ✓ (custom range) | ✓ |
| RobustScaler | ✓ (centering + scaling) | ✓ |
| MaxAbsScaler | ✓ | ✓ |
| Normalizer (L1/L2/Max) | ✓ | ✓ |
| Binarizer | ✓ | ✓ |
| KBinsDiscretizer | ✓ (Uniform/Quantile/KMeans, Ordinal/OneHotDense) | ✓ |
| QuantileTransformer | ✓ (Uniform/Normal output) | ✓ |
| PowerTransformer | ✓ (Yeo-Johnson/Box-Cox + MLE lambda) | ✓ |
| LabelEncoder | ✓ (handle_unknown: Error/Ignore) | ✓ |
| OrdinalEncoder | ✓ (auto + manual) | ✓ |
| OneHotEncoder | ✓ (drop, handle_unknown, sparse CSR) | ✓ |
| TargetEncoder | ✓ (smoothed mean, UnknownTarget: GlobalMean/NaN/Error) | ✓ |
| FrequencyEncoder | ✓ (count/proportion, UnknownFrequency: Zero/Error) | — |
| SimpleImputer | ✓ (mean/median/most_frequent/constant) | ✓ |
| KNN Imputer | ✓ (uniform/distance) | ✓ |
| PolynomialFeatures | ✓ (degree, interaction_only, bias) | ✓ |
| VarianceThreshold | ✓ | ✓ |
| SelectKBest | ✓ (F-classif / Chi2 / Mutual Info) | ✓ |
| PCA | ✓ (Jacobi EV + power-iteration deflation + randomized SVD, count/variance/all, whiten, `PCASolver`) | ✓ |
| TruncatedSVD | ✓ (SVDComponents: Count/Variance/All) | ✓ |
| Pipeline | ✓ (TransformerKind, serde) | ✓ |
| ColumnTransformer | ✓ (Numeric + Categorical + Target specs, Output table, duplicate detection, remainder passthrough) | ✓ |
| FunctionTransformer | ✓ (optional inverse, closure-based) | ✓ |
| FeatureNames | ✓ (trait, all transformers, short-input padding) | ✓ |
| inverse_transform | ✓ (scalers, PowerTransformer, PCA, SVD, OneHotEncoder, OrdinalEncoder, LabelEncoder) | ✓ |
| Pipeline Ergonomics | ✓ (get_step, step, set_step, insert, remove) | — |
| Matrix Slicing | ✓ (select_columns, select_rows) | — |
| Covariance / Correlation | ✓ (ddof-configurable) | — |
| JSON Serialization | ✓ (serde feature) | — (joblib) |
| Sparse Output | ✓ (CSR via SparseMatrix) | ✓ |
| Parallelism | ✓ (rayon feature) | — (joblib) |

## Performance: datarust vs scikit-learn

The numbers below are **measured**, not estimated. The same deterministic synthetic
dataset (xorshift64, seed 42, values in `[-100, 100)`) is fed to both libraries, and
the median `fit_transform` time over 15 runs (after one warmup) is reported. The
benchmark harness lives in `examples/bench_compare_rust.rs` and
`benches/compare_sklearn.py` — re-run them on your own hardware.

**Test setup:** Apple M5 Pro (18 cores, arm64), Rust 1.96.0 (release), Python 3.9.6,
scikit-learn 1.6.1, numpy 2.0.2, scipy 1.13.1. Times are in **milliseconds**. The
`Ratio` column is `sklearn_ms / datarust_ms` — values `> 1` mean **datarust is faster**.
Two datarust columns are shown: the **default** (zero-dependency) build, and the
build with the `rayon` feature enabled (parallel column/row processing). PCA additionally
benefits from the `matrixmultiply` feature, shown in the notes below the table.

| Workload | Size (rows × cols) | datarust default (ms) | datarust +rayon (ms) | sklearn (ms) | best ratio |
|---|---|---:|---:|---:|---:|
| StandardScaler | 1 000 × 10 | 0.023 | 0.016 | 0.280 | **17.5×** |
| StandardScaler | 10 000 × 100 | 1.24 | 1.20 | 2.46 | 2.0× |
| StandardScaler | 50 000 × 200 | 8.2 | 4.7 | 22.4 | **4.8×** |
| MinMaxScaler | 1 000 × 10 | 0.025 | 0.014 | 0.202 | **14.4×** |
| MinMaxScaler | 10 000 × 100 | 1.70 | 1.47 | 1.32 | 0.9× |
| MinMaxScaler | 50 000 × 200 | 10.8 | 7.5 | 11.6 | **1.5×** |
| RobustScaler | 1 000 × 10 | 0.17 | 0.13 | 0.768 | **5.8×** |
| RobustScaler | 10 000 × 100 | 11.2 | 2.09 | 21.4 | **10×** |
| RobustScaler | 50 000 × 200 | 123 | 14.0 | 193.5 | **13.8×** |
| PCA (k = min(10, cols/2)) | 1 000 × 10 | 0.18 | 0.10 | 0.226 | 2.2× |
| PCA | 10 000 × 100 | 45 | 41 | 1.39 | 0.03× |
| PCA | 50 000 × 200 | 838 | 819 | 12.2 | 0.01× |
| Pipeline (Standard→MinMax→Robust) | 1 000 × 10 | 0.20 | 0.21 | 1.02 | **4.9×** |
| Pipeline | 10 000 × 100 | 13.2 | 4.1 | 25.2 | 6.1× |
| Pipeline | 50 000 × 200 | 144 | 26.7 | 229.6 | **8.6×** |
| OneHotEncoder (string) | 1 000 × 5 | 0.38 | 0.55 | 0.800 | 1.5× |
| OneHotEncoder | 10 000 × 10 | 7.4 | 6.8 | 9.9 | 1.5× |
| OneHotEncoder | 50 000 × 20 | 89 | 80 | 205 | **2.6×** |
| ColumnTransformer (num + cat) | 1 000 × 5 | 0.026 | 0.026 | 4.6 | **179×** |
| ColumnTransformer | 10 000 × 10 | 0.23 | 0.24 | 79.8 | **347×** |
| ColumnTransformer | 50 000 × 20 | 1.31 | 1.32 | 812.8 | **620×** |
| LinearRegression (fit+predict) | 1 000 × 10 | 0.16 | 0.16 | — | — |
| LinearRegression | 10 000 × 100 | 14.4 | 14.4 | — | — |
| LinearRegression | 50 000 × 200 | 258 | 258 | — | — |

**PCA with the `matrixmultiply` feature.** The default and `rayon` builds compute the
covariance `Xcᵀ Xc` with a scalar loop; enabling the optional `matrixmultiply` feature
dispatches the covariance **and** the transform/inverse matmuls to a tuned pure-Rust GEMM
(no system BLAS), and a power-iteration + deflation path (`eigh_topk`) replaces the full
Jacobi sweep when `n_components` is small. On 50 000 × 200 this cuts PCA from
**838 ms → 104 ms** (8× faster), and on 10 000 × 100 from **45 ms → 9.3 ms** (4.8×). PCA
remains slower than scikit-learn (which uses LAPACK's full SVD) — see "Where scikit-learn
wins" below — but the gap narrowed from 85× to ~8×.

**Randomized SVD (opt-in).** `PCA::solver(PCASolver::Randomized)` selects the
Halko–Martinsson–Tropp randomized SVD, which is `O(n·p·(k+oversample))` instead of
`O(p³·sweeps)` and is the fast path for tall-and-wide, low-rank data (this is what
sklearn's `svd_solver='randomized'` does). It is currently opt-in while an oversampling
edge case is being verified; `Auto` (the default) uses the exact eigensolver paths.

**LinearRegression with the `matrixmultiply` feature.** `fit` forms the normal-equation
matrices `XᵀX` (p×p) and `Xᵀy` (p) via `Matrix::matmul`, then solves them with a pure-Rust
Cholesky decomposition. Enabling `matrixmultiply` dispatches the matmul to a tuned GEMM,
cutting `fit` from **258 ms → 84 ms** at 50 000 × 200 (3× faster) and **14.4 ms → 5.0 ms**
at 10 000 × 100 (2.9× faster). sklearn timing is not shown here because the Python
comparison harness requires numpy/scipy in the environment; the Rust harness
(`cargo run --release --features matrixmultiply --example bench_compare_rust`) is
reproducible on any machine.

### Reading the results

**Where datarust wins decisively:**

- **Mixed numeric + categorical composition.** `ColumnTransformer` is **160–590×**
  faster than scikit-learn's on large inputs. This is the headline result and reflects
  the cost of sklearn's per-column Python dispatch, dtype coercion, and
  `ColumnTransformer`'s object-array marshalling on mixed-type inputs.
- **String / categorical encoding.** `OneHotEncoder` is ~1.5–2.6× faster because
  datarust operates on a native `StrMatrix` directly — no Python object-array overhead,
  no GIL.
- **Numeric scalers with `rayon`.** Once the data is large enough to amortise thread
  spawn, `StandardScaler`/`RobustScaler`/`Pipeline` all beat sklearn by **4.8–13.8×** at
  50 000 × 200. The single-pass Welford statistics and contiguous flat storage close
  the gap that numpy's vectorised kernels used to dominate.
- **Small data and startup latency.** At 1 000 × 10, datarust is faster on every
  workload — up to **17.5×** on `StandardScaler` (the rayon path now falls back to the
  scalar loop below 4 096 rows, avoiding thread-pool overhead on tiny inputs). There is
  no Python interpreter to spin up and no joblib/numpy import cost — relevant for
  embedded, batch-on-many-small-files, or request-scoped inference paths.

**Where scikit-learn still wins:**

- **PCA on tall-and-wide data (without the `matrixmultiply` feature).** sklearn's `PCA`
  is still faster when comparing default builds (0.01× at 50 000 × 200). It calls into
  LAPACK's full SVD via a shared-library BLAS; datarust implements the covariance
  eigendecomposition with a from-scratch Jacobi sweep. With the `matrixmultiply` feature
  the gap narrows from 85× to ~8×, and `PCA::solver(PCASolver::Randomized)` (randomized
  SVD, the same algorithm sklearn's `svd_solver='randomized'` uses) closes it further for
  low-rank inputs. For PCA on large dense matrices as the hot path, sklearn remains the
  fastest option today.
- **MinMaxScaler at medium width.** At 10 000 × 100 the two are roughly tied (0.9×);
  numpy's contiguous buffer and autovectorisation win narrowly on this particular shape.
  At both smaller and larger sizes datarust leads.

**The honest one-line summary:** for the workloads Rust ML pipelines typically care about
— heterogeneous `ColumnTransformer` composition, categorical encoding, numeric scaling on
medium-to-large data, and latency-sensitive preprocessing — datarust is now the faster
choice; the remaining gap is dense eigendecomposition (PCA/SVD) at scale, where a
dedicated BLAS/LAPACK backend still wins.

### How the speedups were achieved (0.3.0)

Layered optimisations, each measurable:

1. **Single-pass fused statistics.** `StandardScaler`/`MinMaxScaler` previously made
   2–3 full passes over the data (mean, then variance which re-read for mean, then the
   variance sweep). A Welford accumulator now computes mean+variance in one row-major
   pass; min+max are fused similarly; `RobustScaler` sorts each column once instead of
   three times.
2. **Contiguous flat storage.** `Matrix` is now a single `Vec<f64>` (+ rows, cols)
   instead of `Vec<Vec<f64>>` (one heap allocation per row). This unlocks stride-1 cache
   lines and auto-vectorisation across every numeric loop — the dominant win on large
   dense inputs.
3. **Optional tuned GEMM.** The `matrixmultiply` feature (off by default, preserving the
   zero-dependency build) routes `Matrix::matmul`, centered-covariance, and PCA/SVD
   transforms through a micro-optimised pure-Rust kernel.
4. **Flat Jacobi eigensolver + power-iteration deflation.** The eigensolver behind PCA
   and TruncatedSVD now operates on a single contiguous buffer (better cache locality)
   and, when `n_components` is small, a power-iteration + deflation path computes only
   the top-`k` eigenpairs in `O(k·p²·iters)` instead of the full `O(p³·sweeps)` sweep.
5. **Adaptive parallelism threshold.** Scaler `transform` paths now use the scalar loop
   below 4 096 rows and the `rayon` parallel path above it — fixing a regression where
   `rayon`'s thread-pool overhead made small-data transforms slower than the default
   build.
6. **Randomized SVD (opt-in).** `PCA::solver(PCASolver::Randomized)` selects a
   Halko–Martinsson–Tropp randomized SVD — the same family of algorithm sklearn uses for
   its `svd_solver='randomized'`. It is `O(n·p·(k+oversample))` and is the fast path for
   tall-and-wide, low-rank inputs.

See the `[0.3.0]` entry in `CHANGELOG.md` for the per-workload before/after numbers.

### Non-performance advantages over the Python stack

Beyond raw throughput, datarust provides properties scikit-learn cannot offer:

- **Zero external dependencies by default** — no numpy/BLAS/LAPACK/scipy install, no
  shared-library ABI concerns. `cargo add datarust` and you have a working preprocessor.
- **No Python runtime, no GIL** — embeddable in any Rust binary, WASM, or service.
- **Compile-time type safety** — categorical (`StrMatrix`) vs numeric (`Matrix`) inputs
  are enforced by the type system, not discovered at runtime.
- **Single static binary** — deployable preprocessing with no environment drift.
- **Typed `Result<T, DatarustError>`** — no exceptions during inference; the public API
  is panic-free.
- **JSON serde round-trips** — fitted transformers serialize to portable JSON, not
  joblib's Python-specific pickle.

## Complete Examples

### Preprocessing workflow with Pipeline + ColumnTransformer

```rust
use datarust::compose::{ColumnTransformer, Remainder, Table};
use datarust::encoder::OneHotEncoder;
use datarust::pipeline::Pipeline;
use datarust::scaler::StandardScaler;
use datarust::transformer_kind::TransformerKind;
use datarust::Matrix;

// Numeric features: age, salary, bonus
let numeric = Matrix::new(vec![
    vec![25.0, 50000.0, 2000.0],
    vec![30.0, 60000.0, 3000.0],
    vec![35.0, 70000.0, 4000.0],
])?;

// Categorical features: city, department
let categorical = StrMatrix::from_strings(vec![
    vec!["Istanbul", "Engineering"],
    vec!["Ankara", "Sales"],
    vec!["Izmir", "Engineering"],
])?;

let table = Table::new(numeric, categorical)?;

// Mixed-type transformation
let mut ct = ColumnTransformer::new()
    .remainder(Remainder::Passthrough)
    .add_numeric("num", vec![0, 1], TransformerKind::StandardScaler(StandardScaler::new()))
    .add_categorical("dept", vec![1], OneHotEncoder::new());

let transformed = ct.fit_transform(&table)?;

// Feature names
let names = ct.feature_names_out(Some(&["age", "salary", "bonus"]));
assert_eq!(names, vec!["age", "salary", "bonus", "dept_Engineering", "dept_Sales"]);
```

### PCA for dimensionality reduction

```rust
use datarust::decomposition::{PCA, PCAComponents};

let x = Matrix::new(vec![
    vec![2.5, 2.4, 3.0, 5.0],
    vec![0.5, 0.7, 1.0, 8.0],
    vec![2.2, 2.9, 4.0, 3.0],
    vec![1.9, 2.2, 3.5, 4.5],
    vec![3.1, 3.0, 4.5, 6.0],
])?;

// Keep 2 components
let mut pca = PCA::new(PCAComponents::Count(2));
let projected = pca.fit_transform(&x)?;
assert_eq!(projected.ncols(), 2);

// Reconstruct (approximate)
let reconstructed = pca.inverse_transform(&projected)?;

// Explained variance
let ratio: Vec<f64> = pca.explained_variance_ratio().to_vec();
println!("Explained variance ratio: {:?}", ratio);
```

### Missing value imputation

```rust
use datarust::imputer::{SimpleImputer, ImputeStrategy, KnnImputer, KnnWeights};

let mut x = Matrix::new(vec![
    vec![1.0, f64::NAN, 3.0],
    vec![4.0, 5.0, f64::NAN],
    vec![7.0, 8.0, 9.0],
    vec![f64::NAN, 11.0, 12.0],
])?;

// Mean imputation
let mut imp = SimpleImputer::new(ImputeStrategy::Mean);
let filled = imp.fit_transform(&x)?;

// KNN imputation (5 neighbors, uniform weighting)
let mut knn = KnnImputer::new(5, KnnWeights::Uniform);
let imputed = knn.fit_transform(&x)?;
```

### Sparse one-hot encoding

```rust
use datarust::encoder::OneHotEncoder;
use datarust::SparseMatrix;

let s = StrMatrix::from_column(["Istanbul", "Ankara", "Izmir", "Istanbul"])?;
let mut ohe = OneHotEncoder::new();
let sp: SparseMatrix = ohe.fit_transform_sparse(&s)?;

assert_eq!(sp.nnz(), 4);  // 4 ones, rest zeros
assert_eq!(sp.density(), 4.0 / 12.0);
```

### Serialization of a fitted Pipeline

```rust
use datarust::pipeline::Pipeline;
use datarust::serialize::{save_json, load_json};
use datarust::transformer_kind::TransformerKind;
use datarust::scaler::StandardScaler;

let mut pipe = Pipeline::new()
    .push("scale", TransformerKind::StandardScaler(StandardScaler::new()));
pipe.fit(&x)?;

// Save to disk
save_json(&pipe, "pipeline.json")?;

// Load and reuse
let loaded: Pipeline = load_json("pipeline.json")?;
let out = loaded.transform(&new_data)?;
```

### End-to-end: TargetEncoder + ColumnTransformer

```rust,ignore
use datarust::compose::{ColumnTransformer, Table};
use datarust::encoder::{OneHotEncoder, TargetEncoder};
use datarust::scaler::StandardScaler;
use datarust::transformer_kind::TransformerKind;

let numeric = Matrix::new(vec![
    vec![25.0, 50000.0],
    vec![30.0, 60000.0],
    vec![35.0, 70000.0],
])?;
let categorical = StrMatrix::from_strings(vec![
    vec!["Istanbul", "Engineer"],
    vec!["Ankara", "Sales"],
    vec!["Izmir", "Engineer"],
])?;
let targets = vec![100.0, 200.0, 150.0];

let table = Table::new(numeric, categorical)?;
let mut ct = ColumnTransformer::new()
    .add_numeric("nums", vec![0], TransformerKind::StandardScaler(StandardScaler::new()))
    .add_categorical("city", vec![0], OneHotEncoder::new())
    .add_target("te", vec![1], TargetEncoder::new(5.0)?);
ct.fit_with_target(&table, &targets)?;

// Transform with all three spec types
let out = ct.transform(&table)?;
println!("{} columns", out.ncols());
```

### Feature selection + PCA + Pipeline

```rust,ignore
use datarust::decomposition::{PCA, PCAComponents};
use datarust::selection::{SelectKBest, ScoreFunc};
use datarust::scaler::StandardScaler;
use datarust::pipeline::Pipeline;
use datarust::transformer_kind::TransformerKind;

let mut pipe = Pipeline::new()
    .push("scale", TransformerKind::StandardScaler(StandardScaler::new()))
    .push("select", TransformerKind::SelectKBest(SelectKBest::new(ScoreFunc::FClassif, 5)?))
    .push("pca", TransformerKind::PCA(PCA::new(PCAComponents::Count(2))));

pipe.fit_transform_with_labels(&x, &labels)?;

// 2-dimensional output from 50+ feature input
assert_eq!(pipe.transform(&x)?.ncols(), 2);
```

### Inverse transform with error propagation

```rust
use datarust::scaler::{StandardScaler, MinMaxScaler};
use datarust::decomposition::{PCA, PCAComponents};

let x = Matrix::new(vec![
    vec![1.0, 2.0],
    vec![3.0, 4.0],
])?;

// Forward: StandardScaler → MinMaxScaler → PCA
let mut scaler = StandardScaler::new();
let scaled = scaler.fit_transform(&x)?;

let mut mm = MinMaxScaler::new();
let normalized = mm.fit_transform(&scaled)?;

// Inverse: PCA → MinMaxScaler → StandardScaler
let mut pca = PCA::new(PCAComponents::Count(2));
let projected = pca.fit_transform(&normalized)?;

let pca_back = pca.inverse_transform(&projected)?;
let mm_back = mm.inverse_transform(&pca_back)?;
let reconstructed = scaler.inverse_transform(&mm_back)?;

for i in 0..x.nrows() {
    for j in 0..x.ncols() {
        let err = (x.get(i, j) - reconstructed.get(i, j)).abs();
        assert!(err < 1e-10, "reconstruction error at ({},{}): {}", i, j, err);
    }
}
```

### Custom transformer with FunctionTransformer

```rust
use datarust::function_transformer::FunctionTransformer;

fn log_transform(x: &Matrix) -> Result<Matrix> {
    let out: Vec<Vec<f64>> = x.rows_ref()
        .iter()
        .map(|row| row.iter().map(|&v| v.ln()).collect())
        .collect();
    Matrix::new(out)
}

fn exp_transform(x: &Matrix) -> Result<Matrix> {
    let out: Vec<Vec<f64>> = x.rows_ref()
        .iter()
        .map(|row| row.iter().map(|&v| v.exp()).collect())
        .collect();
    Matrix::new(out)
}

let mut ft = FunctionTransformer::new(log_transform)
    .with_inverse(exp_transform);

let x = Matrix::new(vec![vec![1.0, 10.0], vec![100.0, 1000.0]])?;
let log_x = ft.fit_transform(&x)?;
let back = ft.inverse_transform(&log_x)?;
// back ≈ x
```

### Pipeline ergonomics: step inspection and replacement

```rust,ignore
use datarust::decomposition::{PCA, PCAComponents};
use datarust::pipeline::Pipeline;
use datarust::scaler::{StandardScaler, RobustScaler};
use datarust::transformer_kind::TransformerKind;

let mut pipe = Pipeline::new()
    .push("scale", TransformerKind::StandardScaler(StandardScaler::new()))
    .push("reduce", TransformerKind::PCA(PCA::new(PCAComponents::Count(5))));

// Inspect step names
for name in pipe.names() {
    println!("Step: {}", name);
}

// Replace the scaler with a robust alternative
pipe.set_step("scale", TransformerKind::RobustScaler(RobustScaler::new()));

// Mutably access the PCA step to change parameters
if let TransformerKind::PCA(pca) = pipe.get_step_mut("reduce").unwrap() {
    // pca configuration can be modified here (if we had setter methods)
}

// Remove and insert steps dynamically
pipe.remove_step("reduce");
pipe.insert_step(1, "select", TransformerKind::SelectKBest(SelectKBest::new(ScoreFunc::FClassif, 3)?));
```

### Sparse inverse transform with OneHotEncoder

```rust
use datarust::encoder::OneHotEncoder;
use datarust::StrMatrix;

let s = StrMatrix::from_column(["Red", "Blue", "Green", "Red", "Blue"])?;
let mut ohe = OneHotEncoder::new();

// Two round-trip paths: dense → StrMatrix and sparse → StrMatrix
let dense = ohe.fit_transform(&s)?;
let from_dense = ohe.inverse_transform(&dense)?;

let sparse = ohe.transform_sparse(&s)?;
let from_sparse = ohe.inverse_transform_sparse(&sparse)?;

for i in 0..s.nrows() {
    assert_eq!(from_dense.get(i, 0), s.get(i, 0));
    assert_eq!(from_sparse.get(i, 0), s.get(i, 0));
}
```

### QuantileTransformer with NaN rejection

```rust
use datarust::scaler::{QuantileTransformer, OutputDistribution};
use datarust::Matrix;

let x = Matrix::new(vec![
    vec![1.0, f64::NAN],
    vec![3.0, 4.0],
])?;

let mut qt = QuantileTransformer::new(1000)?;
let result = qt.fit_transform(&x);
assert!(result.is_err());  // NaN input is rejected with InvalidInput
```

### Pipeline with feature names

```rust,ignore
use datarust::pipeline::Pipeline;
use datarust::scaler::StandardScaler;
use datarust::selection::VarianceThreshold;
use datarust::decomposition::PCA;
use datarust::decomposition::PCAComponents;
use datarust::transformer_kind::TransformerKind;
use datarust::traits::FeatureNames;

let mut pipe = Pipeline::new()
    .push("scale", TransformerKind::StandardScaler(StandardScaler::new()))
    .push("filter", TransformerKind::VarianceThreshold(VarianceThreshold::new(0.1)?))
    .push("pca", TransformerKind::PCA(PCA::new(PCAComponents::Variance(0.95))));

pipe.fit(&x)?;

// Feature names propagate through the entire pipeline
let input_names = &["age", "salary", "bonus", "years_exp"];
let names = pipe.feature_names_out(Some(input_names));
// e.g. ["pca0", "pca1", "pca2"] — depends on variance threshold + PCA
println!("output columns: {:?}", names);
```
