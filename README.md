# datarust

**Scikit-Learn Preprocessing in Rust** — a modular, dependency-free data preprocessing library built on a lightweight `Matrix` type.

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
| **Encoders** | LabelEncoder, OneHotEncoder (+ CSR sparse output), OrdinalEncoder, TargetEncoder, FrequencyEncoder |
| **Imputers** | SimpleImputer (mean / median / most_frequent / constant), KnnImputer (uniform / distance) |
| **Polynomial** | PolynomialFeatures (degree, interaction_only, include_bias) |
| **Selection** | VarianceThreshold, SelectKBest (ANOVA F / Chi2 / Mutual Information) |
| **Decomposition** | PCA (with whiten, inverse_transform), TruncatedSVD |
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
datarust = "0.1"
```

### Optional features

```toml
[dependencies]
datarust = { version = "0.1", features = ["serde", "rayon"] }
```

- **`serde`** — enables JSON serialization/deserialization of fitted transformers via `datarust::serialize::{save_json, load_json, to_json, from_json}`.
- **`rayon`** — enables parallel column statistics and transforms for large datasets.

## Core Concepts

### Matrix

The fundamental data container is [`Matrix`](https://docs.rs/datarust/latest/datarust/struct.Matrix.html), a row-major `Vec<Vec<f64>>` with validation:

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

### Errors

Operations return `Result<T, DatarustError>` with variants for `NotFitted`, `InvalidInput`, `ShapeMismatch`, `EmptyInput`, `AllMissing`, `UnknownCategory`, `UnknownLabel`, `InvalidConfig`, and `Singular`.

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
use datarust::encoder::LabelEncoder;

let mut encoder = LabelEncoder::new();
encoder.fit(&["dog", "cat", "bird"])?;
let encoded = encoder.transform(&["dog", "bird"])?;
// encoded = [1, 2]
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
```

The CSR [`SparseMatrix`](#sparsematrix-csr) output stores only the `1.0` positions, saving significant memory for high-cardinality columns.

#### OrdinalEncoder

Encode categorical features as integer codes with optional user-defined ordering.

```rust
use datarust::encoder::{OrdinalEncoder, OrdinalCategories};

// Auto: sorted lexicographically
let mut enc = OrdinalEncoder::new();
let out = enc.fit_transform(&s)?;

// Manual: custom order
let mut enc = OrdinalEncoder::new()
    .categories(OrdinalCategories::Manual(vec![
        vec!["small", "medium", "large"],
    ]));
let out = enc.fit_transform(&s)?;
```

#### TargetEncoder

Replace categories with the smoothed mean of the target variable.

```rust
use datarust::encoder::TargetEncoder;

let mut te = TargetEncoder::new(5.0); // smoothing factor
te.fit(&categorical, &target)?;
let out = te.transform(&categorical)?;
```

#### FrequencyEncoder

Replace categories with their frequency (count or proportion).

```rust
use datarust::encoder::FrequencyEncoder;

let mut fe = FrequencyEncoder::new(true); // normalized = proportion
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
Does **not** center the data.

```rust
use datarust::decomposition::TruncatedSVD;

let mut svd = TruncatedSVD::new(5)?;
let out = svd.fit_transform(&x)?;
```

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

All 16 transformer types are available as `TransformerKind` variants, enabling type-safe heterogeneous pipelines.

### ColumnTransformer

Apply different transformers to different columns of a mixed numeric/categorical dataset.

```rust
use datarust::compose::{ColumnTransformer, Remainder, Table};
use datarust::encoder::OneHotEncoder;
use datarust::scaler::StandardScaler;
use datarust::transformer_kind::TransformerKind;

let table = Table::new(numeric, categorical)?;

let mut ct = ColumnTransformer::new()
    .remainder(Remainder::Passthrough)  // retain unselected columns
    .add_numeric("scale", vec![0, 1], TransformerKind::StandardScaler(StandardScaler::new()))
    .add_categorical("city", vec![0], OneHotEncoder::new());
let out = ct.fit_transform(&table)?;
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

## Serialization

Enable the `serde` feature for JSON save/load of fitted transformers.

```toml
datarust = { version = "0.1", features = ["serde"] }
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
datarust = { version = "0.1", features = ["rayon"] }
```

When enabled, the following use parallel iterators:

- **Statistics:** `column_mean`, `column_variance`, `column_min`, `column_max`, `column_median`, `column_mode`, `column_quantile`
- **Scalers:** StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
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
| LabelEncoder | ✓ | ✓ |
| OrdinalEncoder | ✓ (auto + manual) | ✓ |
| OneHotEncoder | ✓ (drop, handle_unknown, sparse CSR) | ✓ |
| TargetEncoder | ✓ (smoothed mean) | ✓ |
| FrequencyEncoder | ✓ (count/proportion) | — |
| SimpleImputer | ✓ (mean/median/most_frequent/constant) | ✓ |
| KNN Imputer | ✓ (uniform/distance) | ✓ |
| PolynomialFeatures | ✓ (degree, interaction_only, bias) | ✓ |
| VarianceThreshold | ✓ | ✓ |
| SelectKBest | ✓ (F-classif / Chi2 / Mutual Info) | ✓ |
| PCA | ✓ (Jacobi EV, count/variance/all, whiten) | ✓ |
| TruncatedSVD | ✓ (via X^T X eigen) | ✓ |
| Pipeline | ✓ (TransformerKind, serde) | ✓ |
| ColumnTransformer | ✓ (numeric + onehot, remainder passthrough) | ✓ |
| FeatureNames | ✓ (trait, all transformers) | ✓ |
| JSON Serialization | ✓ (serde feature) | — (joblib) |
| Sparse Output | ✓ (CSR via SparseMatrix) | ✓ |
| Parallelism | ✓ (rayon feature) | — (joblib) |

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
