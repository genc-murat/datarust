# datarust

**Scikit-Learn Preprocessing in Rust** — a modular, dependency-free machine-learning preprocessing and modeling library built on a lightweight `Matrix` type.

[![crates.io](https://img.shields.io/crates/v/datarust.svg)](https://crates.io/crates/datarust)
[![docs.rs](https://docs.rs/datarust/badge.svg)](https://docs.rs/datarust)
[![CI](https://github.com/genc-murat/datarust/actions/workflows/ci.yml/badge.svg)](https://github.com/genc-murat/datarust/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

```rust
use datarust::scaler::StandardScaler;
use datarust::traits::Transformer;
use datarust::Matrix;

let x = Matrix::new(vec![
    vec![1.0, 10.0],
    vec![2.0, 20.0],
    vec![3.0, 30.0],
    vec![4.0, 40.0],
])?;

// Standardize: (x - mean) / std (population, ddof=0)
let mut scaler = StandardScaler::new();
let standardized = scaler.fit_transform(&x)?;
```

> **Default build has zero external dependencies.** All linear algebra (eigenvalue decomposition, covariance, Cholesky factorization, coordinate descent) is implemented in pure Rust — no system BLAS/LAPACK, no Python runtime, no GIL.

---

## What's included

| Category | Components |
|---|---|
| **Scalers** | StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer (L1/L2/Max), Binarizer |
| **Discretizers** | KBinsDiscretizer (Uniform / Quantile / KMeans), Binarizer |
| **Distribution Transformers** | QuantileTransformer (Uniform / Normal output), PowerTransformer (Yeo-Johnson / Box-Cox) |
| **Encoders** | LabelEncoder, OneHotEncoder (+ CSR sparse output), OrdinalEncoder, TargetEncoder, FrequencyEncoder |
| **Imputers** | SimpleImputer (mean / median / most_frequent / constant), KnnImputer (uniform / distance) |
| **Polynomial** | PolynomialFeatures (degree, interaction_only, include_bias) |
| **Selection** | VarianceThreshold, SelectKBest (ANOVA F / Chi2 / Mutual Information) |
| **Decomposition** | PCA (whiten, inverse_transform, randomized SVD), TruncatedSVD |
| **Linear Models** | LinearRegression (Cholesky & SVD), Ridge (L2), Lasso (L1, coordinate descent, sparse) |
| **Classification** | LogisticRegression (binary, IRLS solver, Cholesky & SVD) |
| **Metrics** | Regression: MSE/RMSE, MAE, R², max_error, explained_variance. Classification: accuracy, precision, recall, F1, confusion_matrix, log_loss |
| **Model Selection** | train_test_split, KFold, StratifiedKFold, cross_val_score |
| **Pipeline** | Sequential Pipeline (serde-serializable), ColumnTransformer (numeric + categorical) |
| **Feature Names** | `FeatureNames` trait on all transformers for output column names |
| **Serialization** | JSON save/load via optional `serde` feature |
| **Parallelism** | Rayon-backed column operations via optional `rayon` feature |
| **Sparse** | CSR `SparseMatrix` type for memory-efficient one-hot output |

---

## Why datarust?

### Zero dependencies by default

`cargo add datarust` pulls in **nothing**. No BLAS, no LAPACK, no numpy, no Python runtime. The default build has a completely empty dependency tree — every algorithm (Jacobi eigendecomposition, Cholesky factorization, one-sided Jacobi SVD, coordinate descent, IRLS) is pure Rust. Opt in to `serde`, `rayon`, or a tuned pure-Rust GEMM via feature flags when you need them.

### Rust-native type safety

sklearn's single duck-typed `fit`/`transform` API becomes **four separate traits** enforced by the type system:

- [`Transformer`](https://docs.rs/datarust/latest/datarust/trait.Transformer.html) — numeric `Matrix → Matrix`
- [`Regressor`](https://docs.rs/datarust/latest/datarust/trait.Regressor.html) — supervised `fit(X, y)` + `predict(X)`
- [`CategoricalTransformer`](https://docs.rs/datarust/latest/datarust/trait.CategoricalTransformer.html) — `StrMatrix → Matrix`
- [`LabelTransformer`](https://docs.rs/datarust/latest/datarust/trait.LabelTransformer.html) — 1-D `&[String] ↔ Vec<usize>`

You cannot accidentally pass categorical strings to a numeric scaler — the compiler catches it.

### Panic-free public API

Every fallible public method returns `Result<T, DatarustError>`. No hidden panics on bad input, no `unwrap()` in the API surface.

### JSON serialization (not pickle)

Fitted models serialize to plain JSON via the `serde` feature — human-readable, language-agnostic, and diff-friendly. No binary pickle blobs.

### Measured speedups vs scikit-learn

On heterogeneous `ColumnTransformer` composition, categorical encoding, and numeric scaling, datarust is **1.5–620× faster** than scikit-learn. See the [Performance](./performance.md) page for the full benchmark tables and methodology.

---

## A taste: end-to-end pipeline

```rust
use datarust::compose::{ColumnTransformer, Table};
use datarust::encoder::OneHotEncoder;
use datarust::categorical_kind::CategoricalTransformerKind;
use datarust::scaler::StandardScaler;
use datarust::transformer_kind::TransformerKind;
use datarust::Matrix;

// Mixed numeric + categorical table
let numeric = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
let table = Table::new(numeric, /* categorical StrMatrix */ /* ... */ /* ) */;

// Build a column transformer: scale numeric, one-hot encode categorical
let mut ct = ColumnTransformer::new()
    .add_numeric("num", vec![0, 1], TransformerKind::StandardScaler(StandardScaler::new()));
    // .add_categorical("cat", vec![2, 3], CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new()));

let out = ct.fit_transform_to_table(&table)?;
// out.numeric and out.categorical hold the transformed columns
```

---

## Next steps

- **New to datarust?** Start with the [Quick Start](./quickstart.md).
- **Want the full feature list vs sklearn?** See the [Feature Comparison](./comparison.md).
- **Looking for a specific module?** Browse the [Module Guide](./guide/scalers.md).
- **Need the API reference?** It's on [docs.rs](https://docs.rs/datarust).
