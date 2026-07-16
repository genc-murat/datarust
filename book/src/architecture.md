# Architecture

How datarust is organized and the design decisions behind it.

## Design philosophy

1. **Zero external dependencies by default.** `cargo add datarust` pulls in nothing. All linear algebra — Jacobi eigendecomposition, one-sided Jacobi SVD, Cholesky factorization, coordinate descent, IRLS — is pure Rust. Feature flags (`serde`, `rayon`, `matrixmultiply`) opt in to extras.

2. **sklearn-inspired but Rust-native.** The familiar `fit`/`transform`/`predict` API, but using `Result` (panic-free), builder-pattern config, and enum-based parameters (`Norm::L2`, `ImputeStrategy::Median`) instead of strings.

3. **Type-safe modality separation.** Four traits enforce that numeric, categorical, target, and label data go to the right transformer. The compiler catches misuse.

4. **Composability via type-erased enums.** `TransformerKind`, `CategoricalTransformerKind`, and `TargetTransformerKind` allow heterogeneous pipelines that are still serializable.

## Module layout

```
src/
├── lib.rs                 # Crate root: re-exports + module declarations
├── error.rs               # DatarustError enum + Result alias
├── traits.rs              # Transformer, Regressor, CategoricalTransformer, TargetTransformer, LabelTransformer, FeatureNames
├── matrix.rs              # Matrix (f64 flat), StrMatrix, SparseMatrix (CSR)
├── stats.rs               # Column statistics, covariance/correlation
├── pipeline.rs            # Sequential Pipeline
├── transformer_kind.rs    # TransformerKind enum (type erasure)
├── categorical_kind.rs    # CategoricalTransformerKind enum
├── target_kind.rs         # TargetTransformerKind enum
├── function_transformer.rs
├── polynomial.rs          # PolynomialFeatures
├── serialize.rs           # JSON save/load (serde feature)
├── linalg/
│   └── cholesky.rs        # Cholesky decomposition + SPD solver
├── scaler/                # 9 transformers (standard, minmax, robust, ...)
├── encoder/               # 5 encoders (onehot, ordinal, label, target, frequency)
├── imputer/               # SimpleImputer, KnnImputer
├── selection/             # VarianceThreshold, SelectKBest
├── decomposition/         # PCA, TruncatedSVD, Jacobi, randomized_svd
├── linear_model/          # LinearRegression, Ridge, Lasso, LogisticRegression
├── metrics/
│   ├── regression.rs      # MSE, MAE, R², max_error, explained_variance
│   └── classification.rs  # accuracy, precision, recall, F1, confusion_matrix, log_loss
├── model_selection/
│   ├── split.rs           # train_test_split
│   ├── kfold.rs           # KFold, StratifiedKFold
│   ├── cross_val.rs       # cross_val_score
│   └── rng.rs             # shared xorshift64 PRNG
└── compose/
    ├── column_transformer.rs
    └── output.rs
```

## Trait hierarchy

| Trait | Data flow | Implementors |
|---|---|---|
| `Transformer` | `Matrix → Matrix` | all scalers, PCA, TruncatedSVD, PolynomialFeatures, VarianceThreshold, SelectKBest, imputers, FunctionTransformer |
| `Regressor` | `fit(X, y)` + `predict(X) → Vec<f64>` | LinearRegression, Ridge, Lasso, LogisticRegression |
| `CategoricalTransformer` | `StrMatrix → Matrix` | OneHotEncoder, OrdinalEncoder, FrequencyEncoder |
| `TargetTransformer` | `fit(StrMatrix, y)` | TargetEncoder |
| `LabelTransformer` | `&[String] ↔ Vec<usize>` | LabelEncoder |
| `FeatureNames` | output column names | every output-producing transformer |

## Solver infrastructure

Three distinct solver families, all pure-Rust, in `linalg/`:

1. **Cholesky** (`linalg::cholesky`) — symmetric positive-definite system solver. Used by `LinearRegression`, `Ridge`, and `LogisticRegression` (per IRLS iteration).
2. **Coordinate descent** (`Lasso`) — soft-thresholding iteration for L1-regularized problems.
3. **IRLS** (`LogisticRegression`) — Newton-Raphson on the logistic loss, solving a weighted least-squares system each iteration.

All three are backed by the shared `Matrix::matmul` for forming Gram matrices, which dispatches to a tuned GEMM under the `matrixmultiply` feature.

## Error handling

Hand-rolled `DatarustError` enum (no `anyhow`/`thiserror`, consistent with the zero-dependency ethos). Every fallible public API returns `Result<T, DatarustError>`. The variant set is ML-domain-specific (`NotFitted`, `UnknownCategory`, `Singular`, etc.), more informative than a generic error blob.
