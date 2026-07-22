# Feature Comparison: datarust vs sklearn

A side-by-side view of what's implemented. `✓` = supported, `—` = no equivalent.

## Preprocessing

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

## Encoders

| Transformer | datarust | sklearn |
|---|---|---|
| LabelEncoder | ✓ (handle_unknown: Error/Ignore) | ✓ |
| OrdinalEncoder | ✓ (auto + manual) | ✓ |
| OneHotEncoder | ✓ (drop, handle_unknown, sparse CSR) | ✓ |
| TargetEncoder | ✓ (smoothed mean, UnknownTarget) | ✓ |
| FrequencyEncoder | ✓ (count/proportion) | — |

## Imputation & Selection

| Component | datarust | sklearn |
|---|---|---|
| SimpleImputer | ✓ (mean/median/most_frequent/constant) | ✓ |
| KNN Imputer | ✓ (uniform/distance) | ✓ |
| PolynomialFeatures | ✓ (degree, interaction_only, bias) | ✓ |
| VarianceThreshold | ✓ | ✓ |
| SelectKBest | ✓ (F-classif / Chi2 / Mutual Info) | ✓ |

## Models & Decomposition

| Component | datarust | sklearn |
|---|---|---|
| LinearRegression | ✓ (Cholesky & SVD) | ✓ |
| Ridge | ✓ (L2, Cholesky & SVD) | ✓ |
| Lasso | ✓ (L1, coordinate descent) | ✓ |
| LogisticRegression | ✓ (IRLS, Cholesky & SVD) | ✓ |
| PCA | ✓ (Jacobi + randomized SVD, whiten, inverse) | ✓ |
| TruncatedSVD | ✓ (Count/Variance/All) | ✓ |

## Model Selection & Metrics

| Component | datarust | sklearn |
|---|---|---|
| train_test_split | ✓ | ✓ |
| KFold / StratifiedKFold | ✓ | ✓ |
| cross_val_score | ✓ | ✓ |
| Regression metrics (MSE, R², ...) | ✓ (5 metrics) | ✓ |
| Classification metrics (accuracy, F1, ROC-AUC, ...) | ✓ (10 metrics) | ✓ |
| Clustering metrics (silhouette) | ✓ | ✓ |
| Hyperparameter introspection (Params trait) | ✓ | — |

## Composition & Infrastructure

| Feature | datarust | sklearn |
|---|---|---|
| Pipeline | ✓ (TransformerKind, serde) | ✓ |
| ColumnTransformer | ✓ (numeric + categorical + target) | ✓ |
| FunctionTransformer | ✓ (optional inverse, closure-based) | ✓ |
| FeatureNames | ✓ (trait, all transformers) | ✓ |
| inverse_transform | ✓ (scalers, PCA, SVD, encoders) | ✓ |
| Pipeline Ergonomics | ✓ (get/set/insert/remove step) | — |
| Matrix Slicing | ✓ (select_columns, select_rows) | — |
| Covariance / Correlation | ✓ (ddof-configurable) | — |
| JSON Serialization | ✓ (serde feature) | — (joblib/pickle) |
| Sparse Output | ✓ (CSR via SparseMatrix) | ✓ |
| Parallelism | ✓ (rayon feature) | — (joblib) |

## Where datarust goes beyond sklearn

- **FrequencyEncoder** — count/proportion encoding not in sklearn.
- **Pipeline runtime editing** — `get_step`, `set_step`, `insert_step`, `remove_step` for live pipeline surgery.
- **JSON serialization** — human-readable, language-agnostic fitted-model persistence (vs sklearn's binary joblib/pickle).
- **Type-safe categorical/numeric separation** — the compiler prevents mixing `StrMatrix` and `Matrix`.

---

# Rust ecosystem comparison

datarust is a **preprocessing-first** classical ML library. Within the Rust
ecosystem its direct peers are **[smartcore]** and **[linfa]**. Deep-learning
frameworks ([candle], [burn], [tch-rs]) solve a different problem and are out of
scope here; [rusty-machine] has been archived and is omitted.

[smartcore]: https://crates.io/crates/smartcore
[linfa]: https://crates.io/crates/linfa
[candle]: https://crates.io/crates/candle-core
[burn]: https://crates.io/crates/burn
[tch-rs]: https://crates.io/crates/tch
[rusty-machine]: https://crates.io/crates/rusty-machine

## Design philosophies

The three libraries were built around different priorities, which shows up in
every design decision:

- **datarust** optimizes for **preprocessing depth and zero-dependency
  deployability.** All linear algebra is pure Rust (Jacobi eigensolver,
  Cholesky, coordinate descent), so the default build has *no* external C
  libraries — it links cleanly into WASM, embedded targets, and CLI tools. The
  trade-off is algorithm breadth: only the four linear models (Linear / Ridge /
  Lasso / Logistic regression) and `KMeans` clustering are implemented so far.
- **smartcore** optimizes for **single-crate algorithm breadth.** One dependency
  gives you SVM, RandomForest, DecisionTree, KMeans, DBSCAN, KNN, NaiveBayes,
  and more, plus model selection and metrics. The trade-off is preprocessing:
  only `StandardScaler` and `OneHotEncoder` are provided, and the crate depends
  on `ndarray` plus a BLAS backend.
- **linfa** optimizes for **modularity.** Each algorithm family lives in its own
  crate (`linfa-svm`, `linfa-trees`, `linfa-clustering`, `linfa-ensemble`, …)
  behind a shared `linfa-core` trait surface, so you only compile what you use.
  `linfa-preprocessing` additionally offers scalers and **text vectorizers**
  (Count / TF-IDF). The trade-off: categorical encoders, imputers, and feature
  selectors are sparse or undocumented, and Ridge/Lasso exist only through
  `linfa-elasticnet`'s `l1_ratio` parameter.

## Feature matrix

Verified against the July 2026 releases: smartcore 0.5.3, linfa 0.8.1.
Legend: `✓` present, `✗` confirmed absent, `?` not clearly documented at the
time of writing — please open an issue or PR if a cell goes stale.

### Preprocessing & Encoders

| Component | datarust | smartcore | linfa |
|---|---|---|---|
| StandardScaler | ✓ | ✓ | ✓ |
| MinMaxScaler | ✓ | ✗ | ✓ |
| RobustScaler | ✓ | ✗ | ? |
| MaxAbsScaler | ✓ | ✗ | ✓ |
| Normalizer (L1/L2/Max) | ✓ | ✗ | ✓ |
| KBinsDiscretizer | ✓ | ✗ | ? |
| QuantileTransformer | ✓ | ✗ | ? |
| PowerTransformer | ✓ | ✗ | ? |
| OneHotEncoder | ✓ | ✓ | ? |
| OrdinalEncoder | ✓ | ✗ | ? |
| LabelEncoder | ✓ | ✗ | ? |
| TargetEncoder | ✓ | ✗ | ✗ |
| FrequencyEncoder | ✓ | ✗ | ✗ |
| SimpleImputer | ✓ | ? | ? |
| KNN Imputer | ✓ | ? | ? |
| PolynomialFeatures | ✓ | ? | ? |
| VarianceThreshold | ✓ | ? | ? |
| SelectKBest | ✓ | ? | ? |
| Text vectorizers (Count/TF-IDF) | ✗ | ✗ | ✓ |

### Models & Decomposition

| Component | datarust | smartcore | linfa |
|---|---|---|---|
| LinearRegression | ✓ | ✓ | ✓ |
| Ridge (dedicated) | ✓ | ✗ | ✗ (via ElasticNet `l1_ratio=0`) |
| Lasso (dedicated) | ✓ | ✗ | ✗ (via ElasticNet `l1_ratio=1`) |
| LogisticRegression | ✓ | ✓ | ✓ |
| PCA | ✓ | ✓ | ✓ |
| TruncatedSVD | ✓ | ✗ | ✓ |
| SVM | ✗ | ✓ | ✓ |
| RandomForest / DecisionTree | ✗ | ✓ | ✓ |
| KMeans | ✓ (k-means++ init) | ✓ | ✓ |
| DBSCAN | ✗ | ✓ | ✓ |

### Infrastructure

| Feature | datarust | smartcore | linfa |
|---|---|---|---|
| Pipeline | ✓ | ? | ? |
| ColumnTransformer | ✓ | ? | ✗ |
| train_test_split | ✓ | ✓ | ? |
| KFold / StratifiedKFold | ✓ | ✓ | ? |
| cross_val_score | ✓ | ✓ | ? |
| Regression + Classification metrics | ✓ (15 metrics + silhouette) | ✓ | ✓ |
| JSON model serialization | ✓ (serde) | ? | ? |
| Zero external deps by default | ✓ | ✗ (ndarray + BLAS) | ✗ (ndarray + BLAS) |
| WASM-friendly (no native BLAS) | ✓ | ? | ? |
| Distribution model | single crate | single crate | per-algorithm crates |

## When to choose which

| Your priority | Reach for |
|---|---|
| Rich preprocessing (scalers, encoders, imputers, selection) | **datarust** |
| WASM / embedded / single-binary deployment, no BLAS | **datarust** |
| SVM, trees, DBSCAN, or the widest single-crate model zoo | **smartcore** |
| Modular compile-what-you-use algorithms, or text vectorizers | **linfa** |
| Deep learning (transformers, CNNs, autograd) | candle / burn / tch-rs |

## Complementary use

These libraries are **not mutually exclusive.** Because all three expose plain
`Vec`/matrix in/out interfaces, you can mix them in a single pipeline: use
datarust for preprocessing (where its coverage is unique), then hand the
transformed features to a smartcore or linfa estimator for an algorithm datarust
doesn't implement yet.
