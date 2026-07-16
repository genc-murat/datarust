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
| Classification metrics (accuracy, F1, ...) | ✓ (6 metrics) | ✓ |

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
