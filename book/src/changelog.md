# Changelog

All notable changes to datarust are documented in the project's [`CHANGELOG.md`](https://github.com/genc-murat/datarust/blob/main/crates/datarust/CHANGELOG.md) on GitHub.

> This page provides a summary. For the full, detailed changelog (including internal refactors and performance tables), see the [canonical source](https://github.com/genc-murat/datarust/blob/main/crates/datarust/CHANGELOG.md).

## Unreleased

## 0.6.5

- Restored enforceable Rust 1.70 compatibility by pinning MSRV-compatible
  Rayon and serde_json dependency lines.
- Hardened deserialized categorical encoder state against mismatched category
  lists, index maps, and output widths.
- Rejected fractional ordinal codes during inverse transformation.
- Completed fitted-state checks for KNN/Simple imputers, MaxAbsScaler, and
  QuantileTransformer.
- Aligned PCA solver and component-count documentation with the implemented
  behavior.

## 0.6.4

- Silhouette scoring compacts arbitrary cluster IDs, handles singleton
  clusters correctly, and rejects invalid one-cluster-per-sample input.
- Numerical estimators, decompositions, scalers, selectors, and regression
  metrics reject non-finite observations; imputers retain `NaN` missing-value
  support while rejecting infinity.
- Non-finite scaler, selector, encoder, and SVD configuration values fail
  before fitting.
- Inconsistent fitted state loaded from JSON returns a recoverable error instead
  of panicking during transform or prediction.
- CI now builds and validates the docs/blog site and denies rustdoc warnings
  explicitly.

## 0.6.3

- Ridge, Lasso, logistic regression, and KMeans now reject non-finite or
  otherwise invalid solver hyperparameters before optimization.
- `Params::set_params` validation no longer mutates KMeans or logistic
  regression when a candidate is invalid.
- CI now executes the intended OS × feature matrix, including datasets and an
  all-features build, and verifies the crates.io package.
- The published crate excludes the blog, book, website, npm, and Cloudflare
  deployment sources, reducing the compressed archive from about 3.0 MiB to
  289 KiB.

## 0.6.2

- Safe dense access now checks bounds in release builds, while dense
  allocations reject dimension overflow.
- Raw and deserialized CSR matrices validate pointer ranges and sorted column
  indices; duplicate triplets are summed.
- Train/test and cross-validation reject undersized or mismatched inputs, and
  fractional test sizes now round up consistently.
- `StratifiedKFold` supports validated, non-contiguous multiclass labels.
- Binary log loss, ROC-AUC, and average precision validate labels and numeric
  inputs. New `*_with_positive_label` functions support label spaces such as
  `{2, 5}`, and average-precision ties are threshold-grouped.

## 0.6.1

- Classification labels are now compacted safely, so metrics and
  `LogisticRegression` support non-contiguous labels such as `{2, 5, 9}` while
  preserving original predictions and probability-column mappings.
- Precision, recall, and F1 now expose explicit binary, macro, weighted, and
  micro averaging, configurable zero-division handling, and per-class reports.
- The optional `datasets` feature adds embedded Iris, Breast Cancer, Wine, and
  Diabetes datasets with no runtime file or network dependency.

## 0.5.0

- **Estimator hierarchy** — `Estimator`, `Predictor`, `Classifier`, and
  `PredictProba` now define common supervised flows. `Regressor` is reserved
  for regression semantics.
- **SupervisedPipeline** — `Pipeline::with_estimator` combines preprocessing,
  target-aware feature selection, and a final predictor in a cloneable,
  serde-serializable pipeline.
- **BREAKING: LogisticRegression classification semantics** — `predict`
  returns hard labels; `predict_proba` returns a two-column probability matrix;
  `predict_positive_proba` exposes the positive-class vector.
- **BREAKING: Predictor imports** — import `Predictor` to call shared
  `fit`/`predict` methods on supervised models.
- **BREAKING: Custom trait implementors** — custom transformers and encoders
  must implement `Estimator`; custom regressors must implement `Estimator` and
  `Predictor` before `Regressor`.

## 0.4.0

### Added
- **`model_selection` module** — `train_test_split`, `KFold`, `StratifiedKFold`, `cross_val_score`. Shared deterministic xorshift64 PRNG, now used by both `model_selection` and `decomposition::randomized_svd`.
- **`LogisticRegression`** — binary classification via IRLS (Cholesky/SVD). The crate's first classifier.
- **`metrics::classification`** — `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `log_loss`.
- **`Ridge`** — L2-regularized regression (Cholesky/SVD). Succeeds on collinear inputs.
- **`Lasso`** — L1-regularized regression via coordinate descent. Produces sparse models (feature selection).
- **`Regressor` trait** — supervised counterpart of `Transformer` (`fit(X, y)` + `predict`).
- **`linalg::cholesky`** module — shared SPD solver foundation.
- **`LinearRegression`** — OLS regression, the crate's first estimator.
- **`metrics::regression`** — MSE/RMSE, MAE, R², max_error, explained_variance.
- `PCASolver` enum (`Auto` / `Full` / `Randomized`) on `PCA`.
- `jacobi::eigh_topk_flat` — power-iteration + deflation for top-k eigenpairs.
- Flat-storage Jacobi eigensolver, flat matmul helpers, flat covariance.

### Performance
- PCA 50 000 × 200 dropped from ~320 ms to ~104 ms with `matrixmultiply`.
- `LinearRegression` fit at 50 000 × 200: 258 ms → 84 ms with `matrixmultiply`.
- Scaler `transform` rayon threshold: scalar loop below 4 096 rows, parallel above.

## 0.3.0

- `matrixmultiply` feature: optional tuned pure-Rust GEMM (no system BLAS).
- **BREAKING:** `Matrix` internal storage switched from `Vec<Vec<f64>>` to a single contiguous `Vec<f64>`. ~13× on `RobustScaler`, ~5× on `StandardScaler` at 50 000 × 200.
- Flat-storage scalers, fused Welford statistics, NaN validation fused into transform loops.

## Earlier versions

See the [full changelog on GitHub](https://github.com/genc-murat/datarust/blob/main/crates/datarust/CHANGELOG.md) for 0.1.x and 0.2.x history.
