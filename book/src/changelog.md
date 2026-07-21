# Changelog

All notable changes to datarust are documented in the project's [`CHANGELOG.md`](https://github.com/genc-murat/datarust/blob/main/CHANGELOG.md) on GitHub.

> This page provides a summary. For the full, detailed changelog (including internal refactors and performance tables), see the [canonical source](https://github.com/genc-murat/datarust/blob/main/CHANGELOG.md).

## Unreleased

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

See the [full changelog on GitHub](https://github.com/genc-murat/datarust/blob/main/CHANGELOG.md) for 0.1.x and 0.2.x history.
