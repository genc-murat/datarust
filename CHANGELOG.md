# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-07-22

The "Core ML foundations" release. This version closes the most painful gaps
in the classifier and clustering story: multiclass (softmax) logistic
regression, k-means clustering, a full multiclass metric suite, and the
hyperparameter-introspection plumbing that future releases will build on.

### Added

- **`roc_auc_score`** and **`average_precision_score`** (`src/metrics/classification.rs`):
  ROC-AUC via the rank-based Mann–Whitney U equivalence (with tie handling)
  and average precision via the step-function PR-curve approximation. Both
  consume `predict_proba` output and are the standard binary-classification
  ranking metrics.
- **`cohen_kappa_score`** and **`matthews_corrcoef`**
  (`src/metrics/classification.rs`): Cohen's kappa (chance-corrected agreement,
  binary + multiclass) and Matthews correlation coefficient (balanced binary
  classification measure, robust to class imbalance).
- **`silhouette_score`** (`src/cluster/metrics.rs`): mean silhouette coefficient
  for clustering evaluation without ground truth, mirroring
  `sklearn.metrics.silhouette_score`.
- **`Params` trait and `ParamValue` enum** (`src/traits.rs`): opt-in
  hyperparameter introspection (`get_params` / `set_params`) for estimators
  whose hyperparameters should be searchable. `KMeans` and `LogisticRegression`
  implement it; this is the foundation for `GridSearchCV` (planned v0.8).
- **Multiclass `LogisticRegression`** (`src/linear_model/logistic_regression.rs`):
  targets with more than two classes (`{0, 1, 2, …}`) are now fit via
  multinomial (softmax) logistic regression with Newton-Raphson on the
  cross-entropy loss. `predict` returns the highest-probability class label;
  `predict_proba` returns an `(n, k)` probability matrix. The last class is the
  reference (coefficients zero); a tiny Levenberg–Marquardt ridge and SVD
  fallback keep the solver stable under near-separation. Binary `{0, 1}` targets
  keep using the existing fast IRLS path. New accessors: `classes()`,
  `coef() -> &[Vec<f64>]` (one row per class), `intercept() -> &[f64]`.
- **Multiclass classification metrics** (`src/metrics/classification.rs`):
  `confusion_matrix` now returns a general `n_classes × n_classes`
  `Vec<Vec<usize>>` (was `[[usize; 2]; 2]`). `precision_score`, `recall_score`,
  and `f1_score` auto-detect binary vs multiclass input and apply macro-averaging
  for multiclass labels. `accuracy_score` uses exact integer-label comparison
  instead of the binary ≥ 0.5 threshold. Binary `{0, 1}` inputs remain fully
  backward-compatible.
- **`Clusterer` trait** (`src/traits.rs`): the unsupervised counterpart to
  `Predictor`. Exposes `fit(&Matrix)`, `predict(&Matrix) -> Result<Vec<usize>>`
  (cluster indices, not regression targets), `fit_predict`, `fit_transform`
  (default one-hot encoding of assignments), `n_clusters`, and `is_fitted`.
  The trait derives from `Estimator`, not `Predictor`, because clustering takes
  no target `y`.
- **`cluster` module** (`src/cluster/`): the new home for clustering
  estimators, parallel to `linear_model` / `decomposition`.
- **`KMeans`** (`src/cluster/kmeans.rs`): k-means clustering via Lloyd's
  algorithm with k-means++ initialization. Mirrors `sklearn.cluster.KMeans`:
  builder API (`with_n_clusters`, `with_init`, `with_max_iter`, `with_tol`,
  `with_n_init`, `with_random_state`), fitted state (`cluster_centers`,
  `labels`, `inertia`, `n_iter`), k-means++ and `Random` initialization
  strategies (`KMeansInit`), `n_init` restarts keeping the lowest-inertia run,
  and a deterministic `random_state`. Serde-serializable under the `serde`
  feature; ships with unit, integration, and doctest coverage.

### Changed

- **`LogisticRegression::coef()`** now returns `&[Vec<f64>]` (one row per class)
  instead of `&[f64]`. Binary models return a single-row slice.
- **`LogisticRegression::intercept()`** now returns `&[f64]` instead of `f64`.
- **`confusion_matrix`** return type changed from `[[usize; 2]; 2]` to
  `Vec<Vec<usize>>` to support arbitrary class counts. Existing binary call
  sites must update from `cm[0][0]` array access to `cm[0][0]` `Vec` access
  (semantically identical, type changes).
- **`precision_score` / `recall_score` / `f1_score`** on binary input now return
  the macro-averaged value across both classes, not the positive-class-only
  value. For binary `{0, 1}` inputs this changes results: e.g. on the classic
  `cm = [[2,1],[1,3]]` example the metrics now report `17/24 ≈ 0.708` instead of
  `0.75`. Use the per-class helper (internal `per_class`) if positive-class-only
  semantics are required.

## [0.5.0] - 2026-07-21

The estimator-contract release. This version makes the first supervised model
and pipeline APIs behave like their scikit-learn counterparts rather than
treating a classifier as a regressor.

### Added
- **`Estimator`, `Predictor`, `Classifier`, and `PredictProba` traits** —
  `Predictor` unifies supervised `fit(X, y)` / `predict(X)` flows;
  `Classifier` represents hard-label predictors; `PredictProba` returns an
  `(n_samples, n_classes)` probability matrix. `Regressor` remains the
  regression semantic marker and default-R² scorer.
- **`SupervisedPipeline<E>`** — a generic, serde-serializable pipeline with
  zero or more `TransformerKind` preprocessing steps and a final
  `E: Predictor` estimator. Construct it with `Pipeline::with_estimator` or
  `SupervisedPipeline::new`.
- **Target-aware transformer fitting** — `Transformer::fit_with_target` has a
  safe unsupervised default. `SelectKBest` now consumes numeric labels through
  this path, so it can run within supervised pipelines without leaking fitted
  preprocessing across a caller-managed training split.

### Changed
- **BREAKING: `LogisticRegression` is a `Classifier`, not a `Regressor`.**
  `predict` now returns hard `0.0` / `1.0` labels. `predict_proba` now returns
  a two-column `Matrix` in `[P(class=0), P(class=1)]` order;
  `predict_positive_proba` provides the former one-vector positive-class view.
  `predict_class` remains as a compatibility alias.
- **BREAKING: import `Predictor` for `fit`, `predict`, `fit_predict`, and
  `is_fitted`.** `Regressor` and `Classifier` express model semantics and
  scoring; `Predictor` owns the shared fitting API.
- **BREAKING: custom trait implementors need the base contract.** Types that
  implement `Transformer`, `CategoricalTransformer`, `TargetTransformer`, or
  `LabelTransformer` must also implement `Estimator`. Custom regressors must
  implement `Estimator` and `Predictor` before implementing `Regressor`.
- `cross_val_score` now accepts any `Predictor + Clone`, so classifiers are
  evaluated with class predictions rather than probability values.

## [0.4.1] - 2026-07-17

A small additive release. The `stats` module now has single-slice (1-D)
counterparts of its column statistics, so a flat `&[f64]` no longer needs to be
wrapped as a `vec![vec![...]]` matrix to summarise it.

### Added
- **1-D statistics in `stats`** — plain-`&[f64]` counterparts of the existing `column_*` functions:
  - `sum`, `mean`, `min`, `max` — scalar reductions. `mean` returns `NaN` on an empty slice (numpy parity); `sum` returns `0.0`; `min`/`max` return their identity element (`+∞` / `−∞`) on empty input.
  - `variance(&[f64], ddof)`, `std(&[f64], ddof)` — with the same delta-degrees-of-freedom semantics as [`column_variance`]/[`column_std`] (`ddof = 0` population, `ddof = 1` sample); `NaN` when `ddof >= n`.
  - `median(&[f64]) -> Option<f64>` — sorts a copy internally (unlike [`median_sorted`], which assumes pre-sorted input); `None` on empty input.
  - `mode(&[f64]) -> Option<f64>` — most frequent value, ties broken by the smallest value (matching [`mode_column`]); `None` on empty input.

## [0.4.0] - 2026-07-16

The "ML toolkit" release. datarust grows from a preprocessing library into a full
scikit-learn-style toolkit: four linear-model estimators (regression +
classification), evaluation metrics, and model-selection infrastructure.

### Added
- **`model_selection` module** — model-evaluation infrastructure mirroring `sklearn.model_selection`:
  - `train_test_split` — split `X`/`y` into train/test with configurable `test_size` (default 0.25), `shuffle`, and `random_state`. Builder via `TrainTestSplit`.
  - `KFold` — K-fold cross-validation splitter producing `(train_indices, test_indices)` folds. Configurable `n_splits`, `shuffle`, `random_state`.
  - `StratifiedKFold` — preserves class balance across folds for binary classification.
  - `cross_val_score` — generic scorer that clones any `Regressor + Clone` estimator, fits each fold, and applies a user-supplied scorer (e.g. `r2_score` or `accuracy_score`). Works with all four shipped estimators.
  - Shared deterministic xorshift64 PRNG (`model_selection::rng`) — now used by both `model_selection` and `decomposition::randomized_svd`, eliminating a code duplication.
- **`linear_model::LogisticRegression`** — binary classification via IRLS (Iteratively Reweighted Least Squares / Newton-Raphson). Mirrors `sklearn.linear_model.LogisticRegression`. Each iteration solves a weighted least-squares system `(XᵀWX) β = XᵀWz` via the shared Cholesky (default) or SVD solver. `predict` returns `P(y=1|x)` in `[0,1]`; `predict_class` thresholds at 0.5. Tunable via `max_iter` (100), `tol` (1e-4); exposes `n_iter()`. The crate's first classifier — opens the classification use case.
- **`metrics::classification`** module — `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix` (2×2), `log_loss` (cross-entropy). Mirrors `sklearn.metrics` with hand-verified parity.
- **`linear_model::Ridge`** — L2-regularized regression (`||Xβ − y||² + α‖β‖²`). Mirrors `sklearn.linear_model.Ridge` with `alpha`, `fit_intercept`, and two solvers:
  - `RidgeSolver::Cholesky` (default) — solves `(XᵀX + αI) β = Xᵀy` via the shared Cholesky solver. Because `α > 0` guarantees the system matrix is positive-definite, Ridge succeeds on rank-deficient / collinear inputs where `LinearRegression` would fail.
  - `RidgeSolver::Svd` — eigendecomposition pseudo-inverse path.
- **`linear_model::Lasso`** — L1-regularized regression (`(1/(2n))||Xβ − y||² + α‖β‖₁`), solved by **coordinate descent** with soft-thresholding. Mirrors `sklearn.linear_model.Lasso`. The L1 penalty drives irrelevant coefficients to **exactly zero**, yielding sparse models that perform implicit feature selection — the key behavioural difference from Ridge. Tunable via `alpha`, `max_iter`, `tol`; exposes `n_iter()`.
- **`linear_model::LinearRegression`** — ordinary least-squares regression, the crate's first `predict`-capable estimator. Mirrors `sklearn.linear_model.LinearRegression` with `fit_intercept` and two solvers:
  - `LinearSolver::Cholesky` (default) — solves the normal equations `XᵀX β = Xᵀy` via a pure-Rust Cholesky decomposition. Fast, zero-dependency; requires full column rank.
  - `LinearSolver::Svd` — eigen-decomposition-based pseudo-inverse, stable for rank-deficient / collinear inputs.
  - Accessors: `coef()`, `intercept()`, `n_features_in()`, `score()` (R² of prediction).
- **`Regressor` trait** (`src/traits.rs`) — the supervised counterpart of `Transformer`: `fit(&mut self, X, y)`, `predict(&self, X) -> Vec<f64>`, `fit_predict`, `is_fitted`. Foundation for Ridge / Lasso / LogisticRegression.
- **`linalg::cholesky`** module — `cholesky_decompose` and `solve_spd` / `solve_spd_system` for symmetric positive-definite systems. Shared solver foundation for future linear models.
- **`metrics::regression`** module — `mean_squared_error` (with `squared=false` → RMSE), `mean_absolute_error`, `r2_score`, `max_error`, `explained_variance_score`. Mirrors `sklearn.metrics` with verified parity on reference inputs.
- `PCASolver` enum (`Auto` / `Full` / `Randomized`) on `PCA`, selectable via `PCA::solver(...)`. `Randomized` uses a Halko–Martinsson–Tropp randomized SVD (`src/decomposition/randomized_svd.rs`) — the fast path for tall-and-wide, low-rank inputs. `Auto` (default) keeps the exact eigensolver paths while the oversampling edge case is verified.
- `jacobi::eigh_topk_flat` — power-iteration + deflation that computes only the top-`k` eigenpairs of a flat symmetric matrix in `O(k·n²·iters)` instead of `O(n³·sweeps)`. Used by `PCA` when `n_components` is small.
- `pca::matmul_flat` — shared flat matmul helper (GEMM when `matrixmultiply` is enabled, scalar otherwise), now used by `PCA` and `TruncatedSVD` transforms.
- `stats::covariance_centered_flat` — flat-storage centered covariance (GEMM-backed when `matrixmultiply` is enabled).

### Changed
- **PCA / TruncatedSVD eigensolver now flat.** `jacobi::eigh_flat` operates on a single contiguous buffer (better cache locality + auto-vectorisation) instead of `Vec<Vec<f64>>`. PCA 50 000 × 200 dropped from ~320 ms to ~104 ms with `matrixmultiply`.
- **PCA / TruncatedSVD transforms now flat + matmul.** `transform`/`inverse_transform` write directly into contiguous output buffers and dispatch to `matmul_flat` (GEMM when enabled), replacing the per-row `Vec` allocation and manual dot products.
- **TruncatedSVD `xtx` now GEMM-backed** when `matrixmultiply` is enabled (was a scalar nested loop).
- **Scaler `transform` rayon threshold.** `StandardScaler`/`MinMaxScaler`/`RobustScaler` now use the scalar loop below 4 096 rows and the `rayon` parallel path above it. This fixes a regression where the rayon build was slower than the default build on small inputs (e.g. 1 000 × 10 `StandardScaler`: 0.36 ms → 0.016 ms, a 22× improvement on the rayon build).

### Performance
Measured on Apple M5 Pro (18 cores, arm64), Rust 1.96 release, median of 11 runs after one warmup, `fit_transform` on deterministic synthetic data (seed 42):

| Workload (50 000 × 200) | 0.3.0 default | 0.3.1 default | 0.3.1 +rayon |
|---|---:|---:|---:|
| StandardScaler | 8.0 ms | 8.2 ms | **4.7 ms** |
| MinMaxScaler | 11.4 ms | 10.8 ms | **7.5 ms** |
| RobustScaler | 127 ms | 123 ms | **14.0 ms** |
| Pipeline (3 scalers) | 148 ms | 144 ms | **26.7 ms** |
| OneHotEncoder | 93 ms | 89 ms | **80 ms** |

| Workload (50 000 × 200) | 0.3.0 +matrixmultiply | 0.3.1 +matrixmultiply |
|---|---:|---:|
| PCA | 303 ms | **104 ms** |

The PCA gap vs scikit-learn narrowed from ~85× (0.2.0) to ~8× (0.3.1 +matrixmultiply); the randomized-SVD solver closes it further for low-rank inputs.

## [0.3.0] - 2026-07-03

### Added
- `matrixmultiply` feature: optional tuned pure-Rust GEMM (via the `matrixmultiply` crate, no system BLAS) for `Matrix::matmul` and centered-covariance computation. ~3.5× speedup on PCA at 50 000 × 200. The default build remains zero-external-dependency.
- `Matrix::as_slice()` / `Matrix::as_mut_slice()` — flat row-major accessors for cache-friendly, auto-vectorizable numeric loops.
- `stats::column_mean_var_flat`, `stats::column_min_max_flat`, `stats::column_quantiles_many_flat` — flat-storage counterparts of the fused statistics functions, used by the scalers.
- `stats::column_mean_flat` — single-pass flat column mean.
- `stats::column_mean_var` — Welford single-pass mean+variance (replaces the 3-pass `column_mean` + `column_variance` pair in scaler `fit`).
- `stats::column_min_max` — fused single-pass min+max (replaces the 2-pass separate calls).
- `stats::column_quantiles_many` — multiple quantiles from a single sort per column (replaces the 3× redundant sort in `RobustScaler`).

### Changed
- **BREAKING:** `Matrix` internal storage switched from `Vec<Vec<f64>>` (one heap allocation per row) to a single contiguous `Vec<f64>` + `(rows, cols)`. This is the dominant performance win for large dense inputs: ~13× on `RobustScaler`, ~5× on `StandardScaler`, ~8× on `Pipeline` at 50 000 × 200 (with `rayon`).
- **BREAKING:** `Matrix::rows_ref()` now returns `Vec<Vec<f64>>` (owned, allocating) instead of `&Vec<Vec<f64>>`, and is marked `#[doc(hidden)]`. Prefer `Matrix::as_slice()` / `Matrix::iter_rows()` in new code; `rows_ref` is retained only for transitional compatibility and will be removed in a future release.
- **BREAKING:** `Matrix::into_rows()` is marked `#[doc(hidden)]`; prefer `as_slice` / `from_flat`.
- `StandardScaler`, `MinMaxScaler`, `RobustScaler` `transform`/`inverse_transform` now write directly into flat output buffers with stride-1 reads (replacing per-row `Vec` allocation).
- `RobustScaler::fit` gathers and sorts each column once for q1/median/q3 (previously three separate gather+sort passes).
- NaN validation is fused into the scaler transform loops (no separate `validate_no_nan` pass).
- `Matrix::from_flat` now stores the flat buffer directly instead of re-chunking into rows.
- `Matrix::get` uses `get_unchecked` after a `debug_assert!` on the hot path.
- `Matrix::matmul` and centered-covariance dispatch to `matrixmultiply::dgemm` when the feature is enabled.

### Performance
Measured on Apple M5 Pro (18 cores, arm64), Rust 1.96 release, median of 15 runs after one warmup, `fit_transform` on deterministic synthetic data (seed 42):

| Workload (50 000 × 200) | 0.2.0 | 0.3.0 (default) | 0.3.0 (+rayon) | 0.3.0 (+matrixmultiply) |
|---|---:|---:|---:|---:|
| StandardScaler | 115 ms | 8.4 ms | 4.6 ms | — |
| MinMaxScaler | 81 ms | 12.2 ms | 7.3 ms | — |
| RobustScaler | 459 ms | 137 ms | 14.6 ms | — |
| Pipeline (3 scalers) | 662 ms | 152 ms | 28 ms | — |
| PCA | 1056 ms | 1008 ms | 1028 ms | **303 ms** |
| OneHotEncoder | 88 ms | 98 ms | 88 ms | — |

For the full sklearn comparison table and methodology, see the "Performance: datarust vs scikit-learn" section of the README.

### Fixed
- `validate_no_nan` correctly reports the flat buffer index → (row, col) position.
- `transform_to_table` no longer constructs an invalid zero-column `Matrix` when only categorical columns are present (builds a dummy `nrows × 1` matrix to satisfy the `Table` row-count invariant).

## [Unreleased]

### Added
- `ImputeStrategy` now derives `Default` (default variant `Mean`), for consistency with the other config enums (`BinStrategy`, `Norm`, `HandleUnknown`, `OrdinalCategories`).
- `examples/bench_compare_rust.rs` and `benches/compare_sklearn.py` — mirrored Rust/Python harnesses for the README performance comparison.
- README "Performance: datarust vs scikit-learn" section with measured median `fit_transform` times across Standard/MinMax/Robust scalers, PCA, Pipeline, OneHotEncoder and ColumnTransformer at three dataset sizes, plus methodology notes and a non-throughput advantages summary.

### Changed
- README installation/feature examples now reference `0.2` instead of the stale `0.1`.

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
