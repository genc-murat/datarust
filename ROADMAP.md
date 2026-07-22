# Roadmap

This document tracks the path from the current release (v0.6.0) toward a
**complete scikit-learn-style ML toolkit** for Rust (v1.0). It is a living
document — priorities may shift, but the principles and the destination stay
fixed.

> **Where we are today (v0.6.0):** the deepest preprocessing coverage in the
> Rust ecosystem (18 transformers/encoders/imputers/selectors), four linear
> models with binary + multiclass (softmax) support, `KMeans` clustering, 15
> metrics (including ROC-AUC, PR-AUC, Cohen's kappa, Matthews corrcoef,
> silhouette), `Pipeline` + `ColumnTransformer`, cross-validation, the `Params`
> trait for hyperparameter introspection — all with zero external dependencies
> by default.

## Guiding principles

These are non-negotiable. Every item on the roadmap respects them.

1. **Zero dependencies by default.** The default build compiles with no
   external crates beyond `std`. Every algorithm ships as pure Rust — no
   BLAS, no LAPACK, no C/C++ runtime. This keeps datarust embeddable in WASM,
   embedded targets, CLI tools, and services without a toolchain headache.
2. **CPU-first.** GPU computing, distributed training, and deep learning are
   deliberately out of scope — [candle], [burn], and [tch-rs] already serve
   that space. datarust owns the *classical ML on CPU* niche.
3. **scikit-learn API parity.** `fit` / `transform` / `predict` /
   `feature_names_out` conventions are preserved so Python practitioners feel
   at home. Where Rust enables a better design (type-safe categorical/numeric
   separation, `Result`-based errors), we take it.
4. **Type safety and no panics.** Public APIs return `Result`; invalid input
   is a recoverable `DatarustError`, never a panic. `#![warn(missing_docs)]`
   and `#![warn(clippy::all)]` are enforced in CI.
5. **Opt-in weight.** Heavier capabilities stay behind feature flags
   (`serde`, `rayon`, `matrixmultiply`). Users who don't need them pay nothing.
6. **Measured against scikit-learn.** Every estimator ships with parity tests
   that assert agreement with known sklearn reference values, not just
   self-consistency.

[candle]: https://crates.io/crates/candle-core
[burn]: https://crates.io/crates/burn
[tch-rs]: https://crates.io/crates/tch

---

## Recently shipped (v0.6.0)

The v0.6 "Core ML foundations" release closed the most painful gaps in the
classifier and clustering story. All of its deliverables are now part of the
stable API:

- ✅ `Clusterer` trait + `KMeans` (Lloyd's algorithm, k-means++ initialization)
- ✅ Multiclass `LogisticRegression` (softmax Newton-Raphson)
- ✅ Multiclass metrics (n×n confusion matrix, macro-averaged precision/recall/F1)
- ✅ ROC-AUC, average precision, Cohen's kappa, Matthews correlation coefficient
- ✅ Silhouette score for clustering evaluation
- ✅ `Params` trait for hyperparameter introspection (foundation for GridSearchCV)

---

## Release track

Progress on each item can be tracked by the checkboxes below.

### v0.7 — Tree-based learning

> *Theme: the single most requested algorithm family.*

Trees and ensembles are the backbone of tabular ML. v0.7 introduces a
decision-tree kernel, then builds the two flagship ensembles on top of it. A
self-contained RNG abstraction (seedable, reproducible) is needed first — the
current private `xorshift64` in `model_selection` should be promoted to a
shared crate-internal utility.

**Deliverables:**

- [ ] `src/tree/` module: `DecisionTreeClassifier` and `DecisionTreeRegressor`
      (CART splitter; Gini / entropy / MSE criteria; max depth, min samples
      split, min samples leaf).
- [ ] `src/ensemble/` module:
  - [ ] `RandomForestClassifier` / `RandomForestRegressor` (bagging +
        random feature subsampling).
  - [ ] `ExtraTreesClassifier` / `ExtraTreesRegressor` (random splits).
  - [ ] `Bagging` meta-estimator (works with any `Predictor`).
- [ ] Tree-based `FeatureNames` support and `feature_importances_` output.
- [ ] Promote the internal PRNG to a shared, seedable utility so ensembles can
      build deterministic bootstrap samples.

### v0.8 — Model selection & text

> *Theme: the missing pieces of production ML workflows.*

Without hyperparameter search, text vectorization, and a couple more
classifiers, datarust cannot serve NLP or rigorous model-tuning workflows.
This phase also broadens sparse-matrix support, which text features depend on.

**Deliverables:**

- [ ] `GridSearchCV` and `RandomizedSearchCV` — built on the `Params` trait
      (`get_params` / `set_params`) from v0.6 and the existing `cross_val_score`
      loop. Must support pipeline parameter addressing (`step__param` naming).
- [ ] `validation_curve` and `learning_curve` diagnostics.
- [ ] `src/feature_extraction/text/` module:
  - [ ] `CountVectorizer` (tokenizer, n-grams, vocabulary).
  - [ ] `TfidfTransformer` and `TfidfVectorizer`.
  - [ ] `HashingVectorizer` (stateless).
- [ ] Sparse-matrix arithmetic: sparse × dense matrix multiplication and
      sparse + sparse addition (needed for TF-IDF output and large one-hot
      features). Today `SparseMatrix` is read-only CSR with no linear algebra.
- [ ] `src/neighbors/` module: `KNeighborsClassifier`, `KNeighborsRegressor`
      (reuses the distance infrastructure already present in `KnnImputer`).
- [ ] `src/naive_bayes/` module: `GaussianNB` and `MultinomialNB` — the natural
      pairing with the text vectorizers for document classification.

### v0.9 — Depth & breadth

> *Theme: converge toward scikit-learn positional parity.*

With the foundations in place, v0.9 fills out the algorithm catalog and adds
the data-loading ergonomics that make the library pleasant to adopt.

**Deliverables:**

- [ ] Ensemble depth: `GradientBoostingClassifier`/`Regressor`, `AdaBoost`,
      `VotingClassifier`/`Regressor`, `StackingClassifier`/`Regressor`.
- [ ] `src/svm/` module: `LinearSVC`/`LinearSVR` (coordinate-descent style,
      no kernel) and `SVC`/`SVR` (RBF/poly kernels via a pure-Rust SMO solver).
- [ ] More clustering: `DBSCAN`, `AgglomerativeClustering`.
- [ ] More linear models: `ElasticNet`, `SGDClassifier`/`Regressor`,
      `RidgeClassifier`.
- [ ] More decomposition: `NMF`, `FactorAnalysis`.
- [x] `src/datasets/` module — Iris, Breast Cancer, Wine, Diabetes datasets
      delivered early (behind the `datasets` feature flag). Loaded from `const`
      arrays with no I/O.
- [ ] CSV reader behind an optional `csv` feature flag (pure-Rust parser, or
      a thin wrapper over the `csv` crate gated behind the feature).

### v1.0 — Stability

> *Theme: lock in the API and ship a stability guarantee.*

v1.0 is not primarily about new features — it is the release where the public
API is frozen and the maintenance promise begins.

**Deliverables:**

- [ ] Remove the `#[doc(hidden)]` legacy APIs (`Matrix::rows_ref`,
      `into_rows`) that have been slated for removal since v0.3.0.
- [ ] Refresh `ARCHITECTURE.md` to reflect the flat-storage `Matrix` (the doc
      still describes the pre-0.3.0 `Vec<Vec<f64>>` layout in places).
- [ ] Full public-API audit: consistent naming, complete doc comments, no
      undocumented public items.
- [ ] MSRV review — decide whether to hold at 1.70 or bump.
- [ ] Publish a v1.0 stability statement: what counts as a breaking change
      going forward, and the SemVer commitment.
- [ ] Ensure every estimator has sklearn-parity tests and a runnable example.

---

## Explicitly out of scope

These will **not** be pursued, to keep the library focused. Each is served
better by another part of the Rust ecosystem.

| Area | Why not |
|---|---|
| **GPU compute** (cuBLAS, wgpu) | Conflicts with the zero-dependency, CPU-first identity. [candle] and [burn] own GPU ML in Rust. |
| **Distributed training** (Ray-style) | Classical ML rarely needs it; the networking/scheduler surface is a different project. |
| **Deep learning** (CNN, RNN, Transformer) | A different scale of problem with autograd, GPU, and model-zoo requirements. Use [burn]/[candle]. A lightweight `MLPClassifier` is the only concession (see "Under consideration"). |
| **pickle / joblib compatibility** | Python's pickle binary format maps object graphs that cannot round-trip through Rust. The realistic bridge is ONNX or `.npy` (below). |
| **SHAP / LIME** | Heavy and niche. Permutation importance (planned for a future `inspection` module) covers the common need more cheaply. |

## Under consideration (post-1.0)

Not committed, but worth evaluating once the core is stable:

- **`f32` support** — a generic `Matrix<T: Float>` would halve memory for
  embedded/large datasets, but it is a sweeping refactor across every
  estimator and trait. High value, high cost.
- **Manifold learning** — `TSNE`, `SpectralEmbedding`, `Isomap`. `TSNE` in
  particular is the visualization standard, but Barnes–Hut approximation is
  substantial work.
- **`HistGradientBoosting`** — the fastest boosting variant (LightGBM-style
  histograms). Complex, but the performance leader for tabular data.
- **ONNX export/import** — a real bridge to and from the Python ecosystem.
  Export requires per-estimator op mapping; import via the [`ort`] crate is
  easier and would enable running sklearn-trained models in Rust.
- **NumPy `.npy`/`.npz` reader** — a small pure-Rust parser that unblocks
  interop with Python-trained data.
- **`MLPClassifier`/`MLPRegressor`** — a minimal feedforward network (~1000
  lines of backprop). The one deep-learning concession worth considering,
  since it stays within classical-ML workloads.

[`ort`]: https://crates.io/crates/ort

---

## How to contribute

The checkboxes above double as a contribution guide. If you want to help:

- Pick an unchecked item that has no architectural prerequisite, or team up
  on one that does.
- Every estimator should land with: doc comments, sklearn-parity tests, and
  (where applicable) a runnable example in `examples/`.
- Open an issue before starting non-trivial work, to avoid duplicated effort
  and to align on API shape.

Progress against this roadmap is tracked in `CHANGELOG.md` under each
release; this file is updated as priorities evolve.
