# Roadmap

The canonical, detailed roadmap lives in [`ROADMAP.md`][repo] inside the
`datarust` crate. This page gives a higher-level tour of *why* each phase is
ordered the way it is, and what it unlocks for users.

[repo]: https://github.com/genc-murat/datarust/blob/main/crates/datarust/ROADMAP.md

## The destination

datarust aims to be the **scikit-learn of Rust** for classical, CPU-bound
machine learning: the library you reach for when you need preprocessing,
trees, clustering, SVM, cross-validation, and text features — without pulling
in BLAS, a Python runtime, or a GPU stack.

The principles that shape every decision:

- **Zero dependencies by default.** Every algorithm ships as pure Rust.
- **CPU-first.** GPU and deep learning are served by [candle] and [burn].
- **scikit-learn API parity** where Rust allows it; type-safe improvements
  where Rust enables them.
- **No panics** — public APIs return `Result`; `missing_docs` is enforced.

[candle]: https://crates.io/crates/candle-core
[burn]: https://crates.io/crates/burn

## Where we are (v0.8.0)

The preprocessing, linear-model, clustering, tree, ensemble, tuning, and
nearest-neighbor foundations are solid:

- **18 transformers/encoders/imputers/selectors** — the deepest preprocessing
  coverage of any Rust ML library.
- **4 linear models** (Linear / Ridge / Lasso / Logistic regression) with
  binary IRLS and multiclass softmax support.
- **Trees, forests, ExtraTrees, and Bagging** (v0.7 + v0.8) — a CART kernel
  plus four ensemble estimators that reproduce scikit-learn's fits
  bit-for-bit under the same seed.
- **Hyperparameter search** (v0.8) — `GridSearchCV` / `RandomizedSearchCV`
  over any `Params` estimator, `CrossValidator` trait for custom splitters,
  and `validation_curve` / `learning_curve` diagnostics.
- **KNeighborsClassifier / KNeighborsRegressor** (v0.8) — brute-force
  Euclidean k-NN with sklearn's tie-breaking.
- **KMeans clustering** (Lloyd's algorithm, k-means++ initialization) +
  silhouette score.
- 15 metrics: regression (MSE, MAE, R², ...), classification (accuracy,
  precision, recall, F1, ROC-AUC, PR-AUC, kappa, Matthews, ...), clustering
  (silhouette).
- `Pipeline`, `ColumnTransformer` (with nested `step__param` tuning),
  cross-validation, the `Params` trait, and a public seedable `Rng`.
- **Zero external dependencies** by default; `serde` / `rayon` /
  `matrixmultiply` are opt-in.

What is *conspicuously absent*: text features, naive Bayes, and SVM. The
roadmap below addresses each, in order of impact.

## The release track

### v0.7 — Tree-based learning (Shipped)

Trees and ensembles are the backbone of tabular ML, and v0.7 delivered the
core of that stack with an unusual guarantee: **bit-exact scikit-learn
parity.** The CART splitter is a transliteration of sklearn 1.6.1's
`BestSplitter` + `DensePartitioner`, and the forests seed their bootstrap
samples through the same MT19937 "legacy `RandomState`" stream sklearn
uses — the same integer `random_state` yields the same trees, node for
node, verified bit-for-bit by the parity test suite.

Shipped: `src/tree/` (`DecisionTreeClassifier` / `DecisionTreeRegressor`,
`TreeStructure` introspection, `feature_importances`), `src/ensemble.rs`
(`RandomForestClassifier` / `RandomForestRegressor` with on-demand
`estimators_samples()`), and the public `Rng` promoted from
`model_selection`. `ExtraTrees` and a generic `Bagging` meta-estimator
shipped in v0.8; out-of-bag scoring and parallel fitting for Bagging
(and the forests) remain tracked follow-ups — see the canonical roadmap's
"v0.8 follow-ups — Bagging completeness" checklist.

### v0.8 — Model selection & ensemble completion (Shipped)

**Why this came second.** Once the algorithm catalog is broader, the next
bottleneck is *tuning*. `GridSearchCV` builds directly on the `Params` trait
(`get_params`/`set_params`) shipped in v0.6 and needed no new numerical
kernels — so it shipped together with the `ExtraTrees` and `Bagging`
ensembles carried over from the tree phase and brute-force `KNeighbors`.
Text features were split into v0.9 because they depend on sparse matrix
arithmetic, a substantial kernel effort that deserves its own release.

Shipped: `ExtraTreesClassifier` / `ExtraTreesRegressor` and
`BaggingClassifier` / `BaggingRegressor` (bit-exact sampling), `GridSearchCV`
/ `RandomizedSearchCV` with the `CrossValidator` trait,
`validation_curve` / `learning_curve`, nested `step__param` addressing for
the pipeline stack, and `src/neighbors/` (`KNeighborsClassifier` /
`KNeighborsRegressor`, brute-force Euclidean with sklearn's tie-breaking).

**Why this comes second.** Once the algorithm catalog is broader, the next
bottleneck is *tuning*. `GridSearchCV` builds directly on the `Params` trait
(`get_params`/`set_params`) shipped in v0.6 and needs no new numerical
kernels — so it ships first, together with the `ExtraTrees` and `Bagging`
ensembles carried over from the tree phase and `KNeighbors` classifiers.
Text features were split into v0.9 because they depend on sparse matrix
arithmetic, a substantial kernel effort that deserves its own release.

Highlights: `GridSearchCV`/`RandomizedSearchCV`, `validation_curve`/
`learning_curve`, `ExtraTrees`, `Bagging`, and `KNeighbors`.

### v0.9 — Text & sparse

**Why text gets its own phase.** Text vectorizers depend on sparse matrix
arithmetic, which today does not exist — this phase widens `SparseMatrix`
from read-only storage into a real linear-algebra type, then builds the NLP
stack on top of it.

Highlights: sparse × dense / sparse + sparse arithmetic,
`CountVectorizer`/`TfidfVectorizer`/`HashingVectorizer`, and Naive Bayes —
together forming a complete document-classification stack.

### v0.10 — Depth & breadth

**Why this is late.** These are high-value but lower-leverage than the
foundational phases. Many users will be productive with v0.7–0.9 alone.

Highlights: `GradientBoosting`/`AdaBoost`/`Voting`/`Stacking`,
`LinearSVC`/`SVC` (SMO solver, pure Rust),
`DBSCAN`/`AgglomerativeClustering`, `ElasticNet`/`SGDClassifier`,
`NMF`/`FactorAnalysis`.

### v1.0 — Stability

**Why v1.0 is mostly cleanup.** By this point the API surface has been
exercised across many estimators and is ready to be frozen. v1.0 removes
legacy `#[doc(hidden)]` APIs, refreshes `ARCHITECTURE.md`, runs a full public-API
audit, and publishes the SemVer stability commitment.

## What is deliberately out of scope

The roadmap is as much about what datarust will *not* become:

- **GPU compute and deep learning** — served by [candle] and [burn].
- **Distributed training** — a networking project, not a classical-ML one.
- **pickle/joblib compatibility** — technically impossible across languages.
- **SHAP/LIME** — permutation importance (planned) covers the common need.

A few items are **under consideration** for post-1.0 but not committed: `f32`
generics, manifold learning (`TSNE`), `HistGradientBoosting`, ONNX
export/import, NumPy `.npy` interop, and a minimal `MLPClassifier`. See
[`ROADMAP.md`][repo] for the full list and rationale.

## Contributing

The checkboxes in [`ROADMAP.md`][repo] double as a contribution guide. Pick an
unchecked item, open an issue to align on API shape, and land it with doc
comments and sklearn-parity tests. Progress is recorded in
[`CHANGELOG.md`](./changelog.md) under each release.
