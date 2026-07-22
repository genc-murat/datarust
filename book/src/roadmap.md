# Roadmap

The canonical, detailed roadmap lives in [`ROADMAP.md`][repo] at the repository
root. This page gives a higher-level tour of *why* each phase is ordered the way
it is, and what it unlocks for users.

[repo]: https://github.com/genc-murat/datarust/blob/main/ROADMAP.md

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

## Where we are (v0.5.0)

The preprocessing and linear-model foundations are solid:

- **18 transformers/encoders/imputers/selectors** — the deepest preprocessing
  coverage of any Rust ML library.
- **4 linear models** (Linear / Ridge / Lasso / Logistic regression).
- `Pipeline`, `ColumnTransformer`, cross-validation, and a binary metric suite.
- **Zero external dependencies** by default; `serde` / `rayon` /
  `matrixmultiply` are opt-in.

What is *conspicuously absent*: multiclass classification, clustering, trees
and ensembles, hyperparameter search, text features, and SVM. The roadmap
below addresses each, in order of impact.

## The release track

### v0.6 — Core ML foundations

**Why this comes first.** The 0.5.x series left classification binary-only
and clustering absent. Almost every real workflow needs multiclass labels and
some form of evaluation beyond accuracy — so this phase closes the most
painful gaps *and* lays two pieces of plumbing that later phases depend on:
the `Clusterer` trait (without which no clustering algorithm can exist) and
parameter introspection on `Estimator` (without which `GridSearchCV` cannot
exist).

Highlights: multiclass `LogisticRegression` (softmax + OvR), ROC-AUC and
PR-AUC, a general multiclass confusion matrix with macro/micro averaging,
`KMeans`, and the silhouette score.

### v0.7 — Tree-based learning

**Why this comes second.** Trees and ensembles are the single most requested
feature and the backbone of tabular ML. It cannot come before v0.6 because the
multiclass metric work and the shared RNG infrastructure (needed for bootstrap
sampling) are prerequisites.

Highlights: `DecisionTree` (CART), `RandomForest`, `ExtraTrees`, a `Bagging`
meta-estimator, and `feature_importances_` output. A new `src/tree/` and
`src/ensemble/` module pair.

### v0.8 — Model selection & text

**Why this comes third.** Once the algorithm catalog is broader, the next
bottleneck is *tuning* and *NLP*. `GridSearchCV` builds directly on the
parameter-introspection work from v0.6. Text vectorizers depend on sparse
matrix arithmetic, which today does not exist — so this phase widens
`SparseMatrix` from read-only storage into a real linear-algebra type.

Highlights: `GridSearchCV`/`RandomizedSearchCV`, `validation_curve`/
`learning_curve`, `CountVectorizer`/`TfidfVectorizer`, `KNeighbors`, and
Naive Bayes — the last three together forming a complete document-classification
stack.

### v0.9 — Depth & breadth

**Why this is late.** These are high-value but lower-leverage than the
foundational phases. Many users will be productive with v0.6–0.8 alone.

Highlights: `GradientBoosting`/`AdaBoost`/`Voting`/`Stacking`, `LinearSVC`/`SVC`
(SMO solver, pure Rust), `DBSCAN`/`AgglomerativeClustering`, `ElasticNet`/
`SGDClassifier`, `NMF`/`FactorAnalysis`, an embedded `datasets` module, and a
`csv` feature flag for data loading.

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
