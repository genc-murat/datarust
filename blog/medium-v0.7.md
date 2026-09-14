# I Ported scikit-learn's Decision Trees to Rust — Bit for Bit. Here's Why That Was the Whole Point.

*How datarust v0.7 ships random forests that reproduce sklearn's fits exactly — same seed, same trees, same importances — in a crate with zero dependencies.*

---

When I wrote about [shipping a scikit-learn workflow in pure Rust](/blog/v0-6/), the response was consistent, encouraging, and slightly painful: *"this is great, but where are the trees?"*

Fair. Nobody deploys a `LogisticRegression` to win a Kaggle tabular competition. The workhorse of tabular ML — the thing people actually reach for when they have messy columns, mixed types, and no patience for feature engineering — is a random forest. And for six minor versions, datarust didn't have one.

v0.7 changes that. `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`. But the interesting part isn't *that* they exist. It's the guarantee they ship with:

> With the same data, the same hyperparameters, and the same integer `random_state`, a datarust forest and a scikit-learn forest produce **the same trees — node for node, bit for bit.**

Same split features. Same thresholds. Same impurities, down to the last ULP. Same predictions, same probabilities, same `feature_importances_`, same bootstrap samples.

That is a strange thing to build, so let me explain why I built it.

## Why "close enough" wasn't good enough

A tree is a simple algorithm, and that's the trap. Gini impurity is one line of math. Sort the feature values, walk the split points, pick the best. Any competent implementation gets *approximately* the same tree as sklearn.

Approximately is where the trouble lives.

Trees are chaos amplifiers. A threshold that differs by one float ULP routes one row to the other side of a split; that row changes a leaf's mean; that leaf's prediction changes; in a forest, the averaged probability shifts. If you're migrating a validated sklearn model to Rust — say, for a WASM deployment, or an edge device, or a service that can't ship Python — "approximately the same" means *revalidating everything*. And if you ever want to claim the Rust model matches the model the data scientists signed off on, you need a bit stronger than "we ran the test suite and it looked fine."

There's also the reproducibility angle. "Same seed, same model" is easy to promise and surprisingly hard to deliver *across implementations*. sklearn's trees are deterministic given a seed, but deterministic through a specific tangle of RNG streams, sort orders, and float accumulation patterns. Reproducing the *outputs* means reproducing all of it.

So the design constraint for v0.7 wasn't "implement CART". It was: **transcribe scikit-learn 1.6.1's tree code so faithfully that the parity is testable in `to_bits()` assertions.** Then build the forest on top.

## What "transcription" actually means

sklearn's tree core is Cython — `_splitter.pyx`, `_criterion.pyx`, `_tree.pyx`. Porting it is mostly unglamorous. The glamour is in the list of things that turn out to matter for bit-exactness, none of which are obvious until you're staring at a fixture that's off by exactly one bit:

- **Feature values are quantized to `f32`, but thresholds are computed as `f64` midpoints.** sklearn's dense splitter compares `f32` representations and computes `(prev + current) / 2` in float64. Miss this and your thresholds drift.
- **Ties in split quality resolve through the RNG, not feature order.** Even with `max_features=All`, sklearn draws features through its `rand_r` chain at *every* split. Two implementations that both find "the best split" can pick *different* best splits on tied data. You don't get parity without porting the 32-bit xorshift and its Fisher–Yates feature draws exactly.
- **Float accumulation order is load-bearing.** The criterion's running class-sum updates use a forward-accumulate-or-reverse-subtract choice rule; entropy uses `ln(p)/ln(2)` per term, contracted by the FMA instructions clang emits for the Cython; `feature_importances_` normalizes with numpy's *pairwise* summation. Rust's `f64::mul_add` reproduces the FMA; the pairwise sum had to be written out by hand.
- **The sort is part of the algorithm.** Samples are ordered by introsort (median-of-3, 3-way partition, heapsort fallback). The sort's *output* affects float accumulation order, so "any correct sort" is not good enough.

Each of those is a small detail. Together they're the difference between `assert_eq!(a, b)` and `assert!((a - b).abs() < 1e-12)` — and the second one is a lie you tell yourself right before a one-row divergence shows up in production.

The parity suite pins it: six fixtures generated from seeded sklearn 1.6.1 / numpy 2.0.2 runs — iris (three configurations, including entropy and `max_features=sqrt`), breast cancer at depth 4, diabetes regression at two depths — with every node feature, threshold, impurity, value, prediction, probability, and importance compared **by bit**.

```rust
let mut clf = DecisionTreeClassifier::new()
    .with_criterion(Criterion::Entropy)
    .with_max_depth(Some(3))
    .with_random_state(42);

clf.fit(&x, &y)?;

let tree = clf.tree().unwrap();
tree.n_nodes();              // same as sklearn's tree_.node_count
tree.node_feature(0);        // same split feature
tree.node_threshold(0);      // same threshold, same bits
clf.feature_importances()?;  // same importances, same bits
```

## The forest is three RNG streams deep

A random forest adds two more layers of randomness on top of the tree, and both had to be reproduced:

1. **The seed stream.** sklearn draws one seed per tree from numpy's legacy `RandomState` — an MT19937 with specific seeding and "masked rejection" integer draws. datarust embeds a minimal clone of that generator. Same master seed, same per-tree seeds.
2. **The bootstrap.** Each tree trains on `n_samples` indices drawn with replacement from its per-tree stream. datarust applies those indices as sample-weight multiplicities, so the tree kernel sees *exactly* the resample sklearn's trees see.

```rust
let mut forest = RandomForestClassifier::new()
    .with_n_estimators(100)
    .with_random_state(42);

forest.fit(&x_train, &y_train)?;
let proba = forest.predict_proba(&x_test)?;  // averaged tree probabilities
let importances = forest.feature_importances()?;
```

One design note I'm happy about: the fitted forest stores only the per-tree *seeds* (8 bytes each) and regenerates the bootstrap index lists on demand in `estimators_samples()` — sklearn's `estimators_samples_` semantics. The obvious implementation stores the indices, which quietly costs you `n_estimators × n_samples × 8 bytes` for the model's lifetime. On a million rows with the default hundred trees, that's the difference between ~800 MB and ~800 bytes.

The aggregation follows sklearn too: classifiers average per-tree *probabilities* (then argmax, ties to the smallest class label) — not a majority vote of hard predictions, which is a different estimator. Regressors average predictions. Importances are averaged per tree and re-normalized.

## What's deliberately not in v0.7

Honesty section, because the omissions are as designed as the inclusions:

- **No out-of-bag scoring** yet. The bootstrap machinery makes it natural follow-up work; it didn't make the cut for correctness-first.
- **No parallel fitting.** The trees are embarrassingly parallel and the crate already has an optional `rayon` feature, but v0.7 fits sequentially. Parity first, throughput second.
- **No `ExtraTrees`, no generic `Bagging`.** The kernel supports them; the estimators don't exist yet.
- **No sample weights, no missing-value routing, no sparse input.** Same as the rest of datarust: dense, finite, single-output.

Also shipping in v0.7, quieter but useful: the internal PRNG is now a **public, seedable `Rng`** (`datarust::rng`, xorshift64* with `next_normal` and Fisher–Yates `shuffle`), and the diabetes dataset grew `load_normalized()` — the sklearn `load_diabetes(scaled=True)` layout, computed from the embedded raw data, whose literals were re-precisioned to sklearn's exact `f64` values.

## Try it

```toml
[dependencies]
datarust = "0.7"
```

The trees and forests are in the default build — no features, no dependencies, nothing but the standard library. If you have a validated sklearn forest and a reason to leave Python behind (WASM, a 5 MB container, a platform without a Python runtime), the migration story is now boring in the best way: set the same `random_state`, assert `to_bits()` equality against your reference predictions, and ship.

The [Trees](/docs/guide/trees/) and [Ensembles](/docs/guide/ensembles/) guide pages have the full API. If you find data where parity breaks, that's a bug and I want it.
