# Ensembles

Random forests were the reason the tree kernel was built the way it was:
not "a CART implementation", but a *transcription* of scikit-learn's, so
that a datarust forest and a sklearn forest trained on the same data with
the same integer `random_state` produce the same trees — node for node,
bit for bit. `src/ensemble.rs` ships the two flagship ensembles on top of
that kernel.

```rust
use datarust::ensemble::RandomForestClassifier;
use datarust::traits::{Classifier, PredictProba, Predictor};

let mut forest = RandomForestClassifier::new()
    .with_n_estimators(100)               // default
    .with_criterion(Criterion::Gini)      // default; or Entropy
    .with_max_features(MaxFeatures::Sqrt) // default; All, Log2, Count(n)
    .with_bootstrap(true)                 // default
    .with_random_state(42);

forest.fit(&x, &y)?;
let pred = forest.predict(&new_x)?;
let proba = forest.predict_proba(&new_x)?;   // averaged tree probabilities
let importances = forest.feature_importances()?;
```

## How the parity works

A forest's randomness has three layers, and all three are reproduced
exactly:

1. **The seed stream.** sklearn's forest draws one seed per tree from a
   legacy numpy `RandomState` (MT19937). datarust embeds a minimal clone of
   that generator, so the master stream splits into the same per-tree
   seeds.
2. **The bootstrap samples.** each tree's row indices are drawn from the
   same per-tree stream, with the same masked-rejection arithmetic. The
   indices are applied as sample-weight multiplicities, so the tree kernel
   sees exactly the bootstrap resample sklearn's trees see.
3. **The splits.** each tree's splitter consumes its seed through the same
   `rand_r` chain and Fisher–Yates feature draws as sklearn's
   `BestSplitter` (see [Trees](./trees.md)).

The result is pinned by `tests/integration_forest.rs`: predictions,
probabilities, `feature_importances`, and bootstrap samples all match a
reference sklearn forest via `to_bits()`.

## Aggregation

- **Classifier**: per-tree class probabilities are averaged, then argmax
  is taken (ties break to the smallest class label, like sklearn). This is
  probability averaging, *not* majority voting of hard predictions.
- **Regressor**: predictions are averaged arithmetically.
- **`feature_importances()`**: per-tree normalized importances are averaged
  and re-normalized, matching sklearn's forest-level
  `feature_importances_`.

## `estimators_samples()`

Each tree's bootstrap indices are *derived*, not stored: the fitted model
keeps only the per-tree seeds (8 bytes each) and regenerates the index
lists on demand — the same on-demand scheme sklearn's
`estimators_samples_` uses. A 100-tree forest on a million rows therefore
doesn't carry ~800 MB of indices around:

```rust
let samples = forest.estimators_samples();  // Vec<Vec<usize>>, n_estimators lists
let first_tree_rows = &samples[0];
```

## RandomForestRegressor

The regressor shares every builder except `with_criterion` — regression
trees are squared-error only:

```rust
use datarust::ensemble::RandomForestRegressor;
use datarust::traits::{Predictor, Regressor};

let mut forest = RandomForestRegressor::new()
    .with_n_estimators(200)
    .with_max_features(MaxFeatures::All)  // sklearn's regressor default
    .with_random_state(7);

forest.fit(&x, &y)?;
let r2 = forest.score(&x_test, &y_test)?;
```

Defaults follow sklearn: 100 trees, bootstrap on, `max_features = Sqrt`
for the classifier and `All` for the regressor.

## Parameters

| Parameter | Default (clf / reg) | Notes |
|---|---|---|
| `n_estimators` | 100 / 100 | Number of trees; must be ≥ 1 |
| `criterion` | `Gini` / — | Classifier only; `SquaredError` is rejected by `fit` |
| `max_depth` | `None` / `None` | Unlimited; must be ≥ 1 when set |
| `min_samples_split` | 2 / 2 | Minimum samples to consider splitting |
| `min_samples_leaf` | 1 / 1 | Minimum samples in each leaf |
| `min_impurity_decrease` | 0.0 / 0.0 | Split must improve impurity by more than this |
| `max_features` | `Sqrt` / `All` | Features drawn per split: `All`, `Sqrt`, `Log2`, `Count(n)` |
| `bootstrap` | true / true | Disable to fit every tree on the full sample |
| `random_state` | 0 / 0 | Deterministic master seed |

Both forests implement the `Params` trait (`get_params`/`set_params`) for
hyperparameter introspection.

## Scope (this first release)

Out of bag scoring, weighted fits, and parallel tree fitting are
intentionally outside the first forest release; `ExtraTrees` and a generic
`Bagging` meta-estimator are on the roadmap. Fitting is sequential —
embarrassingly parallel, and a natural future `rayon` win.
