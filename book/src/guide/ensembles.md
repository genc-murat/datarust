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

## ExtraTrees

`ExtraTreesClassifier` and `ExtraTreesRegressor` (extremely-randomized trees)
delegate to the same per-tree `splitter="random"` kernel that sklearn uses.
Instead of scanning every midpoint, each split draws *one* random threshold
between the feature's min and max. Per-node feature subsampling works
identically to the random forest: the splitter consumes the same Fisher–Yates
stream for feature draws, then a single `rand_uniform` call per non-constant
feature.

sklearn's defaults differ from `RandomForest`:
- `bootstrap = false` (fit every tree on the full sample)
- `max_features = Sqrt` (clf) / `All` (reg)

```rust
use datarust::ensemble::{ExtraTreesClassifier, ExtraTreesRegressor};
use datarust::traits::{Classifier, PredictProba, Predictor};

let mut et = ExtraTreesClassifier::new()
    .with_n_estimators(100)
    .with_bootstrap(false)   // sklearn default
    .with_random_state(7);

et.fit(&x, &y)?;
let pred = et.predict(&new_x)?;
```

Both ExtraTrees types wrap the corresponding `RandomForest*` internally, so
the parameter table above applies — only `random_split` and `bootstrap`
defaults change.

## Bagging

`BaggingClassifier` and `BaggingRegressor` are generic meta-estimators that
fit a base estimator on bootstrap (or random subset) samples of the training
data, with an optional random feature subset per member. The first release
uses `DecisionTree*` as the base, matching sklearn's default.

Sampling details (all sklearn-bit-exact):

- **Feature indices** are drawn *before* sample indices, both from the same
  per-estimator `RandomState`.
- `bootstrap = true`: indices drawn with replacement via `randint`.
- `bootstrap = false`: indices drawn without replacement via sklearn's
  auto-dispatched SWOR (permutation, tracking selection, or reservoir
  sampling).
- `max_samples` and `max_features` accept both float fractions (truncated to
  integer, e.g. `0.33 × 150 → 49`) and absolute integer counts.
- `bootstrap_features` controls whether feature indices are drawn with or
  without replacement.
- The per-tree `random_state` is **not** the forest-level seed directly:
  sklearn's `_set_random_states` derives a fresh seed via
  `RandomState(forest_seed).randint(MAX_INT)` before passing it to the base
  estimator. This wrapping is reproduced exactly.

Aggregation:
- Classifier: per-estimator class probabilities are summed and divided by
  `n_estimators`; missing classes (when a bootstrap sample omits a label) are
  aligned on the ensemble's global `classes_`.
- Regressor: predictions are averaged.

```rust
use datarust::ensemble::{BaggingClassifier, BaggingRegressor,
                          BaggingMaxSamples, BaggingMaxFeatures};
use datarust::traits::{Predictor, PredictProba};

let mut bag = BaggingClassifier::new()
    .with_n_estimators(10)                     // default
    .with_max_samples(BaggingMaxSamples::Fraction(1.0))  // all
    .with_max_features(BaggingMaxFeatures::Fraction(1.0))
    .with_bootstrap(true)
    .with_random_state(42);

bag.fit(&x, &y)?;
let pred = bag.predict(&new_x)?;
let proba = bag.predict_proba(&new_x)?;

// Inspect per-estimator state
let samples = bag.estimators_samples();
let features = bag.estimators_features();
```

## Scope (this first release)

Out of bag scoring, weighted fits, and parallel tree fitting are
intentionally outside the first ensemble release. Fitting is sequential —
embarrassingly parallel, and a natural future `rayon` win.
ExtraTrees and Bagging are built on the tree kernel without `rayon` support;
the generic `Bagging` wrapper currently targets `DecisionTree*` bases only.
