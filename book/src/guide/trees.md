# Trees

Decision trees are the first supervised learners in datarust beyond linear
models, and the foundation the [ensembles](./ensembles.md) build on. The
`tree` module ships a CART implementation whose splitter is a
line-by-line transliteration of scikit-learn 1.6.1's `BestSplitter`, so a
fitted tree matches sklearn **node-for-node, bit-for-bit** when given the same
data, parameters, and integer seed — split features, thresholds, impurities,
predictions, probabilities, and `feature_importances_` (verified by
`tests/tree_parity.rs` against seeded sklearn fixtures).

```rust
use datarust::tree::{Criterion, DecisionTreeClassifier, MaxFeatures};
use datarust::traits::{Predictor, PredictProba};

let mut clf = DecisionTreeClassifier::new()
    .with_criterion(Criterion::Gini)      // default; or Criterion::Entropy
    .with_max_depth(None)                 // None = unlimited (default)
    .with_min_samples_split(2)            // default
    .with_min_samples_leaf(1)             // default
    .with_min_impurity_decrease(0.0)      // default
    .with_max_features(MaxFeatures::All)  // default; Sqrt, Log2, Count(n)
    .with_random_state(7);                // seed for per-split feature draws

clf.fit(&x, &y)?;
let pred = clf.predict(&new_x)?;
let proba = clf.predict_proba(&new_x)?;
let importances = clf.feature_importances()?;
```

## Why a seed matters for trees

Even with `max_features = All`, sklearn's tree draws features through its RNG
at *every* split; ties in split quality therefore resolve through the seeded
stream, not feature order. datarust reproduces this exactly — the same
`random_state` yields the same tree as sklearn, and different seeds may yield
different (equally valid) trees on tied data. For reproducible models, always
set `with_random_state`.

## DecisionTreeClassifier

CART classification tree (Gini by default, or entropy — both use sklearn's
exact base-2 definitions). Implements `Predictor` and `PredictProba`;
`predict` returns the majority class of the reached leaf (ties break to the
smallest class label, like sklearn), `predict_proba` returns the leaf's class
frequencies.

```rust
use datarust::tree::DecisionTreeClassifier;
use datarust::traits::{Classifier, PredictProba, Predictor};

let mut clf = DecisionTreeClassifier::new().with_random_state(7);
clf.fit(&x_train, &y_train)?;

let pred = clf.predict(&x_test)?;
let proba = clf.predict_proba(&x_test)?;   // n_rows × n_classes
let r2_like_acc = clf.score(&x_test, &y_test)?;
```

## DecisionTreeRegressor

CART regression tree with the squared-error (variance-reduction) criterion.
Leaves predict the mean target of their training samples.

```rust
use datarust::tree::DecisionTreeRegressor;
use datarust::traits::{Predictor, Regressor};

let mut reg = DecisionTreeRegressor::new()
    .with_max_depth(Some(5))
    .with_random_state(7);

reg.fit(&x, &y)?;
let pred = reg.predict(&new_x)?;
let r2 = reg.score(&x_test, &y_test)?;
```

With the default `min_samples_leaf = 1` the tree grows to pure leaves and
reproduces the training targets exactly; constrain leaf size or depth to
regularize.

## Feature importances

`feature_importances()` computes sklearn's normalized impurity reduction:

$$\text{importance}_j = \frac{\sum_{t \in \text{splits on } j} \frac{N_t}{N}\,\Delta i(t)}{\sum_{t \in \text{all splits}} \frac{N_t}{N}\,\Delta i(t)}$$

The vector sums to 1 (or is all-zero for a pure root). The summation and
per-split arithmetic reproduce sklearn's compiled float behavior exactly, so
importances match `feature_importances_` bit-for-bit.

## Inspecting the tree

`tree()` exposes the fitted `TreeStructure`:

```rust
let tree = clf.tree().unwrap();
tree.n_nodes();        // arena size
tree.get_depth();      // maximum depth
tree.n_leaves();       // leaf count
tree.classes();        // sorted class labels (classification)

// Per-node introspection (preorder arena, root = 0):
tree.node_feature(0);        // Some(feature) for internal nodes, None for leaves
tree.node_threshold(0);      // split threshold
tree.node_impurity(0);       // impurity at fit time
tree.node_improvement(0);    // impurity improvement over the parent
tree.node_n_samples(0);      // training samples reaching the node
tree.node_regression_value(0); // leaf/regression mean value

// Route one feature row to its leaf:
let leaf = tree.leaf_for(&row)?;
```

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| `criterion` | `Gini` | Classification only; `Entropy` uses base-2 logs |
| `max_depth` | `None` | Unlimited; must be ≥ 1 when set |
| `min_samples_split` | `2` | Minimum samples to consider splitting |
| `min_samples_leaf` | `1` | Minimum samples in each leaf |
| `min_impurity_decrease` | `0.0` | Split must improve impurity by more than this |
| `max_features` | `All` | Features drawn per split: `All`, `Sqrt`, `Log2`, `Count(n)` |
| `random_state` | `0` | Seed for the per-split feature-draw RNG |

Divergences from sklearn (by design): no `sample_weight`, `ccp_alpha`
pruning, missing-value routing, or sparse input; `X` must be finite and
single-output. The `Params` trait exposes the numeric subset for
hyperparameter introspection, ready for v0.8's grid search.

## The shared RNG

The per-split feature draws use datarust's public, seedable
[`Rng`](https://docs.rs/datarust/latest/datarust/struct.Rng.html)
(`datarust::rng`), promoted to a top-level module in v0.7 so ensembles can
build deterministic bootstrap samples from the same utility. See
[Model Selection](./model-selection.md) for its use in splitting, and
[Ensembles](./ensembles.md) for the forests built on this kernel.
