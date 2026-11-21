# I Shipped GridSearchCV to Rust — and Found Out How Much sklearn Encodes in the Word "Best"

*How datarust v0.8 adds hyperparameter search, CV curves, ExtraTrees/Bagging, and k-NN — with the search layer pinned against sklearn to 1e-8 and one tie-breaking rule that took longer to find than to implement.*

---

The [v0.7 release](/blog/v0-7/) ended with a confession: no `ExtraTrees`, no generic `Bagging`, and — the loudest gap in the whole issue tracker — no way to tune anything. Every estimator had builders and a `Params` trait for introspection, but you had to be your own grid search.

v0.8 closes that loop. `ExtraTreesClassifier`, `ExtraTreesRegressor`, `BaggingClassifier`, `BaggingRegressor`, `GridSearchCV`, `RandomizedSearchCV`, `validation_curve`, `learning_curve`, and brute-force `KNeighborsClassifier` / `KNeighborsRegressor`. But the part worth writing down is not the checklist. It's what "matching sklearn" turned out to mean when the thing you're matching is a *search procedure* rather than a numerical kernel.

## A tree is math. A search is a set of promises.

When I ported the CART splitter, parity meant bit equality: same seed in, same tree out, `to_bits()` assertions all green. That's hard, but it's *legible* — either the floats agree or they don't.

`GridSearchCV` is a different kind of contract. The algorithm is trivial: enumerate the cartesian product of the grid, cross-validate each candidate, pick the best mean score, refit. Any competent implementation produces a good model. Almost none produce **the same model as sklearn**, because the selection is full of silent tie-breaking decisions:

- Two candidates whose mean scores agree to the last ulp — which one wins? sklearn's `argmax` says: the *first*. Not "either". The first.
- The rank column of `cv_results_` — is it `1, 2, 3` or `1, 1, 3` for tied candidates? sklearn uses competition ranking (minimum rank). Your dashboard, your paper, your regression report all read differently depending on this.
- "Best" is defined on *means of fold scores*. Fold assignment order therefore leaks into which candidate wins. If your splitter assigns folds differently than sklearn's, you can tune two implementations on identical data and get different winners with no bug in either.

So the design constraint for the search layer was: **same splitter semantics, same aggregation, same tie-breaking, same refit semantics — verified against a generated fixture, not against vibes.** The parity test fits sklearn's `GridSearchCV(Ridge, alpha=[1.0, 0.1, 1.0], cv=KFold(3))` on a seeded dataset and pins the per-fold scores, means, standard deviations, and the rank vector `[2, 1, 2]` — including the deliberate duplicate alpha whose two identical candidates must tie at rank 2 while the middle candidate wins outright. Scores match sklearn 1.6.1 within 1e-8; ordering matches exactly.

That last number deserves a caveat, because precision claims are where Rust ML libraries oversell. The tree parity in v0.7 is *bit-exact* — `assert_eq!` on `to_bits()`. The search parity is *tolerance parity* — the fold scores are sums of floats accumulated in a different order, so the honest claim is 1e-8, and the changelog says exactly that. Ordering and tie-breaks are exact because they're integers. The mean score of a candidate is not, and pretending otherwise would be the kind of overclaim that erodes the whole guarantee.

## The trait that makes every estimator searchable

The mechanical foundation is the `Params` trait that shipped in v0.6 — estimators enumerate their tunable hyperparameters as `(name, value)` pairs and the search driver reads and writes them back before each refit. v0.8 finishes the rollout: `Pipeline`, `SupervisedPipeline`, `ColumnTransformer`, and `TransformerKind` now implement it with sklearn's nested addressing, so one search tunes preprocessing *and* the estimator together:

```rust
let mut gs = GridSearchCV::new(
    SupervisedPipeline::new(Ridge::new()).push(
        "scaler",
        TransformerKind::StandardScaler(StandardScaler::new()),
    ),
    vec![
        ("estimator__alpha", vec![ParamValue::Float(0.1), ParamValue::Float(1.0)]),
        ("scaler__with_mean", vec![ParamValue::Bool(true), ParamValue::Bool(false)]),
    ],
    r2_score,
)
.with_cv(KFold::new().with_n_splits(5));
gs.fit(&x, &y)?;
```

One design decision here is invisible unless you've maintained a library: `get_params` returns `Cow<'static, str>`. Leaf estimators return borrowed static names, zero allocation. Composed names like `scaler__with_mean` are built per call into owned strings — and the first draft of this code called `Box::leak` to satisfy a `&'static str` signature. That's the kind of wart that survives for years because it *works*: a leaked string is a correct string. But `get_params` is an introspection call people will put in loops, and a library that leaks on every introspection is a library you quietly stop trusting. Now is the only cheap time to fix the signature, so that's what v0.8 is for.

There's also a new `CrossValidator` trait. `cross_val_score` used to hard-code `KFold`; now `KFold`, `StratifiedKFold`, and your own splitter all plug into it, into both searches, and into the curves.

## The tie-break that wasn't where the docs said it was

The k-NN port taught me more than the search layer did, because its edge case hid in a helper function.

Brute-force k-NN is an afternoon of work: squared Euclidean distances, partial-select the k nearest, aggregate. The subtlety is the weighting. sklearn's `weights="distance"` votes with `1/d` — and for a query point that *coincides* with a training point, `1/d` is division by zero. What happens?

I read the docstring, which says the zero-distance neighbor gets weight 1. I implemented exactly that: coincident point votes with weight 1, everyone else with their real `1/d`, and a query sitting on a training point gets a prediction dominated by that point's class. Sensible. Testable. Shipped in my head.

Then I generated the parity fixture, and sklearn returned `[1.0, 0.0]` for the coincident query's probabilities. Not `0.987`. Exactly `[1.0, 0.0]`.

The answer is in `_get_weights`, and it's two lines long:

```python
dist[inf_row] = inf_mask[inf_row]
```

If **any** neighbor of a query has zero distance, sklearn doesn't weight that neighbor 1.0 and the rest normally — it zeroes out *every other neighbor in the row*. Winner-takes-all: the coincident points vote, everything else is discarded. The docstring describes the `inf → 1.0` substitution; the code implements a rule the docstring never states. My "sensible" implementation was a faithful port of the documentation and a bug against the behavior.

This is precisely why the release plan has a cut-line rule — *"never ship an estimator without its sklearn-parity fixtures"* — that you're not allowed to waive when the estimator is easy. k-NN is the simplest algorithm in the crate, and the fixture caught a semantic difference that code review, unit tests, and reading the reference implementation's docs all missed. The parity suite now pins the winner-takes-all rule, the uniform-vote tie (exact ties break to the smallest class, sklearn's `argmax` over sorted `classes_`), and the doubly-degenerate case where *two* training points coincide with the query and tie with each other.

The rest of the neighbors parity is the same discipline at boring scale: iris predictions and probabilities within 1e-9 for both weighting schemes, diabetes regression within 1e-6, equidistant-corner tie cases exact. And `n_neighbors` gets the sklearn treatment for post-fit mutation — change `k` after `fit` and `predict` re-checks the bound instead of panicking on a slice index.

## ExtraTrees and Bagging, same seed, same forest

The carried-over ensembles shipped first in this cycle, and they keep the v0.7 guarantee where it's honest. ExtraTrees draws one uniform-random threshold per candidate feature from the same MT19937 "legacy `RandomState`" stream the forests clone — fitted trees match sklearn node-for-node, bit-for-bit, pinned per tree. Bagging reproduces the sampling stream exactly: feature indices drawn before sample indices from the same per-estimator `RandomState`, `randint` bootstrap versus sklearn's auto-dispatched sampling-without-replacement, truncated float fractions, and the `_set_random_states` per-tree seed derivation.

One API fix landed here that matters for tuning: bagging's base-tree parameters were previously merged into `get_params` *unprefixed*, which collided with the ensemble's own names and silently made the base tree's `max_features` unreachable through `set_params`. v0.8 addresses base parameters with sklearn's `estimator__` prefix, so a grid can mix `"n_estimators"` with `"estimator__max_depth"` without ambiguity.

```rust
let mut rs = RandomizedSearchCV::new(
    BaggingClassifier::new(),
    vec![
        ("n_estimators", ParamDistribution::Choices(vec![ParamValue::Int(10), ParamValue::Int(50)])),
        ("estimator__max_depth", ParamDistribution::Choices(vec![ParamValue::Int(4), ParamValue::Int(8)])),
    ],
    accuracy_score,
)
.with_n_iter(8)
.with_random_state(42);
```

`RandomizedSearchCV` itself comes with an honesty note of its own: it samples from `Choices` / `Uniform` / `RandInt` distributions with datarust's own seeded xorshift stream. Choices-only spaces sample *without replacement* from the exhaustive product, like sklearn's list-only spaces; anything with a continuous distribution samples per parameter independently. It is deterministic for a fixed `random_state` (default `0` — reproducible by default, unlike sklearn's `None`). It is *not* bit-exact with sklearn's `ParameterSampler`, and the docs say so instead of implying it.

## What's deliberately not in v0.8

- **No ball/KD-tree acceleration.** Brute force is the honest first release for k-NN at the scale datarust targets; trees are post-0.8 and explicitly on the roadmap.
- **No bit-exact `ParameterSampler` stream.** Documented contract instead.
- **No OOB scoring, sample weights, or parallel fitting** for the ensembles — same cut as v0.7, unchanged.
- **Text and sparse stay in v0.9.** They need a real sparse CSR kernel first, and that deserves its own parity budget.

Also shipping, quieter: `validation_curve` and `learning_curve` reuse the shared CV fold loop (parity-tested within 1e-8), `KFold`/`StratifiedKFold` grew the shared `CrossValidator` trait, and `set_params` on `Ridge`, `Lasso`, and the centering scalers now invalidates fitted state instead of leaving you predicting from stale coefficients.

## Try it

```toml
[dependencies]
datarust = "0.8"
```

Zero dependencies in the default build — the search layer, the ensembles, and k-NN are all there. Start with the [Model Selection guide](/docs/guide/model-selection/) for the tuning tour, [Ensembles](/docs/guide/ensembles/) for the carried-over estimators, and [Nearest Neighbors](/docs/guide/neighbors/) for the new module. There's a runnable end-to-end example in `crates/datarust/examples/hyperparameter_search.rs`.

And if you find a case where the tie-breaking disagrees with sklearn — that's a bug, and I want it. The v0.7 rule stands: the parity claims are only as good as the fixtures that can falsify them.
