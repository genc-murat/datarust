# Model Selection

Train/test splitting and cross-validation. Mirrors `sklearn.model_selection`. Live in [`datarust::model_selection`](https://docs.rs/datarust/latest/datarust/model_selection/index.html).

## train_test_split

Split `X` and `y` into train and test subsets.

```rust
use datarust::model_selection::{train_test_split, TrainTestSplit};

// Quick split with defaults (25% test, shuffled):
let (x_tr, x_te, y_tr, y_te) = train_test_split(&x, &y)?;

// Or configure via the builder:
let (x_tr, x_te, y_tr, y_te) = TrainTestSplit::new()
    .with_test_size(0.2)        // fraction in (0, 1)
    .with_shuffle(true)
    .with_random_state(42)      // deterministic seed
    .split(&x, &y)?;
```

The split requires at least two samples. Fractional test sizes round up, so a
25% test share uses two rows when the dataset contains five rows.

## KFold

K-fold cross-validation. Each sample serves as the test set exactly once.

```rust
use datarust::model_selection::KFold;

let cv = KFold::new()
    .with_n_splits(5)
    .with_shuffle(true)
    .with_random_state(42);

for (train_idx, test_idx) in cv.split(n_samples)? {
    let x_train = x.select_rows(&train_idx)?;
    let x_test = x.select_rows(&test_idx)?;
    // ...
}
```

## StratifiedKFold

Approximately preserves class balance across folds. It accepts two or more
non-negative integer-valued classes, including gapped labels such as
`{2, 5, 9}`. Essential for imbalanced classification.

```rust
use datarust::model_selection::StratifiedKFold;

let cv = StratifiedKFold::new().with_n_splits(5);

for (train_idx, test_idx) in cv.split(&y)? {  // note: takes y, not n_samples
    // each fold has roughly the same class ratio as the full dataset
}
```

## cross_val_score

Evaluate any `Predictor + Clone` estimator by K-fold cross-validation with a user-supplied scorer.

```rust
use datarust::model_selection::{cross_val_score, KFold};
use datarust::linear_model::LinearRegression;
use datarust::metrics::regression::r2_score;

let cv = KFold::new().with_n_splits(5);
let scores = cross_val_score(&LinearRegression::new(), &x, &y, &cv, r2_score)?;
// scores.len() == 5, one R² per fold
```

For classification, pass `accuracy_score` instead:

```rust
use datarust::linear_model::LogisticRegression;
use datarust::metrics::classification::accuracy_score;

let scores = cross_val_score(&LogisticRegression::new(), &x, &y, &cv, accuracy_score)?;
```

The scorer is any closure `Fn(&[f64], &[f64]) -> Result<f64>`, so you can pass custom metrics too.

## CrossValidator

`KFold` and `StratifiedKFold` both implement the [`CrossValidator`](https://docs.rs/datarust/latest/datarust/model_selection/trait.CrossValidator.html) trait, so `cross_val_score`, `GridSearchCV`, `RandomizedSearchCV`, and the curve diagnostics all accept either splitter — or your own:

```rust
use datarust::model_selection::{CrossValidator, cross_val_score, KFold, StratifiedKFold};

let regression_cv: &dyn CrossValidator = &KFold::new().with_n_splits(5);
let classification_cv: &dyn CrossValidator = &StratifiedKFold::new().with_n_splits(5);
```

A custom splitter only needs `n_splits()` and `split(x, y)` plus `Debug`.

## GridSearchCV

Exhaustive search over a parameter grid, mirroring `sklearn.model_selection.GridSearchCV`. Candidates come from the cartesian product of the grid, are evaluated by cross-validation, and the best is refit on the full dataset (`refit=true` by default, ties break to the first candidate).

```rust
use datarust::model_selection::{GridSearchCV, KFold, ParamValue};
use datarust::linear_model::Ridge;
use datarust::metrics::regression::r2_score;

let mut gs = GridSearchCV::new(
    Ridge::new(),
    vec![(
        "alpha",
        vec![
            ParamValue::Float(0.1),
            ParamValue::Float(1.0),
            ParamValue::Float(10.0),
        ],
    )],
    r2_score,
)
.with_cv(KFold::new().with_n_splits(5))
.with_refit(true);
gs.fit(&x, &y)?;

gs.best_params();   // Some([("alpha", Float(0.1))])
gs.best_score();    // mean CV score of the winner
gs.best_index();    // index into cv_results().params
let model = gs.best_estimator().unwrap(); // refit on all data
model.predict(&x_new)?;
```

`cv_results()` exposes the full table — per-candidate mean, standard deviation, fold scores, and competition ranks (`1, 1, 3` for exact ties, like sklearn's `rank_test_score`).

Every estimator that implements the [`Params`](https://docs.rs/datarust/latest/datarust/traits/trait.Params.html) trait is searchable: the linear models, trees, forests, bagging ensembles, `KMeans`, the scalers, and the pipeline wrappers.

## Pipeline parameter grids

`Pipeline` and `SupervisedPipeline` implement `Params` with nested `step__param` addressing, so a single search tunes preprocessing and the estimator together:

```rust
use datarust::model_selection::{GridSearchCV, ParamValue};
use datarust::pipeline::SupervisedPipeline;
use datarust::transformer_kind::TransformerKind;
use datarust::scaler::StandardScaler;
use datarust::linear_model::Ridge;

let pipeline = SupervisedPipeline::new(Ridge::new()).push(
    "scaler",
    TransformerKind::StandardScaler(StandardScaler::new()),
);

let mut gs = GridSearchCV::new(
    pipeline,
    vec![
        ("estimator__alpha", vec![ParamValue::Float(0.1), ParamValue::Float(1.0)]),
        ("scaler__with_mean", vec![ParamValue::Bool(true), ParamValue::Bool(false)]),
    ],
    r2_score,
);
gs.fit(&x, &y)?;
```

## RandomizedSearchCV

For large spaces, sample `n_iter` candidates from parameter distributions instead of enumerating them. Sampling uses datarust's seeded xorshift64 [`Rng`](https://docs.rs/datarust/latest/datarust/struct.Rng.html) — deterministic for a fixed `random_state` (default `0`, unlike sklearn's unseeded default), but **not** bit-exact with sklearn's `ParameterSampler`.

- `Choices` — one value drawn uniformly per candidate.
- `Uniform(lo, hi)` — uniform float in `[lo, hi)`.
- `RandInt(lo, hi)` — uniform integer in the inclusive range.

If every entry is `Choices`, candidates are drawn *without replacement* from the exhaustive product (sklearn's list-only behavior); otherwise each candidate samples every parameter independently.

```rust
use datarust::model_selection::{ParamDistribution, RandomizedSearchCV};
use datarust::linear_model::Lasso;
use datarust::metrics::regression::r2_score;

let mut rs = RandomizedSearchCV::new(
    Lasso::new(),
    vec![
        ("alpha", ParamDistribution::Uniform(1e-4, 10.0)),
        ("max_iter", ParamDistribution::RandInt(100, 1000)),
    ],
    r2_score,
)
.with_n_iter(20)
.with_random_state(42);
rs.fit(&x, &y)?;

rs.best_params(); // e.g. Some([("alpha", Float(0.83)), ("max_iter", Int(517))])
```

`cv_results()` / `best_*()` mirror `GridSearchCV`.

## validation_curve

Scores train and CV-test folds across a range of one hyperparameter — the classic under/over-fitting picture. Returns `(train_scores, validation_scores)`, each indexed `[param_value][fold]`.

```rust
use datarust::model_selection::{KFold, ParamValue, validation_curve};
use datarust::linear_model::Ridge;
use datarust::metrics::regression::r2_score;

let (train, validation) = validation_curve(
    &Ridge::new(),
    &x, &y,
    "alpha",
    &[ParamValue::Float(0.1), ParamValue::Float(1.0), ParamValue::Float(10.0)],
    &KFold::new().with_n_splits(5),
    r2_score,
)?;
```

## learning_curve

Scores train and CV-test folds over increasing training-set sizes to reveal whether more data would help. `train_sizes` are fractions of each fold's training split; the training split is truncated to its first `floor(fraction × n_train)` samples.

```rust
use datarust::model_selection::{KFold, learning_curve};
use datarust::linear_model::LinearRegression;
use datarust::metrics::regression::r2_score;

let (sizes, train, validation) = learning_curve(
    &LinearRegression::new(),
    &x, &y,
    &[0.25, 0.5, 0.75, 1.0],
    &KFold::new().with_n_splits(5),
    r2_score,
)?;
// sizes: deduplicated absolute sample counts, ascending
```

A runnable end-to-end tour lives in
`crates/datarust/examples/hyperparameter_search.rs`
(`cargo run --example hyperparameter_search`).
