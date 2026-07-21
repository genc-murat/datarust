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

Preserves class balance across folds. Essential for imbalanced classification.

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
