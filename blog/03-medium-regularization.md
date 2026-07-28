# My Regression Model Had 40 Features. Only Five of Them Knew Anything.

*An honest guide to Ridge, Lasso, cross-validation, and the uncomfortable moment when a lower training score is actually good news.*

---

There is a dangerous kind of satisfaction in fitting a regression model with a lot of columns.

The training score climbs. The residuals shrink. Every new feature seems to make the model a little more intelligent. By the time the table has 40 columns, the model can explain patterns nobody remembers adding.

Then you show it new data.

The score falls apart, and the impressive training result turns out to be a very detailed memory of the past.

I've learned to be suspicious when a model becomes dramatically better every time the dataset becomes wider. Sometimes the new columns contain signal. Sometimes they simply give the model more ways to explain noise.

In this article, we'll make that failure visible with [**datarust**](https://crates.io/crates/datarust), then use Ridge and Lasso regularization to improve it. The example is deliberately controlled:

- 120 rows
- 40 numeric features
- Only the first 5 features affect the target
- The other 35 columns are pure noise
- Extra noise is added to the target too

Because we created the data, we know the truth. That makes this a useful laboratory: if a model assigns importance to feature 27, we know it did not discover a secret business insight. It got attached to an accident.

## The short version

We are going to:

1. Reserve a test set and leave it untouched
2. Tune Ridge and Lasso using cross-validation on the training set
3. Fit each selected model once on all training rows
4. Compare ordinary linear regression, Ridge, and Lasso on the same test set
5. Count how many coefficients Lasso keeps

Create a project and add datarust:

```sh
cargo new regularized_regression
cd regularized_regression
cargo add datarust
```

Replace `src/main.rs` with this:

```rust
use datarust::linear_model::{Lasso, LinearRegression, LinearSolver, Ridge};
use datarust::metrics::regression::{mean_squared_error, r2_score};
use datarust::model_selection::{cross_val_score, KFold, TrainTestSplit};
use datarust::pipeline::Pipeline;
use datarust::scaler::StandardScaler;
use datarust::traits::Predictor;
use datarust::transformer_kind::TransformerKind;
use datarust::Matrix;

// A tiny deterministic PRNG keeps the example dependency-free and repeatable.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self, sigma: f64) -> f64 {
        let u = self.next_f64().max(f64::MIN_POSITIVE);
        let v = self.next_f64();
        sigma
            * (-2.0 * u.ln()).sqrt()
            * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate 40 features, but build y from x0..x4 only.
    let mut rng = Rng::new(2026);
    let mut rows = Vec::new();
    let mut y = Vec::new();

    for _ in 0..120 {
        let row: Vec<f64> = (0..40).map(|_| rng.normal(1.0)).collect();
        let target = 4.0 * row[0]
            - 3.0 * row[1]
            + 2.0 * row[2]
            + 1.5 * row[3]
            - 1.0 * row[4]
            + rng.normal(4.0);

        rows.push(row);
        y.push(target);
    }

    let x = Matrix::new(rows)?;

    // The test set is used once, after model and alpha selection are finished.
    let (x_train, x_test, y_train, y_test) = TrainTestSplit::new()
        .with_test_size(0.20)
        .with_random_state(42)
        .split(&x, &y)?;

    let cv = KFold::new()
        .with_n_splits(5)
        .with_shuffle(true)
        .with_random_state(7);

    // Tune Ridge on training folds only.
    println!("Ridge alpha   mean CV R²");
    let mut best_ridge_alpha = 0.0;
    let mut best_ridge_cv = f64::NEG_INFINITY;

    for &alpha in &[0.1, 1.0, 10.0, 100.0, 1000.0] {
        let candidate = Pipeline::new()
            .push(
                "scale",
                TransformerKind::StandardScaler(StandardScaler::new()),
            )
            .with_estimator(Ridge::new().with_alpha(alpha));

        let scores = cross_val_score(
            &candidate,
            &x_train,
            &y_train,
            &cv,
            r2_score,
        )?;
        let score = mean(&scores);

        println!("{alpha:>11.1}   {score:>10.3}");

        if score > best_ridge_cv {
            best_ridge_alpha = alpha;
            best_ridge_cv = score;
        }
    }

    // Tune Lasso independently; alpha values are not comparable across models.
    println!("\nLasso alpha   mean CV R²");
    let mut best_lasso_alpha = 0.0;
    let mut best_lasso_cv = f64::NEG_INFINITY;

    for &alpha in &[0.01, 0.1, 0.5, 1.0, 2.0] {
        let candidate = Pipeline::new()
            .push(
                "scale",
                TransformerKind::StandardScaler(StandardScaler::new()),
            )
            .with_estimator(
                Lasso::new()
                    .with_alpha(alpha)
                    .with_max_iter(5000),
            );

        let scores = cross_val_score(
            &candidate,
            &x_train,
            &y_train,
            &cv,
            r2_score,
        )?;
        let score = mean(&scores);

        println!("{alpha:>11.2}   {score:>10.3}");

        if score > best_lasso_cv {
            best_lasso_alpha = alpha;
            best_lasso_cv = score;
        }
    }

    // Baseline: unregularized least squares with a robust SVD solver.
    let mut linear = Pipeline::new()
        .push(
            "scale",
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        .with_estimator(
            LinearRegression::new().with_solver(LinearSolver::Svd),
        );
    linear.fit(&x_train, &y_train)?;
    let linear_train_pred = linear.predict(&x_train)?;
    let linear_test_pred = linear.predict(&x_test)?;

    // Refit the selected Ridge configuration on all training rows.
    let mut ridge = Pipeline::new()
        .push(
            "scale",
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        .with_estimator(Ridge::new().with_alpha(best_ridge_alpha));
    ridge.fit(&x_train, &y_train)?;
    let ridge_train_pred = ridge.predict(&x_train)?;
    let ridge_test_pred = ridge.predict(&x_test)?;

    // Refit the selected Lasso configuration on all training rows.
    let mut lasso = Pipeline::new()
        .push(
            "scale",
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        .with_estimator(
            Lasso::new()
                .with_alpha(best_lasso_alpha)
                .with_max_iter(5000),
        );
    lasso.fit(&x_train, &y_train)?;
    let lasso_train_pred = lasso.predict(&x_train)?;
    let lasso_test_pred = lasso.predict(&x_test)?;

    println!("\nHeld-out test set");
    println!("model             train R²  test R²   test RMSE");
    println!(
        "LinearRegression    {:>6.3}    {:>6.3}     {:>7.2}",
        r2_score(&y_train, &linear_train_pred)?,
        r2_score(&y_test, &linear_test_pred)?,
        mean_squared_error(&y_test, &linear_test_pred, false)?,
    );
    println!(
        "Ridge               {:>6.3}    {:>6.3}     {:>7.2}",
        r2_score(&y_train, &ridge_train_pred)?,
        r2_score(&y_test, &ridge_test_pred)?,
        mean_squared_error(&y_test, &ridge_test_pred, false)?,
    );
    println!(
        "Lasso               {:>6.3}    {:>6.3}     {:>7.2}",
        r2_score(&y_train, &lasso_train_pred)?,
        r2_score(&y_test, &lasso_test_pred)?,
        mean_squared_error(&y_test, &lasso_test_pred, false)?,
    );

    let nonzero = lasso
        .estimator()
        .coef()
        .iter()
        .filter(|value| value.abs() > 1e-10)
        .count();

    println!("\nLasso kept {nonzero} of {} features", x.ncols());
    Ok(())
}
```

Run it:

```sh
cargo run --release
```

With datarust v0.6 and the fixed seeds above, the output is:

```text
Ridge alpha   mean CV R²
        0.1        0.220
        1.0        0.249
       10.0        0.362
      100.0        0.274
     1000.0        0.025

Lasso alpha   mean CV R²
       0.01        0.245
       0.10        0.423
       0.50        0.598
       1.00        0.529
       2.00        0.297

Held-out test set
model             train R²  test R²   test RMSE
LinearRegression     0.821     0.272        5.38
Ridge                0.809     0.425        4.78
Lasso                0.713     0.561        4.18

Lasso kept 8 of 40 features
```

The model with the worst training score wins on new data.

That sentence is the whole point of regularization.

## Ordinary regression did exactly what we asked

Linear regression minimizes squared error:

```text
minimize ||y - Xβ||²
```

It has no reason to prefer a simple explanation over a complicated one. If feature 23 happens to align with a few training residuals, the model is allowed to give it a coefficient. With 40 opportunities to find accidental relationships in only 96 training rows, some of those coincidences will look convincing.

Our baseline reaches a training `R²` of `0.821`. That sounds respectable. On the held-out test rows, it manages only `0.272`.

Nothing is broken. The SVD solver found a valid least-squares solution. The model optimized the objective we gave it. The problem is that “fit these rows as closely as possible” was not the same as “learn the relationship that generated future rows.”

This is an important distinction because overfitting rarely arrives as an error message. It arrives as a flattering metric in the wrong place.

## Ridge adds a price to large coefficients

Ridge regression changes the objective:

```text
minimize ||y - Xβ||² + α||β||²
```

The second term is an L2 penalty. Every coefficient is allowed to exist, but large values now have a cost. The model must decide whether a slightly better fit is worth a more aggressive set of weights.

With `alpha = 10`, selected by cross-validation, Ridge gives up a little training performance: `0.809` instead of `0.821`. Its test `R²` improves from `0.272` to `0.425`, and RMSE falls from `5.38` to `4.78`.

Ridge is often my first regularized baseline. It behaves well when many features carry small amounts of signal or when predictors are correlated. Instead of choosing one winner from a group of similar columns, it tends to share weight and shrink the group together.

It does not usually make coefficients exactly zero. If the goal is a sparse model or explicit feature selection, that is where Lasso becomes interesting.

## Lasso is willing to delete a feature

Lasso uses an L1 penalty:

```text
minimize (1 / 2n)||y - Xβ||² + α||β||₁
```

The geometry of that penalty allows coefficients to land exactly on zero. A feature with a zero coefficient contributes nothing to the prediction, so Lasso performs a form of feature selection while fitting the model.

At the cross-validated `alpha = 0.5`, our Lasso model keeps 8 of the 40 features. Its training `R²` drops to `0.713`, considerably below ordinary regression. Its test `R²` rises to `0.561`, the best result of the three.

It stopped spending model capacity on most of the noise columns.

Notice that it kept eight features, not the true five. Lasso is not a truth detector. Finite samples contain accidental correlations, the target contains noise, and the selected penalty balances sparsity against prediction. A nonzero coefficient is evidence that the feature helped this fitted model, not proof of causality or permanent importance.

That caveat matters. Sparse models are easier to inspect, but “easy to inspect” and “scientifically correct” are different promises.

## Why scaling belongs inside the pipeline

Both Ridge and Lasso penalize coefficient magnitude. Coefficient magnitude depends on feature units.

Imagine the same quantity expressed once in dollars and once in millions of dollars. The predictions can be identical, but the coefficients differ by a factor of one million. Without scaling, the penalty treats those mathematically equivalent representations very differently.

That is why every candidate begins with `StandardScaler`:

```rust
let candidate = Pipeline::new()
    .push(
        "scale",
        TransformerKind::StandardScaler(StandardScaler::new()),
    )
    .with_estimator(Ridge::new().with_alpha(alpha));
```

Putting the scaler inside the supervised pipeline does more than keep the code tidy. During cross-validation, `cross_val_score` clones and fits the entire pipeline for each fold. The scaler learns its mean and standard deviation from that fold's training rows only.

If we standardized the complete dataset before cross-validation, statistics from each validation fold would leak into its training transformation. It is a subtle leak, and usually a smaller one than target leakage, but it still makes the evaluation less honest.

Preprocessing that learns parameters belongs inside the validation boundary.

## The test set does not help choose `alpha`

The most common tuning mistake looks perfectly reasonable:

1. Fit five alpha values.
2. Check all five on the test set.
3. Report the best result as test performance.

At step two, the test set quietly became a validation set. We used its answers to make a modeling decision, so it is no longer an independent estimate of final performance.

The example separates the roles:

```text
all data
   ├── training set
   │      └── 5-fold cross-validation chooses alpha
   └── test set
          └── used once after selection
```

For every alpha, four training folds fit the pipeline and the fifth scores it. That rotates five times, and we compare mean validation `R²`. Only after choosing `alpha = 10` for Ridge and `alpha = 0.5` for Lasso do we fit on the complete training set and touch the held-out test rows.

For a small experiment, this is a solid workflow. If you repeatedly revise the feature set after seeing the final test result, the test set begins influencing the project anyway. Serious comparisons may need nested cross-validation or a genuinely untouched final evaluation dataset.

## `alpha` is not a universal knob

The output shows `10.0` working best for Ridge and `0.5` for Lasso. It would be a mistake to conclude that Ridge “needs more regularization.”

The two implementations optimize different objective forms, and alpha's effect depends on sample count, feature scaling, target scale, noise, and the model itself. An alpha value has meaning inside a specific pipeline and dataset. It is not a score you compare across algorithm families.

Use a logarithmic search range, inspect whether the best candidate sits at an edge, and expand the range if necessary. Our small grids are enough to demonstrate the curve:

- Too little penalty behaves like the overfit baseline.
- A moderate penalty improves validation performance.
- Too much penalty erases useful signal and underfits.

Regularization is not “more is safer.” It is a bias-variance tradeoff controlled by data.

## What the metrics are actually saying

We report both `R²` and RMSE:

```rust
let r2 = r2_score(&y_test, &prediction)?;
let rmse = mean_squared_error(&y_test, &prediction, false)?;
```

`R²` compares the model with a baseline that always predicts the test target mean. A score of `1` is perfect, `0` is no better than that mean baseline, and negative values are possible.

RMSE stays in target units. In this synthetic example, Lasso's typical squared-error scale is about `4.18` target units versus `5.38` for ordinary regression. If the target were delivery time in days or revenue in dollars, that number would be much easier to connect to product consequences.

The train/test gap tells another story. Ordinary regression falls from `0.821` to `0.272`; Lasso falls from `0.713` to `0.561`. Lasso fits the training set less closely, but the behavior it learns travels better.

I would rather deploy the modest model that survives contact with new data than the brilliant one that only recognizes its childhood.

## Synthetic data is useful — and unusually polite

We know the first five columns are informative because the target is literally constructed from them:

```rust
let target = 4.0 * row[0]
    - 3.0 * row[1]
    + 2.0 * row[2]
    + 1.5 * row[3]
    - 1.0 * row[4]
    + rng.normal(4.0);
```

Real data does not label its noise features for us. Predictors may interact, drift over time, contain missing values, duplicate each other, or act as proxies for things we should not model. A linear relationship may only be an approximation.

Before carrying this workflow into production, I would also check:

- Whether the split matches how future data arrives — random, chronological, grouped by user, or grouped by site
- Whether feature generation is identical during training and prediction
- Whether outliers dominate squared error
- Whether selected coefficients remain stable across folds and time periods
- Whether the metric reflects the actual cost of mistakes
- Whether any apparently useful feature creates fairness, privacy, or leakage concerns

Regularization helps with variance. It does not repair a broken evaluation design or make inappropriate features safe.

## The practical takeaway

The three models tell a clean story:

| Model | Train R² | Test R² | Test RMSE | Behavior |
|---|---:|---:|---:|---|
| LinearRegression | 0.821 | 0.272 | 5.38 | Fits noise freely |
| Ridge | 0.809 | 0.425 | 4.78 | Shrinks every coefficient |
| Lasso | 0.713 | 0.561 | 4.18 | Zeroes most coefficients |

There is no rule that Lasso always wins. Change the data so all 40 features carry a little signal, and Ridge may be the better bias. Add strong interactions, and all three linear models may be the wrong family.

What survives is the workflow:

1. Keep a final test set out of model selection.
2. Put learned preprocessing inside the pipeline.
3. Choose regularization strength with cross-validation.
4. Compare generalization, not just training fit.
5. Treat sparsity as a modeling behavior, not proof of truth.

The hardest part is emotional: accepting that the model with the lower training score may be the one that learned more.

Once you get comfortable with that, regularization stops looking like a penalty.

It starts looking like discipline.

```sh
cargo add datarust
```

---

*datarust is MIT-licensed and available on [crates.io](https://crates.io/crates/datarust). Documentation lives at [genc-murat.github.io/datarust](https://genc-murat.github.io/datarust/), and the repository includes a [runnable Ridge-vs-Lasso example](https://github.com/genc-murat/datarust/blob/main/crates/datarust/examples/regularization_comparison.rs).*
