# My Random Split Scored 0.71 R². Tomorrow Scored −0.90.

*A practical datarust experiment with temporal leakage, regime change, chronological holdouts, and the validation set that accidentally taught my model about the future.*

---

The random test score looked respectable:

```text
R² = 0.7113
```

Then I evaluated the same modeling idea the way production would actually experience it: train on the past and predict the next period.

```text
R² = -0.9024
```

The model had not suddenly changed. The question had.

Random splitting asked whether the model could predict held-out rows from a population it had already sampled across the entire timeline. Chronological splitting asked whether a model trained before a regime change could survive after it.

Only the second question resembled deployment.

Random train/test splits are excellent when rows are exchangeable — when order carries no information and future observations come from the same stable process as past observations. Time-ordered product data often violates that assumption through price changes, seasonality, policy updates, hardware revisions, market shifts, and user behavior that evolves.

Let's build a controlled example with [datarust](https://crates.io/crates/datarust) and watch a reasonable metric hide a future failure.

## One thousand rows and a change at row 800

The dataset has one predictive feature named `signal` and one timestamp used only for splitting diagnostics.

For the first 800 observations, the target follows:

```text
target = 20 + 5 × signal + noise
```

At row 800, the system moves to a new level:

```text
target = 40 + 5 × signal + noise
```

The slope remains the same. The target is now 20 units higher for the same feature value. Importantly, the model does not receive a feature that announces the regime change.

I evaluate two workflows:

1. A reproducible random 80/20 split
2. A chronological split using rows 0–799 for training and 800–999 for testing

I also isolate the future-regime rows inside the random test set. That slice shows what the aggregate random metric is averaging away.

Here is the complete Rust program:

```rust
use datarust::linear_model::LinearRegression;
use datarust::metrics::regression::{
    mean_absolute_error, mean_squared_error, r2_score,
};
use datarust::model_selection::TrainTestSplit;
use datarust::traits::Predictor;
use datarust::Matrix;

fn feature_only(
    x_with_time: &Matrix,
) -> Result<Matrix, Box<dyn std::error::Error>> {
    Ok(Matrix::from_columns(vec![x_with_time.col(0)])?)
}

fn future_count(x_with_time: &Matrix) -> usize {
    x_with_time
        .col(1)
        .iter()
        .filter(|&&time| time >= 800.0)
        .count()
}

fn report(
    name: &str,
    actual: &[f64],
    predicted: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let bias = predicted
        .iter()
        .zip(actual)
        .map(|(p, a)| p - a)
        .sum::<f64>()
        / actual.len() as f64;

    println!(
        "{name:<24} {:>5}  {:>7.3}  {:>7.3}  \
         {:>8.4}  {:>8.3}",
        actual.len(),
        mean_absolute_error(actual, predicted)?,
        mean_squared_error(actual, predicted, false)?,
        r2_score(actual, predicted)?,
        bias,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let noise = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut rows = Vec::new();
    let mut targets = Vec::new();

    for time in 0..1_000 {
        let signal = ((time * 37) % 100) as f64 / 10.0;
        let level_shift = if time >= 800 { 20.0 } else { 0.0 };
        let target = 20.0
            + 5.0 * signal
            + level_shift
            + noise[time % noise.len()];

        // Column 1 is retained for split diagnostics, not modeling.
        rows.push(vec![signal, time as f64]);
        targets.push(target);
    }

    let x_with_time = Matrix::new(rows)?;

    let (
        random_train,
        random_test,
        y_random_train,
        y_random_test,
    ) = TrainTestSplit::new()
        .with_test_size(0.20)
        .with_shuffle(true)
        .with_random_state(42)
        .split(&x_with_time, &targets)?;

    let mut random_model = LinearRegression::new();
    random_model.fit(
        &feature_only(&random_train)?,
        &y_random_train,
    )?;
    let random_predictions =
        random_model.predict(&feature_only(&random_test)?)?;

    let random_future_indices: Vec<usize> =
        (0..random_test.nrows())
            .filter(|&i| random_test.get(i, 1) >= 800.0)
            .collect();
    let random_future =
        random_test.select_rows(&random_future_indices)?;
    let y_random_future: Vec<f64> = random_future_indices
        .iter()
        .map(|&i| y_random_test[i])
        .collect();
    let random_future_predictions = random_model.predict(
        &feature_only(&random_future)?,
    )?;

    let chronological_train_indices: Vec<usize> =
        (0..800).collect();
    let chronological_test_indices: Vec<usize> =
        (800..1_000).collect();
    let chronological_train = x_with_time
        .select_rows(&chronological_train_indices)?;
    let chronological_test = x_with_time
        .select_rows(&chronological_test_indices)?;
    let y_chronological_train = targets[..800].to_vec();
    let y_chronological_test = targets[800..].to_vec();

    let mut chronological_model = LinearRegression::new();
    chronological_model.fit(
        &feature_only(&chronological_train)?,
        &y_chronological_train,
    )?;
    let chronological_predictions = chronological_model.predict(
        &feature_only(&chronological_test)?,
    )?;

    println!("Regime starts at row 800 of 1000\n");
    println!("split                     train future   test future");
    println!(
        "random                    {:>12}   {:>11}",
        future_count(&random_train),
        future_count(&random_test),
    );
    println!(
        "chronological             {:>12}   {:>11}\n",
        future_count(&chronological_train),
        future_count(&chronological_test),
    );

    println!("fitted line               intercept    slope");
    println!(
        "random                    {:>9.3}  {:>7.3}",
        random_model.intercept(),
        random_model.coef()[0],
    );
    println!(
        "chronological             {:>9.3}  {:>7.3}\n",
        chronological_model.intercept(),
        chronological_model.coef()[0],
    );

    println!(
        "evaluation                 rows      MAE     RMSE \
         R²      bias"
    );
    report(
        "random test (mixed)",
        &y_random_test,
        &random_predictions,
    )?;
    report(
        "random test (future)",
        &y_random_future,
        &random_future_predictions,
    )?;
    report(
        "chronological future",
        &y_chronological_test,
        &chronological_predictions,
    )?;

    Ok(())
}
```

This is the output I measured:

```text
Regime starts at row 800 of 1000

split                     train future   test future
random                             156            44
chronological                        0           200

fitted line               intercept    slope
random                       23.797    5.018
chronological                19.941    5.012

evaluation                 rows      MAE     RMSE        R²      bias
random test (mixed)        200    6.558    8.452    0.7113    -0.572
random test (future)        44   16.206   16.281   -0.7220   -16.206
chronological future       200   20.000   20.050   -0.9024   -20.000
```

The random score is numerically correct. It belongs to an easier deployment story.

## Random splitting let training visit the future

The new regime occupies 200 of 1,000 rows, or 20% of the dataset.

After shuffling, the random training set contains 156 future-regime observations. Its test set contains the remaining 44. The model has already seen examples from both sides of the shift before it is evaluated.

Because `signal` alone cannot identify the regime, the random model compromises between the two target levels. Its fitted intercept is `23.797`, close to the weighted population average:

```text
old level: 20
new level: 40
mixed learned level: about 24
```

Most random test rows come from the old regime, where this compromise is wrong by only about four units. A smaller group comes from the new regime, where it is wrong by about sixteen.

The aggregate result is:

```text
MAE  = 6.558
RMSE = 8.452
R²   = 0.7113
```

That score answers a legitimate question if future requests will be randomly drawn from the same stationary mixture and training will contain that mixture too.

It does not answer “How will a model trained today perform after the system changes tomorrow?”

## The future slice was already failing inside the random test

The random test set contains 44 future rows. When I score only those rows:

```text
MAE  = 16.206
RMSE = 16.281
R²   = -0.7220
bias = -16.206
```

The positive aggregate R² and negative future-only R² come from the same fitted model and same random test set.

Aggregation hid the failure because 78% of the random test rows belonged to the easier old regime. A global metric weighted that majority accordingly.

This is why temporal slicing remains useful even when a random split is otherwise defensible. I want performance by period, not only across periods.

It is also a reminder that one mean score can hide a cohort whose errors have a consistent direction. The near-zero overall bias of `-0.572` is not evidence that the model is unbiased everywhere. Positive and negative segment biases can cancel.

## Chronological holdout recreated the deployment boundary

The chronological model trains on rows 0 through 799. It never sees the new level. Its fitted line remains close to the original relationship:

```text
intercept = 19.941
slope     = 5.012
```

The test set is exactly the next 200 rows, all from the shifted regime. Predictions are systematically 20 units too low:

```text
MAE  = 20.000
RMSE = 20.050
R²   = -0.9024
bias = -20.000
```

Chronological splitting did not make the model worse. It removed information the deployed model would not have had and exposed the consequence.

That is what validation is supposed to do.

## A reproducible seed does not create a valid boundary

The random example uses:

```rust
.with_shuffle(true)
.with_random_state(42)
```

The fixed seed is valuable. It makes the partition repeatable, which helps debugging and comparison.

It does not make future rows appropriate training data.

Reproducibility answers “Can I recreate this experiment?” Experimental design answers “Does this experiment represent the decision I will make?” I need both.

Changing seeds and averaging random splits would reduce sampling noise around the wrong estimand. It would not turn a random mixture into a forecast.

## `shuffle(false)` is not automatically a future holdout

There is a datarust API detail worth knowing.

`TrainTestSplit` defaults to shuffling. If shuffling is disabled, its current implementation assigns the *first* requested fraction to the test set and the remaining rows to training.

For ascending time order, this would test on the oldest 20% and train on the newer 80%:

```text
test:  rows 0–199
train: rows 200–999
```

That reverses the production direction and still lets training see the future of every test row.

This is why the example constructs chronological indices explicitly:

```rust
let train_idx: Vec<usize> = (0..800).collect();
let test_idx: Vec<usize> = (800..1_000).collect();

let x_train = x.select_rows(&train_idx)?;
let x_test = x.select_rows(&test_idx)?;
```

“Do not shuffle” preserves order. It does not define which end represents the future. I verify the returned dates and boundaries rather than inferring them from a builder flag.

## The timestamp was not a model feature

The example stores time in column 1 so we can count future rows after the random split. Before fitting, `feature_only` selects column 0.

That distinction is deliberate. Giving a model a timestamp does not automatically solve drift. A linear time coefficient can learn a gradual trend but may fail on a discontinuous policy change. Calendar features can express seasonality but cannot know an unseen product launch will shift demand by 20.

Time can be predictive, a split key, both, or neither. The roles should be explicit.

More importantly, a feature must be available at prediction time. “This row belongs to the post-change regime” is easy to label after the change and impossible for a pre-change model to learn without some observable cause.

## Preprocessing has to respect time too

This controlled model has one numeric feature and no preprocessing. A real pipeline might include scaling, imputation, encoding, PCA, or feature selection.

Every fitted preprocessing step must learn from the chronological training window only:

```text
past rows -> fit scaler/encoder/model
future rows -> transform and predict only
```

Computing category vocabularies, medians, quantiles, or selected features on the full timeline lets future distribution information leak backward even if the final estimator uses a chronological split.

The unit I validate is the whole fitted pipeline.

Target-derived features require even more care. A rolling average must use values available before the prediction timestamp, with label delays respected. A perfectly chronological row split can still leak through a feature computed with tomorrow's outcome.

## One holdout date is not enough

The final 20% holdout answers one historically specific question. Maybe row 800 contains an unusual shock. Maybe the model performs well across most months and fails only there. Maybe the latest window is exactly the risk production needs us to see.

I usually add walk-forward backtests:

```text
train 0–399   -> validate 400–499
train 0–499   -> validate 500–599
train 0–599   -> validate 600–699
train 0–699   -> validate 700–799
train 0–799   -> validate 800–899
```

An expanding window answers how performance evolves as more history becomes available. A rolling window can be better when old regimes become actively misleading.

The window length, prediction horizon, retraining cadence, and label delay should match production. Predicting tomorrow from data through today is not the same task as predicting the next quarter from data that arrives two weeks late.

datarust currently provides the matrix selection primitives needed to build these indices manually. The important part is making the temporal policy visible rather than hiding it behind a generic cross-validation mean.

## When random splitting is still correct

Chronological evaluation is not universally superior.

If observations are genuinely exchangeable and production samples from the same stable population, random splitting uses data efficiently. Randomization can reduce accidental ordering effects and produce representative train/test mixtures.

Examples may include a controlled laboratory dataset, independently sampled manufactured parts from a stable process, or a one-time static inference task where time has no deployment meaning.

The deciding question is not whether a timestamp column exists. It is whether the system must generalize across time.

Other boundaries may matter more:

- Keep all rows from one customer in the same split.
- Hold out entire devices, hospitals, stores, or geographies.
- Separate batches produced under different processes.
- Combine group and time constraints.

The validation boundary should match the source of independence production requires.

## What the random score forgot to say

The random split's `0.7113` R² was not fake. It measured interpolation across a timeline whose future regime was already represented in training.

The score became misleading only when interpreted as a forecast.

The chronological result told the operational story:

- Training contained zero post-change rows.
- Testing contained all 200 post-change rows.
- The feature slope remained correct.
- The unseen 20-unit level shift created 20 units of bias.
- R² fell below zero because the shifted model was worse than the future target's mean reference.

The lesson I keep is this:

> A test row is not truly unseen if its future world has already leaked into training.

Before trusting a validation metric, I now print the minimum and maximum timestamp in every split, count rows by regime, and ask whether the training set contains information that would not exist on the real prediction date.

The split is not preparation for evaluation.

The split defines what “unseen” means.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
