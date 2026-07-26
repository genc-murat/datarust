# My R² Was −3.5. The Model Still Got the Slope Exactly Right.

*A practical datarust guide to negative R², MAE, RMSE, explained variance, and the constant bias that made one metric look perfect while another fell through the floor.*

---

The first metric looked impossible.

```text
R² = -3.5000
```

I had seen R² described as “the percentage of variance explained,” so my instinct was to read this as negative 350% accuracy. That interpretation did not survive contact with the predictions.

The model had learned the slope exactly. Every ten-unit increase in the feature produced the correct twenty-unit increase in the target. The ordering was perfect. The shape was perfect.

Every prediction was also 60 units too low.

That constant bias is enough to make R² strongly negative. It is also invisible to explained variance in this particular example, which reports a perfect `1.0000`.

Neither metric is broken. They penalize different properties of the residuals.

Let's make the disagreement explicit with [datarust](https://crates.io/crates/datarust).

## A model trained before the market moved

Imagine a simple price model with one feature. During training, the relationship is:

```text
price = 100 + 2 × x
```

The model sees values from 0 through 50 and learns that line exactly.

By test time, the market has shifted upward. The slope is still 2, but the intercept is now 160:

```text
price = 160 + 2 × x
```

The model understands how prices move with `x`; it does not know that the whole market moved by 60.

I compare three prediction sets:

1. The fitted but systematically shifted model
2. A constant predictor equal to the test-target mean
3. A centered model with small alternating errors

Here is the complete Rust program:

```rust
use datarust::linear_model::LinearRegression;
use datarust::metrics::regression::{
    explained_variance_score, max_error, mean_absolute_error,
    mean_squared_error, r2_score,
};
use datarust::traits::Predictor;
use datarust::Matrix;

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
        "{name:<22} {:>7.2} {:>7.2} {:>7.2} \
         {:>8.4} {:>10.4} {:>8.2}",
        mean_absolute_error(actual, predicted)?,
        mean_squared_error(actual, predicted, false)?,
        max_error(actual, predicted)?,
        r2_score(actual, predicted)?,
        explained_variance_score(actual, predicted)?,
        bias,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let train_x = Matrix::new(
        (0..=50)
            .map(|x| vec![x as f64])
            .collect(),
    )?;
    let train_y: Vec<f64> = (0..=50)
        .map(|x| 100.0 + 2.0 * x as f64)
        .collect();

    let test_x = Matrix::new(vec![
        vec![55.0],
        vec![65.0],
        vec![75.0],
        vec![85.0],
        vec![95.0],
    ])?;

    // The slope is unchanged, but the market level rose by 60.
    let test_y: Vec<f64> = test_x
        .col(0)
        .iter()
        .map(|x| 160.0 + 2.0 * x)
        .collect();

    let mut model = LinearRegression::new();
    model.fit(&train_x, &train_y)?;
    let shifted_predictions = model.predict(&test_x)?;

    let test_mean =
        test_y.iter().sum::<f64>() / test_y.len() as f64;
    let mean_baseline = vec![test_mean; test_y.len()];

    let centered_noisy: Vec<f64> = test_y
        .iter()
        .zip([5.0, -5.0, 5.0, -5.0, 5.0])
        .map(|(actual, error)| actual + error)
        .collect();

    println!(
        "Fitted training line: y = {:.2} + {:.2}x",
        model.intercept(),
        model.coef()[0],
    );
    println!("Test actual:    {:?}", test_y);
    println!("Model predicts: {:?}\n", shifted_predictions);

    println!(
        "model                    MAE    RMSE  MaxErr \
         R²  Expl.Var     Bias"
    );
    report(
        "shifted trained model",
        &test_y,
        &shifted_predictions,
    )?;
    report("test-mean baseline", &test_y, &mean_baseline)?;
    report("centered noisy model", &test_y, &centered_noisy)?;

    Ok(())
}
```

This is the output I measured:

```text
Fitted training line: y = 100.00 + 2.00x
Test actual:    [270.0, 290.0, 310.0, 330.0, 350.0]
Model predicts: [210.0, 230.0, 250.0, 270.0, 290.0]

model                    MAE    RMSE  MaxErr       R²  Expl.Var     Bias
shifted trained model    60.00   60.00   60.00  -3.5000     1.0000   -60.00
test-mean baseline       24.00   28.28   40.00   0.0000     0.0000     0.00
centered noisy model      5.00    5.00    5.00   0.9688     0.9700     1.00
```

The negative R² is not mysterious once we look at what it compares.

## R² carries its own baseline

R² is defined as:

```text
           sum((actual - predicted)²)
R² = 1 - --------------------------------
           sum((actual - test_mean)²)
```

The numerator is the model's squared error. The denominator is the squared error from always predicting the mean of the evaluated targets.

For our test values, the mean is 310:

```text
actual:     270, 290, 310, 330, 350
deviation:  -40, -20,   0,  20,  40
```

The total squared deviation is:

```text
40² + 20² + 0² + 20² + 40² = 4000
```

The trained model is wrong by 60 on all five rows, so its residual sum of squares is:

```text
5 × 60² = 18000
```

Therefore:

```text
R² = 1 - 18000 / 4000
   = 1 - 4.5
   = -3.5
```

The score means the model's squared error is 4.5 times the test-mean reference error. It does not mean the model has negative accuracy, and it is not a percentage bounded at zero.

R² has an upper bound of 1 and no finite lower bound. Make predictions sufficiently bad and it can become arbitrarily negative.

## The mean baseline is mathematical, not necessarily deployable

The `test-mean baseline` in the table predicts 310 for every test row and receives R² of exactly zero by definition.

That does not mean I could have deployed it before seeing the labels. The true test-target mean is only known after evaluation.

R² uses this hindsight baseline to normalize error against the variation in the evaluated target. For an operational baseline, I would use something available at prediction time: the training mean, last month's average, a seasonal forecast, or an existing business rule.

This distinction matters when describing the result. “Worse than predicting the test mean” is mathematically accurate. “Worse than our production baseline” requires measuring the actual production baseline.

Still, a strongly negative R² is a valuable alarm. The model is performing poorly relative to the target spread in the cohort, regardless of whether the reference predictor itself is deployable.

## Explained variance ignored the constant offset

Explained variance uses a related but different formula:

```text
                            variance(actual - predicted)
explained variance = 1 - --------------------------------
                                  variance(actual)
```

Our residuals are:

```text
actual - predicted = [60, 60, 60, 60, 60]
```

Their variance is zero. Every error is identical.

So explained variance reports:

```text
1 - 0 / variance(actual) = 1.0
```

The model perfectly captures how the target varies from row to row. It gets the level completely wrong.

That is why a perfect explained-variance score does not guarantee accurate predictions. It can coexist with enormous systematic bias.

The metric is useful when I specifically care about reproducing variation, but I never read it without an absolute error metric and residual bias.

## R² contains a bias penalty that explained variance does not

Using population variance notation, mean squared residual can be decomposed into:

```text
mean(residual²)
    = variance(residual) + mean(residual)²
```

That gives a useful relationship:

```text
R² = 1 - [variance(residual) + residual_bias²]
         ---------------------------------------
                    variance(actual)

Explained variance = 1 - variance(residual)
                         ------------------
                          variance(actual)
```

When residual mean is zero, the metrics coincide. As systematic bias grows, R² falls below explained variance.

The centered noisy model demonstrates the small version of this. Its residuals alternate around zero but have a mean prediction bias of `1.00`, so explained variance is `0.9700` while R² is slightly lower at `0.9688`.

The shifted model demonstrates the extreme version: all residual variation disappears, yet squared bias overwhelms the target variance.

## MAE said what a person needed to hear

Mean absolute error for the shifted model is 60:

```text
MAE = 60 target units
```

That is immediately interpretable. If the target is measured in thousands of dollars, the average miss is $60,000. If it is delivery minutes, the average miss is one hour.

MAE weights every absolute error linearly. It is often a good description of a typical miss and aligns naturally with businesses where cost grows roughly in proportion to error size.

Unlike R², MAE depends on target units. I cannot compare an MAE of 5 across revenue dollars and temperature degrees without domain context. That apparent limitation is also its strength: it forces the metric back into the world where the prediction is used.

## RMSE agreed because every error was the same

Root mean squared error squares errors, averages them, and returns to the original units:

```text
RMSE = sqrt(mean((actual - predicted)²))
```

The shifted model has five identical 60-unit errors, so MAE and RMSE both equal 60.

They separate when error sizes vary. One catastrophic miss influences RMSE more strongly than MAE. That makes RMSE useful when large errors are disproportionately painful, but also more sensitive to outliers and label mistakes.

I often report both:

- MAE for the typical linear error burden
- RMSE to reveal whether large misses are pulling the distribution outward

The maximum error adds the worst observed miss, though it is unstable and highly sample-dependent.

## The target's spread changes R²

The same 60-unit RMSE can produce very different R² values on different test cohorts.

If actual targets vary across thousands of units, a 60-unit miss may explain most of their variation. If the cohort lies in a narrow 80-unit band, the same miss can be worse than the mean reference and produce a strongly negative score.

That makes R² dimensionless, but not context-free. It depends on the variance of the evaluated targets.

A model evaluated only on a narrow customer segment may have lower R² than on the full population while delivering the same MAE. Conversely, a broad target range can make R² look strong even when absolute errors remain commercially unacceptable.

I keep the cohort definition and target distribution next to the score.

## What a negative R² makes me investigate

Negative R² is a symptom, not a diagnosis. I look for:

- Distribution shift in the target level or feature relationships
- Extrapolation beyond the training feature range
- A missing intercept or incorrect target transformation
- Train–serve preprocessing mismatch
- Feature column reordering or unit changes
- Overfitting that fails on held-out data
- Segments with systematically different baselines
- Labels delayed, revised, or defined differently after deployment

In this controlled example, the residual plot would make the problem obvious: every residual is the same 60-unit horizontal line.

A validation-period intercept recalibration could correct that specific shift while preserving the slope. But I would estimate the correction from an approved calibration window, never from the final test labels, and I would still investigate why the market level moved.

If residual bias varies by feature value or segment, a single intercept correction is not enough. The relationship itself has changed.

## The metric set I would ship

For a regression service, my compact evaluation report includes:

```rust
let mae = mean_absolute_error(&y_test, &predictions)?;
let rmse = mean_squared_error(&y_test, &predictions, false)?;
let r2 = r2_score(&y_test, &predictions)?;
let ev = explained_variance_score(&y_test, &predictions)?;
let worst = max_error(&y_test, &predictions)?;
```

I add mean residual bias, error quantiles, and slices by time and important segments. One global average can hide one group being consistently overpredicted and another consistently underpredicted.

The metrics answer different questions:

```text
MAE                 typical absolute miss in target units
RMSE                squared-error burden, emphasizing large misses
Max error           worst observed absolute miss
R²                  improvement over the evaluated-target mean reference
Explained variance  how well residual variation is controlled
Bias                systematic direction and level of error
```

No single number replaces the others.

## Negative was the useful result

The model in this experiment learned `y = 100 + 2x` perfectly. The new environment followed `y = 160 + 2x`.

Explained variance celebrated the correct slope. MAE and RMSE exposed the 60-unit miss. R² compared that miss with the test target's modest spread and returned `-3.5000`.

All of those statements are true at the same time.

The negative score was not an invalid percentage to clip at zero. It was the metric refusing to hide systematic failure behind correct ranking and variation.

The lesson I keep is simple:

> A model can understand movement and still be wrong about level.

When R² turns negative, I do not repair the dashboard. I inspect the residuals, the baseline, the target spread, and the world that changed between training and evaluation.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
