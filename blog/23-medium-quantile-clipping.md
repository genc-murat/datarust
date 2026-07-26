# The Training R² Was 1.0. Every New Extreme Still Got the Same Prediction.

*A practical datarust guide to QuantileTransformer, empirical ranks, and the quiet ceiling waiting beyond the training range.*

---

The model fit the training data almost perfectly.

```text
training R²:  1.000000000000
training MAE: 1.300e-14
```

Then four organizations arrived with monthly event volumes of `10,000`, `12,100`, `22,500`, and `100,000`.

The model predicted the same support workload for all four:

```text
220.00 hours
```

This was not a broken linear regression. It was not a floating-point problem, and the feature order was correct.

The `QuantileTransformer` before the model had learned that `10,000` was the largest value in training. Every value at or above that maximum became the same upper percentile. Once `12,100` and `100,000` both turned into `1.0`, no downstream estimator could tell them apart.

That is the bargain behind an empirical quantile transform. It can make a skewed training distribution beautifully regular without learning how magnitude should continue beyond the observations it has seen.

Let's reproduce the good fit and the production ceiling with [datarust](https://crates.io/crates/datarust).

## A relationship designed to be easy by rank

I use a controlled workload example with 100 training organizations.

Their monthly event volumes are squared values:

```text
1², 2², 3², ... 100²
```

The expected support workload grows with the square root of volume:

```text
hours = 20 + 2 × sqrt(volume)
```

That gives training targets from 22 to 220 hours.

The construction is deliberate. Every next organization has the next larger event volume, so its empirical percentile rises by exactly `1/99`. A uniform quantile transform turns the curved raw relationship into a straight line:

```text
volume:       1, 4, 9, ... 10,000
quantile:     0, 1/99, 2/99, ... 1
hours:       22, 24, 26, ... 220
```

Linear regression should be nearly perfect inside that learned range. The interesting question is what happens after it.

I also fit a second linear model on the domain-informed `sqrt(volume)` feature. That model is not presented as a universally better preprocessing recipe; I have intentionally given it the correct functional form. It is here to make the quantile model's inability to extrapolate visible.

Here is the complete Rust program:

```rust
use datarust::linear_model::LinearRegression;
use datarust::metrics::regression::{
    mean_absolute_error, r2_score,
};
use datarust::pipeline::Pipeline;
use datarust::scaler::{
    OutputDistribution, QuantileTransformer,
};
use datarust::traits::{Predictor, Transformer};
use datarust::transformer_kind::TransformerKind;
use datarust::Matrix;

fn sqrt_features(
    x: &Matrix,
) -> Result<Matrix, Box<dyn std::error::Error>> {
    Ok(Matrix::new(
        (0..x.nrows())
            .map(|i| vec![x.get(i, 0).sqrt()])
            .collect(),
    )?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let train_rows: Vec<Vec<f64>> = (1..=100)
        .map(|size| vec![(size * size) as f64])
        .collect();
    let y_train: Vec<f64> = (1..=100)
        .map(|size| 20.0 + 2.0 * size as f64)
        .collect();
    let x_train = Matrix::new(train_rows)?;

    let mut rank_model = Pipeline::new()
        .push(
            "quantiles",
            TransformerKind::QuantileTransformer(
                QuantileTransformer::new(100)?
                    .output_distribution(
                        OutputDistribution::Uniform,
                    ),
            ),
        )
        .with_estimator(LinearRegression::new());
    rank_model.fit(&x_train, &y_train)?;
    let train_pred = rank_model.predict(&x_train)?;

    let future = Matrix::new(vec![
        vec![10_000.0],
        vec![12_100.0],
        vec![22_500.0],
        vec![100_000.0],
    ])?;
    let future_truth: Vec<f64> = (0..future.nrows())
        .map(|i| 20.0 + 2.0 * future.get(i, 0).sqrt())
        .collect();
    let rank_values = rank_model.transform(&future)?;
    let rank_pred = rank_model.predict(&future)?;

    let mut sqrt_model = LinearRegression::new();
    sqrt_model.fit(
        &sqrt_features(&x_train)?,
        &y_train,
    )?;
    let sqrt_pred =
        sqrt_model.predict(&sqrt_features(&future)?)?;

    println!(
        "training R2:  {:.12}",
        r2_score(&y_train, &train_pred)?,
    );
    println!(
        "training MAE: {:.3e}\n",
        mean_absolute_error(&y_train, &train_pred)?,
    );

    println!(
        "volume     quantile   rank prediction   \
         sqrt prediction   expected"
    );
    for i in 0..future.nrows() {
        println!(
            "{:>8.0}     {:>5.3}         {:>7.2}   \
             {:>15.2}      {:>7.2}",
            future.get(i, 0),
            rank_values.get(i, 0),
            rank_pred[i],
            sqrt_pred[i],
            future_truth[i],
        );
    }
    println!(
        "\nfuture MAE — quantile: {:.2}, sqrt: {:.2}",
        mean_absolute_error(&future_truth, &rank_pred)?,
        mean_absolute_error(&future_truth, &sqrt_pred)?,
    );

    let probes = Matrix::new(vec![
        vec![0.0],
        vec![1.0],
        vec![2_500.0],
        vec![10_000.0],
        vec![12_100.0],
        vec![22_500.0],
        vec![100_000.0],
    ])?;

    let mut uniform = QuantileTransformer::new(100)?;
    uniform.fit(&x_train)?;
    let uniform_values = uniform.transform(&probes)?;

    let mut normal = QuantileTransformer::new(100)?
        .output_distribution(OutputDistribution::Normal);
    normal.fit(&x_train)?;
    let normal_values = normal.transform(&probes)?;

    println!("\nprobe      uniform      normal");
    for i in 0..probes.nrows() {
        println!(
            "{:>8.0}     {:>7.4}     {:>8.4}",
            probes.get(i, 0),
            uniform_values.get(i, 0),
            normal_values.get(i, 0),
        );
    }

    Ok(())
}
```

This is the output I measured:

```text
training R2:  1.000000000000
training MAE: 1.300e-14

volume     quantile   rank prediction   sqrt prediction   expected
   10000     1.000          220.00            220.00       220.00
   12100     1.000          220.00            240.00       240.00
   22500     1.000          220.00            320.00       320.00
  100000     1.000          220.00            652.46       652.46

future MAE — quantile: 138.11, sqrt: 0.00

probe      uniform      normal
       0      0.0000      -5.9978
       1      0.0000      -5.9978
    2500      0.4949      -0.0127
   10000      1.0000       5.9978
   12100      1.0000       5.9978
   22500      1.0000       5.9978
  100000      1.0000       5.9978
```

The training metric was accurate. It was also answering a much narrower question than production asked.

## QuantileTransformer learned an empirical CDF

During `fit`, datarust sorts each feature independently and stores reference values at evenly spaced quantile positions. With 100 observations and `n_quantiles = 100`, every training value becomes a reference.

During `transform`, a value between references is located with binary search and linearly interpolated to a percentile between zero and one.

For example, `2,500` is `50²`, the 50th training observation:

```text
zero-based rank = 49
percentile      = 49 / 99
                = 0.4949
```

This rank-like representation compresses the enormous gaps in the raw upper tail. The difference between `9,801` and `10,000` receives the same percentile spacing as the difference between `1` and `4`.

That is often the point. Quantile transformation can reduce the influence of extreme magnitudes, spread dense regions apart, and produce a uniform or approximately normal marginal distribution.

It does not learn a formula for values beyond its reference table.

## The empirical CDF has hard edges

The fitted feature range is:

```text
minimum:     1
maximum: 10,000
```

The transform applies these boundary rules:

```text
value <= training minimum  -> percentile 0
value >= training maximum  -> percentile 1
```

That is why all of these collapse to the same output:

```text
10,000   -> 1.0
12,100   -> 1.0
22,500   -> 1.0
100,000  -> 1.0
```

Once the values have collapsed, the loss is downstream and irreversible. A more flexible estimator after the transformer cannot recover the missing magnitude. A neural network, tree, or polynomial model would still receive four identical feature values.

The lower boundary behaves the same way. In the probe table, both `0` and the training minimum `1` become `0.0`.

## Normal output changes the label, not the ceiling

Switching to `OutputDistribution::Normal` does not restore extrapolation.

The transformer first finds the same empirical percentile. It then passes that percentile through the inverse standard-normal CDF. Because exact zero and one would map to negative and positive infinity, datarust clamps them slightly inward.

In this run:

```text
uniform upper boundary: 1.0000
normal upper boundary:  5.9978
```

Every above-maximum value still receives `5.9978`. The coordinate looks different, but it contains the same amount of distinguishing information: none.

Choosing a normal output distribution can be useful when the next estimator benefits from a bell-shaped marginal. It is not an extrapolation strategy.

## More quantiles do not extend the range

It is tempting to increase `n_quantiles` and expect a more detailed tail.

More reference points can improve resolution *between* the observed minimum and maximum. They cannot create references outside the fitted data.

In datarust, the effective number of quantiles is also capped at the number of training rows:

```text
effective quantiles = min(n_quantiles, training rows)
```

Setting `n_quantiles` to 10,000 for these 100 rows would still produce at most 100 reference positions. More importantly, `100,000` would still be above the final reference and still map to the upper boundary.

Resolution and extrapolation are separate properties.

## A perfect random split could miss the failure

If I randomly split observations drawn from the same bounded distribution, training and validation will probably share similar ranges. The validation maximum may fall below the training maximum, and interpolation will work as intended.

The held-out score can look excellent while the model remains untested on future scale.

This becomes especially dangerous when a feature grows over time:

- transaction volume rises with adoption,
- account size increases as the product moves upmarket,
- sensor readings shift after new hardware is deployed,
- prices rise with inflation,
- traffic peaks exceed anything in the launch-period training set.

A chronological split is more revealing because it preserves that growth. I also add explicit range-stress cases rather than hoping the validation sample contains them by chance.

The question is not only “Does validation come from the same distribution?” It is also “What will the preprocessing contract do when it does not?”

## The square-root model had information the rank model did not

The comparison model transformed volume using a known relationship:

```text
sqrt(10,000)  = 100
sqrt(12,100)  = 110
sqrt(22,500)  = 150
sqrt(100,000) ≈ 316.23
```

These remain distinct beyond the training range, so linear regression continues the learned relationship and matches the controlled targets.

Real systems rarely hand me the correct formula this cleanly. The lesson is not “always take a square root.” It is that a functional transform and an empirical rank transform encode different assumptions.

A square root says:

> Magnitude continues to matter outside the observed range, but with diminishing effect.

An empirical quantile transform says:

> Position within the fitted distribution matters; values outside its boundaries share an endpoint.

I want that assumption to be explicit because the training metric alone will not reveal it.

## When the ceiling is desirable

Clipping is not automatically a defect.

It can be a feature when:

- Raw extremes are mostly measurement errors or unstable outliers.
- I intentionally want bounded influence from any single feature.
- Ordering matters more than absolute distance.
- The model should not extrapolate into an unsupported region.
- A separate business rule handles observations outside the training envelope.

For a fraud feature, treating every extremely large transfer as “top percentile” may be safer than allowing one enormous value to dominate a linear score. For a quality-control process, an above-limit measurement may all trigger the same manual review path.

In those systems, saturation is policy.

The mistake is allowing it to become policy accidentally.

## What I monitor in production

When I deploy a quantile-transformed feature, I keep the raw value available for observability. Monitoring only the transformed value hides the size of an excursion because every upper overflow looks identical.

I track:

- the fraction of requests below the fitted minimum,
- the fraction above the fitted maximum,
- raw p50, p95, p99, minimum, and maximum,
- the proportion mapped exactly to zero or one,
- performance for in-range and out-of-range cohorts,
- how those rates change over time.

I also store the fitted training bounds with the model metadata. An alert such as “8.4% of event-volume values exceeded the training maximum this week” is far more actionable than a stable transformed p99 of `1.0`.

Depending on the application, an overflow can trigger a fallback model, a domain rule, a warning, a capped prediction with an explicit flag, or retraining. Silently returning the same ordinary-looking prediction is the least informative option.

## Fit it inside the training boundary

The response to range clipping is not to fit the transformer on validation or production data before evaluation. That would leak future distribution information into preprocessing and make the measured model impossible to reproduce honestly.

The safe sequence remains:

1. Fit `QuantileTransformer` on training features only.
2. Transform validation data with those frozen references.
3. Treat boundary hits as part of the validation result.
4. Fit the final production artifact according to the chosen retraining policy.
5. Reuse that exact fitted artifact during serving.

The `Pipeline` in the example keeps preprocessing and regression together, which prevents a serving path from accidentally using a separately refitted quantile table.

No-leakage evaluation and range monitoring solve different problems. I need both.

## The transformer answered a rank question perfectly

Our model did not contradict its training score.

Inside the observed range, event volume rank and support workload were perfectly aligned. `QuantileTransformer` exposed that rank, and linear regression learned it with an R² of `1.0`.

Outside the range, the transformer had only one answer: upper endpoint.

```text
12,100  -> 1.0 -> 220 hours
22,500  -> 1.0 -> 220 hours
100,000 -> 1.0 -> 220 hours
```

Those values were numerically different and operationally different. The model never saw the distinction because preprocessing removed it first.

So when a quantile transform improves my validation score, I now ask one more question before celebrating:

> If tomorrow's largest value is ten times today's maximum, do I want the model to know the difference?

If the answer is yes, an empirical rank alone is not enough.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
