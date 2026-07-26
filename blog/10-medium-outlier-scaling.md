# One $10,000 Transaction Changed Everyone Else's Scale

*A practical datarust comparison of StandardScaler, MinMaxScaler, RobustScaler, and QuantileTransformer — and what “robust” actually means.*

---

I once found a transaction in a dataset that was almost one hundred times larger than everything around it.

My first reaction was predictable: *that has to be bad data.*

It wasn't. The customer really had spent that much.

The value was unusual, but legitimate — which left me with a more interesting problem. I couldn't delete it, yet I also didn't want one large transaction to decide how every ordinary transaction should be represented.

This is where the apparently simple choice of a feature scaler becomes a modeling decision.

`StandardScaler`, `MinMaxScaler`, `RobustScaler`, and `QuantileTransformer` can all turn a numeric column into something easier for a model to work with. But they do not mean the same thing, and an outlier exposes those differences immediately.

So instead of discussing them in the abstract, let's give all four scalers exactly the same awkward dataset and see what they learn.

## The experiment: 100 ordinary values and one very large one

Imagine a transaction feature with 100 values evenly spread from $80.00 to $119.60. Then one real $10,000 transaction arrives.

The ordinary values cover a range of $39.60. The outlier sits $9,880.40 above the largest of them.

Here is a complete Rust program using [datarust](https://crates.io/crates/datarust). It fits each scaler on the contaminated data, but it also fits the first three on the clean data so we can see how much their learned statistics move.

```rust
use datarust::scaler::{
    MinMaxScaler, OutputDistribution, QuantileTransformer, RobustScaler,
    StandardScaler,
};
use datarust::traits::Transformer;
use datarust::Matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let clean_values: Vec<Vec<f64>> = (0..100)
        .map(|i| vec![80.0 + i as f64 * 0.4])
        .collect();

    let mut contaminated_values = clean_values.clone();
    contaminated_values.push(vec![10_000.0]);

    let clean = Matrix::new(clean_values)?;
    let contaminated = Matrix::new(contaminated_values)?;
    let checkpoints = Matrix::new(vec![
        vec![80.0],
        vec![100.0],
        vec![119.6],
        vec![10_000.0],
    ])?;

    let mut standard_clean = StandardScaler::new();
    standard_clean.fit(&clean)?;
    let mut standard = StandardScaler::new();
    standard.fit(&contaminated)?;

    let mut minmax_clean = MinMaxScaler::new();
    minmax_clean.fit(&clean)?;
    let mut minmax = MinMaxScaler::new();
    minmax.fit(&contaminated)?;

    let mut robust_clean = RobustScaler::new();
    robust_clean.fit(&clean)?;
    let mut robust = RobustScaler::new();
    robust.fit(&contaminated)?;

    let mut quantile = QuantileTransformer::new(101)?
        .output_distribution(OutputDistribution::Uniform);
    quantile.fit(&contaminated)?;

    println!("Learned statistics (clean -> with the $10,000 value)");
    println!(
        "Standard mean: {:.3} -> {:.3}",
        standard_clean.mean()[0],
        standard.mean()[0]
    );
    println!(
        "Standard std:  {:.3} -> {:.3}",
        standard_clean.std()[0],
        standard.std()[0]
    );
    println!(
        "MinMax range:  {:.3} -> {:.3}",
        minmax_clean.data_range()[0],
        minmax.data_range()[0]
    );
    println!(
        "Robust center: {:.3} -> {:.3}",
        robust_clean.center()[0],
        robust.center()[0]
    );
    println!(
        "Robust scale:  {:.3} -> {:.3}",
        robust_clean.scale()[0],
        robust.scale()[0]
    );

    let standard_values = standard.transform(&checkpoints)?;
    let minmax_values = minmax.transform(&checkpoints)?;
    let robust_values = robust.transform(&checkpoints)?;
    let quantile_values = quantile.transform(&checkpoints)?;

    println!("\nvalue      standard      minmax       robust     quantile");
    for row in 0..checkpoints.nrows() {
        println!(
            "{:8.1}  {:12.6}  {:10.6}  {:11.6}  {:11.6}",
            checkpoints.get(row, 0),
            standard_values.get(row, 0),
            minmax_values.get(row, 0),
            robust_values.get(row, 0),
            quantile_values.get(row, 0),
        );
    }

    Ok(())
}
```

And this is the actual output:

```text
Learned statistics (clean -> with the $10,000 value)
Standard mean: 99.800 -> 197.822
Standard std:  11.546 -> 980.285
MinMax range:  39.600 -> 9920.000
Robust center: 99.800 -> 100.000
Robust scale:  19.800 -> 20.000

value      standard      minmax       robust     quantile
    80.0     -0.120191    0.000000    -1.000000     0.000000
   100.0     -0.099789    0.002016     0.000000     0.500000
   119.6     -0.079795    0.003992     0.980000     0.990000
 10000.0      9.999313    1.000000   495.000000     1.000000
```

There is a lot hiding in those few lines.

## StandardScaler: the outlier changes both the ruler and its origin

`StandardScaler` subtracts the mean and divides by the standard deviation:

```text
z = (x - mean) / standard_deviation
```

Without the outlier, the mean is $99.80 and the standard deviation is about $11.55. Both are reasonable summaries of the ordinary transactions.

After adding $10,000, the mean jumps to $197.82 and the standard deviation to $980.29. The outlier has changed both the center and the unit of measurement.

That leaves all three ordinary checkpoints packed into a tiny interval:

```text
$80.00   -> -0.120191
$100.00  -> -0.099789
$119.60  -> -0.079795
```

The original $39.60 difference between the lowest and highest ordinary values has become roughly `0.04` standard units. For a distance-based model, that feature now barely separates normal customers at all.

The $10,000 value becomes almost exactly `10.0`, which may look surprisingly tame. That is not evidence that the value stopped being extreme. The same observation inflated the standard deviation used to measure its own extremeness.

There is a neat statistical detail here. When one observation dominates a sample of 101 values, its standardized value approaches the square root of 100 — about 10. The outlier effectively brings a larger ruler to its own measurement.

Standard scaling is still a perfectly sensible default when a feature is reasonably well behaved. It is just not outlier-resistant.

## MinMaxScaler: one value owns almost the entire interval

`MinMaxScaler` maps the fitted minimum to 0 and the fitted maximum to 1:

```text
scaled = (x - minimum) / (maximum - minimum)
```

Before the outlier, the range is $39.60. After it, the range is $9,920.

The result is visually brutal. The entire ordinary population occupies less than four thousandths of the available `[0, 1]` interval:

```text
$80.00   -> 0.000000
$100.00  -> 0.002016
$119.60  -> 0.003992
$10,000  -> 1.000000
```

Technically, MinMaxScaler has done exactly what we asked. Practically, it has devoted almost all of its resolution to the empty space between $119.60 and $10,000.

This is why “my neural network expects values between zero and one” is not, by itself, a good reason to choose min-max scaling. A bounded output during training says nothing about how usefully the ordinary values are distributed inside those bounds.

There is another operational detail worth knowing: datarust's `MinMaxScaler` does not clamp future values. If it is fitted with a maximum of $10,000 and later sees $20,000, the transformed value will be greater than 1. That extrapolation is often the mathematically honest behavior, but your downstream code should be ready for it.

Min-max scaling works best when the feature has meaningful, stable bounds and surprising extremes are unlikely to redefine them.

## RobustScaler: protect the ruler, not the data

`RobustScaler` replaces the mean with the median and the standard deviation with the interquartile range, or IQR:

```text
scaled = (x - median) / IQR
```

The median describes the center of the ordered data. The IQR measures the distance between the 25th and 75th percentiles. A single value at the far end has very little influence on either.

Our output makes that visible:

```text
median: 99.800 -> 100.000
IQR:    19.800 -> 20.000
```

The $10,000 transaction barely changes the learned coordinate system. Ordinary values retain a useful spread:

```text
$80.00   -> -1.00
$100.00  ->  0.00
$119.60  ->  0.98
```

This is usually what people want when they ask for scaling that is robust to outliers.

But look at the outlier itself:

```text
$10,000 -> 495.00
```

That number is the most important result in this article.

**RobustScaler does not remove, clip, cap, repair, or make an outlier harmless.** It prevents the outlier from deciding the center and scale used for everybody else. The transformed outlier remains 495 IQRs above the median because it really is that far away under this definition.

If I pass this feature into K-means or another method based on Euclidean distance, that one point can still exert enormous influence. Robust preprocessing statistics do not automatically make the downstream algorithm robust.

If the business decision is to cap transactions at a known limit, winsorize the top percentile, apply `log1p`, or remove data-entry errors, that is a separate transformation backed by separate reasoning. A scaler should not quietly make that decision for us.

## QuantileTransformer: keep the order, rewrite the distance

`QuantileTransformer` takes a different approach. Instead of asking how many dollars or standard deviations separate two values, it asks where each value sits in the fitted distribution.

With a uniform output distribution, the smallest fitted value maps to 0, the median lands around 0.5, and the largest maps to 1:

```text
$80.00   -> 0.00
$100.00  -> 0.50
$119.60  -> 0.99
$10,000  -> 1.00
```

The outlier can no longer create an enormous numeric distance. It occupies the top of the rank scale, while ordinary observations remain spread across nearly the whole interval.

That sounds ideal, but it comes with a tradeoff: the transformation is nonlinear.

The distance from $119.60 to $10,000 is $9,880.40, yet their transformed distance is only `0.01`. Meanwhile, $20 differences in the ordinary range become roughly half of the transformed interval. Quantile transformation preserves order, not the original spacing.

For a strongly skewed feature, that may be exactly what the model needs. For a feature where absolute differences carry important meaning, it may erase valuable structure.

Values outside the fitted range also behave differently from MinMaxScaler. With a uniform output, datarust's `QuantileTransformer` maps observations above the fitted maximum to 1 and observations below the fitted minimum to 0. In production, multiple new extremes can therefore become indistinguishable at the boundary.

This is not better or worse in isolation. It is a statement about what information we choose to preserve.

## So which scaler should I use?

I don't think there is a universal ranking. I use the scaler whose assumptions match the feature and the model.

- **StandardScaler** is a strong baseline for roughly symmetric features without severe outliers. It preserves linear spacing, but its learned mean and standard deviation are sensitive to extremes.

- **MinMaxScaler** is useful when known bounds matter and placing values on a fixed interval is meaningful. It preserves linear spacing, but one unexpected minimum or maximum can compress everything else.

- **RobustScaler** is my usual first experiment when legitimate outliers should not control the center and scale. It preserves linear relationships after rescaling, but the outliers themselves remain extreme.

- **QuantileTransformer** is compelling for heavily skewed distributions when relative rank matters more than raw distance. It limits the numeric effect of extremes, but rewrites spacing and saturates outside the fitted range.

The downstream model matters too. Trees generally care much less about monotonic scaling than linear models, neural networks, K-means, k-nearest neighbors, and optimization methods based on gradients or distances. I would not add a nonlinear quantile transform to a tree pipeline just because it makes a histogram look tidy.

## The scaler cannot tell you whether the outlier is wrong

Before choosing any transformation, I want to know what the large value represents.

Is $10,000 a misplaced decimal point? A currency conversion bug? A wholesale customer in a mostly retail dataset? A real event we specifically want the model to recognize?

Those cases deserve different responses.

If it is a data error, correct it at the source. If it is a different population, a segment indicator or separate model may be more honest. If transaction amounts are naturally right-skewed, a logarithmic transform may represent the process better than any scaler. If it is rare but important, suppressing its effect might be the wrong objective entirely.

Preprocessing is not a substitute for understanding the column.

## One rule that applies to all four

Whichever scaler I choose, I fit it on the training data only.

```rust
let mut scaler = RobustScaler::new();
let x_train_scaled = scaler.fit_transform(&x_train)?;
let x_test_scaled = scaler.transform(&x_test)?;
```

Fitting on the full dataset lets validation or test observations influence the learned mean, range, median, IQR, or quantiles. That is data leakage. The metrics may look slightly better, but they no longer describe how the pipeline will behave on genuinely unseen data.

In production, I also monitor the raw feature — not just the transformed one. A scaler can keep returning valid numbers while the underlying distribution drifts into a completely different regime.

## The part I now remember

Before running this example, I understood the definitions of all four scalers. But definitions are easy to file away and forget.

The numbers are harder to forget:

- Standard scaling pushed ordinary values into a band only `0.04` wide.
- Min-max scaling squeezed them below `0.004`.
- Robust scaling kept their useful spread, while leaving the outlier at `495`.
- Quantile transformation spread them by rank, but turned a $9,880 gap into `0.01`.

Each result is correct. Each tells a different story about what distance should mean.

That is the real choice behind scaling. We are not merely making numbers smaller. We are deciding which observations get to define the ruler — and what the model is allowed to forget.

---

*The complete example above was run against the current datarust codebase. If you're building preprocessing and classical machine-learning workflows in Rust, you can find the crate on [crates.io](https://crates.io/crates/datarust).*
