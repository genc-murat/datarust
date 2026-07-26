# Two Orders Were Two Cents Apart. My Feature Said They Were Different Categories.

*A practical datarust guide to Uniform, Quantile, and KMeans binning — and the cliffs we create when continuous values become buckets.*

---

The two orders were almost identical.

```text
$22.365
$22.385
```

The difference was two cents.

After preprocessing, one became `bin 1` and the other became `bin 2`.

That is not a rounding bug. It is what discretization does: it takes a continuous line and draws borders across it. Values on the same side of a border become identical. Values on opposite sides become different categories, no matter how close they were before.

Sometimes that simplification is useful. Bins can make a noisy feature more stable, let a linear model express nonlinear effects, and produce human-readable groups such as “small,” “medium,” and “large.” They can also create arbitrary cliffs, erase distances, and behave strangely when the data has a long tail.

The choice of binning strategy decides where those tradeoffs land.

To see the difference, I gave datarust's `KBinsDiscretizer` the same skewed order-value data three times.

## One hundred orders and a very long tail

The training feature contains 100 order values:

- 90 ordinary orders from $10.00 to $32.25
- 8 larger orders from $60.00 to $130.00
- One $300 order
- One $600 order

I ask each strategy for four bins:

1. `Uniform`: equal-width intervals between the minimum and maximum
2. `Quantile`: intervals intended to contain equal numbers of observations
3. `KMeans`: one-dimensional clusters based on the observed values

I also transform five checkpoints, including the two nearly identical orders and a new $700 order outside the fitted range.

Here is the complete Rust program:

```rust
use datarust::scaler::{
    BinStrategy, KBinsDiscretizer, KBinsEncode,
};
use datarust::traits::{FeatureNames, Transformer};
use datarust::Matrix;

fn summarize(
    name: &str,
    strategy: BinStrategy,
    train: &Matrix,
    checkpoints: &Matrix,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut kb = KBinsDiscretizer::new(4)?
        .strategy(strategy)
        .encode(KBinsEncode::Ordinal);
    let train_bins = kb.fit_transform(train)?;
    let checkpoint_bins = kb.transform(checkpoints)?;

    let mut counts = vec![0_usize; kb.n_actual_bins()[0]];
    for bin in train_bins.col(0) {
        counts[bin as usize] += 1;
    }

    println!("{name}");
    println!(
        "edges:  [{}]",
        kb.bin_edges()[0]
            .iter()
            .map(|edge| format!("{edge:.4}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("counts: {counts:?}");
    println!(
        "checkpoints: {}\n",
        checkpoints
            .col(0)
            .iter()
            .zip(checkpoint_bins.col(0))
            .map(|(value, bin)| {
                format!("${value:.3}->bin{bin:.0}")
            })
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut values: Vec<Vec<f64>> = (0..90)
        .map(|i| vec![10.0 + i as f64 * 0.25])
        .collect();
    values.extend(
        (0..8).map(|i| vec![60.0 + i as f64 * 10.0]),
    );
    values.push(vec![300.0]);
    values.push(vec![600.0]);

    let train = Matrix::new(values)?;
    let checkpoints = Matrix::new(vec![
        vec![22.365],
        vec![22.385],
        vec![100.0],
        vec![300.0],
        vec![700.0],
    ])?;

    println!(
        "100 order values: $10.00 to $600.00, four requested bins\n"
    );
    summarize(
        "Uniform",
        BinStrategy::Uniform,
        &train,
        &checkpoints,
    )?;
    summarize(
        "Quantile",
        BinStrategy::Quantile,
        &train,
        &checkpoints,
    )?;
    summarize(
        "KMeans",
        BinStrategy::KMeans,
        &train,
        &checkpoints,
    )?;

    let mut one_hot = KBinsDiscretizer::new(4)?
        .strategy(BinStrategy::Quantile)
        .encode(KBinsEncode::OneHotDense);
    one_hot.fit(&train)?;
    let encoded = one_hot.transform(&checkpoints)?;

    println!("Quantile one-hot features:");
    println!(
        "{:?}",
        one_hot.feature_names_out(
            Some(&["order_value".to_string()])
        )
    );
    for i in 0..encoded.nrows() {
        println!(
            "${:.3} -> {:?}",
            checkpoints.get(i, 0),
            encoded.row(i),
        );
    }

    Ok(())
}
```

This is the actual output:

```text
100 order values: $10.00 to $600.00, four requested bins

Uniform
edges:  [10.0000, 157.5000, 305.0000, 452.5000, 600.0000]
counts: [98, 1, 0, 1]
checkpoints: $22.365->bin0, $22.385->bin0, $100.000->bin0,
             $300.000->bin1, $700.000->bin3

Quantile
edges:  [10.0000, 16.1875, 22.3750, 28.5625, 600.0000]
counts: [25, 25, 25, 25]
checkpoints: $22.365->bin1, $22.385->bin2, $100.000->bin3,
             $300.000->bin3, $700.000->bin3

KMeans
edges:  [10.0000, 24.7059, 86.9142, 370.8333, 600.0000]
counts: [59, 34, 6, 1]
checkpoints: $22.365->bin0, $22.385->bin0, $100.000->bin2,
             $300.000->bin2, $700.000->bin3

Quantile one-hot features:
["order_value_bin0", "order_value_bin1",
 "order_value_bin2", "order_value_bin3"]
$22.365 -> [0.0, 1.0, 0.0, 0.0]
$22.385 -> [0.0, 0.0, 1.0, 0.0]
$100.000 -> [0.0, 0.0, 0.0, 1.0]
$300.000 -> [0.0, 0.0, 0.0, 1.0]
$700.000 -> [0.0, 0.0, 0.0, 1.0]
```

The strategies all returned four bins. That is almost the only thing they have in common.

## Uniform bins gave the outlier the floor plan

Uniform binning takes the fitted minimum and maximum, divides the range into equal-width intervals, and ignores how observations are distributed inside them.

Our range is:

```text
$600 - $10 = $590
```

With four bins, each interval is $147.50 wide:

```text
[10.00, 157.50)
[157.50, 305.00)
[305.00, 452.50)
[452.50, 600.00]
```

Ninety-eight of the one hundred training orders land in the first bin. The $300 order lands in the second, the third bin is empty, and only the $600 order occupies the fourth.

The implementation is correct. The question was wrong for this distribution.

Uniform binning gives equal space to equal dollar ranges. That can be meaningful when the scale itself defines the product — temperature bands, regulated dosage intervals, or price tiers with fixed widths. In this long-tailed dataset, one extreme order decides the width for everybody else.

The result looks like a four-level feature but behaves almost like “ordinary versus a few unusual values.” An empty training bin contributes a column with no learned evidence if one-hot encoded, and 98% of the data loses all within-bin variation.

## Quantile bins balanced occupancy by stretching distance

Quantile binning uses the observed distribution. Its edges are the minimum, 25th percentile, median, 75th percentile, and maximum:

```text
$10.0000
$16.1875
$22.3750
$28.5625
$600.0000
```

Every training bin receives exactly 25 orders. That balance is attractive. A model gets evidence for every level, and one $600 observation cannot force 98 ordinary orders into the same bucket.

But equal frequency requires unequal widths.

The first three bins each cover about $6.19. The final bin stretches from $28.56 to $600. Under the ordinal output, a $30 order and a $600 order become the same value: `3`.

This strategy preserves rank groups, not monetary distance.

It also creates the two-cent cliff that opened this article. The learned median edge is $22.375:

```text
$22.365 -> bin 1
$22.385 -> bin 2
```

Each value is one cent from the boundary, yet their representations share nothing under one-hot encoding. Meanwhile, $30 and $600 can share the same representation despite being $570 apart.

Quantile bins are useful when relative standing matters — bottom quartile, middle deciles, top percentile — and approximately balanced groups are valuable. They are not a claim that observations within a group are genuinely interchangeable.

## KMeans bins followed clusters rather than width or headcount

The KMeans strategy fits one-dimensional cluster centers and places interior edges halfway between neighboring centers.

For this data, it learns:

```text
[10.0000, 24.7059, 86.9142, 370.8333, 600.0000]
```

The resulting counts are deliberately unequal:

```text
[59, 34, 6, 1]
```

KMeans is trying to keep values close to their assigned cluster center. It is not trying to give each bin equal width or equal membership.

That lets it reserve structure for the dense ordinary region while separating the delayed and extreme values more naturally than Uniform did. But squared-distance optimization remains sensitive to extremes. The isolated $600 order earns an entire bin, and the learned borders may move substantially when the tail changes.

KMeans bins are descriptive clusters of the fitted numeric distribution. Calling them “small,” “medium,” and “large” afterward is an interpretation we add, not something the algorithm understood.

## A boundary belongs to the upper bin

datarust uses left-inclusive interior bins. If a value equals an interior edge, it enters the bin above that edge.

For the quantile boundary:

```text
value < 22.375  -> bin 1
value = 22.375  -> bin 2
value > 22.375  -> bin 2
```

This deterministic rule matters when business logic and model preprocessing meet. A dashboard might round both $22.365 and $22.385 to $22.37 while the model places them in different bins. Currency should generally be represented at an appropriate fixed precision upstream, and displayed rounding should not be confused with stored numeric values.

More broadly, any hard boundary creates local instability. If tiny measurement noise can move an observation across a high-impact edge, the feature may produce brittle decisions.

## Ordinal output adds another assumption

With `KBinsEncode::Ordinal`, four bins become the numbers `0`, `1`, `2`, and `3`.

They are genuinely ordered, unlike arbitrary category labels. But a linear model sees more than order. It sees equal numeric steps:

```text
effect(bin 1) - effect(bin 0)
    == effect(bin 3) - effect(bin 2)
```

That may be false, especially with quantile bins whose dollar widths differ dramatically.

One-hot encoding gives each bin an independent indicator:

```rust
let mut kb = KBinsDiscretizer::new(4)?
    .strategy(BinStrategy::Quantile)
    .encode(KBinsEncode::OneHotDense);
```

The model can then learn a separate effect for every interval without assuming equal jumps. It still inherits the hard boundaries and within-bin information loss, but it does not treat bin indices as a ruler.

Trees can often learn thresholds directly from the continuous feature, making manual binning unnecessary. For a linear model, one-hot bins can provide a simple piecewise-constant nonlinear effect at the cost of discontinuities.

## Values outside training are clamped, not given new bins

The discretizer learns a fixed schema. It does not create a fifth bin when production sees something larger than the training maximum.

Our fitted maximum is $600, yet:

```text
$700 -> bin 3
```

Values below the fitted minimum similarly enter the first bin. This is operationally convenient because the output shape stays stable, but it saturates the tails. Under quantile encoding, $28.57, $300, $600, and $700 can all become the same final-bin indicator.

If the difference between “large” and “far beyond anything seen in training” matters, I preserve the raw value, add an explicit out-of-range flag, use domain-defined overflow bins, or choose a transform that extrapolates.

A valid bin code does not mean the production distribution still resembles training.

## Fit the borders on training data only

Every strategy learns from data:

- Uniform learns the minimum and maximum.
- Quantile learns empirical percentiles.
- KMeans learns cluster centers and their midpoint boundaries.

Fitting those values before the train/test split leaks information from the test distribution into preprocessing. In cross-validation, the discretizer belongs inside the fold's pipeline so every training fold learns its own edges.

That also means edge movement is expected across retraining runs. A customer at $22.38 may move from bin 2 to bin 1 even when their order value does not change, simply because the population quantiles shifted.

For user-facing tiers, eligibility rules, or audited decisions, learned boundaries may be too unstable. Fixed domain thresholds can be more appropriate even if their training counts are uneven.

## When I bin a continuous feature

I consider discretization when:

- Domain thresholds already exist and are meaningful.
- Measurement precision is noisier than the distinctions in the raw value.
- A simple model needs a controlled nonlinear effect.
- Interpretability of broad ranges matters more than exact distance.
- Tail values should share a category rather than extrapolate indefinitely.

I avoid it by default when small changes should produce small prediction changes, exact magnitude carries useful signal, or the model can learn smooth nonlinear structure directly.

And I rarely evaluate a binning strategy from its edges alone. I inspect at least four things:

```text
bin edges
training counts per bin
target behavior per bin
production out-of-range and boundary rates
```

An interval with no training rows is not merely untidy. A boundary with a sharp target jump may be genuine signal or leakage. A rapidly growing last-bin population may indicate distribution drift hidden by a stable output schema.

## The simplification has to earn its loss

Our three four-bin features summarized the same 100 values in radically different ways:

- Uniform let the $600 maximum define equal widths and placed 98 orders together.
- Quantile produced perfectly balanced counts by making the final interval enormous.
- KMeans adapted to numeric clusters but devoted one entire bin to one extreme order.
- Every strategy erased within-bin distance and introduced hard boundaries.

Binning is often described as reducing precision. I think that phrasing is too gentle.

It replaces one geometry with another.

The useful question is not “How many bins should I use?” It is:

> Which differences should the model forget, and where am I willing to create a cliff?

If I cannot answer that, the raw continuous feature is usually the more honest place to start.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
