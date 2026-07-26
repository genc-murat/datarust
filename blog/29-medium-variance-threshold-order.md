# I Put StandardScaler First. My Low-Variance Filter Stopped Filtering.

*A practical datarust guide to pipeline order, VarianceThreshold, unit variance, and the preprocessing steps that looked interchangeable until they selected different columns.*

---

I built two pipelines from the same components.

```text
Pipeline A: VarianceThreshold -> StandardScaler
Pipeline B: StandardScaler -> VarianceThreshold
```

Both used a variance threshold of `0.05`. Both saw the same 100 rows.

Pipeline A removed the almost-constant feature.

Pipeline B kept it.

The reason was hiding in the name of the first step. `StandardScaler` gave every non-constant feature variance one. By the time `VarianceThreshold` looked at the matrix, the original variance differences no longer existed.

```text
raw selector input:    [0.0, 0.000025, 0.09, 833.25]
scaled selector input: [0.0, 1.0,      1.0,  1.0]
```

With threshold `0.05`, the second selector could still remove the exactly constant column. Every other feature passed comfortably.

Nothing was wrong with either transformer. The pipeline order changed the question being asked.

Let's reproduce the selection masks and output feature names with [datarust](https://crates.io/crates/datarust).

## Four features with four kinds of variation

The controlled matrix has 100 rows and four columns:

```text
constant:
    always 7

almost_constant:
    alternates between 100.00 and 100.01

rare_flag:
    1 for 10% of rows, otherwise 0

signal:
    counts from 0 to 99
```

Their population variances are approximately:

```text
constant:          0.000000
almost_constant:   0.000025
rare_flag:         0.090000
signal:           833.250000
```

At threshold `0.05`, the intended raw-space decision is easy to inspect:

- Drop `constant`.
- Drop `almost_constant`.
- Keep `rare_flag`.
- Keep `signal`.

The rare binary flag is useful for another reason: it reminds us that low numeric variance is not automatically low semantic value. We keep it here because `0.09 > 0.05`; a slightly higher threshold could remove it without ever consulting the prediction target.

I fit both pipeline orders, inspect the variance stored inside each fitted selector, and ask the pipeline to propagate the original feature names.

Here is the complete Rust program:

```rust
use datarust::pipeline::Pipeline;
use datarust::scaler::StandardScaler;
use datarust::selection::VarianceThreshold;
use datarust::traits::{FeatureNames, Transformer};
use datarust::transformer_kind::TransformerKind;
use datarust::Matrix;

fn population_variances(x: &Matrix) -> Vec<f64> {
    let means = x.column_mean();
    (0..x.ncols())
        .map(|j| {
            (0..x.nrows())
                .map(|i| {
                    (x.get(i, j) - means[j]).powi(2)
                })
                .sum::<f64>()
                / x.nrows() as f64
        })
        .collect()
}

fn selector_state(
    pipeline: &Pipeline,
) -> (&[f64], &[bool]) {
    match pipeline.get_step("select").unwrap() {
        TransformerKind::VarianceThreshold(selector) => {
            (selector.variances(), selector.get_support())
        }
        _ => unreachable!(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rows: Vec<Vec<f64>> = (0..100)
        .map(|i| {
            vec![
                7.0,
                100.0
                    + if i % 2 == 0 { 0.0 } else { 0.01 },
                if i % 10 == 0 { 1.0 } else { 0.0 },
                i as f64,
            ]
        })
        .collect();
    let x = Matrix::new(rows)?;

    let names: Vec<String> = [
        "constant",
        "almost_constant",
        "rare_flag",
        "signal",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let mut select_then_scale = Pipeline::new()
        .push(
            "select",
            TransformerKind::VarianceThreshold(
                VarianceThreshold::new(0.05)?,
            ),
        )
        .push(
            "scale",
            TransformerKind::StandardScaler(
                StandardScaler::new(),
            ),
        );
    let selected_first =
        select_then_scale.fit_transform(&x)?;

    let mut scale_then_select = Pipeline::new()
        .push(
            "scale",
            TransformerKind::StandardScaler(
                StandardScaler::new(),
            ),
        )
        .push(
            "select",
            TransformerKind::VarianceThreshold(
                VarianceThreshold::new(0.05)?,
            ),
        );
    let scaled_first =
        scale_then_select.fit_transform(&x)?;

    let (raw_seen_variances, raw_support) =
        selector_state(&select_then_scale);
    let (scaled_seen_variances, scaled_support) =
        selector_state(&scale_then_select);

    println!(
        "raw feature variances: {:?}\n",
        population_variances(&x),
    );

    println!("select -> scale");
    println!("selector saw:     {:?}", raw_seen_variances);
    println!("support mask:     {:?}", raw_support);
    println!(
        "output names:     {:?}",
        select_then_scale
            .feature_names_out(Some(&names)),
    );
    println!(
        "output variances: {:?}\n",
        population_variances(&selected_first),
    );

    println!("scale -> select");
    println!(
        "selector saw:     {:?}",
        scaled_seen_variances,
    );
    println!("support mask:     {:?}", scaled_support);
    println!(
        "output names:     {:?}",
        scale_then_select
            .feature_names_out(Some(&names)),
    );
    println!(
        "output variances: {:?}",
        population_variances(&scaled_first),
    );

    Ok(())
}
```

This is the output I measured:

```text
raw feature variances: [0.0, 2.5000000000025578e-5, 0.08999999999999987, 833.25]

select -> scale
selector saw:     [0.0, 2.5000000000027672e-5, 0.09000000000000002, 833.25]
support mask:     [false, false, true, true]
output names:     ["rare_flag", "signal"]
output variances: [1.0000000000000018, 1.0000000000000002]

scale -> select
selector saw:     [0.0, 0.9999999999999162, 0.9999999999999996, 1.000000000000002]
support mask:     [false, true, true, true]
output names:     ["almost_constant", "rare_flag", "signal"]
output variances: [0.9999999999999151, 1.0000000000000018, 1.0000000000000002]
```

The tiny deviations from exact one are floating-point arithmetic. The transformed geometry is doing what `StandardScaler` promised.

The selector is also doing exactly what it promised—on the matrix it actually received.

## StandardScaler erased the original variance scale

With default settings, `StandardScaler` transforms each non-constant column independently:

```text
z = (x - training_mean) / training_standard_deviation
```

The transformed population variance is:

```text
variance(z) = variance(x) / variance(x) = 1
```

That holds whether the original variance was `0.000025`, `0.09`, or `833.25`.

An exactly constant feature has standard deviation zero. datarust maps its centered values to zero, so it retains variance zero and the later selector removes it.

After scaling, the selector sees only two categories:

```text
constant feature:     variance 0
non-constant feature: variance approximately 1
```

A threshold strictly between zero and one therefore behaves almost like a constant-column filter. It cannot distinguish “barely moved” from “varied enormously” in the original units.

## VarianceThreshold uses the matrix in front of it

`VarianceThreshold` computes population variance for every input column and keeps features whose value is strictly greater than the configured threshold:

```text
keep feature j when variance_j > threshold
```

It has no memory of upstream raw values. It does not ask whether a scaler previously divided one feature by `0.005` and another by `28.866`.

In the first pipeline, its support mask is:

```text
[false, false, true, true]
```

In the second:

```text
[false, true, true, true]
```

Both masks are correct for their own inputs.

This is why pipeline order is part of model behavior rather than code organization. Transformations generally do not commute:

```text
select(scale(x)) != scale(select(x))
```

The component list alone does not specify the feature space. The arrows do.

## Selecting first preserved the intended question

Our stated goal was to remove features with low variance in their raw training representation.

That requires the selector to see raw training columns:

```rust
Pipeline::new()
    .push(
        "select",
        TransformerKind::VarianceThreshold(
            VarianceThreshold::new(0.05)?,
        ),
    )
    .push(
        "scale",
        TransformerKind::StandardScaler(
            StandardScaler::new(),
        ),
    )
```

The selector drops `constant` and `almost_constant`. The scaler then learns means and standard deviations only for the two surviving features.

The final model still receives standardized columns:

```text
rare_flag variance ≈ 1
signal variance    ≈ 1
```

Selection and scaling each answer their intended question in the coordinate system where that question makes sense.

## A threshold of zero is a special case

The default `VarianceThreshold` uses threshold zero. It removes exactly constant columns and keeps features with any positive measured variance.

For ordinary finite data, placing that selector before or after default standardization will often produce the same support mask:

- Raw constant features have variance zero.
- Scaled constant features remain zero.
- Raw non-constant features have positive variance.
- Scaled non-constant features have variance one.

That can make the ordering mistake invisible during early testing. The difference appears when a positive threshold is intended to remove near-constant features.

It can also appear through numerical edge cases, missing-value handling, or an upstream transformation that collapses distinct values. I still place the selector according to its semantic input, even when today's threshold happens to make two orders agree.

## Raw variance depends on units

Selecting before scaling preserves raw variance, but raw variance is not unit-free.

If a measurement is multiplied by 100, its variance is multiplied by 10,000:

```text
variance(100 × x) = 100² × variance(x)
```

The same physical distance expressed in meters or centimeters can therefore land on opposite sides of a single global threshold.

This does not mean scaling first is the solution; standardization turns every non-constant variance into one and destroys the filter's intended ranking.

It means an absolute variance threshold across mixed units needs justification. Alternatives include:

- apply thresholds within groups of comparable features,
- convert columns to reviewed canonical units first,
- use domain-specific tolerances per feature,
- filter binary features by prevalence or support count,
- remove exact constants with threshold zero,
- use supervised selection when the goal is target relevance.

“Low variance” only has meaning relative to a representation.

## The rare flag could still be the best feature

`VarianceThreshold` is unsupervised. It reads `X`, not `y`.

Our rare flag has Bernoulli variance:

```text
p × (1 - p) = 0.1 × 0.9 = 0.09
```

It survives threshold `0.05`. If only 1% of rows contained the flag, its variance would be `0.0099` and the same filter would remove it.

That may be desirable for a noisy indicator with too little support. It may be disastrous if the flag identifies the only true failure cases.

Variance measures how often or how far a feature changes. It does not measure predictive value, causal importance, business cost, or data quality.

Before removing a rare binary feature, I inspect its meaning, support in every split, relationship with the target, and whether its rarity is exactly why it matters.

Unsupervised filtering is a computational and representation choice, not a relevance oracle.

## Centering alone would not create this result

The operation that equalizes variance is division by standard deviation, not subtraction of the mean.

If `StandardScaler` is configured with standard-deviation scaling disabled:

```rust
StandardScaler::new().with_std(false)
```

then it only centers columns. Centering changes the mean but preserves variance:

```text
variance(x - mean(x)) = variance(x)
```

A later variance selector would see the same variances as before centering, apart from floating-point detail.

Default `StandardScaler::new()` enables both mean removal and standard-deviation scaling, which is why the experiment produces unit variances.

Being precise about which part of a transformer changes the diagnostic prevents vague rules such as “feature selection must always happen before preprocessing.” Some selectors require non-negative input; some transformations create the features that should be selected; supervised selection belongs inside cross-validation. Order follows semantics and API requirements.

## Feature names exposed the difference

Both pipelines produce valid numeric matrices. Their widths reveal that something changed—two columns versus three—but names show exactly what changed:

```text
select -> scale:
["rare_flag", "signal"]

scale -> select:
["almost_constant", "rare_flag", "signal"]
```

datarust's `Pipeline` chains the `FeatureNames` implementation of each step. `VarianceThreshold` filters names with its fitted support mask; `StandardScaler` preserves the names it receives.

I record transformed feature names with the model artifact and assert them in tests. A downstream estimator only sees positions. It cannot tell whether column zero used to be `rare_flag` or `almost_constant` after a pipeline refactor.

A support-mask change is a schema change, even if prediction code still compiles.

## The selector must fit inside validation folds

Variance selection does not read labels, but it still learns from data.

If I fit it once on the full dataset before cross-validation, validation rows influence which columns are kept. A feature that is constant in the training fold but varies in the held-out fold can cross the threshold because the selector already saw both.

The honest pattern puts both selection and scaling inside the fitted pipeline passed to each fold:

```text
fold training rows
    -> fit VarianceThreshold
    -> fit StandardScaler on survivors
    -> fit estimator

fold validation rows
    -> transform with that fitted selector
    -> transform with that fitted scaler
    -> predict
```

After choosing the threshold and model configuration, I fit the complete pipeline on all permitted training data and preserve it for test and production use.

The order must be correct inside every fold, not repaired after evaluation.

## What I test before shipping the pipeline

For a variance filter, I log and test:

- raw input feature names,
- variances seen by the fitted selector,
- configured threshold,
- support mask,
- output feature names and width,
- behavior at the strict threshold boundary,
- support stability across folds and time windows,
- downstream metric changes with and without the filter.

I also keep a deliberately constant column and a deliberately near-constant column in a small pipeline unit test. They make step-order regressions obvious.

If a selector after standardization reports every surviving variance near one, that is not proof the dataset naturally contains equally variable features. It is evidence that the scaler completed its job before the selector inspected them.

## The filter answered the question created upstream

The two pipelines contained the same classes and the same threshold:

```text
VarianceThreshold(0.05)
StandardScaler()
```

Only the order changed.

In raw space, the selector could distinguish `0.000025` from `0.09` and `833.25`. It removed the almost-constant feature.

In standardized space, those three values had all become approximately one. It kept every non-constant feature.

Neither result was mysterious after inspecting the intermediate matrix. The surprising part was how easy it would have been to review the pipeline by component names and miss the changed question.

So when a preprocessing step depends on a statistic, I now ask:

> Is this statistic being measured before or after another step deliberately rewrites it?

For `VarianceThreshold` after default `StandardScaler`, the original variance had already been normalized away.

The filter did not stop working.

It filtered the representation I accidentally gave it.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
