# Every Training Value Was Missing. Zero Made the Pipeline Run.

*A practical datarust guide to all-missing columns, SimpleImputer failure, constant fallbacks, unlearnable coefficients, and the sensor that came online after the model had already learned to ignore it.*

---

The imputer refused to fit.

```text
Mean: all values missing: column 1
Median: all values missing: column 1
MostFrequent: all values missing: column 1
```

Every training value in the second feature was `NaN`. There was no mean, median, or most frequent observed value to learn.

I replaced the missing column with zero. The pipeline ran. Ridge fitted successfully. Production predictions came back without an error.

Then I inspected the coefficients:

```text
[2.0, 0.0]
```

The model had learned a slope for the first feature and exactly nothing for the sensor column.

That sensor started reporting real values in production. One request carried `-2`; another carried `+2`. Both had the same first feature. Both received the same prediction:

```text
1.000 and 1.000
```

Constant imputation had repaired the matrix shape. It had not created training information.

This is the uncomfortable boundary of missing-value handling: an imputer can replace absent cells, but no statistic can recover a feature that was absent from every training row. Making the code run is not the same as making the feature learnable.

Let's reproduce the behavior with [datarust](https://crates.io/crates/datarust), compare the constant-fill model with a controlled observed-data model, and decide what an all-missing column should mean operationally.

## One visible feature and one invisible sensor

The true relationship in our synthetic data is:

```text
target = 2 × x + 5 × sensor
```

Both features matter. During training, however, the sensor column is missing in every row:

```text
[x, NaN]
```

We retain a private observed copy of the synthetic training data only to fit an oracle comparison later. A real project would not have that secret copy; if it did, the sensor would not be missing.

Production rows contain both values:

```text
[x, sensor]
```

This simulates a sensor integration that was broken throughout model training and repaired before deployment.

We try four `SimpleImputer` strategies:

- Mean
- Median
- Most frequent
- Constant zero

The first three need at least one observed training value. Constant filling does not.

Here is the complete Rust program:

```rust
use datarust::imputer::{
    ImputeStrategy, SimpleImputer,
};
use datarust::linear_model::Ridge;
use datarust::metrics::regression::{
    mean_absolute_error, r2_score,
};
use datarust::traits::{Predictor, Transformer};
use datarust::Matrix;

fn data(
    x_values: &[f64],
    sensor_values: &[f64],
) -> (Matrix, Matrix, Vec<f64>) {
    let mut observed = Vec::new();
    let mut missing = Vec::new();
    let mut targets = Vec::new();

    for &x in x_values {
        for &sensor in sensor_values {
            observed.push(vec![x, sensor]);
            missing.push(vec![x, f64::NAN]);
            targets.push(2.0 * x + 5.0 * sensor);
        }
    }

    (
        Matrix::new(observed).unwrap(),
        Matrix::new(missing).unwrap(),
        targets,
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (train_observed, train_missing, y_train) = data(
        &[-2.0, -1.0, 0.0, 1.0, 2.0],
        &[-3.0, -1.0, 1.0, 3.0],
    );
    let (production, _, y_production) = data(
        &[-2.5, -0.5, 0.5, 2.5],
        &[-2.0, -0.5, 0.5, 2.0],
    );

    for strategy in [
        ImputeStrategy::Mean,
        ImputeStrategy::Median,
        ImputeStrategy::MostFrequent,
    ] {
        let name = format!("{strategy:?}");
        let mut imputer =
            SimpleImputer::new(strategy);

        match imputer.fit(&train_missing) {
            Ok(()) => println!(
                "{name}: unexpectedly fitted"
            ),
            Err(error) => println!(
                "{name}: {error}"
            ),
        }
    }

    let mut constant = SimpleImputer::new(
        ImputeStrategy::Constant(0.0),
    );
    let x_train_constant =
        constant.fit_transform(&train_missing)?;
    let x_production_constant =
        constant.transform(&production)?;

    let mut fallback_model =
        Ridge::new().with_alpha(1e-9);
    fallback_model.fit(&x_train_constant, &y_train)?;
    let fallback_predictions = fallback_model
        .predict(&x_production_constant)?;

    let mut oracle_model =
        Ridge::new().with_alpha(1e-9);
    oracle_model.fit(&train_observed, &y_train)?;
    let oracle_predictions =
        oracle_model.predict(&production)?;

    println!(
        "\nconstant fill values: {:?}",
        constant.fill_values()
    );
    println!(
        "constant-fill coefficients: {:?}",
        fallback_model.coef()
    );
    println!(
        "observed-data coefficients: {:?}",
        oracle_model.coef()
    );
    println!(
        "constant-fill production: R2 {:.3}, MAE {:.3}",
        r2_score(
            &y_production,
            &fallback_predictions,
        )?,
        mean_absolute_error(
            &y_production,
            &fallback_predictions,
        )?
    );
    println!(
        "observed-data production: R2 {:.3}, MAE {:.3}",
        r2_score(
            &y_production,
            &oracle_predictions,
        )?,
        mean_absolute_error(
            &y_production,
            &oracle_predictions,
        )?
    );

    let same_x = Matrix::new(vec![
        vec![0.5, -2.0],
        vec![0.5, 2.0],
    ])?;
    let ignored = fallback_model.predict(
        &constant.transform(&same_x)?,
    )?;
    println!(
        "same x, sensor -2 vs +2 -> \
         predictions {:.3} and {:.3}",
        ignored[0], ignored[1]
    );

    Ok(())
}
```

Run it:

```sh
cargo run --release
```

This is the output I measured against the current datarust codebase:

```text
Mean: all values missing: column 1
Median: all values missing: column 1
MostFrequent: all values missing: column 1

constant fill values: [0.0, 0.0]
constant-fill coefficients: [1.9999999999500004, 0.0]
observed-data coefficients: [1.9999999999500004, 4.999999999949999]
constant-fill production: R2 0.197, MAE 6.250
observed-data production: R2 1.000, MAE 0.000
same x, sensor -2 vs +2 -> predictions 1.000 and 1.000
```

The erroring strategies were honest about missing evidence. The constant strategy was honest about its configured fill value. The mistake would be interpreting successful fitting as proof that the second feature participated in learning.

## A mean requires at least one value

Mean imputation computes a per-column statistic from observed training cells:

```text
fill_j = sum(observed values in column j)
         / count(observed values in column j)
```

For the sensor column:

```text
count = 0
```

There is no finite statistic to return. Median and most-frequent strategies have the same fundamental problem: no ordered values exist from which to choose a middle or a mode.

datarust reports `AllMissing` rather than silently inventing a default.

That error is useful information about the training dataset:

- the upstream source may be broken,
- the feature may not exist in the selected time window,
- a join may have failed,
- a column mapping may be wrong,
- or the feature may not be ready for modeling.

Changing the imputation strategy can suppress the error. It does not answer which of those causes occurred.

I investigate the column before choosing a fallback.

## Constant filling solved a representation problem

`ImputeStrategy::Constant(0.0)` does not need observed values. It stores zero as the fill value for every input column:

```text
constant fill values: [0.0, 0.0]
```

The first column contains no missing cells, so its configured zero is never used. Every training sensor `NaN` becomes zero:

```text
[x, NaN] -> [x, 0]
```

The matrix is now finite and rectangular. Ridge can fit it.

That is a genuine success at the representation layer. Many estimators reject `NaN`; constant imputation produces acceptable numeric input.

It is not a success at the information layer. All sensor values are identical after filling, so the feature has zero training variance.

No relationship between sensor and target can be estimated from a constant column.

## Ridge had no sensor direction to learn

The fallback model's coefficients are:

```text
[approximately 2.0, exactly 0.0]
```

The first coefficient recovers the visible part of the true relationship:

```text
target contribution = 2 × x
```

The second training column is always zero. Changing its coefficient would not change a single training prediction:

```text
coefficient × 0 = 0
```

Ridge's L2 penalty then favors the minimum-norm choice: a zero coefficient.

This is not specific to Ridge in spirit. A tree cannot learn a split from a column with one value. A feature selector may remove it as constant. A neural model receives no gradient evidence connecting variation in that input with variation in the target.

Different estimators may store different internal parameters, but none can learn the effect of variation that training never contained.

## Production values passed through—and still did nothing

`SimpleImputer::transform` replaces only `NaN` cells. The production sensor values are observed, so they remain `-2`, `-0.5`, `0.5`, and `2`.

This means the imputer is not erasing the new values at serving time.

The model ignores them because their learned coefficient is zero.

Our direct probe holds `x = 0.5` constant and changes the sensor from `-2` to `+2`:

```text
[0.5, -2.0] -> 1.000
[0.5,  2.0] -> 1.000
```

The raw requests differ. The transformed requests differ. The predictions do not.

This is a subtle train–serve problem because ordinary input monitoring may celebrate that sensor coverage improved. The model still behaves as if the integration were broken.

A feature becoming available after training does not retrofit knowledge into stored coefficients.

## The oracle comparison is counterfactual

The observed-data model receives the private complete synthetic training matrix:

```text
[x, sensor]
```

It recovers both true slopes:

```text
[approximately 2.0, approximately 5.0]
```

Its production result is perfect to displayed precision:

```text
R2 1.000, MAE 0.000
```

The constant-fill model reaches:

```text
R2 0.197, MAE 6.250
```

The gap shows how much signal the missing sensor could carry if it were genuinely available during training. It is not evidence that we can deploy the oracle.

In a real project, I cannot train on values that were never collected. The correct response is to obtain historical coverage, wait for new labeled data, or ship a model that explicitly excludes the unavailable feature.

An oracle is a diagnostic upper bound, not an imputation technique.

## Zero is a value, not evidence

There are domains where constant zero is semantically correct:

- an absent purchase count truly means no purchases,
- an optional component contributes zero capacity when not installed,
- a missing one-hot indicator means the category is inactive under a declared encoding.

There are many domains where zero is merely convenient:

- an unreported salary,
- a disconnected temperature sensor,
- a lab test that was never ordered,
- a latency metric missing because logging failed.

For the sensor in this experiment, zero means a real neutral reading. Replacing “not observed” with zero conflates two states.

Even if downstream performance were acceptable, I would name and document that assumption rather than calling zero a generic missing-value repair.

The fact that a constant strategy works on an all-missing column says only that the API has a value to write.

## A missingness indicator cannot invent variation either

A common pattern adds a binary flag:

```text
sensor_is_missing = 1 when sensor is NaN
```

For our training set, that flag is always one. It is another constant column.

In production, when the sensor comes online, the flag becomes zero. The model has never seen that state and cannot learn how predictions should change when it occurs.

The indicator is still valuable for:

- schema validation,
- monitoring coverage,
- routing rows to a fallback,
- and modeling partially missing features where both states occur during training.

It does not solve the all-missing case by itself.

Missingness indicators are informative only when training data contains relevant variation in missingness and labels with which to estimate its effect.

## Dropping the feature can be the honest model

If a column is entirely missing at training time, I often remove it from that model version.

That makes the contract truthful:

```text
model v1 inputs: [x]
sensor not used
```

When the sensor becomes reliable and enough labeled observations accumulate, I can train a new candidate:

```text
model v2 inputs: [x, sensor]
```

The new version needs fresh validation, artifact metadata, and a rollout comparison. Adding a column to live requests does not automatically turn v1 into v2.

Keeping an all-zero placeholder may be operationally useful when a fixed schema must remain stable, but I still mark it as inactive and assert its fitted coefficient or support state. Otherwise teams can believe a named feature is contributing simply because it appears in the matrix.

Feature presence and feature use are different facts.

## Historical backfill needs a time boundary

Sometimes the sensor values can be reconstructed from raw logs or another trusted source. That may create legitimate training coverage.

I ask two questions before backfilling:

1. Would this value have been available at the original prediction time?
2. Is its reconstruction independent of future outcomes?

A feature rebuilt from a later investigation, final status, or post-event database can leak the answer into historical training. The new column looks wonderfully predictive because it contains information the live system would not have had.

The correct historical feature is an as-of value, not the best value known today.

Backfilling can repair collection. It can also create a time-travel dataset. The pipeline cannot tell the difference for us.

## Cross-validation may expose fold-specific absence

A feature can have some observed values globally and still be entirely missing inside one training fold.

That is common when:

- a sensor was introduced late,
- one site never collected the field,
- a rare test is ordered for a small subgroup,
- or grouped validation isolates a provider or region.

If `SimpleImputer::Mean` lives correctly inside the pipeline, that fold may return `AllMissing`. Fitting the imputer globally before cross-validation can hide the error by borrowing observed values from the validation fold—but that is leakage.

The honest failure reveals that the proposed model cannot be trained using only the information available in that split.

Possible responses include:

- dropping the unstable feature,
- using a justified constant plus explicit monitoring,
- redesigning the split only if it misrepresents deployment,
- collecting more coverage,
- or using separate models for populations with different schemas.

I do not repair a fold by letting it study held-out rows.

## What I check before accepting an imputed column

For every feature, I record:

- training missing count and rate,
- observed count in each fold and important group,
- fitted fill value,
- whether the fill is a learned statistic or domain constant,
- variance after imputation,
- downstream coefficient or feature support,
- validation performance with and without the feature,
- production missingness and coverage drift,
- behavior when a previously absent value appears.

For an all-missing column, the minimum observed count is not merely low. It is zero, and that deserves a separate decision rather than another decimal in a missing-rate report.

I also probe the fitted model directly, as we did here:

```text
hold every other input fixed
change the newly observed feature
verify whether prediction changes
```

If it does not, the feature may be present only cosmetically.

## The successful fit was the beginning of the diagnosis

Mean, median, and most-frequent imputation refused to invent a statistic for an empty column. Constant zero allowed the numeric pipeline to proceed.

That produced a valid model with a clear limitation:

```text
sensor coefficient = 0
```

When the sensor came online, real values passed through the imputer. The model still returned identical predictions for `-2` and `+2` because training contained no sensor variation from which to learn.

The zero fill did exactly what it promised. It replaced missing cells.

It did not replace missing history.

So when an all-missing column blocks fitting, I no longer ask only:

> Which constant will make this matrix acceptable?

I ask:

> Should this feature exist in the current model at all, and what evidence would let a future model use it?

In this experiment, zero made the pipeline run.

Only observed training data could have made the sensor matter.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
