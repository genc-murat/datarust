# Product C Became 70% of Traffic. Its Frequency Feature Stayed 0.05.

*A practical datarust guide to FrequencyEncoder, frozen training prevalence, categorical drift, batch-dependent refitting, and the one-column matrix whose meaning changed without changing shape.*

---

Product C had become the dominant category in production.

```text
current traffic share: 0.70
```

The model's frequency feature still said:

```text
encoded value: 0.05
```

At first, that looked stale or broken. It was neither.

The `FrequencyEncoder` had learned that C represented 5% of the training rows. Calling `transform` later performed a lookup in that fitted mapping. It did not recount the current request batch, and it was not supposed to.

I tried the obvious “fix”: fit a new encoder on production traffic so the feature would say `0.70`.

The transformed matrix still had one column. Ridge accepted it without a shape error. Production MAE jumped from `0.000` to `26.750`.

The model coefficients had been learned in a coordinate system where:

```text
A -> 0.80
B -> 0.15
C -> 0.05
```

The refitted encoder silently changed that coordinate system to:

```text
A -> 0.10
B -> 0.20
C -> 0.70
```

Same categories. Same matrix width. Completely different numeric meaning.

Let's reproduce the failure with [datarust](https://crates.io/crates/datarust), then separate three ideas that are easy to mix together: frozen serving features, live drift monitoring, and model retraining.

## A category distribution that reverses

The training sample contains 100 rows:

```text
A: 80 rows
B: 15 rows
C:  5 rows
```

With normalized frequency encoding, their fitted values are simply their training proportions:

```text
A -> 0.80
B -> 0.15
C -> 0.05
```

The controlled target is constructed as:

```text
target = 60 - 50 × fitted_frequency
```

That gives each category a stable target:

```text
A -> 20.0
B -> 52.5
C -> 57.5
```

This formula is deliberately artificial. It lets a one-feature Ridge model learn the relationship almost exactly and isolates what happens when only the encoder mapping changes.

Production also contains 100 rows, but the category mix has reversed:

```text
A: 10 rows
B: 20 rows
C: 70 rows
```

For the experiment, the expected target attached to each category remains stable. That is another deliberate control. In real life, changed prevalence may arrive with changed behavior too; we will return to that problem after separating the refitting bug.

We score production in two ways:

```text
correct serving path:
    fitted training encoder.transform(production)

incorrect serving path:
    new encoder.fit_transform(production)
```

Here is the complete Rust program:

```rust
use datarust::encoder::FrequencyEncoder;
use datarust::linear_model::Ridge;
use datarust::metrics::regression::mean_absolute_error;
use datarust::traits::Predictor;
use datarust::StrMatrix;

fn batch(counts: [usize; 3]) -> (StrMatrix, Vec<f64>) {
    let categories = ["A", "B", "C"];
    let targets = [20.0, 52.5, 57.5];
    let mut rows = Vec::new();
    let mut y = Vec::new();

    for index in 0..3 {
        for _ in 0..counts[index] {
            rows.push(categories[index]);
            y.push(targets[index]);
        }
    }

    (StrMatrix::from_column(rows).unwrap(), y)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (train_categories, y_train) =
        batch([80, 15, 5]);
    let (production_categories, y_production) =
        batch([10, 20, 70]);

    let mut fitted_encoder =
        FrequencyEncoder::new(true);
    let x_train = fitted_encoder
        .fit_transform(&train_categories)?;

    let mut model = Ridge::new().with_alpha(1e-6);
    model.fit(&x_train, &y_train)?;

    let frozen_features = fitted_encoder
        .transform(&production_categories)?;
    let frozen_predictions =
        model.predict(&frozen_features)?;

    let mut refitted_encoder =
        FrequencyEncoder::new(true);
    let refitted_features = refitted_encoder
        .fit_transform(&production_categories)?;
    let refitted_predictions =
        model.predict(&refitted_features)?;

    println!(
        "production MAE with frozen encoder:  {:.3}",
        mean_absolute_error(
            &y_production,
            &frozen_predictions,
        )?
    );
    println!(
        "production MAE with refitted encoder: {:.3}\n",
        mean_absolute_error(
            &y_production,
            &refitted_predictions,
        )?
    );

    let probes =
        StrMatrix::from_column(["A", "B", "C"])?;
    let training_values =
        fitted_encoder.transform(&probes)?;
    let current_values =
        refitted_encoder.transform(&probes)?;
    let frozen_probe_predictions =
        model.predict(&training_values)?;
    let refitted_probe_predictions =
        model.predict(&current_values)?;

    println!(
        "category  fitted value  current share  \
         frozen pred  refit pred  expected"
    );
    for row in 0..3 {
        println!(
            "{:>8}       {:>6.3}         {:>6.3}       \
             {:>7.3}    {:>7.3}    {:>7.3}",
            probes.get(row, 0),
            training_values.get(row, 0),
            current_values.get(row, 0),
            frozen_probe_predictions[row],
            refitted_probe_predictions[row],
            [20.0, 52.5, 57.5][row],
        );
    }

    Ok(())
}
```

Run it:

```sh
cargo run --release
```

This is the output I measured against the current datarust codebase:

```text
production MAE with frozen encoder:  0.000
production MAE with refitted encoder: 26.750

category  fitted value  current share  frozen pred  refit pred  expected
       A        0.800          0.100        20.000     55.000     20.000
       B        0.150          0.200        52.500     50.000     52.500
       C        0.050          0.700        57.500     25.000     57.500
```

Refitting made the values look current. It made the fitted model semantically obsolete.

## Frequency means frequency during fit

`FrequencyEncoder::new(true)` learns normalized proportions. During `fit`, datarust counts each category and divides by the number of fitted rows.

Conceptually, it stores:

```text
mapping["A"] = 0.80
mapping["B"] = 0.15
mapping["C"] = 0.05
```

`transform` does not receive permission to update that state. It replaces each incoming category with its stored value:

```text
production C -> lookup mapping["C"] -> 0.05
```

The feature is therefore more accurately named:

```text
category_frequency_in_training_window
```

It is not:

```text
category_frequency_in_whatever_batch_is_being_scored
```

That longer name clarifies why the value stays fixed and why the encoder belongs in the fitted model artifact.

The behavior matches scalers and other stateful preprocessing. A `StandardScaler` does not recalculate its mean for every request. A one-hot encoder does not reorder columns when a new category becomes popular. A frequency encoder likewise preserves the statistics learned during fitting.

## The model learned the old coordinate system

Ridge sees one floating-point column. It does not receive the original category names.

During training, it learns that lower encoded frequency corresponds to a larger target:

```text
0.80 -> 20.0
0.15 -> 52.5
0.05 -> 57.5
```

With the frozen encoder, production categories occupy those same coordinates. The stable targets in our controlled setup are predicted exactly to the displayed precision.

Refitting reverses the geometry:

```text
A moves from 0.80 to 0.10
C moves from 0.05 to 0.70
```

The coefficient does what it learned to do:

```text
low numeric value  -> high prediction
high numeric value -> low prediction
```

So A receives `55.000` instead of `20.000`, while C receives `25.000` instead of `57.500`.

Nothing is wrong with Ridge's arithmetic. It was paired with a transformer that redefined its only input feature.

## Shape validation could not help

Both transformations produce a `100 × 1` matrix:

```text
frozen encoder output:  100 rows × 1 column
refitted encoder output: 100 rows × 1 column
```

The model expects one feature and receives one feature. Every value is finite. No category is unknown. Prediction succeeds.

This is more dangerous than a width mismatch because the interface looks completely healthy.

Frequency encoding compresses a categorical field to one number. That compact shape is stable even when every category-to-number mapping changes.

I therefore version more than the output width:

- fitted encoder state,
- normalized-versus-count mode,
- training window,
- category normalization rules,
- unknown policy,
- model parameters,
- and reference transformations for known categories.

A golden check such as `C -> 0.05` would catch the incompatible encoder before the model served traffic.

## The encoder can accidentally use prevalence as identity

Frequency encoding is often described as a way to represent popularity. In this experiment, every category has a unique training share:

```text
A: 0.80
B: 0.15
C: 0.05
```

Those values also act as compact category identifiers. Ridge can associate `0.05` with C's target even if rarity itself is not the causal reason C behaves that way.

This proxy identity is fragile. If prevalence order changes, the same semantic category moves to a different number under refitting. If two categories share the same frequency, they collide and become indistinguishable.

That leads to an important validation question:

> Is the model learning a relationship with popularity, or memorizing category behavior through unique popularity values?

I test that by:

- evaluating on later time windows with shifted category shares,
- grouping or perturbing similar frequencies,
- comparing against identity-preserving one-hot encoding,
- inspecting whether categories with equal frequency have different outcomes,
- and validating genuinely unseen categories separately.

One numeric column can still hide a brittle lookup table.

## Frozen encoding does not prove the model is still good

Our frozen path achieves zero production MAE because the experiment deliberately keeps each category's target stable.

Real distribution drift can change more than counts.

Product C may become common because:

- a promotion attracted a different population,
- the product itself changed,
- another product was retired,
- traffic routing moved between regions,
- logging rules changed,
- or seasonality altered who buys it.

The old `0.05` is the correct input for the old fitted pipeline. The relationship learned from that pipeline may still be obsolete.

Freezing prevents an accidental train–serve coordinate mismatch. It does not immunize the model against concept drift.

That is why I separate two questions:

```text
Can I reproduce the trained computation correctly?
Is the trained computation still useful on the current population?
```

The frozen encoder addresses the first. Monitoring and labeled evaluation address the second.

## Live prevalence belongs in monitoring

The refitted encoder in the example is useful as a diagnostic calculator. It tells us the current batch proportions:

```text
A: 0.10
B: 0.20
C: 0.70
```

I do not feed those diagnostic values into the old model.

Instead, I compare them with the fitted reference distribution:

| Category | Training share | Current share | Change |
|---|---:|---:|---:|
| A | 0.80 | 0.10 | −0.70 |
| B | 0.15 | 0.20 | +0.05 |
| C | 0.05 | 0.70 | +0.65 |

This is a large, obvious shift. Real monitoring may also aggregate:

- total variation distance,
- population stability measures,
- top-category rank changes,
- new-category rate,
- disappearing-category rate,
- and segment-specific prevalence.

The important architectural separation is:

```text
serving feature:
    frozen training mapping

monitoring statistic:
    current observed distribution
```

Both numbers are valuable. They answer different questions.

## Refitting on a request batch creates batch-dependent predictions

Suppose the service fits a frequency encoder on each incoming batch.

The same category C could receive:

```text
0.70 in a morning batch
0.20 in an afternoon batch
1.00 when scored alone
```

Its prediction would depend on which unrelated rows happened to arrive beside it.

For a single-row request, the only observed category has normalized frequency `1.0`. Every category would therefore look identical when scored alone. Batch size and composition would become hidden model inputs.

This creates several operational problems:

- online and batch predictions disagree,
- replaying one row cannot reproduce its original score without the full batch,
- request grouping changes model behavior,
- traffic spikes alter feature values,
- and latency may require waiting for a batch to form.

If live prevalence is genuinely intended as a feature, it needs a defined as-of window and a reproducible feature-store contract—perhaps “category share over the previous seven complete days.” It should be constructed consistently during historical training and online serving without using future data.

That is a different feature from `FrequencyEncoder` fitted on the model-training sample.

## Count mode has an additional dependency

`FrequencyEncoder::new(false)` stores raw counts rather than normalized proportions.

Then the feature depends on both prevalence and training sample size:

```text
same 5% category
    -> count 50 in 1,000 fitted rows
    -> count 5,000 in 100,000 fitted rows
```

A model coefficient trained on counts is tied to that fitting-window scale. Comparing models trained on different sample sizes or refitting on differently sized batches becomes even harder to interpret.

Normalized frequency removes the direct sample-size factor. It does not remove distribution drift, category collisions, or the requirement to keep the fitted mapping paired with the model.

I usually prefer normalized values when proportion is the intended concept and raw support count is not separately meaningful. If reliability depends on support, I may preserve both count and proportion as explicitly named features rather than hiding the distinction in configuration.

## Validation must respect time

Random cross-validation can make frequency encoding look more stable than deployment.

If every fold samples from the same mixed time period, category proportions in each training fold resemble those in every validation fold. A future launch that reverses prevalence is absent from the experiment.

For evolving catalogs or traffic sources, I backtest chronologically:

```text
past window
    -> fit frequency encoder and model

next window
    -> transform with frozen past mapping
    -> score
    -> compare past and current category distributions
```

This tests the exact behavior we care about: whether a mapping learned in the past remains useful later.

Fitting the encoder on the combined past and future data would leak future prevalence into the historical model. Fitting separately on the future validation window would test a pipeline that cannot exist at the original prediction time.

The split boundary applies to prevalence statistics just as it applies to scaler means, imputation values, and category vocabularies.

## Retrain the pair, not the encoder

When drift is large and labeled performance degrades, retraining may be appropriate.

The new workflow is:

```text
new approved training window
    -> fit new FrequencyEncoder
    -> transform with new mapping
    -> fit new model on those coordinates
    -> validate complete new pipeline
    -> promote encoder and model together
```

Updating only the encoder gives old coefficients new meanings. Updating only the model while keeping an unintended mapping can preserve stale assumptions.

The artifact boundary includes both.

Before promotion, I compare old and new systems on the same evaluation windows and inspect category-level behavior. A model can improve aggregate error by following the new majority while getting smaller but important categories worse.

Retraining is not “make the frequency number current.” It is “learn a new computation from a declared data window and validate it as one unit.”

## What I log beside every frequency feature

For a production frequency-encoded field, I record:

- feature name that states the reference window,
- normalized or raw-count mode,
- fitted row count,
- fitted category count,
- top and tail fitted frequencies,
- unknown-category policy and observed unknown rate,
- current category distribution,
- categories with the largest prevalence changes,
- prediction and error slices by fitted-frequency band,
- encoder and model artifact versions.

I also keep fixed probes:

```text
A -> 0.80
B -> 0.15
C -> 0.05
```

Those values are not expected to follow production traffic. They are expected to remain stable for that model version.

The monitoring dashboard holds the changing numbers.

## The stale-looking value was the correct frozen value

Product C moved from 5% of training data to 70% of production traffic.

The fitted encoder continued to emit `0.05`. That was faithful execution of the trained pipeline.

Refitting made the feature look current, but the old Ridge model interpreted `0.70` using a coefficient learned from the old mapping. C's prediction fell from the expected `57.500` to `25.000`, and production MAE rose to `26.750`.

The one-column interface never warned us.

The lesson is not to ignore prevalence drift. It is to observe drift without mutating a fitted feature underneath a fitted model.

So when a frequency feature looks stale, I ask two separate questions:

> Is this value supposed to describe the training reference population or the live population?

> If the live population changed, does the complete model need monitoring, fallback, or retraining?

In this experiment, `0.05` was not a failed update.

It was the history the coefficient had learned to understand.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
