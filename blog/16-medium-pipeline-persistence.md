# My Model Survived the Restart. Its Preprocessing Didn't.

*A practical datarust guide to saving fitted pipelines, preserving training coordinates, and avoiding predictions that are structurally valid but semantically wrong.*

---

The service restarted cleanly.

The model file loaded. The input had the expected three columns. Prediction returned an `Ok` value.

It was wrong by more than 300,000.

There was no corrupted weight file and no numerical overflow. I had saved the estimator but not the scaler that defined what its inputs meant.

During training, the model saw standardized coordinates such as `-0.329` and `1.155`. After restart, the service handed it raw values such as `1500` and `2500`. The matrix shape was still correct, so the type system had no reason to complain. Three floating-point columns went in; a floating-point prediction came out.

That is a particularly dangerous production bug: the interface is valid while the semantics have changed.

In this article, I want to show why fitted preprocessing is model state, how a datarust `SupervisedPipeline` preserves it, and why re-fitting a scaler inside a prediction service is not a repair.

## A small property-price pipeline

The synthetic training data has three features on very different scales:

```text
area in square feet
property age in years
number of rooms
```

The pipeline applies `StandardScaler` and then fits a `Ridge` regression model. After training, I save two artifacts:

1. The complete fitted pipeline
2. The fitted Ridge estimator alone

I simulate a restart by loading both files and predicting the same three raw requests in four ways:

- The original in-memory pipeline
- The restored full pipeline
- The estimator alone receiving raw features
- The estimator alone receiving features from a newly fitted request-batch scaler

JSON persistence is behind datarust's optional `serde` feature, so the dependency needs to enable it:

```toml
[dependencies]
datarust = { version = "0.6", features = ["serde"] }
```

Here is the complete Rust program:

```rust
use datarust::linear_model::Ridge;
use datarust::pipeline::{Pipeline, SupervisedPipeline};
use datarust::scaler::StandardScaler;
use datarust::traits::{Predictor, Transformer};
use datarust::transformer_kind::TransformerKind;
use datarust::Matrix;

fn max_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    let mut targets = Vec::new();

    for i in 0..240 {
        let area = 600.0 + ((i * 37) % 2400) as f64;
        let age = ((i * 17) % 60) as f64;
        let rooms = 1.0 + ((i * 11) % 5) as f64;
        let noise = (((i * 13) % 11) as f64 - 5.0) * 2.0;
        let price =
            25.0 + 0.18 * area - 2.5 * age + 18.0 * rooms + noise;

        rows.push(vec![area, age, rooms]);
        targets.push(price);
    }

    let x_train = Matrix::new(rows)?;
    let requests = Matrix::new(vec![
        vec![1_500.0, 10.0, 3.0],
        vec![2_500.0, 30.0, 4.0],
        vec![900.0, 45.0, 2.0],
    ])?;

    let mut pipeline = Pipeline::new()
        .push(
            "scale",
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        .with_estimator(Ridge::new().with_alpha(1.0));
    pipeline.fit(&x_train, &targets)?;

    let original = pipeline.predict(&requests)?;
    let trained_coordinates = pipeline.transform(&requests)?;

    let suffix = std::process::id();
    let full_path = std::env::temp_dir().join(
        format!("datarust_full_pipeline_{suffix}.json")
    );
    let model_path = std::env::temp_dir().join(
        format!("datarust_estimator_only_{suffix}.json")
    );

    datarust::serialize::save_json(&pipeline, &full_path)?;
    datarust::serialize::save_json(
        pipeline.estimator(),
        &model_path,
    )?;

    let restored: SupervisedPipeline<Ridge> =
        datarust::serialize::load_json(&full_path)?;
    let restored_predictions = restored.predict(&requests)?;

    let estimator_only: Ridge =
        datarust::serialize::load_json(&model_path)?;
    let raw_into_estimator = estimator_only.predict(&requests)?;

    let mut request_scaler = StandardScaler::new();
    let request_coordinates =
        request_scaler.fit_transform(&requests)?;
    let refitted_scaler_predictions =
        estimator_only.predict(&request_coordinates)?;

    println!(
        "case   original pipeline   restored pipeline   \
         raw -> estimator   refit-on-request"
    );
    for i in 0..requests.nrows() {
        println!(
            "{:>4}   {:17.3}   {:17.3}   {:16.3}   {:16.3}",
            i + 1,
            original[i],
            restored_predictions[i],
            raw_into_estimator[i],
            refitted_scaler_predictions[i],
        );
    }

    println!(
        "\nFull-pipeline restart max difference: {:.2e}",
        max_difference(&original, &restored_predictions)
    );
    println!(
        "Estimator-only, raw input max difference: {:.3}",
        max_difference(&original, &raw_into_estimator)
    );
    println!(
        "Estimator-only, refitted scaler max difference: {:.3}",
        max_difference(&original, &refitted_scaler_predictions)
    );

    println!(
        "\nCoordinates expected by the estimator \
         vs refit on request batch"
    );
    for i in 0..requests.nrows() {
        println!(
            "case {} expected [{:7.3}, {:7.3}, {:7.3}]   \
             refit [{:7.3}, {:7.3}, {:7.3}]",
            i + 1,
            trained_coordinates.get(i, 0),
            trained_coordinates.get(i, 1),
            trained_coordinates.get(i, 2),
            request_coordinates.get(i, 0),
            request_coordinates.get(i, 1),
            request_coordinates.get(i, 2),
        );
    }

    println!(
        "\nJSON sizes: full pipeline = {} bytes, \
         estimator only = {} bytes",
        std::fs::metadata(&full_path)?.len(),
        std::fs::metadata(&model_path)?.len(),
    );

    std::fs::remove_file(full_path)?;
    std::fs::remove_file(model_path)?;

    Ok(())
}
```

Run it with:

```sh
cargo run --release --features serde
```

This is the output I measured:

```text
case   original pipeline   restored pipeline   raw -> estimator   refit-on-request
   1             323.932             323.932         181317.708            345.816
   2             471.386             471.386         301385.449            499.932
   3             111.180             111.180         107239.016             99.462

Full-pipeline restart max difference: 5.68e-14
Estimator-only, raw input max difference: 300914.063
Estimator-only, refitted scaler max difference: 28.546

Coordinates expected by the estimator vs refit on request batch
case 1 expected [ -0.329,  -1.126,  -0.000]   refit [ -0.202,  -1.279,   0.000]
case 2 expected [  1.155,   0.029,   0.707]   refit [  1.313,   0.116,   1.225]
case 3 expected [ -1.218,   0.895,  -0.707]   refit [ -1.111,   1.162,  -1.225]

JSON sizes: full pipeline = 772 bytes, estimator only = 221 bytes
```

One path preserved a model. The others preserved only pieces.

## The estimator learned in a coordinate system

The Ridge model did not train on raw area, age, and room count. It trained on values transformed by the fitted `StandardScaler`:

```text
z_area  = (area  - training_area_mean)  / training_area_std
z_age   = (age   - training_age_mean)   / training_age_std
z_rooms = (rooms - training_rooms_mean) / training_rooms_std
```

Its coefficients therefore describe changes in standardized feature units. A coefficient multiplies `1.155`, not `2500`.

When I load the estimator alone and pass raw request 2 directly into it, the input still has three columns. `Ridge` correctly checks the shape and sees the expected feature count. It cannot know that column zero is in square feet rather than training standard deviations.

The result is `301385.449` instead of `471.386`.

This is why shape validation is necessary and insufficient. A matrix can have the correct dimensions, finite values, and completely wrong semantics.

## Fitted preprocessing is learned state

It is tempting to think of a scaler as a formula and the estimator as the actual model.

But the formula is only half the scaler. The fitted mean and standard deviation are learned from training data. An imputer stores fill values. A one-hot encoder stores category vocabulary and column order. PCA stores projection directions. A quantile transformer stores an empirical distribution.

Those learned parameters define the input space where the estimator's weights make sense.

The full JSON artifact contains both sides of that contract:

```text
StandardScaler
  fitted feature count
  training means
  training standard deviations

Ridge
  fitted feature count
  coefficients
  intercept
  alpha
```

Loading a `SupervisedPipeline<Ridge>` restores that fitted state. It does not need the training rows and does not call `fit` again.

## Re-fitting on the request batch is not recovery

The second workaround looks more reasonable:

```rust
let mut scaler = StandardScaler::new();
let scaled_request_batch = scaler.fit_transform(&requests)?;
let predictions = estimator.predict(&scaled_request_batch)?;
```

At least the estimator now receives small standardized values. The predictions even look plausible: `345.816`, `499.932`, and `99.462`.

They are still wrong by as much as `28.546` because the batch scaler learned different coordinates.

The training scaler asks, “How far is this property from the training population?” The request scaler asks, “How far is this property from the two or three requests that happened to arrive beside it?” Those are different features.

Worse, a prediction becomes batch-dependent. Send property A alone and its standardized values may collapse toward zero. Send it beside two luxury properties and it moves to the low end of the batch. The raw property did not change, but its prediction can.

An online service should transform each request with frozen training parameters. `transform` belongs in serving. `fit` does not.

## The restored pipeline was effectively identical

The original and restored pipeline predictions differ by at most:

```text
5.68e-14
```

That is floating-point rounding noise, not behavioral drift. The same raw rows followed the same fitted transformation and estimator after the simulated restart.

I use this as a deployment invariant:

> A model artifact loaded in a fresh process must reproduce reference predictions for fixed raw inputs within an explicit tolerance.

The test should begin with raw application inputs, not already transformed matrices. Testing only the estimator would miss the exact failure this article demonstrates.

A small golden set can catch missing preprocessing, changed feature order, serialization mistakes, and numerical regressions before the artifact receives production traffic.

## One file is not the only valid design

It is possible to save the scaler and estimator as separate artifacts and restore them correctly. Sometimes separate components are useful for inspection or reuse.

The operational burden is higher:

- Both files need the same model version.
- Deployment must update them atomically.
- Rollback must restore the matching pair.
- Serving code must apply them in the right order.
- Monitoring must identify which combination produced a prediction.

Saving the supervised pipeline makes the common path smaller. One artifact represents one fitted computation graph, and `predict(&raw_matrix)` is the public interface.

In this experiment, the difference between the complete artifact and estimator-only JSON is only 551 bytes. Real pipelines can be much larger, especially with category vocabularies or decomposition matrices, but removing necessary state is not a useful optimization.

## JSON helps inspection, not schema magic

datarust writes pretty-printed JSON rather than an opaque runtime-specific blob. I can inspect it, diff two fitted artifacts, store a checksum, and review which parameters changed.

The concrete Rust type also has to match when loading:

```rust
let model: SupervisedPipeline<Ridge> =
    datarust::serialize::load_json("property_model.json")?;
```

That prevents loading the artifact as an unrelated estimator type. It does not solve every versioning problem automatically. If application releases change the artifact structure, feature definitions, or library compatibility, the deployment still needs an explicit migration or rollback policy.

JSON also does not know that column zero means `area`, column one means `age`, and column two means `rooms`. A `Matrix` with `[age, rooms, area]` has the same shape and the wrong meaning.

I keep an input schema beside the artifact contract:

```text
model version
library/application version
feature names and exact order
units and categorical policies
training-data timestamp or identifier
artifact checksum
reference inputs and expected predictions
```

Serialization preserves fitted numeric state. Deployment metadata preserves the larger semantic contract.

## Write and promote artifacts deliberately

The example calls `save_json` directly because it is a small local program. In production, I prefer a safer promotion flow:

1. Write the new artifact to a versioned temporary path.
2. Load it back into a fresh model instance.
3. Run schema and golden-prediction checks.
4. Compute and record a checksum.
5. Atomically point serving traffic to the verified version.
6. Keep the prior artifact available for rollback.

I also load only trusted artifacts. Human-readable does not mean automatically trustworthy, and successful deserialization does not prove that the model was approved for deployment.

The file write is the easy part. Artifact lifecycle is part of the model system.

## What should happen in training and serving

The boundary I use is simple.

During training:

```text
fit preprocessing on training features
transform training features
fit estimator
evaluate complete pipeline
save complete fitted pipeline
```

During serving:

```text
load fitted pipeline once
validate raw input schema
call predict on raw input
never refit
record model version with the prediction
```

Any code path that constructs a new scaler inside a request handler deserves immediate suspicion.

## The bug was not that the numbers were huge

The absurd `301385.449` prediction made the raw-input mistake easy to notice. The request-batch scaler produced a subtler failure: plausible results that drifted by up to `28.546`.

That second case is closer to the bugs I fear in production. A plausible number passes dashboards, range checks, and casual review. It can stay wrong for a long time.

The complete pipeline avoided both failures because it carried the meaning of its inputs across the restart:

- Same preprocessing steps
- Same learned means and standard deviations
- Same step order
- Same estimator parameters
- Same raw-to-prediction interface

The model was never just the Ridge coefficients.

It was the entire path that made those coefficients meaningful.

---

*The complete example and its reported output were run against the current datarust codebase with the `serde` feature enabled. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
