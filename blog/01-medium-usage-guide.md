# Your First Machine-Learning App in Rust, Without Rebuilding Half of Python

*A practical datarust walkthrough: load a real dataset, build a preprocessing pipeline, evaluate it honestly, and save the fitted model as JSON.*

---

There is a particular kind of optimism that appears right after you create a new Rust project.

The directory is clean. The binary is tiny. `cargo build` is quiet. Then you decide the program should make one small machine-learning prediction, and suddenly you are comparing linear-algebra backends, converting between three matrix types, and wondering whether shipping a Python sidecar would really be so bad.

I've had that moment.

So this is the guide I wish I had: not a tour of every algorithm in the library, and not a benchmark designed to make Rust look heroic. Just one complete, copy-pasteable workflow using [**datarust**](https://crates.io/crates/datarust): data in, prediction out, fitted model saved to disk.

We are going to classify wines from their chemical measurements. The dataset is small enough to understand, but the workflow is the same one you would use for a service, a CLI tool, or a WASM application:

1. Load data
2. Hold out a test set
3. Scale the features
4. Train a multiclass logistic-regression model
5. Measure it on data it has not seen
6. Predict one new sample
7. Save and reload the whole pipeline

No Python process. No BLAS installation. No mystery model object hiding in a pickle file.

## Start with an empty Rust application

Create a project and add datarust with two optional features:

```sh
cargo new wine_classifier
cd wine_classifier
cargo add datarust --features datasets,serde
```

The `datasets` feature gives us a few classic datasets compiled directly into the binary. The `serde` feature lets us write a fitted pipeline to JSON. Neither feature requires a runtime service or a data download.

If you are adding the dependency by hand, the relevant part of `Cargo.toml` is:

```toml
[dependencies]
datarust = { version = "0.6", features = ["datasets", "serde"] }
```

Now replace `src/main.rs` with this:

```rust
use datarust::datasets::wine;
use datarust::linear_model::{LogisticRegression, LogisticSolver};
use datarust::metrics::classification::{
    accuracy_score, confusion_matrix, f1_score,
};
use datarust::model_selection::TrainTestSplit;
use datarust::pipeline::{Pipeline, SupervisedPipeline};
use datarust::scaler::StandardScaler;
use datarust::traits::{PredictProba, Predictor};
use datarust::transformer_kind::TransformerKind;
use datarust::Matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load 178 wine samples with 13 numeric features and 3 classes.
    let data = wine::load();
    let x = data.features();
    let y = data.targets().to_vec();

    println!(
        "Loaded {} samples with {} features",
        data.n_samples(),
        data.n_features()
    );

    // 2. Keep 20% of the rows aside for an honest final check.
    let (x_train, x_test, y_train, y_test) = TrainTestSplit::new()
        .with_test_size(0.20)
        .with_random_state(42)
        .split(&x, &y)?;

    // 3. Build one object that owns preprocessing and the final model.
    let mut model = Pipeline::new()
        .push(
            "scale",
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        .with_estimator(
            LogisticRegression::new()
                .with_solver(LogisticSolver::Svd)
                .with_max_iter(200),
        );

    // The scaler is fitted on x_train only, then its output trains the model.
    model.fit(&x_train, &y_train)?;

    // 4. Evaluate on untouched test rows.
    let predictions = model.predict(&x_test)?;
    let accuracy = accuracy_score(&y_test, &predictions)?;
    let macro_f1 = f1_score(&y_test, &predictions)?;
    let matrix = confusion_matrix(&y_test, &predictions)?;

    println!("Test accuracy: {:.1}%", accuracy * 100.0);
    println!("Macro F1:      {:.3}", macro_f1);
    println!("Confusion matrix: {matrix:?}");

    // 5. Predict one new laboratory sample.
    let new_sample = Matrix::new(vec![vec![
        13.20, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76,
        0.26, 1.28, 4.38, 1.05, 3.40, 1050.0,
    ]])?;

    let predicted_class = model.predict(&new_sample)?[0] as usize;
    let probabilities = model.predict_proba(&new_sample)?;

    println!("Predicted class: {}", data.target_names()[predicted_class]);
    println!("Probabilities:   {:?}", probabilities.row(0));

    // 6. Save every fitted parameter: scaler statistics and model weights.
    datarust::serialize::save_json(&model, "wine-model.json")?;

    // The concrete estimator type is stated when loading the JSON back.
    let restored: SupervisedPipeline<LogisticRegression> =
        datarust::serialize::load_json("wine-model.json")?;

    let restored_prediction = restored.predict(&new_sample)?;
    assert_eq!(restored_prediction[0], predicted_class as f64);

    println!("Reloaded model made the same prediction.");
    Ok(())
}
```

Run it:

```sh
cargo run --release
```

With datarust v0.6 and the fixed split seed above, the interesting part of the output looks like this:

```text
Loaded 178 samples with 13 features
Test accuracy: 91.7%
Macro F1:      0.918
Confusion matrix: [[11, 1, 0], [0, 13, 2], [0, 0, 9]]
Predicted class: class_0
Reloaded model made the same prediction.
```

You now have a trained classifier and a `wine-model.json` file. More importantly, you have a workflow that keeps the boring-but-critical parts together.

Let's unpack why each part is there.

## `Matrix` is the front door

All dense numeric input in datarust arrives as a `Matrix`. It is a deliberately small abstraction: rows, columns, and a contiguous `Vec<f64>` underneath.

In this example, `data.features()` builds the matrix for us. With your own data, the boundary looks like this:

```rust
let x = Matrix::new(vec![
    vec![1.2, 18.0, 0.7],
    vec![0.8, 25.0, 1.1],
    vec![1.6, 11.0, 0.4],
])?;
```

`Matrix::new` validates the shape immediately. If row three quietly has four values while the others have three, construction fails there instead of producing a confusing error during training.

That may sound like a small thing. It stops feeling small the first time a malformed CSV row reaches production.

datarust intentionally does not force a file format on you. Read from the `csv` crate, a database client, Arrow, an HTTP request, or your own structs; at the library boundary, turn the numeric values into rows and construct a `Matrix`.

## Split before you fit anything

The most important line in the example may be the least exciting one:

```rust
let (x_train, x_test, y_train, y_test) = TrainTestSplit::new()
    .with_test_size(0.20)
    .with_random_state(42)
    .split(&x, &y)?;
```

The test set is put aside before the scaler learns a mean or standard deviation. That matters because preprocessing learns from data too. If you scale the full dataset and split afterward, information from the test rows has already leaked into training.

It is a very polite kind of cheating — nothing crashes, the API does not complain, and your metric becomes slightly better than reality.

The fixed random state is there for reproducibility. During development, the same code should mean the same split. Once the workflow is stable, cross-validation gives you a better view than trusting a single split.

## The pipeline is more than tidy syntax

This part should look familiar if you have used scikit-learn:

```rust
let mut model = Pipeline::new()
    .push(
        "scale",
        TransformerKind::StandardScaler(StandardScaler::new()),
    )
    .with_estimator(
        LogisticRegression::new()
            .with_solver(LogisticSolver::Svd)
            .with_max_iter(200),
    );
```

Calling `fit` does two jobs in order:

- `StandardScaler` learns its statistics from the training matrix and transforms it.
- `LogisticRegression` learns from the transformed matrix and the training labels.

Calling `predict` later reuses the fitted scaler before it reaches the classifier. You cannot accidentally train on standardized features and send raw values into prediction — an easy mistake when preprocessing and the model live in unrelated variables.

I also like that the final type is still concrete: `SupervisedPipeline<LogisticRegression>`. Rust knows which estimator sits at the end, so methods such as probability prediction remain type-checked without a runtime plugin system.

Why the SVD solver? Wine features differ wildly in scale and can be correlated. Scaling handles the first problem; SVD is a robust choice for the linear solve when columns are close to dependent. For a well-conditioned dataset, the default Cholesky solver is usually the faster option.

## Accuracy is a start, not a conclusion

We print three views of performance:

```rust
let accuracy = accuracy_score(&y_test, &predictions)?;
let macro_f1 = f1_score(&y_test, &predictions)?;
let matrix = confusion_matrix(&y_test, &predictions)?;
```

Accuracy answers, “How many test rows did we classify correctly?” Macro F1 calculates F1 for each wine class and gives every class equal weight. The confusion matrix shows which classes the model is mixing up.

That last one is often where the useful conversation begins. A single score can tell you the model is wrong 8% of the time. A confusion matrix can tell you it is almost always confusing class 1 with class 2 — which may point to better features, mislabeled data, or a perfectly reasonable overlap in the real world.

The important detail is that every one of these numbers comes from `x_test`, not the rows used to fit the model. Training accuracy is useful for debugging. It is not evidence that the model generalizes.

## Prediction should use the exact same path

The new sample contains the same 13 measurements, in the same order as the training matrix:

```rust
let predicted_class = model.predict(&new_sample)?[0] as usize;
let probabilities = model.predict_proba(&new_sample)?;
```

The pipeline applies its stored training-set scaling statistics and then asks logistic regression for a class. `predict_proba` returns one probability per class, which is often more useful than the winning label alone.

A prediction of `class_0` at 0.99 confidence and one at 0.38 confidence are both `class_0`, but they are not the same product decision. Keep the probabilities if you need thresholds, review queues, or a useful “not sure” state in the interface.

One practical warning: datarust checks matrix shape, but it cannot know that you swapped `alcohol` and `malic_acid`. In a real application, define the feature order once and keep it next to the code that converts your domain object into a row.

## Saving JSON changes the deployment conversation

The final step writes both halves of the fitted pipeline:

```rust
datarust::serialize::save_json(&model, "wine-model.json")?;
```

That file contains the scaler's learned means and scales, the classifier's weights, its class labels, and the fitted state needed for prediction. Loading it restores a model that can predict immediately:

```rust
let restored: SupervisedPipeline<LogisticRegression> =
    datarust::serialize::load_json("wine-model.json")?;
```

There is no training data in the serving process and no need to call `fit` again. Train in one Rust binary, copy the JSON file, and load it in another.

JSON is not the smallest model format on earth, but it has an underrated feature: you can inspect it. You can put it in artifact storage, diff versions, attach a checksum, and see whether a retraining run changed scaler statistics or model coefficients. For classical models, that transparency is often worth more than shaving a few kilobytes.

Treat the file as a versioned artifact anyway. The safest production setup records the datarust version, feature order, training-data version, and evaluation metrics beside the model. A readable file is helpful; it is not a substitute for model lineage.

## What changes when the data is yours?

The embedded Wine dataset makes the example pleasantly short. A real application usually adds three pieces around it:

**An input adapter.** Read your database rows, CSV records, or request structs and turn them into `Matrix` rows. Keep feature ordering explicit.

**A richer preprocessing stage.** Numeric data may need imputation, robust scaling, feature selection, or PCA. Mixed data can use `Table` and `ColumnTransformer` to scale numeric columns and one-hot encode categorical columns in one operation.

**A validation boundary.** Check units, allowed ranges, missing values, and schema version before constructing the prediction row. A model will happily make a mathematically valid prediction for an age of `-400`; your application should not ask it to.

The core lifecycle stays the same:

```text
domain data → Matrix → fitted pipeline → prediction/probabilities
                              ↓
                           JSON model
```

Once that boundary is clean, the same model can sit behind an HTTP endpoint, run inside a batch job, power a command-line tool, or compile into a WASM target.

## A few mistakes worth avoiding

After building a few of these workflows, these are the ones I would watch for:

- **Fitting preprocessing before the split.** Fit every learned transformation on training data only.
- **Changing feature order at inference.** Shape checks catch missing columns, not swapped meanings.
- **Reporting only training metrics.** They answer whether the model learned the seen rows, not whether it handles new ones.
- **Saving the estimator without its preprocessing.** The scaler is part of the model. Keep it in the pipeline and serialize them together.
- **Treating probabilities as guarantees.** A softmax output is a model score, not a promise. Check calibration when the numeric confidence drives real decisions.
- **Expecting datarust to read every data source.** It is an ML and preprocessing library, not a dataframe framework. Bring the I/O crate that fits your application.

None of these are Rust-specific. Rust simply gives us a good place to make the boundaries visible.

## Where I would use this today

datarust makes the most sense when the workflow is classical ML and deployment simplicity matters: linear or logistic regression in a service, preprocessing at the edge, a small clustering job, a fitted transformation embedded in a CLI, or a model that needs to run without a Python runtime.

It is not trying to replace the entire Python data-science ecosystem. If I were exploring a new dataset interactively, drawing twenty charts, or training gradient-boosted trees, I would probably still begin in Python. If the final workflow needed to become a small, predictable, self-contained program, that is where I would reach for Rust.

That distinction is useful. The goal is not to prove that every notebook should be rewritten. The goal is to make the production path boring.

And at the end of this example, it is: one binary, one JSON model, one call to `predict`.

```sh
cargo add datarust --features datasets,serde
```

---

*datarust is MIT-licensed and available on [crates.io](https://crates.io/crates/datarust). The full guide is at [genc-murat.github.io/datarust](https://genc-murat.github.io/datarust/), and runnable examples live in the [GitHub repository](https://github.com/genc-murat/datarust/tree/main/examples).*
