# My Model Predicted 1. One Service Called It Deny. Another Called It Review.

*A practical datarust guide to LabelEncoder, target-code mappings, lexicographic class order, unknown-label sentinels, and the deployment contract hiding behind a tiny integer.*

---

The classifier returned `1`.

The training process decoded it as:

```text
1 -> deny
```

A second service rebuilt its own label encoder from the two labels it happened to know. It decoded the same model output as:

```text
1 -> review
```

Neither service had a parsing bug. They had different fitted class mappings.

The model's numeric output was never a complete business answer. It was an index into the `LabelEncoder` vocabulary used during training. Once that vocabulary was lost or refitted independently, the integer remained valid while its meaning changed.

The problem became even more obvious for model output `2`. The training encoder mapped it to `review`; the partial encoder had only two classes and returned an out-of-range error.

This is the target-label version of a feature-schema bug. The matrix can be correct, the model can predict correctly, and the application can still show the wrong action because a separate artifact silently redefined the class IDs.

Let's fit a small multiclass model with [datarust](https://crates.io/crates/datarust), decode its outputs correctly, then refit an incompatible label encoder and watch the semantics move.

## Three actions become three integers

Our application uses three string labels:

```text
approve
deny
review
```

datarust's numeric classifiers accept numeric target values. `LabelEncoder` bridges the application vocabulary and the model target:

```text
approve -> 0
deny    -> 1
review  -> 2
```

The mapping is not based on workflow severity or a hand-written business order. `LabelEncoder` deduplicates the fitted strings and sorts them lexicographically.

The feature data contains three well-separated two-dimensional groups. That keeps the modeling part deliberately easy: each request should be classified correctly, leaving the label contract as the only thing we are trying to break.

After fitting, we score three representative requests and decode the numeric results with the original encoder.

Then we simulate a second service that fits a new encoder on only:

```text
approve
review
```

Its local mapping becomes:

```text
approve -> 0
review  -> 1
```

The integer `1` did not change. Its meaning did.

Here is the complete Rust program:

```rust
use datarust::encoder::label::LabelHandleUnknown;
use datarust::encoder::LabelEncoder;
use datarust::linear_model::{
    LogisticRegression, LogisticSolver,
};
use datarust::traits::Predictor;
use datarust::Matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    let mut string_labels = Vec::new();

    for (label, center_x, center_y) in [
        ("approve", -3.0, 0.0),
        ("deny", 3.0, 0.0),
        ("review", 0.0, 3.0),
    ] {
        for i in 0..30 {
            let dx =
                ((i % 5) as f64 - 2.0) * 0.08;
            let dy =
                ((i / 5) as f64 - 2.5) * 0.06;
            rows.push(vec![
                center_x + dx,
                center_y + dy,
            ]);
            string_labels.push(label);
        }
    }

    let x_train = Matrix::new(rows)?;
    let mut labels = LabelEncoder::new();
    let encoded =
        labels.fit_transform(string_labels)?;
    let y_train: Vec<f64> = encoded
        .iter()
        .map(|&value| value as f64)
        .collect();

    let mut model = LogisticRegression::new()
        .with_solver(LogisticSolver::Svd)
        .with_max_iter(200);
    model.fit(&x_train, &y_train)?;

    let requests = Matrix::new(vec![
        vec![-3.0, 0.0],
        vec![3.0, 0.0],
        vec![0.0, 3.0],
    ])?;
    let predicted_f64 = model.predict(&requests)?;
    let predicted_codes: Vec<usize> = predicted_f64
        .iter()
        .map(|value| *value as usize)
        .collect();
    let decoded =
        labels.inverse_transform(&predicted_codes)?;

    println!("training mapping");
    for (code, label) in
        labels.classes().iter().enumerate()
    {
        println!("{code} -> {label}");
    }

    println!("\nmodel output");
    for row in 0..requests.nrows() {
        println!(
            "request {} -> code {} -> {}",
            row + 1,
            predicted_codes[row],
            decoded[row]
        );
    }

    let mut partial = LabelEncoder::new();
    partial.fit(["approve", "review"])?;
    println!(
        "\nrefitted partial mapping: {:?}",
        partial.classes()
    );
    for &code in &predicted_codes {
        match partial.inverse_transform(&[code]) {
            Ok(value) => println!(
                "model code {code} -> {}",
                value[0]
            ),
            Err(error) => println!(
                "model code {code} -> ERROR: {error}"
            ),
        }
    }

    let mut numeric_strings = LabelEncoder::new();
    let numeric_codes = numeric_strings
        .fit_transform(["1", "2", "10"])?;
    println!(
        "\nnumeric-looking labels: classes {:?}, \
         input codes {:?}",
        numeric_strings.classes(),
        numeric_codes
    );

    let mut tolerant = LabelEncoder::new()
        .handle_unknown(LabelHandleUnknown::Ignore);
    tolerant.fit(["approve", "deny", "review"])?;
    let unknown = tolerant.transform(["appeal"])?;
    let unknown_decoded =
        tolerant.inverse_transform(&unknown)?;
    println!(
        "unknown label uses sentinel: {}, decoded {:?}",
        unknown[0] == usize::MAX,
        unknown_decoded[0]
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
training mapping
0 -> approve
1 -> deny
2 -> review

model output
request 1 -> code 0 -> approve
request 2 -> code 1 -> deny
request 3 -> code 2 -> review

refitted partial mapping: ["approve", "review"]
model code 0 -> approve
model code 1 -> review
model code 2 -> ERROR: unknown label: index 2

numeric-looking labels: classes ["1", "10", "2"], input codes [0, 2, 1]
unknown label uses sentinel: true, decoded ""
```

The classifier learned the easy geometry correctly. The semantic failure appeared only when another component interpreted its output with the wrong vocabulary.

## The codes are identifiers, not meanings

The original encoder reports its fitted classes:

```text
["approve", "deny", "review"]
```

The position in that array defines the numeric code:

```text
classes[0] = "approve"
classes[1] = "deny"
classes[2] = "review"
```

Code `2` does not mean “more review” than code `1`, and code `0` does not mean “lowest priority.” They are categorical identifiers chosen to satisfy a numeric model interface.

For multiclass logistic regression, the model learns a probability and decision structure over those encoded classes. When it predicts `1.0`, the correct application label comes from the same mapping used to create training target `1.0`.

The complete inference path is:

```text
raw request
    -> numeric feature pipeline
    -> classifier predicts numeric class code
    -> fitted LabelEncoder inverse transform
    -> application action
```

Stopping at the numeric code exposes an internal implementation detail as if it were a stable product API.

## Refitting changed the dictionary

The partial service sees only two strings:

```text
approve
review
```

It fits a perfectly valid local encoder:

```text
0 -> approve
1 -> review
```

But the model was not trained against that dictionary. Its code `1` means `deny`.

The result is a silent semantic substitution:

```text
model intended: 1 -> deny
service decoded: 1 -> review
```

Code `2` at least fails loudly because the partial vocabulary has no third entry. Code `1` is more dangerous: it is within range and decodes successfully to the wrong action.

This is why a fresh `LabelEncoder` in a prediction service is not a harmless reconstruction. `fit` creates a mapping. Serving needs the mapping already used by training.

I never fit a target decoder from:

- the labels present in the current request batch,
- the subset of classes known to one microservice,
- a sample configuration file with missing values,
- or a hand-maintained array whose ordering merely looks equivalent.

The fitted encoder is model state.

## Sorting is deterministic, not semantic

Lexicographic sorting makes the same complete string set produce the same mapping. That is useful. It is not a promise of business order.

The numeric-looking example makes this visible:

```text
input:   ["1", "2", "10"]
classes: ["1", "10", "2"]
codes:   [0, 2, 1]
```

As strings, `"10"` sorts before `"2"` because the first character `1` sorts before `2`.

If these are category names, that order is harmless as long as the mapping is preserved. If they are genuinely numeric or ordinal values, converting them to categorical string IDs and later interpreting their codes as magnitude invents the wrong order.

The correct choices differ:

- Use `LabelEncoder` when strings are names of target classes.
- Parse actual numeric targets as numbers.
- Define an explicit ordinal mapping when order has domain meaning.
- Never infer business priority from the automatically assigned code.

Deterministic sorting answers “which ID should this string receive?” It does not answer “which outcome is more severe?”

## Decoding belongs next to the model artifact

With datarust's `serde` feature enabled, `LabelEncoder` can be serialized with its fitted `classes`, lookup indices, unknown policy, and fitted state.

A deployment can save the model and label mapping as versioned components:

```rust
datarust::serialize::save_json(
    &model,
    "decision-model.json",
)?;
datarust::serialize::save_json(
    &labels,
    "decision-labels.json",
)?;
```

At serving time:

```rust
let model: LogisticRegression =
    datarust::serialize::load_json(
        "decision-model.json"
    )?;
let labels: LabelEncoder =
    datarust::serialize::load_json(
        "decision-labels.json"
    )?;
```

Two files still represent one logical artifact. They need the same model version, atomic promotion, matching rollback, and a compatibility check.

An application-level manifest can record:

```text
model checksum
label-encoder checksum
ordered class list
application version
training-data version
golden requests and decoded outputs
```

Readable JSON helps inspection. It does not make mismatched files compatible.

## Numeric predictions deserve validation before casting

The example converts hard predictions from `f64` to `usize`:

```rust
let predicted_codes: Vec<usize> = predicted_f64
    .iter()
    .map(|value| *value as usize)
    .collect();
```

That is safe in this controlled program because the fitted classifier returns its learned integer class labels.

At a production boundary, I validate before casting:

- the value is finite,
- it is non-negative,
- it is integer-valued within a tolerance,
- and the resulting index is smaller than `labels.classes().len()`.

An unchecked float-to-integer cast can turn an upstream model or parsing error into a confusing downstream label failure. The decoder's range error is useful, but catching the invalid model output at the boundary gives better diagnostics.

Probability output needs the same mapping discipline. Column `j` belongs to encoded class `j`, which belongs to `labels.classes()[j]`. Exporting a probability array without its ordered labels creates another opportunity for semantic drift.

## Unknown labels are not ordinary classes

`LabelHandleUnknown::Error` is the default. It rejects a string that was absent during fitting.

The tolerant policy is explicit:

```rust
LabelEncoder::new()
    .handle_unknown(LabelHandleUnknown::Ignore)
```

For unknown input `"appeal"`, datarust returns the sentinel:

```text
usize::MAX
```

The example prints:

```text
unknown label uses sentinel: true, decoded ""
```

The empty decoded string signals that the sentinel has no known class name. It must not be treated as a fourth business class or casually converted into a numeric training label.

Unknown target labels usually mean the taxonomy changed:

- a new outcome was introduced,
- annotation policy changed,
- spelling normalization drifted,
- or evaluation data is incompatible with the model version.

Ignoring that event can make metrics meaningless. A classifier trained for `approve / deny / review` cannot be evaluated honestly by silently hiding new `appeal` outcomes.

I either reject and investigate the unknown label or handle it through an explicit versioned taxonomy migration.

## Target encoding is different from feature encoding

`LabelEncoder` is for one-dimensional target labels. It turns outcome names into class IDs and can reverse those IDs after prediction.

It is not a shortcut for converting an unordered input feature such as plan, city, or browser into one numeric column. A feature encoder affects geometry consumed by the estimator; treating arbitrary categories as `0`, `1`, and `2` can invent distance and direction.

For input columns, datarust provides one-hot, ordinal, frequency, and target-aware encoders with different assumptions.

For output labels, the code is an identifier. The model does not interpret the difference between class IDs as a regression distance. The application uses the fitted mapping to recover the name.

Keeping those two jobs separate prevents a surprising amount of confusion around the word “encoding.”

## Taxonomy changes require model-version decisions

Suppose the product team replaces `review` with two outcomes:

```text
manual_review
automated_review
```

Updating only the label encoder cannot make the existing classifier predict the new distinction. Its training targets contained one `review` class, and its learned output space reflects that taxonomy.

Likewise, renaming `deny` to `reject` may be a display-only migration or a genuine label-definition change. The team has to decide which:

- If semantics are identical, preserve the model mapping and translate the display name through versioned metadata.
- If semantics changed, rebuild the training labels, retrain, and evaluate a new model artifact.

The ordered class list is part of the model interface, but the larger label definition is part of the data contract.

A string rename can be operationally small and statistically enormous.

## The golden test starts with names and ends with names

Before promotion, I run fixed examples through the complete semantic path:

```text
known raw features
    -> fitted feature preprocessing
    -> model numeric prediction
    -> fitted label inverse transform
    -> expected business label
```

For this experiment, the invariant is:

```text
request near (-3, 0) -> approve
request near ( 3, 0) -> deny
request near ( 0, 3) -> review
```

Testing only the numeric outputs `[0, 1, 2]` would miss a swapped decoder. Testing only the encoder mapping would miss a changed model. The end-to-end assertion catches both.

I also assert the exact ordered class list at startup:

```text
["approve", "deny", "review"]
```

If the serving binary expects another taxonomy, startup should fail before a correct numeric prediction becomes a wrong business action.

## The integer was an index into history

The original system behaved correctly:

```text
0 -> approve
1 -> deny
2 -> review
```

The partial encoder also behaved correctly relative to its own fitted data:

```text
0 -> approve
1 -> review
```

The failure came from combining a model trained against the first dictionary with a decoder fitted against the second.

Code `1` did not carry the word `deny` inside it. Its meaning lived in the fitted class array and the training run that produced both artifacts.

So I no longer treat target encoding as a disposable preprocessing detail. I preserve the encoder, version the taxonomy, export probability columns with ordered labels, and test decoded outcomes after every restart.

The classifier predicted `1`.

Only the correct historical mapping knew what that meant.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
