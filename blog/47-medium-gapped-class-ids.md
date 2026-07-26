# My Predictions Were Perfect. Macro F1 Was 0.095.

*A practical datarust guide to zero-based class indices, gapped business IDs, phantom classes, oversized confusion matrices, LabelEncoder, comparable evaluation slices, and the two integer spaces I should never have treated as interchangeable.*

---

> **Version note — July 26, 2026:** This incident reproduces datarust `0.6.0`.
> Version `0.6.1` fixes the underlying bug: metrics compact observed numeric
> labels, labeled confusion matrices expose the original row/column mapping,
> and `LogisticRegression` maps predictions back to the labels used in `fit`.
> The encoding boundary described here is still useful when business taxonomy
> and model vocabulary have different lifecycles.

The classifier predicted every label correctly.

Our two business classes were numbered `10` and `20`:

```text
actual:    10, 10, 10, 20, 20, 20
predicted: 10, 10, 10, 20, 20, 20
```

Accuracy was exactly `1.0`.

Macro F1 was `0.095238`.

The confusion matrix had 21 rows and 21 columns even though the application had only two classes. Just two cells contained data.

I had passed valid business identifiers into an API that expected compact class indices. datarust interpreted the largest label, `20`, as evidence that the label space ran from `0` through `20`. It calculated perfect F1 for classes 10 and 20, zero for the other 19 positions, and averaged all 21 values:

```text
(1 + 1) / 21 = 0.095238
```

The model was not bad. The prediction vector was not wrong. My evaluation representation was.

Let's reproduce the historical result with datarust `0.6.0`, normalize the same labels with `LabelEncoder`, and then look at how `0.6.1` makes the safe behavior the default.

## A class ID and a class index are different things

The application inherited two status IDs from a database:

```text
10 = approved
20 = review
```

Those numbers are identifiers. Their magnitudes and the gap between them have no statistical meaning. Status 20 is not twice status 10, and statuses 0 through 9 do not secretly exist because 10 does.

A model-facing class index has a narrower contract. For `K` classes, it should be one of:

```text
0, 1, 2, ..., K - 1
```

For our two classes, that means:

```text
business ID 10 -> model index 0
business ID 20 -> model index 1
```

The mapping must be fitted or configured once, persisted, and reused in both directions. The model sees compact indices. The application sees stable business names or IDs.

I skipped that boundary because both sides happened to use numbers. The type system saw `f64` values either way. The metric saw a very different class universe.

## The complete Rust experiment

Create a small application and add datarust:

```sh
cargo new class_index_contract
cd class_index_contract
cargo add datarust@=0.6.0
```

Replace `src/main.rs` with this:

```rust
use datarust::encoder::LabelEncoder;
use datarust::metrics::classification::{
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
};

fn as_f64(values: &[usize]) -> Vec<f64> {
    values
        .iter()
        .map(|&value| value as f64)
        .collect()
}

fn raw(values: &[&str]) -> Vec<f64> {
    values
        .iter()
        .map(|value| value.parse::<f64>().unwrap())
        .collect()
}

fn report(
    name: &str,
    truth: &[f64],
    predicted: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let cm = confusion_matrix(truth, predicted)?;
    let nonzero = cm
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&count| count > 0)
        .count();

    println!("{name}");
    println!(
        "  accuracy:        {:.6}",
        accuracy_score(truth, predicted)?,
    );
    println!(
        "  macro precision: {:.6}",
        precision_score(truth, predicted)?,
    );
    println!(
        "  macro recall:    {:.6}",
        recall_score(truth, predicted)?,
    );
    println!(
        "  macro F1:        {:.6}",
        f1_score(truth, predicted)?,
    );
    println!(
        "  confusion shape: {}x{} \
         ({nonzero} nonzero cells)",
        cm.len(),
        cm[0].len(),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let truth = ["10", "10", "10", "20", "20", "20"];
    let perfect = ["10", "10", "10", "20", "20", "20"];
    let one_miss = ["10", "10", "10", "20", "20", "10"];

    report(
        "raw IDs, perfect",
        &raw(&truth),
        &raw(&perfect),
    )?;
    report(
        "raw IDs, one miss",
        &raw(&truth),
        &raw(&one_miss),
    )?;

    // In a real workflow, fit this on the training target and
    // reuse the same encoder for validation and production.
    let mut labels = LabelEncoder::new();
    labels.fit(["10", "20"])?;
    let truth_codes = labels.transform(truth)?;
    let perfect_codes = labels.transform(perfect)?;
    let miss_codes = labels.transform(one_miss)?;

    println!(
        "mapping: {:?} -> [0, 1]",
        labels.classes(),
    );
    report(
        "encoded IDs, perfect",
        &as_f64(&truth_codes),
        &as_f64(&perfect_codes),
    )?;
    report(
        "encoded IDs, one miss",
        &as_f64(&truth_codes),
        &as_f64(&miss_codes),
    )?;
    Ok(())
}
```

Run it:

```sh
cargo run --release
```

The output is:

```text
raw IDs, perfect
  accuracy:        1.000000
  macro precision: 0.095238
  macro recall:    0.095238
  macro F1:        0.095238
  confusion shape: 21x21 (2 nonzero cells)
raw IDs, one miss
  accuracy:        0.833333
  macro precision: 0.083333
  macro recall:    0.079365
  macro F1:        0.078912
  confusion shape: 21x21 (3 nonzero cells)
mapping: ["10", "20"] -> [0, 1]
encoded IDs, perfect
  accuracy:        1.000000
  macro precision: 1.000000
  macro recall:    1.000000
  macro F1:        1.000000
  confusion shape: 2x2 (2 nonzero cells)
encoded IDs, one miss
  accuracy:        0.833333
  macro precision: 0.875000
  macro recall:    0.833333
  macro F1:        0.828571
  confusion shape: 2x2 (3 nonzero cells)
```

Encoding does not change which six predictions are correct. It changes whether the evaluator understands that there are two classes or imagines 21 possible index positions.

## Why the confusion matrix became 21×21 in v0.6.0

In datarust v0.6.0, classification labels are represented as non-negative integer-valued floats. The confusion-matrix implementation determines its dimension from the largest label present in either `y_true` or `y_pred`:

```text
n_classes = max_label + 1
```

The largest label in our arrays is 20, so:

```text
n_classes = 20 + 1 = 21
```

The matrix therefore reserves rows and columns for:

```text
0, 1, 2, ..., 9, 10, 11, ..., 19, 20
```

Only these diagonal cells are nonzero:

```text
cm[10][10] = 3
cm[20][20] = 3
```

The empty positions are not discovered business classes. They are a consequence of treating an external identifier as an array index.

With `LabelEncoder`, the maximum label is 1:

```text
n_classes = 1 + 1 = 2
```

Now the matrix has exactly the two rows and columns the problem needs.

## Macro averaging made the representation bug visible

Macro F1 calculates an F1 score per class and gives every class equal weight.

For the raw perfect predictions:

```text
class 10 F1 = 1
class 20 F1 = 1
classes 0..9 and 11..19 F1 = 0
```

There are 19 empty class positions and two populated ones:

```text
macro F1 = (19 × 0 + 1 + 1) / 21
         = 0.095238
```

This does not mean macro averaging is broken. Macro averaging can only operate on the class set supplied by the confusion matrix. The bad class set came from the labels.

After compact encoding, the same arithmetic becomes:

```text
macro F1 = (1 + 1) / 2 = 1
```

The metric is now answering the intended question: how well did the classifier perform when approved and review each receive equal weight?

## Accuracy did not care about the gaps

Accuracy compares labels row by row:

```text
correct rows / all rows
```

For the perfect raw arrays, each `10` equals `10` and each `20` equals `20`. All six rows match, so accuracy is `1.0`.

It never needs to construct a score for the absent integers between them. That is why accuracy looked correct while macro precision, recall, and F1 collapsed.

The disagreement between metrics was useful evidence. When accuracy says every row is correct but F1 says performance is near zero, I inspect label encoding and confusion-matrix shape before blaming the estimator.

I do not “resolve” the disagreement by reporting only accuracy. The metric mismatch exposed a broken evaluation contract that could affect more than one dashboard.

## One wrong prediction did not explain the tiny F1

Our second prediction vector makes one real mistake:

```text
actual 20 -> predicted 10
```

The compact representation reports:

```text
accuracy: 0.833333
macro F1: 0.828571
```

Those values are easy to reconcile. Five of six rows are correct, and both classes still have reasonably strong F1 scores.

The raw-ID representation reports macro F1 `0.078912`. The single mistake reduced performance, but it did not create 19 additional real classes. Most of that tiny result still comes from averaging zeros for index positions that should never have entered the evaluation.

This is why I keep the confusion matrix beside summary metrics. Its dimensions are schema information. A `21×21` matrix for a two-class product is not merely inconvenient formatting; it is a diagnostic signal.

## Large external IDs can turn a score bug into a memory problem

The 21×21 matrix is small. Real business identifiers are often not.

Imagine two classes with IDs:

```text
100
10000
```

A maximum-label-based confusion matrix would require `10001 × 10001` counters—more than 100 million cells. At eight bytes per `usize` on a 64-bit system, the cell storage alone is roughly 800 MB, before row allocations and other overhead.

An ID in the millions can imply terabytes of requested storage. The process may slow dramatically, fail allocation, or be terminated before any metric is returned.

I do not run a “quick test” with a giant label to prove this in production. The allocation formula is proof enough:

```text
memory grows approximately with (max_label + 1)²
```

Compact encoding changes that to growth with the number of actual classes:

```text
memory grows approximately with K²
```

For two real classes, the correct matrix is four counters regardless of the database IDs attached to them.

## What v0.6.1 changes

Version `0.6.1` moves that compact vocabulary into the classification core.
The same external numeric labels now produce the intended result directly:

```rust
use datarust::metrics::classification::{
    confusion_matrix_labeled, f1_score,
};

let truth = vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0];
let matrix = confusion_matrix_labeled(&truth, &truth)?;

assert_eq!(matrix.labels, vec![10.0, 20.0]);
assert_eq!(matrix.counts, vec![vec![3, 0], vec![0, 3]]);
assert_eq!(f1_score(&truth, &truth)?, 1.0);
# Ok::<_, datarust::DatarustError>(())
```

The compact indices remain an implementation detail. When the mapping matters,
`confusion_matrix_labeled` returns it alongside the counts. The plain
`confusion_matrix` helper returns only the compact count matrix.

`LogisticRegression` uses the same idea. A model fitted on `{2, 5, 9}` predicts
`2`, `5`, or `9`, and probability column `i` corresponds to `classes()[i]`.
`predict_proba_for_class` retrieves a column without making the caller guess its
position.

## LabelEncoder creates the right model vocabulary

`LabelEncoder` learns the unique string labels, sorts them, and maps them to contiguous indices.

For our numeric-looking strings:

```text
classes: ["10", "20"]
mapping: 10 -> 0, 20 -> 1
```

The lexicographic ordering is not a statement that one class is safer, larger, or more important. The codes are array positions, not feature values.

That distinction matters:

- An input feature encoded as `0` and `1` may introduce numeric geometry to a model.
- A target class code tells the classifier and evaluator which class slot a row belongs to.

For target labels, compact indices are exactly what we want—as long as the original mapping remains available to recover the business meaning.

## Fit the mapping on training labels, then freeze it

The encoder is fitted state.

The correct lifecycle is:

```text
training:
    fit LabelEncoder on training target labels
    transform training targets
    fit classifier

validation and test:
    transform labels with the training encoder
    predict model indices
    calculate metrics in the same index space

production:
    predict model indices
    inverse-transform with the persisted training encoder
```

I do not independently fit one encoder for the actual labels and another for the predictions. Two encoders can assign different integers when their observed vocabularies differ. The arrays may then compare equal or unequal for the wrong semantic reason.

I also do not refit the encoder on each evaluation slice. A geography that happens to contain only one class would create a different local vocabulary from the complete model.

One fitted mapping travels with the classifier.

## Unknown target labels should stop evaluation

Suppose the deployed product adds a new status:

```text
30 = manual escalation
```

If the fitted classifier knows only approved and review, silently assigning manual escalation an available numeric code does not make the model capable of predicting it.

For evaluation, an unknown ground-truth class is an important compatibility event. I usually fail clearly and decide one of the following:

- exclude the period with an explicit reason while reporting coverage
- map the new status into an approved existing taxonomy upstream
- retrain a model whose target vocabulary includes the new class
- evaluate the old and new taxonomies separately

`LabelHandleUnknown::Ignore` can return a sentinel for application-controlled handling. I never cast that sentinel into a floating-point label and pass it directly to classification metrics. A sentinel is not a model class, and a very large sentinel is especially dangerous for a maximum-label-sized confusion matrix.

Unknown-class handling belongs before metric allocation.

## Validate integer labels before scoring

datarust metric APIs accept `f64` slices because classifier predictions use numeric floats. The semantic contract is still non-negative integer-valued class labels.

I validate each value before evaluation:

```text
finite
not negative
has no fractional part
known to the fitted model vocabulary, when scoring a model
```

That catches mistakes such as:

```text
probability 0.83 passed where class index 1 was expected
an unknown business ID passed as though the model had fitted it
unknown-label sentinel passed as an ordinary class
NaN produced by an upstream conversion
```

Starting with v0.6.1, the metric implementation rejects fractional, negative,
and non-finite labels instead of rounding them. I still validate at the
application boundary because an integer can be well formed while belonging to
the wrong taxonomy.

Probabilities belong in probability-aware metrics such as log loss, ROC-AUC, or average precision. Hard-label precision, recall, F1, and confusion matrices need discrete class indices.

## Evaluation slices need a declared class universe

Compact encoding solves the ID-gap problem, but sliced evaluation introduces another question:

```text
Which classes should contribute to this slice's macro average?
```

Imagine a three-class model with indices `0`, `1`, and `2`. One regional test slice happens to contain only classes 0 and 1, and the model never predicts 2 there. An auto-sized confusion matrix sees a maximum label of 1 and becomes `2×2`. Another slice containing class 2 becomes `3×3`.

Their macro F1 values now average over different observed class sets.

Neither convention is universally wrong:

- Observed-class macro F1 asks how the model handled classes present in that slice.
- Global-class macro F1 gives every model class a position, even when a slice contains no support for estimating some of them.

What is wrong is comparing the two numbers as if they used the same denominator.

For release evaluation, I record:

```text
global class vocabulary
support count per class and slice
confusion-matrix dimensions
per-class precision, recall, and F1
macro aggregation convention
```

If I need a fixed global class set across slices, I calculate the per-class counts against that known vocabulary rather than relying on each slice to infer its own dimensions.

Zero support is information, not a score to hide.

## Business IDs can change without changing the modeling problem

External identifiers have their own lifecycle. A database migration may renumber status 20 to 200. Two companies merging may have conflicting code systems. A downstream service may replace numeric codes with UUIDs.

Before v0.6.1, passing those identifiers directly to model metrics could change
confusion-matrix size and macro scores. Metrics now compact the observed IDs,
but an administrative migration can still change the vocabulary expected by a
fitted model or make reports from two periods describe different taxonomies.

With an explicit target vocabulary, the modeling space remains stable:

```text
"approved" -> 0
"review"   -> 1
```

The application can separately map current database IDs to those canonical names. This adds one visible translation layer and removes many invisible assumptions.

I prefer stable semantic labels at that boundary. A name such as `manual_review` is harder to confuse with an array offset than the number `20`.

## Persist both directions of the mapping

The model artifact needs more than a coefficient matrix or solver state.

I store:

```text
class index -> canonical label
canonical label -> class index
business ID -> canonical label, when applicable
encoder and model version
training taxonomy version
unknown-label policy
```

The forward mapping prepares training and evaluation targets. The inverse mapping turns model predictions back into application decisions.

During deployment validation, I test both:

```text
business ID 10 -> approved -> model index 0
model index 0 -> approved -> current business ID 10
```

A round trip should preserve meaning, not merely produce a valid integer.

## Tests I now keep near every classifier

The useful tests are small enough to run on every build:

```text
K fitted classes produce exactly indices 0..K-1
all training, validation, and predicted indices are < K
perfect predictions produce accuracy 1 and macro F1 1
confusion-matrix shape is K×K for the full evaluation set
inverse-transform restores every canonical label
unknown labels follow the documented error path
serialized and in-memory mappings are identical
business-ID changes do not alter compact model indices
```

The perfect-prediction assertion is especially valuable. It sounds almost too obvious to test. In this experiment, it would have caught the broken label space immediately.

For sliced reports, I add support counts and explicitly test the chosen global-versus-observed class convention.

## The 0.095 score belonged to 19 imaginary classes

Our prediction vector was perfect:

```text
10, 10, 10, 20, 20, 20
```

But `10` and `20` were database identifiers, not zero-based class indices. In
datarust v0.6.0, that allocated a `21×21` confusion matrix and macro-averaged
across 21 positions. Nineteen empty positions contributed zero, pulling perfect
macro F1 down to `0.095238`.

`LabelEncoder` translated the same two semantic labels to `0` and `1`. The
confusion matrix became `2×2`, macro F1 became `1.0`, and no prediction changed.
Version `0.6.1` now performs that compaction internally for numeric labels while
retaining the original mapping when requested.

My safeguards are now straightforward:

- use v0.6.1 or newer for compact numeric-label metrics
- keep a deliberate boundary between external IDs and model vocabulary
- use `LabelEncoder` when semantic or string labels need a persisted mapping
- fit the encoder on training labels and reuse it everywhere
- persist the mapping with the classifier
- reject fractional, out-of-range, and sentinel labels before scoring
- inspect confusion-matrix dimensions and per-class support
- declare the class universe used for sliced macro metrics
- test that perfect predictions really receive perfect scores

Both sides used integers. That did not make them the same language.
