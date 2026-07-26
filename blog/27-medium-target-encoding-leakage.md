# My Customer IDs Predicted Churn With 100% Accuracy. New Customers Got a Coin Flip.

*A practical datarust guide to target encoding, singleton categories, smoothing, cross-fitting, and the feature that quietly carried its own label.*

---

The training result looked flawless.

```text
training accuracy: 100.0%
training R²:         1.000
```

The model had one feature: target-encoded `customer_id`.

Every customer appeared once.

That detail turned the feature into a disguised copy of the target. A customer who churned received one encoded value; a customer who stayed received another. Linear regression only had to translate those two numbers back to zero and one.

I added smoothing of 20. The encodings moved much closer together:

```text
stayed:  0.476190
churned: 0.523810
```

Training accuracy remained 100%.

Then I sent 100 genuinely new customer IDs through the fitted encoder. Every ID became the global churn mean, `0.5`. Test accuracy fell to 50%.

This is the dangerous elegance of target leakage. The training transformation was mathematically valid, the model fit was real, and the metric calculation was correct. The experimental boundary was wrong.

Let's reproduce the leak, show why smoothing cannot remove it, and build an out-of-fold encoding with [datarust](https://crates.io/crates/datarust).

## Two hundred IDs with no predictive signal

The training data contains 200 unique customer IDs:

```text
customer_000
customer_001
...
customer_199
```

The binary labels are deterministically balanced: 100 zeros and 100 ones. The IDs themselves contain no reusable churn information. A separate test set contains 100 different identifiers:

```text
new_customer_000
...
new_customer_099
```

It is balanced too.

I deliberately use linear regression as a scoring model and classify scores at `0.5`. This is not a recommendation to replace logistic regression. It makes the leakage algebra exact and easy to inspect. A logistic model, tree, or sufficiently flexible downstream estimator can exploit the same target-derived separation.

The experiment has two parts:

1. Fit and transform the same training rows with smoothing values 0, 20, and 100.
2. Construct five-fold out-of-fold encodings so no training row contributes its own label to its encoded feature.

Here is the complete Rust program:

```rust
use datarust::encoder::TargetEncoder;
use datarust::linear_model::{
    LinearRegression, LinearSolver,
};
use datarust::metrics::{
    classification::accuracy_score,
    regression::r2_score,
};
use datarust::traits::Predictor;
use datarust::{Matrix, StrMatrix};

fn make_ids(
    prefix: &str,
    n: usize,
) -> (StrMatrix, Vec<f64>) {
    let ids: Vec<String> = (0..n)
        .map(|i| format!("{prefix}_{i:03}"))
        .collect();

    // Five deterministic folds; every fold has the
    // same 50/50 class balance.
    let labels: Vec<f64> = (0..n)
        .map(|i| ((i / 5) % 2) as f64)
        .collect();

    (StrMatrix::from_column(ids).unwrap(), labels)
}

fn hard_labels(scores: &[f64]) -> Vec<f64> {
    scores
        .iter()
        .map(|&score| {
            if score >= 0.5 { 1.0 } else { 0.0 }
        })
        .collect()
}

fn min_max(x: &Matrix) -> (f64, f64) {
    x.as_slice().iter().copied().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(min, max), value| {
            (min.min(value), max.max(value))
        },
    )
}

fn out_of_fold_encode(
    ids: &StrMatrix,
    labels: &[f64],
    smoothing: f64,
    folds: usize,
) -> Result<Matrix, Box<dyn std::error::Error>> {
    let mut encoded = vec![0.0; ids.nrows()];

    for fold in 0..folds {
        let train_ids: Vec<String> = (0..ids.nrows())
            .filter(|i| i % folds != fold)
            .map(|i| ids.get(i, 0).to_string())
            .collect();
        let train_y: Vec<f64> = (0..ids.nrows())
            .filter(|i| i % folds != fold)
            .map(|i| labels[i])
            .collect();
        let valid_indices: Vec<usize> =
            (0..ids.nrows())
                .filter(|i| i % folds == fold)
                .collect();
        let valid_ids: Vec<String> = valid_indices
            .iter()
            .map(|&i| ids.get(i, 0).to_string())
            .collect();

        let train_matrix =
            StrMatrix::from_column(train_ids)?;
        let valid_matrix =
            StrMatrix::from_column(valid_ids)?;

        let mut encoder =
            TargetEncoder::new(smoothing)?;
        encoder.fit(&train_matrix, &train_y)?;
        let fold_values =
            encoder.transform(&valid_matrix)?;

        for (row, &original_index) in
            valid_indices.iter().enumerate()
        {
            encoded[original_index] =
                fold_values.get(row, 0);
        }
    }

    Ok(Matrix::new(
        encoded
            .into_iter()
            .map(|value| vec![value])
            .collect(),
    )?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (train_ids, y_train) =
        make_ids("customer", 200);
    let (test_ids, y_test) =
        make_ids("new_customer", 100);

    println!(
        "smoothing  train encoded range   coefficient  \
         intercept  train R2  train acc  new-ID acc"
    );

    for smoothing in [0.0, 20.0, 100.0] {
        let mut encoder =
            TargetEncoder::new(smoothing)?;
        let x_train =
            encoder.fit_transform(&train_ids, &y_train)?;
        let x_test = encoder.transform(&test_ids)?;

        let mut model = LinearRegression::new()
            .with_solver(LinearSolver::Svd);
        model.fit(&x_train, &y_train)?;

        let train_scores = model.predict(&x_train)?;
        let test_scores = model.predict(&x_test)?;
        let (low, high) = min_max(&x_train);

        println!(
            "{:>9.0}  [{:.6}, {:.6}]  {:>11.3}  \
             {:>9.3}  {:>8.3}  {:>9.1}%  {:>10.1}%",
            smoothing,
            low,
            high,
            model.coef()[0],
            model.intercept(),
            r2_score(&y_train, &train_scores)?,
            100.0
                * accuracy_score(
                    &y_train,
                    &hard_labels(&train_scores),
                )?,
            100.0
                * accuracy_score(
                    &y_test,
                    &hard_labels(&test_scores),
                )?,
        );
    }

    let oof_train = out_of_fold_encode(
        &train_ids,
        &y_train,
        20.0,
        5,
    )?;

    let mut final_encoder =
        TargetEncoder::new(20.0)?;
    final_encoder.fit(&train_ids, &y_train)?;
    let honest_test =
        final_encoder.transform(&test_ids)?;

    let mut honest_model = LinearRegression::new()
        .with_solver(LinearSolver::Svd);
    honest_model.fit(&oof_train, &y_train)?;

    let honest_train_scores =
        honest_model.predict(&oof_train)?;
    let honest_test_scores =
        honest_model.predict(&honest_test)?;
    let (oof_low, oof_high) = min_max(&oof_train);

    println!(
        "\nout-of-fold encoding with smoothing 20"
    );
    println!(
        "OOF encoded range: \
         [{oof_low:.6}, {oof_high:.6}]"
    );
    println!(
        "coefficient: {:.3}, intercept: {:.3}",
        honest_model.coef()[0],
        honest_model.intercept(),
    );
    println!(
        "OOF train accuracy: {:.1}%, \
         new-ID accuracy: {:.1}%",
        100.0
            * accuracy_score(
                &y_train,
                &hard_labels(&honest_train_scores),
            )?,
        100.0
            * accuracy_score(
                &y_test,
                &hard_labels(&honest_test_scores),
            )?,
    );

    let unseen_values = final_encoder.transform(
        &StrMatrix::from_column([
            "new_customer_A",
            "new_customer_B",
            "new_customer_C",
        ])?,
    )?;
    println!(
        "unseen categories after final fit: {:?}",
        unseen_values.as_slice(),
    );

    Ok(())
}
```

This is the output I measured:

```text
smoothing  train encoded range   coefficient  intercept  train R2  train acc  new-ID acc
        0  [0.000000, 1.000000]        1.000      0.000     1.000      100.0%        50.0%
       20  [0.476190, 0.523810]       21.000    -10.000     1.000      100.0%        50.0%
      100  [0.495050, 0.504950]      101.000    -50.000     1.000      100.0%        50.0%

out-of-fold encoding with smoothing 20
OOF encoded range: [0.500000, 0.500000]
coefficient: 0.000, intercept: 0.500
OOF train accuracy: 50.0%, new-ID accuracy: 50.0%
unseen categories after final fit: [0.5, 0.5, 0.5]
```

The out-of-fold result looked worse because it was the first result that answered the deployment question honestly.

## The encoder used the label it was supposed to predict

For category `c`, datarust computes:

```text
(count_c × mean_target_c + smoothing × global_mean)
-------------------------------------------------------
               count_c + smoothing
```

Every customer ID appears once, so:

```text
count_c = 1
mean_target_c = that customer's own label
global_mean = 0.5
```

With no smoothing, the formula becomes:

```text
encoded_customer = label
```

The target was not subtly correlated with the feature. It was the feature.

Calling `fit_transform` is perfectly appropriate for transformations that do not read `y`, such as standard scaling fitted on training features. Target encoding is supervised preprocessing. Transforming the same rows used to compute their category target means gives each row a path to influence its own input.

The estimator never sees the raw label column as a feature. The information crosses the boundary through fitted preprocessing state.

## Smoothing narrowed the gap and preserved the answer

With smoothing 20, the two singleton encodings are:

```text
label 0: (0 + 20 × 0.5) / 21 = 10 / 21 = 0.476190
label 1: (1 + 20 × 0.5) / 21 = 11 / 21 = 0.523810
```

They look conservatively close to the global mean. They still sit on opposite sides of `0.5`, with no overlap.

The downstream linear model learns:

```text
predicted label = 21 × encoded value - 10
```

For smoothing 100, the gap shrinks further:

```text
label 0 -> 0.495050
label 1 -> 0.504950
```

The model responds with a coefficient of 101 and an intercept of -50. It reverses the smoothing exactly.

For any finite smoothing value in this controlled singleton setup, the encoded value remains an affine function of the row's label. Smoothing reduces how far a rare category moves away from the global mean. It does not prevent a flexible downstream model from detecting which side it moved toward.

Smoothing addresses variance and overconfidence in low-count category estimates. Cross-fitting addresses self-influence. They are not substitutes.

## New categories all received the global mean

The default unknown-category policy for `TargetEncoder` is `UnknownTarget::GlobalMean`.

None of the test IDs appeared during fitting, so every row received:

```text
0.5
```

The naively trained model had only seen two feature values tied perfectly to the labels. At `0.5`, its score also becomes `0.5`. Our threshold sends every test row to one class, producing 50% accuracy on the balanced test set.

The other unknown policies change failure handling, not information content:

- `GlobalMean` supplies a neutral fitted baseline.
- `NaN` marks the unknown and requires a downstream missing-value policy.
- `Error` rejects the transformation.

None can infer a customer-specific churn rate for an ID with no observed history.

That is not a limitation to work around with a clever sentinel. It is the honest information boundary.

## Out-of-fold encoding removed self-influence

The five-fold function creates the training feature in pieces.

For each fold:

1. Fit `TargetEncoder` on the other four folds.
2. Transform the held-out fold.
3. Write those values into the held-out rows' positions.

Every ID is unique, so a held-out ID is unknown to its fold encoder. Because each fold is class-balanced by construction, the remaining training portion also has a global mean of `0.5`.

The complete out-of-fold feature is therefore constant:

```text
OOF encoded range: [0.5, 0.5]
```

SVD-based linear regression assigns it a zero coefficient and uses the target mean as its intercept. Both out-of-fold training accuracy and new-ID accuracy are 50%.

This is not cross-fitting destroying a useful feature. It is cross-fitting revealing that unique identity had no transferable signal in the experiment.

## Repeated categories behave differently

Real target-encoding candidates are more often merchant, neighborhood, device family, campaign, or product category—not a perfectly unique row ID.

Suppose `merchant_17` appears in several folds. When its row is held out, the fold encoder can still estimate that merchant's outcome rate from *other* rows. The encoded value may carry legitimate historical signal without reading the held-out row's own label.

Rare categories remain noisy, which is where smoothing helps. Categories that occur only once behave like our customer IDs and fall back to the fold's global mean under honest cross-fitting.

The fold design must match the prediction boundary. Random row folds may still leak when:

- multiple rows belong to the same customer,
- later events reveal earlier outcomes,
- repeated measurements from one device appear in both sides,
- labels arrive after a delay,
- the task predicts a genuinely new entity rather than a future event for a known entity.

In those cases I use group-aware or chronological folds. Cross-fitting is only as honest as the split that defines “unseen.”

## The final encoder still fits all training rows

After producing out-of-fold features and fitting the downstream model, I fit one final encoder on all available training rows. That encoder transforms validation, test, and production categories without reading their labels.

The sequence is:

```text
training model inputs:
out-of-fold target encodings

validation / test / production inputs:
transform with final encoder fitted on all training rows
```

This is the same pattern used by cross-fitted target-encoding implementations: the training representation prevents self-leakage, while the final mapping uses all permitted history for future transforms.

I do not call `fit_transform` independently on validation or production data. That would either require labels that should not be available or build an incompatible mapping from the batch being predicted.

The final encoder, fold recipe, downstream model, feature order, smoothing value, and unknown policy form one model contract.

## A random holdout was necessary but not sufficient

We did reserve 100 new IDs, and their 50% result exposed the failure. But the 100% training metric was already invalid as an estimate because the downstream model trained on self-encoded rows.

A common workflow still gets this wrong:

```text
split raw rows
fit TargetEncoder on training rows and their labels
transform those same training rows
fit estimator
transform validation rows
score
```

The validation score itself does not directly leak validation labels. Yet the estimator learned from a training representation richer than the representation available for a rare or unseen validation category. That train–serve mismatch can cause extreme overfitting even with an untouched holdout.

Out-of-fold encoding makes the estimator train on values produced under the same information restriction each held-out row faces.

## Customer ID was the wrong abstraction

Even with a leakage-safe encoder, a unique customer ID remains useless for predicting a new customer in this experiment.

The better inputs describe properties that can exist at prediction time and recur across observations:

- tenure,
- plan type,
- verified acquisition channel,
- usage trends computed only from past events,
- payment history available before the prediction timestamp,
- support interactions with a clear lookback window.

An identifier can be legitimate when the task predicts another event for a known entity and prior entity outcomes are permitted information. Even then, I usually create explicit historical features with timestamps and minimum support rather than letting a target encoder hide the temporal logic inside a category mean.

Raw identifiers also raise privacy, memorization, fairness, and deletion concerns. A suspiciously powerful ID-like feature deserves more review, not less.

## What I log for a target-encoded feature

Before trusting target encoding, I inspect:

- unique-category count and singleton rate,
- category counts in every training fold,
- global mean per fold,
- smoothing value,
- unknown-category rate in validation and production,
- encoded-value ranges for OOF and final transforms,
- performance for frequent, rare, and unknown categories,
- whether the fold boundary respects entities and time.

I also compare against simple baselines: one-hot for manageable cardinality, frequency encoding, grouped rare categories, and removal of the feature.

If target encoding produces a spectacular gain concentrated in singletons, I assume leakage until the out-of-fold evidence proves otherwise.

## The worse score was the trustworthy one

The naive experiment produced a seductive table:

```text
smoothing 0:   100% training accuracy
smoothing 20:  100% training accuracy
smoothing 100: 100% training accuracy
```

Increasing smoothing made the encodings look safer while the downstream coefficient grew to undo it.

Cross-fitting removed each row's access to its own label. Every unique ID became the only defensible estimate available from the other folds: the global mean. Accuracy fell to 50%, exactly where a signal-free balanced feature belongs.

That result was not a failed encoder. It was a successful experiment.

So when target encoding improves a model, I ask a question that smoothing alone cannot answer:

> Was this row encoded from other observations, or did its own outcome help build the feature used to predict it?

For the 100%-accurate customer IDs, the answer was hidden in plain sight.

Each feature carried its own label.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
