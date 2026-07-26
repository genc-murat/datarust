# My Test Set Had 100 Rows. It Had Zero New Customers.

*A practical datarust guide to entity overlap, repeated observations, group-aware holdouts, and the row split that measured recognition instead of generalization.*

---

The test accuracy was perfect.

```text
accuracy: 1.000
```

The model received 100 test rows. At first glance, that sounded like 100 opportunities to prove itself on unseen data.

Those rows belonged to 71 customers.

All 71 customers also appeared in training.

The model had not learned a pattern that generalized to new customers. It had learned a one-hot coefficient for each known customer identity, then recognized more rows from those same identities.

I rebuilt the test set with 20 entirely held-out customers while keeping its size at 100 rows. Accuracy fell to `0.500`.

Nothing about the estimator changed. The evaluation question changed from:

> Can the model score another row from a customer it has already seen?

to:

> Can the model score a customer it has never seen?

Those are both legitimate tasks. They are not interchangeable.

Let's make the entity overlap visible with [datarust](https://crates.io/crates/datarust), a row-level `TrainTestSplit`, and a manually constructed customer-group holdout.

## One hundred customers, five rows each

The synthetic dataset contains:

```text
100 customers
5 observations per customer
500 rows total
```

Each customer has a stable binary label. Even-numbered customers belong to class zero; odd-numbered customers belong to class one. There is no transferable customer feature—only `customer_id`.

The model pipeline is intentionally capable of memorization:

1. Fit a one-hot encoder on training customer IDs.
2. Fit Ridge regression on those indicator columns.
3. Treat scores at or above `0.5` as class one.

Ridge is used here as a transparent scoring model, not as a calibrated probability classifier. With repeated, stable labels, a known customer's one-hot coefficient can push its score close to zero or one. An unknown customer becomes an all-zero encoded row and receives only the fitted intercept.

I compare two boundaries with identical row counts:

```text
random row split:       400 train, 100 test
customer-group holdout: 400 train, 100 test
```

The first uses datarust's `TrainTestSplit`. The second assigns 80 whole customers to training and 20 whole customers to test.

Here is the complete Rust program:

```rust
use std::collections::BTreeSet;

use datarust::encoder::{
    HandleUnknown, OneHotEncoder,
};
use datarust::linear_model::Ridge;
use datarust::metrics::classification::accuracy_score;
use datarust::model_selection::TrainTestSplit;
use datarust::traits::Predictor;
use datarust::{Matrix, StrMatrix};

fn customer_id(row_index: usize) -> String {
    format!("customer_{:03}", row_index / 5)
}

fn ids_from_index_matrix(indices: &Matrix) -> StrMatrix {
    let ids: Vec<String> = (0..indices.nrows())
        .map(|i| {
            customer_id(indices.get(i, 0) as usize)
        })
        .collect();
    StrMatrix::from_column(ids).unwrap()
}

fn unique_ids(ids: &StrMatrix) -> BTreeSet<String> {
    (0..ids.nrows())
        .map(|i| ids.get(i, 0).to_string())
        .collect()
}

fn hard_labels(scores: &[f64]) -> Vec<f64> {
    scores
        .iter()
        .map(|&score| {
            if score >= 0.5 { 1.0 } else { 0.0 }
        })
        .collect()
}

fn fit_and_score(
    name: &str,
    train_ids: &StrMatrix,
    test_ids: &StrMatrix,
    y_train: &[f64],
    y_test: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let train_customers = unique_ids(train_ids);
    let test_customers = unique_ids(test_ids);
    let overlap = train_customers
        .intersection(&test_customers)
        .count();
    let seen_test_rows = (0..test_ids.nrows())
        .filter(|&i| {
            train_customers.contains(test_ids.get(i, 0))
        })
        .count();

    let mut encoder = OneHotEncoder::new()
        .handle_unknown(HandleUnknown::Ignore);
    let x_train = encoder.fit_transform(train_ids)?;
    let x_test = encoder.transform(test_ids)?;

    let mut model = Ridge::new().with_alpha(0.01);
    model.fit(&x_train, y_train)?;
    let scores = model.predict(&x_test)?;
    let predictions = hard_labels(&scores);

    println!("{name}");
    println!(
        "rows train/test: {}/{} | \
         customers train/test: {}/{} | \
         overlapping customers: {}",
        train_ids.nrows(),
        test_ids.nrows(),
        train_customers.len(),
        test_customers.len(),
        overlap,
    );
    println!(
        "seen test rows: {}/{} ({:.1}%) | \
         encoded columns: {}",
        seen_test_rows,
        test_ids.nrows(),
        100.0 * seen_test_rows as f64
            / test_ids.nrows() as f64,
        x_train.ncols(),
    );
    println!(
        "test accuracy: {:.3} | \
         score range: [{:.3}, {:.3}]\n",
        accuracy_score(y_test, &predictions)?,
        scores
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
        scores
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let row_indices = Matrix::new(
        (0..500).map(|i| vec![i as f64]).collect(),
    )?;
    let labels: Vec<f64> = (0..500)
        .map(|row| ((row / 5) % 2) as f64)
        .collect();

    let (
        row_train_idx,
        row_test_idx,
        row_y_train,
        row_y_test,
    ) = TrainTestSplit::new()
        .with_test_size(0.20)
        .with_random_state(42)
        .split(&row_indices, &labels)?;
    let row_train_ids =
        ids_from_index_matrix(&row_train_idx);
    let row_test_ids =
        ids_from_index_matrix(&row_test_idx);

    fit_and_score(
        "random row split",
        &row_train_ids,
        &row_test_ids,
        &row_y_train,
        &row_y_test,
    )?;

    let mut group_train_ids = Vec::new();
    let mut group_test_ids = Vec::new();
    let mut group_y_train = Vec::new();
    let mut group_y_test = Vec::new();

    for row in 0..500 {
        let customer = row / 5;
        let id = customer_id(row);
        let label = (customer % 2) as f64;

        if customer < 80 {
            group_train_ids.push(id);
            group_y_train.push(label);
        } else {
            group_test_ids.push(id);
            group_y_test.push(label);
        }
    }

    fit_and_score(
        "customer-group holdout",
        &StrMatrix::from_column(group_train_ids)?,
        &StrMatrix::from_column(group_test_ids)?,
        &group_y_train,
        &group_y_test,
    )?;

    Ok(())
}
```

This is the output I measured:

```text
random row split
rows train/test: 400/100 | customers train/test: 100/71 | overlapping customers: 71
seen test rows: 100/100 (100.0%) | encoded columns: 100
test accuracy: 1.000 | score range: [0.001, 0.999]

customer-group holdout
rows train/test: 400/100 | customers train/test: 80/20 | overlapping customers: 0
seen test rows: 0/100 (0.0%) | encoded columns: 80
test accuracy: 0.500 | score range: [0.500, 0.500]
```

The row count stayed constant. The meaning of “unseen” did not.

## TrainTestSplit honored the row-level request

`TrainTestSplit` receives a `Matrix` and target vector. It shuffles row indices, assigns the requested fraction to test, and returns the corresponding rows.

It does not know that five indices belong to one customer. We never supplied that relationship to the splitter.

With five rows per customer and an 80/20 row split, a test observation is very likely to have at least one sibling observation in training. In this particular seeded split:

```text
test rows:             100
unique test customers:  71
test customers in train: 71
seen test rows:        100
```

The splitter did not leak data behind our back. It correctly sampled rows from the population we defined. The modeling error was defining each row as independent when the deployment question concerned customers.

A library cannot infer entity boundaries from repeated numeric patterns safely. Group identity belongs in the experiment design.

## One-hot encoding turned overlap into memory

The row-split encoder fits on IDs from all 100 customers. Its output has 100 columns—one per learned identity.

For customer 37, every row activates the same indicator column:

```text
customer_037 -> [0, 0, ..., 1, ..., 0]
```

All five rows also share the same label. If four land in training and one in test, the model can learn customer 37's outcome from the four training rows and reproduce it for the fifth.

The score range shows near-perfect memorization:

```text
[0.001, 0.999]
```

This is not evidence that Ridge discovered customer behavior. Identity itself was enough.

The same effect can arise without an explicit ID feature. Highly specific address, device fingerprint, account configuration, image background, or repeated measurement signature can act as a proxy for entity identity.

Removing the ID column does not prove the split is group-safe.

## The group holdout removed every identity column at test time

The customer-level split assigns customers 0 through 79 to training and 80 through 99 to test. Both sides remain balanced, and each still contains five rows per customer.

The training encoder learns 80 categories. Every test customer is unknown.

We configured:

```rust
.handle_unknown(HandleUnknown::Ignore)
```

For one-hot encoding without a dropped category, datarust represents an unknown ID as an all-zero row. Ridge therefore receives no active customer coefficient and returns its balanced training intercept:

```text
score range: [0.500, 0.500]
```

There are no other features, so chance-level accuracy is the correct result.

`HandleUnknown::Ignore` prevented a runtime error. It did not create information for a new entity. An all-zero encoding policy keeps the serving schema stable; it cannot make identity generalize.

## The perfect score answered a legitimate different question

Suppose the production system predicts the next support escalation for customers with established history. Known-customer identity or prior customer behavior may be permitted and useful. A grouped holdout containing only new customers could then be too pessimistic for the main traffic mix.

But a random row split can still be too optimistic if it allows information from later observations to help earlier ones. For known-customer prediction, I usually need a temporal boundary:

```text
past rows for a customer -> training/history
future row for that customer -> validation
```

The correct evaluation may require multiple cohorts:

- Future events for known customers
- First events for new customers
- Returning customers after a long gap
- New regions, devices, or organizations

Each cohort answers a different product question. I report them separately instead of averaging them into one comforting number.

The problem is not entity overlap by definition. It is unmeasured mismatch between evaluation overlap and deployment overlap.

## Repeated labels are the sharpest version

Our example copies one stable customer-level label onto five rows. That makes the leak dramatic.

This pattern appears in real data:

- A patient diagnosis copied onto multiple scans
- A household outcome attached to every household member
- A machine failure label attached to overlapping sensor windows
- A fraudulent account label copied to all transactions
- A product defect label attached to multiple images of the same item

If train and test contain different rows from the same entity, the model may recognize entity-specific details and reproduce the shared label.

When row-level outcomes genuinely vary within an entity, overlap can still inflate performance by exposing stable entity propensity. The effect is less absolute but follows the same mechanism.

I decide the split at the level where labels and nuisance signatures are correlated, not merely at the table's physical row boundary.

## Duplicate detection is not enough

Exact duplicate rows are easy to search for. Group leakage often survives deduplication.

Five visits from one customer can have different timestamps, transaction amounts, text, and usage values while sharing a stable identity and outcome tendency. Two medical images from one patient are not byte-identical. Overlapping time windows from one machine contain different values.

The relevant question is:

> Could these rows share information that will not be available for a genuinely new test entity?

Entity keys, parent relationships, session IDs, device IDs, source files, geographic units, and temporal windows may all define groups.

Sometimes groups are nested: images within visits, visits within patients, patients within hospitals. The deployment boundary determines which level must remain isolated.

## Preprocessing must stay inside the group boundary

The example fits `OneHotEncoder` separately after each split. That matters.

Fitting the encoder on all IDs before splitting would leak the full category vocabulary into training preprocessing. It would also produce columns for held-out customers even though their training columns contained only zeros.

More dangerous supervised operations—target encoding, feature selection, calibration—can leak outcomes directly if fitted outside the grouped folds.

The correct order is:

```text
choose train/test groups
    -> fit encoder on training groups
    -> fit scaler/imputer/selector on training groups
    -> fit model
    -> transform held-out groups with frozen state
    -> evaluate
```

For cross-validation, every fold needs its own preprocessing fit. A globally fitted pipeline followed by group-aware scoring is still globally informed.

## Group splitting needs label balance too

Moving whole entities can create difficult class distributions. If all positive examples belong to a few groups, a fold may contain one class or no positive training example.

Our manual holdout deliberately selects ten even and ten odd customers, so both train and test are balanced. Real grouping requires more care:

- count labels by group before splitting,
- inspect group sizes,
- preserve chronology where required,
- avoid allowing one large group to dominate a fold,
- report per-fold class counts,
- reduce fold count if the number of positive groups is small.

Row-level stratification cannot repair entity leakage because it may deliberately distribute one entity's rows across folds to improve class proportions.

Grouping, stratification, and time are constraints that sometimes conflict. I prioritize the deployment boundary and accept wider uncertainty rather than violate it for prettier fold balance.

## datarust currently provides the row primitive

The current codebase provides `TrainTestSplit`, `KFold`, and `StratifiedKFold`, but not a dedicated group-aware splitter.

The example therefore partitions IDs before model fitting. In a real application I create group-index lists from the source table, choose train/test group sets, and use those indices consistently across numeric features, categorical features, labels, and metadata.

That manual code deserves tests:

```text
intersection(train_groups, test_groups) is empty
every row appears exactly once
row and label counts match
class counts are reported
time constraints are respected
```

I would rather own a small explicit grouping function than force a row splitter to answer a question it was not designed to receive.

## What I print next to every score

A test metric without split diagnostics is hard to trust.

For grouped data, I record:

- train and test row counts,
- train and test entity counts,
- entity intersection count,
- percentage of test rows from seen entities,
- label distribution by entity and by row,
- minimum, median, and maximum rows per entity,
- time range on each side,
- unknown-category rate after preprocessing.

The most important line in our random split was not accuracy:

```text
seen test rows: 100/100 (100.0%)
```

That line told us what the score measured.

## The unit of generalization was missing from the API call

Both experiments returned 100 test rows.

The random split returned rows from 71 customers the model already knew and scored 100%.

The group holdout returned rows from 20 new customers and scored 50%.

Neither number is universally correct. The first estimates recognition for known identities under a row-random boundary. The second estimates generalization to new identities in a signal-free setup.

The dangerous step is naming either one “test accuracy” without saying which population it represents.

So before splitting a repeated-observation table, I now ask:

> What is the smallest unit that must be genuinely unseen when this model is deployed?

Sometimes it is a row. Often it is a customer, patient, device, household, store, document, or point in time.

Our original test set contained 100 unseen rows.

It contained zero unseen customers.

That was enough to turn chance performance into perfection.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
