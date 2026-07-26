# My Classifier Was 90% Accurate. It Found One Failure.

*A practical datarust guide to probability thresholds, precision, recall, and choosing a decision rule with validation data instead of habit.*

---

The first number looked great: `90.2%` accuracy.

The second number changed the mood: there were 40 failures in the test set, and the model found one of them.

Technically, nothing was broken. The dataset was imbalanced. Most machines did not fail, so a classifier could be right most of the time by saying “no failure” to almost everyone. The default probability threshold of `0.5` made that conservative behavior look like success.

I've seen accuracy presented this way in churn prediction, fraud detection, incident alerting, and medical screening. The model is praised for being correct on the abundant easy class while quietly missing the rare class that justified building it.

In this article, we'll train one logistic-regression model with [**datarust**](https://crates.io/crates/datarust), then make two different decision systems from its probabilities:

- The default threshold: predict failure when `P(failure) ≥ 0.50`
- A threshold selected on validation data using an explicit error cost

The model weights will not change. The test rows will not change. Only the point where a probability becomes an action will move.

That small distinction is one of the most important ideas in applied classification:

> A probability model estimates risk. A threshold decides what to do about it.

Those are related jobs, not the same job.

## A rare-event experiment

We will generate 2,000 rows with four numeric features. The target is sampled from a logistic probability, so the classes overlap: some high-risk rows do not fail, and some apparently safe rows do. That is closer to real decision-making than a perfectly separable classroom dataset.

The positive rate ends up around 11%.

We split the data three ways:

- **Training set:** fit the scaler and logistic regression
- **Validation set:** choose the probability threshold
- **Test set:** compare the final default and selected decisions once

Create a Rust project:

```sh
cargo new failure_threshold
cd failure_threshold
cargo add datarust
```

Replace `src/main.rs` with this:

```rust
use datarust::linear_model::{LogisticRegression, LogisticSolver};
use datarust::metrics::classification::{
    accuracy_score, confusion_matrix,
};
use datarust::model_selection::TrainTestSplit;
use datarust::pipeline::Pipeline;
use datarust::scaler::StandardScaler;
use datarust::traits::{PredictProba, Predictor};
use datarust::transformer_kind::TransformerKind;
use datarust::Matrix;

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u = self.next_f64().max(f64::MIN_POSITIVE);
        let v = self.next_f64();
        (-2.0 * u.ln()).sqrt()
            * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn make_data(n: usize) -> (Matrix, Vec<f64>) {
    let mut rng = Rng::new(2026);
    let mut rows = Vec::with_capacity(n);
    let mut targets = Vec::with_capacity(n);

    for _ in 0..n {
        let x0 = rng.normal();
        let x1 = rng.normal();
        let x2 = rng.normal();
        let x3 = rng.normal();

        let score =
            -3.0 + 1.5 * x0 - 1.0 * x1 + 0.8 * x2 + 0.2 * x3;
        let probability = 1.0 / (1.0 + (-score).exp());
        let label = if rng.next_f64() < probability {
            1.0
        } else {
            0.0
        };

        rows.push(vec![x0, x1, x2, x3]);
        targets.push(label);
    }

    (Matrix::new(rows).unwrap(), targets)
}

fn labels_at(probabilities: &[f64], threshold: f64) -> Vec<f64> {
    probabilities
        .iter()
        .map(|&p| if p >= threshold { 1.0 } else { 0.0 })
        .collect()
}

// One missed failure costs 10 units; one false alarm costs 1.
fn cost(y_true: &[f64], y_pred: &[f64]) -> datarust::Result<usize> {
    let cm = confusion_matrix(y_true, y_pred)?;
    let false_positives = cm[0][1];
    let false_negatives = cm[1][0];
    Ok(false_positives + 10 * false_negatives)
}

// Derive positive-class metrics explicitly because failure is the class whose
// operating behavior we want to inspect.
fn positive_metrics(
    y_true: &[f64],
    y_pred: &[f64],
) -> datarust::Result<(f64, f64, f64)> {
    let cm = confusion_matrix(y_true, y_pred)?;
    let false_positives = cm[0][1] as f64;
    let false_negatives = cm[1][0] as f64;
    let true_positives = cm[1][1] as f64;

    let precision = if true_positives + false_positives == 0.0 {
        0.0
    } else {
        true_positives / (true_positives + false_positives)
    };
    let recall = if true_positives + false_negatives == 0.0 {
        0.0
    } else {
        true_positives / (true_positives + false_negatives)
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    Ok((precision, recall, f1))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (x, y) = make_data(2_000);
    let positive_rate =
        y.iter().filter(|&&label| label == 1.0).count() as f64
            / y.len() as f64;

    // Hold out 20% for the final test. Split the remaining 80% into
    // 1,200 training rows and 400 validation rows.
    let (x_development, x_test, y_development, y_test) =
        TrainTestSplit::new()
            .with_test_size(0.20)
            .with_random_state(42)
            .split(&x, &y)?;
    let (x_train, x_validation, y_train, y_validation) =
        TrainTestSplit::new()
            .with_test_size(0.25)
            .with_random_state(7)
            .split(&x_development, &y_development)?;

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
    model.fit(&x_train, &y_train)?;

    // Select a threshold using validation probabilities only.
    let validation_proba =
        model.predict_proba(&x_validation)?.col(1);
    let thresholds = [
        0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.40, 0.50, 0.60,
    ];
    let mut best_threshold = 0.5;
    let mut best_cost = usize::MAX;

    println!("Positive rate: {:.1}%", positive_rate * 100.0);
    println!("threshold  precision  recall  F1     cost");

    for threshold in thresholds {
        let predictions = labels_at(&validation_proba, threshold);
        let current_cost = cost(&y_validation, &predictions)?;
        let (precision, recall, f1) =
            positive_metrics(&y_validation, &predictions)?;

        println!(
            "{threshold:>9.2}  {:>9.3}  {:>6.3}  {:>5.3}  {current_cost:>4}",
            precision,
            recall,
            f1,
        );

        if current_cost < best_cost {
            best_cost = current_cost;
            best_threshold = threshold;
        }
    }

    println!(
        "\nSelected threshold: {best_threshold:.2} \
         (validation cost {best_cost})"
    );

    // Touch the held-out test set only after the threshold is fixed.
    let test_proba = model.predict_proba(&x_test)?.col(1);
    println!("\nHeld-out test set");
    println!(
        "threshold  accuracy  precision  recall  F1     cost  \
         confusion [[TN, FP], [FN, TP]]"
    );

    for threshold in [0.50, best_threshold] {
        let predictions = labels_at(&test_proba, threshold);
        let (precision, recall, f1) =
            positive_metrics(&y_test, &predictions)?;

        println!(
            "{threshold:>9.2}  {:>8.3}  {:>9.3}  {:>6.3}  \
             {:>5.3}  {:>4}  {:?}",
            accuracy_score(&y_test, &predictions)?,
            precision,
            recall,
            f1,
            cost(&y_test, &predictions)?,
            confusion_matrix(&y_test, &predictions)?,
        );
    }

    Ok(())
}
```

Run it:

```sh
cargo run --release
```

The validation sweep selects `0.15`:

```text
Positive rate: 11.2%
threshold  precision  recall  F1     cost
     0.05      0.103   1.000  0.187   331
     0.10      0.150   1.000  0.260   216
     0.15      0.265   0.921  0.412   127
     0.20      0.407   0.632  0.495   175
     0.25      0.567   0.447  0.500   223
     0.30      0.615   0.211  0.314   305
     0.40      0.000   0.000  0.000   380
     0.50      0.000   0.000  0.000   380
     0.60      0.000   0.000  0.000   380

Selected threshold: 0.15 (validation cost 127)
```

Then the untouched test set gives us the comparison that matters:

```text
threshold  accuracy  precision  recall  F1     cost  confusion [[TN, FP], [FN, TP]]
     0.50     0.902      1.000   0.025  0.049   390  [[360, 0], [39, 1]]
     0.15     0.743      0.263   0.875  0.405   148  [[262, 98], [5, 35]]
```

The selected threshold lowers accuracy from `90.2%` to `74.3%`.

It also catches 35 failures instead of one.

## Accuracy answered the easiest question

The test set contains 360 negative rows and 40 positive rows. A classifier that predicts “no failure” 400 times would be 90% accurate without identifying a single failure.

Our default threshold is only slightly less passive. It produces:

```text
TN = 360   FP = 0
FN = 39    TP = 1
```

Accuracy combines those four numbers into one fraction:

```text
(TN + TP) / total = (360 + 1) / 400 = 90.25%
```

Most of that success comes from correctly handling the abundant negative class. If “find impending failures” is the product requirement, accuracy is reporting excellent performance on the supporting cast.

This does not make accuracy useless. It makes it incomplete. A metric is only informative relative to the error distribution you care about.

## The model returned probabilities before we forced a decision

`LogisticRegression::predict` returns hard labels using its built-in decision rule. For threshold tuning, we ask for probabilities instead:

```rust
let probabilities = model.predict_proba(&x_validation)?.col(1);
```

The matrix has one probability column per class. Column `1` contains `P(y = 1)` for the failure class.

We create labels ourselves:

```rust
fn labels_at(probabilities: &[f64], threshold: f64) -> Vec<f64> {
    probabilities
        .iter()
        .map(|&p| if p >= threshold { 1.0 } else { 0.0 })
        .collect()
}
```

At `0.50`, a row with predicted failure probability `0.49` is negative. At `0.15`, the same row is positive. The underlying risk score did not move; the action boundary did.

That is why threshold tuning can be performed without retraining. We are not changing what the model believes. We are changing how cautious the surrounding system is.

## Precision and recall pull in different directions

For the positive class:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
```

Precision asks: of the rows we flagged, how many actually failed?

Recall asks: of the rows that failed, how many did we flag?

Lowering the threshold usually increases recall because more rows qualify as positive. It usually decreases precision because some of those additional alerts are false positives.

Our test results show the exchange clearly:

| Threshold | Precision | Recall | Meaning |
|---:|---:|---:|---|
| 0.50 | 1.000 | 0.025 | One alert, correct — 39 failures missed |
| 0.15 | 0.263 | 0.875 | 133 alerts — 35 of 40 failures caught |

A precision of `1.000` looks perfect until you notice the system achieved it by almost never speaking.

At `0.15`, roughly one in four alerts corresponds to a real failure. Whether that is acceptable depends on what an alert triggers. A cheap automated check may tolerate it. An expensive shutdown or invasive intervention may not.

Metrics cannot decide that tradeoff without context.

## F1 is a compromise, not a business model

F1 is the harmonic mean of precision and recall:

```text
F1 = 2 × precision × recall / (precision + recall)
```

It punishes a classifier when either side is very low. The default threshold's positive-class F1 is only `0.049`; the selected threshold reaches `0.405`.

But our validation sweep did not choose the maximum F1. Threshold `0.25` has validation F1 `0.500`, higher than `0.412` at `0.15`. We still select `0.15` because the stated error costs prefer catching more failures.

F1 gives precision and recall a particular symmetric relationship. Your application may not.

If a false negative is ten times more expensive than a false positive, turning both errors into one explicit objective is more honest than hoping a generic metric represents the economics.

## A tiny cost matrix made the decision explicit

For this experiment:

```text
false positive cost = 1
false negative cost = 10
```

The total is:

```rust
cost = false_positives + 10 * false_negatives;
```

At the default threshold, the test set has zero false positives and 39 false negatives:

```text
0 × 1 + 39 × 10 = 390
```

At `0.15`, it has 98 false positives and five false negatives:

```text
98 × 1 + 5 × 10 = 148
```

The selected decision rule reduces defined test cost by about 62%.

Those cost values are illustrative, not discovered by the model. In a real system, they might represent technician time, customer friction, chargeback loss, missed revenue, safety exposure, or a carefully designed utility score. Some consequences cannot responsibly be collapsed into money at all.

What matters is that the assumption is visible. “Use 0.5 because that is the default” is also a cost decision; it is simply an undocumented one.

## The threshold belongs to validation, not the test set

It would be easy to try every threshold on `x_test`, choose the cheapest, and report that same cost as final performance.

That would be optimistic. The test labels would have helped select the decision rule, so the test set would no longer be independent.

Our workflow keeps the roles separate:

```text
training rows   → fit scaler + logistic regression
validation rows → choose threshold 0.15
test rows       → evaluate threshold 0.15 once
```

The `0.15` threshold wins on validation with cost `127`. It then produces test cost `148`. The numbers differ, as honest estimates usually do.

If you search hundreds of thresholds on a small validation set, that choice can overfit too. Use more validation data, cross-validated out-of-fold probabilities, or a coarser decision grid when sample size is limited.

The threshold is a learned configuration even though it is not a model coefficient.

## A good threshold still needs useful probabilities

Threshold tuning cannot rescue a model that ranks positives randomly. Moving the boundary only changes how many rows are selected from that ranking.

Threshold-independent metrics such as ROC-AUC or average precision help evaluate ranking quality across operating points. They still do not choose the operating point for you.

Calibration matters too. A model that assigns `0.20` should ideally be correct about 20% of the time among similar predictions. Our synthetic labels come from a logistic process and the model family matches that process unusually well. Real data may produce overconfident or underconfident probabilities.

If the numeric probability is shown to users, combined with external costs, or compared across time, evaluate calibration explicitly. A threshold can still work on an imperfectly calibrated ranking, but the number `0.15` should not be described as a literal 15% risk without evidence.

## The operating point will move

A production threshold is not permanent.

The best decision boundary can change when:

- Failure prevalence rises or falls
- Sensor distributions drift
- The cost of an inspection changes
- Alert capacity becomes constrained
- The intervention becomes more or less effective
- The model is retrained or recalibrated

Imagine the maintenance team can investigate only 20 machines per day. A fixed threshold may generate 10 alerts in one season and 80 in another. A capacity-aware policy might use a ranked queue, a dynamic cutoff, or separate thresholds by operational context.

Monitor more than accuracy. Track alert volume, positive rate, precision with label delay, recall where measurable, probability distribution, and the confusion matrix over time.

And version the threshold beside the model. Deploying `model.json` without the decision configuration is like shipping a thermometer without saying which temperature triggers the alarm.

## What changed — and what did not

The final comparison is intentionally uncomfortable:

| Threshold | Accuracy | Positive precision | Positive recall | Cost |
|---:|---:|---:|---:|---:|
| 0.50 | 0.902 | 1.000 | 0.025 | 390 |
| 0.15 | 0.743 | 0.263 | 0.875 | 148 |

The lower-accuracy system is much better for the objective we wrote down.

We did not:

- Add more training data
- Change logistic-regression coefficients
- Resample the classes
- Introduce class weights
- Build a more complicated model

We used the probabilities the model already produced and stopped treating `0.5` as a law of nature.

Threshold tuning is not a substitute for better features, good validation, calibrated probabilities, or a model that ranks risk well. It is the final layer that connects all of them to an action.

That layer deserves the same care as training.

Because the model can be statistically correct and operationally useless at the same time.

```sh
cargo add datarust
```

---

*datarust is MIT-licensed and available on [crates.io](https://crates.io/crates/datarust). Documentation lives at [genc-murat.github.io/datarust](https://genc-murat.github.io/datarust/), including the [classification metrics guide](https://genc-murat.github.io/datarust/guide/metrics.html).*
