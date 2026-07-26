# My Classifier Was 95% Accurate. It Had Learned One Word: “Clear.”

*A practical datarust guide to imbalanced labels, confusion matrices, Cohen’s kappa, Matthews correlation coefficient, and the model that won on accuracy by refusing to find anything.*

---

The dashboard said `95.0%`.

That number survived the data export, the evaluation script, and a second run from a clean build. There was no rounding mistake. The classifier had predicted 950 of 1,000 payment decisions correctly.

It had also found zero of the 50 payments the model existed to flag.

```text
true review cases found: 0 / 50
```

The model had discovered a remarkably effective rule:

```text
always say “clear”
```

Because 95% of the evaluation rows were genuinely clear, that rule earned 95% accuracy. It also produced an empty review queue, missed every risky payment, and added no value beyond knowing which label was common.

Then I compared it with a less accurate strategy.

The second strategy made 55 mistakes instead of 50, so its accuracy fell to `94.5%`. But it caught 35 of the 50 review cases.

If I sorted only by accuracy, I would deploy the model that found nothing.

That was the moment I stopped treating a correct-prediction percentage as a complete classifier report.

In this article, we will reproduce the comparison with [datarust](https://crates.io/crates/datarust), inspect the confusion matrices, and use two metrics designed to notice structure that accuracy ignores:

- Cohen’s kappa, which discounts agreement expected from the label marginals
- Matthews correlation coefficient, which summarizes all four cells of a binary confusion matrix

Neither metric will choose a production policy for us. They will make it much harder for a constant prediction to look useful.

## One thousand decisions, fifty reasons for the model

Our controlled evaluation set contains:

```text
950 clear payments       class 0
 50 review payments      class 1
```

The positive prevalence is 5%.

I compare two fixed prediction strategies rather than training an estimator. That is deliberate. Training would introduce questions about features, optimization, thresholds, and generalization. Here I want to isolate the evaluation layer: given two prediction vectors, what story does each metric tell?

The first strategy predicts class `0` for every row:

```text
Always clear
[[950, 0],
 [ 50, 0]]
```

In datarust, confusion-matrix rows are true classes and columns are predicted classes. For a binary problem, the layout is:

```text
[[true negatives,  false positives],
 [false negatives, true positives ]]
```

So the first strategy has 950 true negatives and 50 false negatives.

The second strategy sends 75 payments to review:

```text
Review some
[[910, 40],
 [ 15, 35]]
```

It creates 40 unnecessary reviews, misses 15 real cases, and catches 35.

That tradeoff may or may not be acceptable. It depends on the cost of review, the loss from a missed case, queue capacity, and the consequences of delaying a legitimate payment. But it is at least a functioning classifier rather than a restatement of the majority label.

## The complete Rust experiment

Create a small binary and add datarust:

```sh
cargo new metric_audit
cd metric_audit
cargo add datarust
```

Replace `src/main.rs` with this:

```rust
use datarust::metrics::classification::{
    accuracy_score, cohen_kappa_score,
    confusion_matrix, matthews_corrcoef,
};

fn report(
    name: &str,
    truth: &[f64],
    predicted: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let cm = confusion_matrix(truth, predicted)?;
    let tn = cm[0][0] as f64;
    let fp = cm[0][1] as f64;
    let fn_ = cm[1][0] as f64;
    let tp = cm[1][1] as f64;

    let review_precision = if tp + fp == 0.0 {
        0.0
    } else {
        tp / (tp + fp)
    };
    let review_recall = if tp + fn_ == 0.0 {
        0.0
    } else {
        tp / (tp + fn_)
    };

    println!("{name}");
    println!("  confusion:       {cm:?}");
    println!(
        "  accuracy:        {:.3}",
        accuracy_score(truth, predicted)?,
    );
    println!(
        "  review precision: {:.3}",
        review_precision,
    );
    println!("  review recall:    {:.3}", review_recall);
    println!(
        "  kappa:           {:.3}",
        cohen_kappa_score(truth, predicted)?,
    );
    println!(
        "  MCC:             {:.3}",
        matthews_corrcoef(truth, predicted)?,
    );

    // Keep the names visible so the four cells remain concrete.
    println!(
        "  TN={tn:.0} FP={fp:.0} FN={fn_:.0} TP={tp:.0}"
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Ground truth: 950 clear rows followed by 50 review rows.
    let mut truth = vec![0.0; 950];
    truth.extend(vec![1.0; 50]);

    // Strategy A predicts the majority class for every row.
    let always_clear = vec![0.0; 1_000];

    // Strategy B produces this confusion matrix:
    // [[910 true negatives, 40 false positives],
    //  [ 15 false negatives, 35 true positives ]]
    let mut review_some = vec![1.0; 40];
    review_some.extend(vec![0.0; 910]);
    review_some.extend(vec![0.0; 15]);
    review_some.extend(vec![1.0; 35]);

    report("Always clear", &truth, &always_clear)?;
    println!();
    report("Review some", &truth, &review_some)?;
    Ok(())
}
```

Run it:

```sh
cargo run --release
```

The output is:

```text
Always clear
  confusion:       [[950, 0], [50, 0]]
  accuracy:        0.950
  review precision: 0.000
  review recall:    0.000
  kappa:           0.000
  MCC:             0.000
  TN=950 FP=0 FN=50 TP=0

Review some
  confusion:       [[910, 40], [15, 35]]
  accuracy:        0.945
  review precision: 0.467
  review recall:    0.700
  kappa:           0.532
  MCC:             0.544
  TN=910 FP=40 FN=15 TP=35
```

The first strategy wins the accuracy contest.

It loses every contest connected to the reason we built the classifier.

## Accuracy counts agreements, not usefulness

Accuracy asks one clean question:

```text
How many predicted labels equal the true labels?
```

For the always-clear strategy:

```text
correct = 950
total   = 1000

accuracy = 950 / 1000 = 0.95
```

There is nothing mathematically wrong with that calculation. The problem is the question.

Each correct clear decision contributes one point. Missing a review case loses one point. Accuracy does not know that the second event may cost hundreds of times more, trigger a compliance incident, or be the only case anyone cares about detecting.

It also does not ask whether the classifier used any information. A constant vector can score extremely well whenever one class dominates.

If review prevalence falls from 5% to 1%, the same always-clear rule becomes 99% accurate:

```text
990 clear
 10 review

always clear accuracy = 99%
```

The classifier did not improve. The population became easier to ignore.

This is why I put the majority-class baseline beside every accuracy number. If a fitted model scores 95.2% where “always clear” scores 95.0%, the relevant improvement is not 95.2 percentage points. It is two additional correct decisions per thousand, and I still need to learn which decisions they were.

## Kappa asks how much agreement was available for free

Cohen’s kappa starts with observed agreement, which is accuracy:

```text
p_observed = 0.95
```

Then it estimates agreement expected from the true and predicted class proportions:

```text
kappa = (p_observed - p_expected)
        / (1 - p_expected)
```

The always-clear strategy predicts:

```text
100% clear
  0% review
```

The ground truth is:

```text
95% clear
 5% review
```

Given those marginals, expected agreement is already 95%:

```text
p_expected
  = 0.95 × 1.00
  + 0.05 × 0.00
  = 0.95
```

So:

```text
kappa = (0.95 - 0.95) / (1.00 - 0.95)
      = 0.00
```

The strategy achieved no agreement beyond what its class proportions made available automatically.

That is the useful correction. Accuracy sees 950 matches. Kappa sees that a predictor using only one label was already expected to match 950 rows.

For the review-some strategy, the predicted distribution is:

```text
92.5% clear
 7.5% review
```

Expected agreement from the marginals is:

```text
p_expected
  = 0.95 × 0.925
  + 0.05 × 0.075
  = 0.8825
```

Observed agreement is `0.945`, so:

```text
kappa
  = (0.945 - 0.8825) / (1 - 0.8825)
  ≈ 0.532
```

The second strategy earns less raw agreement but substantially more agreement beyond its marginal baseline.

## “Chance corrected” does not mean “production corrected”

Kappa is often described as chance-corrected agreement. That phrase can sound more magical than the calculation is.

The expected term is derived from class marginals. It is not a simulation of how our production model makes random decisions, and it does not contain the cost of a false negative or false positive.

Kappa can also change when prevalence or prediction proportions change, even if some underlying rates remain similar. That is not necessarily a defect. The metric is explicitly sensitive to those marginals.

I use it as a warning and comparison tool:

- Did the predictions agree beyond the dominant-label baseline?
- Did a model’s apparent accuracy come mostly from prevalence?
- Did a new threshold collapse predictions toward one class?

I do not translate `κ = 0.532` into “53.2% business value.” It has no such unit.

## MCC asks whether truth and prediction move together

For binary classification, Matthews correlation coefficient uses every confusion-matrix cell:

```text
MCC = (TP × TN - FP × FN)
      / sqrt(
          (TP + FP)
        × (TP + FN)
        × (TN + FP)
        × (TN + FN)
      )
```

Its range is:

```text
 1   perfect prediction
 0   no correlation
-1   perfectly inverse prediction
```

The numerator rewards the diagonal products and subtracts the off-diagonal products. The denominator includes the true and predicted totals for both classes.

For the review-some strategy:

```text
TP = 35
TN = 910
FP = 40
FN = 15
```

Both diagonals matter. The 910 true negatives cannot hide the 15 false negatives because all four counts participate in the ratio.

The result is:

```text
MCC = 0.544
```

For the always-clear strategy, the predicted-positive total is zero. The usual denominator therefore collapses to zero. datarust follows the conventional practical behavior and returns `0.0`: a constant prediction has no correlation with the varying target.

That is exactly the distinction accuracy missed.

## Kappa and MCC are related, not interchangeable

Both metrics give the constant strategy zero and the useful strategy a value a little above `0.5`. It is tempting to treat them as two spellings of the same idea.

They are not.

Kappa compares observed agreement with agreement expected from the marginals. MCC behaves like a correlation between truth and prediction. When prevalence and prediction bias move, the two metrics can respond differently.

That difference is useful. If accuracy stays stable while kappa and MCC separate, I inspect:

- whether the prediction rate changed,
- whether the true prevalence changed,
- whether one error direction grew,
- and whether the evaluation sample represents production.

I do not choose whichever metric makes the model look best. I use the disagreement as a reason to reopen the confusion matrix.

## The confusion matrix is the report I can operate

Kappa and MCC reveal that the second strategy contains real signal. They still do not tell the review team what Monday morning looks like.

The confusion matrix does:

```text
75 payments enter review
35 are genuine review cases
40 are unnecessary reviews
15 genuine cases are missed
```

From those counts:

```text
review precision = 35 / 75 = 46.7%
review recall    = 35 / 50 = 70.0%
```

Now I can ask operational questions:

- Can the team process 75 reviews per 1,000 payments?
- Is a 53.3% unnecessary-review rate acceptable?
- What is the loss associated with the 15 missed cases?
- Do errors concentrate in one customer or payment segment?
- Would a different threshold create a better capacity tradeoff?

A scalar score is convenient for comparison. Counts are easier to staff, price, and challenge.

## The threshold is part of the model report

Our two prediction vectors look like two classifiers, but a probabilistic model could produce both from the same scores at different thresholds.

A high threshold may push nearly everything into “clear,” raising accuracy while destroying review recall. A lower threshold may catch more cases while creating additional false reviews.

So whenever I report hard-label metrics, I record:

```text
model artifact
score definition
decision threshold
class mapping
evaluation population
```

Without the threshold, a confusion matrix is difficult to reproduce. Without the class mapping, “positive” can become ambiguous. Without the evaluation population, prevalence can make two reports incomparable.

Kappa and MCC evaluate the hard decisions we supplied. They cannot tell whether a nearby threshold would be better.

## A better acceptance check than “accuracy went up”

Before accepting a binary classifier, I now use a short sequence.

First, calculate the trivial baselines:

```text
always class 0
always class 1
current production rule
```

Second, print the confusion counts for the proposed operating threshold.

Third, report at least:

```text
accuracy
positive precision and recall
kappa
MCC
predicted-positive count
```

Fourth, translate the cells into operational volume and cost.

Fifth, repeat the report across important slices. An overall MCC can hide a model that works for one region and fails for another, just as overall accuracy can.

Finally, evaluate on data that represents the deployment boundary. Perfect metrics on rows from customers already present in training may measure recognition rather than generalization. No metric repairs a leaking split.

The purpose is not to create a wall of numbers. It is to prevent one comfortable number from ending the conversation too early.

## The lower score was the better classifier

Our first strategy was 95% accurate because 95% of the rows carried the label it always predicted.

```text
accuracy = 0.950
kappa   = 0.000
MCC     = 0.000
```

It found none of the cases the classifier existed to find.

The second strategy made five more mistakes overall:

```text
accuracy = 0.945
kappa   = 0.532
MCC     = 0.544
```

It also found 70% of the review cases.

That does not automatically make it deployable. Forty false reviews may still be too expensive, and fifteen misses may still be unacceptable. But the decision is now about a real operating tradeoff rather than an accuracy illusion.

The lesson I keep is simple:

> A classifier can be correct most of the time by refusing to classify the cases that matter.

Accuracy told me how often the labels matched.

The confusion matrix told me what the system did.

Kappa and MCC told me the 95% winner had learned nothing beyond one word.

That word was “clear.”

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
