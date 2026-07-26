# ROC-AUC Stayed the Same. Most of My Alerts Became Wrong.

*A practical datarust experiment with class prevalence, average precision, false-positive rates, and the ranking metric that did exactly what it promised.*

---

The model's ROC-AUC had not moved.

```text
before: 0.7945
after:  0.7945
```

The ranking quality looked stable. The deployment dashboard was less comforting.

At the same score threshold, alert precision had fallen from 70.9% to 11.4%. Almost nine out of ten alerts were false alarms.

Nothing is contradictory about those numbers.

ROC-AUC measures how well scores rank positives above negatives. Alert precision asks how many positive predictions are actually positive. The first can remain unchanged when class prevalence moves; the second cannot.

This distinction matters in fraud detection, incident alerting, medical screening, content moderation, churn prevention, and almost every system where the event we care about is rare. A model can have respectable discrimination and still create an operational queue dominated by negatives.

Let's isolate the effect with [datarust](https://crates.io/crates/datarust).

## The same scores in two different worlds

I start with 100 positive scores and 100 negative scores. They overlap, but positives tend to rank higher.

That produces a balanced evaluation cohort:

```text
100 positives
100 negatives
50% prevalence
```

Then I construct a rare-event cohort using the exact same 100 positive scores while repeating each negative score 19 times:

```text
 100 positives
1900 negatives
5% prevalence
```

Duplicating negatives is not intended to imitate independent production observations. It is a controlled device: every positive-versus-negative score comparison is repeated equally, so the ranking relationship stays exactly the same while prevalence changes.

I calculate ROC-AUC, datarust's `average_precision_score`, and threshold metrics at five operating points.

Here is the complete Rust program:

```rust
use datarust::metrics::classification::{
    average_precision_score, roc_auc_score,
};

fn decisions(scores: &[f64], threshold: f64) -> Vec<f64> {
    scores
        .iter()
        .map(|&score| if score >= threshold { 1.0 } else { 0.0 })
        .collect()
}

fn operating_metrics(
    y: &[f64],
    pred: &[f64],
) -> (f64, f64, f64, f64) {
    let mut tp = 0_usize;
    let mut fp = 0_usize;
    let mut tn = 0_usize;
    let mut fn_ = 0_usize;

    for (&actual, &predicted) in y.iter().zip(pred) {
        match (actual == 1.0, predicted == 1.0) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (false, false) => tn += 1,
            (true, false) => fn_ += 1,
        }
    }

    let precision = tp as f64 / (tp + fp) as f64;
    let recall = tp as f64 / (tp + fn_) as f64;
    let false_positive_rate =
        fp as f64 / (fp + tn) as f64;
    let accuracy = (tp + tn) as f64 / y.len() as f64;

    (precision, recall, false_positive_rate, accuracy)
}

fn report(
    name: &str,
    y: &[f64],
    scores: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let positives =
        y.iter().filter(|&&label| label == 1.0).count();
    let prevalence = positives as f64 / y.len() as f64;
    let auc = roc_auc_score(y, scores)?;
    let ap = average_precision_score(y, scores)?;

    println!(
        "{name}: {} rows, {positives} positives ({:.1}%)",
        y.len(),
        prevalence * 100.0,
    );
    println!("ROC-AUC = {auc:.4}");
    println!(
        "Average precision = {ap:.4} ({:.2}x prevalence)\n",
        ap / prevalence,
    );
    println!(
        "threshold  precision  recall   FPR    accuracy  alerts"
    );

    for threshold in [0.40, 0.50, 0.60, 0.65, 0.70] {
        let pred = decisions(scores, threshold);
        let alerts =
            pred.iter().filter(|&&label| label == 1.0).count();
        let (precision, recall, fpr, accuracy) =
            operating_metrics(y, &pred);
        println!(
            "{threshold:>9.2}  {precision:>9.3}  \
             {recall:>6.3}  {fpr:>5.3}  \
             {accuracy:>8.3}  {alerts:>6}",
        );
    }
    println!();

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let positive_scores: Vec<f64> = (0..100)
        .map(|i| 0.25 + i as f64 * 0.0065)
        .collect();
    let negative_scores: Vec<f64> = (0..100)
        .map(|i| 0.05 + i as f64 * 0.0060)
        .collect();

    let mut balanced_y = vec![1.0; positive_scores.len()];
    balanced_y.extend(vec![0.0; negative_scores.len()]);
    let mut balanced_scores = positive_scores.clone();
    balanced_scores.extend(negative_scores.iter().copied());

    let mut rare_y = vec![1.0; positive_scores.len()];
    let mut rare_scores = positive_scores;
    for score in negative_scores {
        for _ in 0..19 {
            rare_y.push(0.0);
            rare_scores.push(score);
        }
    }

    report(
        "Balanced cohort",
        &balanced_y,
        &balanced_scores,
    )?;
    report("Rare-event cohort", &rare_y, &rare_scores)?;

    Ok(())
}
```

This is the output I measured:

```text
Balanced cohort: 200 rows, 100 positives (50.0%)
ROC-AUC = 0.7945
Average precision = 0.8215 (1.64x prevalence)

threshold  precision  recall   FPR    accuracy  alerts
     0.40      0.650   0.760  0.410     0.675     117
     0.50      0.709   0.610  0.250     0.680      86
     0.60      0.852   0.460  0.080     0.690      54
     0.65      1.000   0.380  0.000     0.690      38
     0.70      1.000   0.300  0.000     0.650      30

Rare-event cohort: 2000 rows, 100 positives (5.0%)
ROC-AUC = 0.7945
Average precision = 0.4753 (9.51x prevalence)

threshold  precision  recall   FPR    accuracy  alerts
     0.40      0.089   0.760  0.410     0.599     855
     0.50      0.114   0.610  0.250     0.743     536
     0.60      0.232   0.460  0.080     0.897     198
     0.65      1.000   0.380  0.000     0.969      38
     0.70      1.000   0.300  0.000     0.965      30
```

The score ordering stayed fixed. The operational meaning did not.

## What ROC-AUC actually held constant

ROC-AUC has a useful probabilistic interpretation:

> Pick one random positive and one random negative. ROC-AUC is the probability that the positive receives the higher score, with half credit for a tie.

In this experiment, that probability is `0.7945`.

Repeating every negative 19 times does not change the fraction of positive-negative pairs that are correctly ordered. Each original comparison is simply counted 19 times. That is why datarust's rank-based `roc_auc_score` returns exactly the same value in both cohorts.

ROC coordinates have the same prevalence-insensitive structure. At threshold `0.50`:

```text
recall / true-positive rate = 0.610 in both cohorts
false-positive rate         = 0.250 in both cohorts
```

Recall divides by the number of positives. Repeating negatives changes neither the positive scores nor that denominator.

False-positive rate divides by all negatives. Repeating every negative repeats both false positives and true negatives proportionally, so the rate remains 25%.

ROC-AUC did not miss the prevalence shift. Prevalence is simply not part of the question it answers.

## Precision counted people, not rates within each class

Precision asks:

```text
true positives / all positive predictions
```

At threshold `0.50`, the balanced cohort produces:

```text
true positives:  61
false positives: 25
precision:        61 / 86 = 70.9%
```

The rare-event cohort has the same 61 true positives. But each negative pattern appears 19 times, so the 25 false positives become 475:

```text
true positives:   61
false positives: 475
precision:         61 / 536 = 11.4%
```

The false-positive *rate* is still 25%. There are simply far more negatives available to turn that rate into false alerts.

This is the base-rate effect in operational form. A modest false-positive rate applied to a large negative population can overwhelm the true positives from a rare class.

## Accuracy improved while the alert queue got worse

The same threshold produces another apparently reassuring movement:

```text
balanced accuracy: 68.0%
rare-event accuracy: 74.3%
```

The model did not improve. The dataset gained many negatives, and even with a 25% false-positive rate it still labels most of them correctly. Those abundant true negatives raise accuracy.

Meanwhile, alert precision collapses from 70.9% to 11.4%.

If the product is an investigation queue, the second number describes the reviewers' experience. Accuracy rewards the system for correctly leaving many routine cases alone, which may be valuable, but it does not tell me whether the alerts are usable.

This is why I do not select a rare-event classifier from one aggregate metric.

## Average precision moved with prevalence

`average_precision_score` summarizes precision as recall increases down the ranked list. In datarust, it uses the standard step-function calculation: each time another positive is encountered, that precision contributes according to the increase in recall.

Unlike ROC-AUC, the precision values inside that calculation depend on the number of negatives mixed into the ranking.

The result falls from:

```text
0.8215 at 50% prevalence
```

to:

```text
0.4753 at 5% prevalence
```

That does not mean the underlying score ordering became worse — we constructed the experiment so it could not. It means the precision-recall view reports the practical competition between positives and negatives at this prevalence.

The baseline matters. A random ranking has expected average precision near the positive prevalence:

```text
balanced random baseline: 0.50
rare-event random baseline: 0.05
```

So `0.4753` in the rare cohort is still about 9.51 times its prevalence baseline. Absolute average precision fell, yet the model remains substantially more useful than random ranking.

I report both AP and prevalence. An AP number without its base rate is missing context.

## Average precision still does not choose a threshold

ROC-AUC and average precision use the whole score ranking. Neither says which cases the product should act on.

Our table shows the choices directly. In the rare cohort:

```text
threshold 0.50 -> 536 alerts, 11.4% precision, 61% recall
threshold 0.60 -> 198 alerts, 23.2% precision, 46% recall
threshold 0.65 ->  38 alerts, 100% precision, 38% recall
```

The perfect precision at `0.65` is a property of this controlled score construction: no negative score exceeds `0.644`. A real test set with finite negatives cannot prove that production false positives will remain zero.

Still, the tradeoff is real. A fraud team with capacity for 50 investigations wants a different operating point from an automated account-blocking system where false positives are extremely costly.

I choose the threshold on validation data that resembles deployment prevalence, using explicit costs, capacity, precision constraints, or recall requirements. The final test set then evaluates that locked decision rule once.

## A balanced test sample can answer the wrong production question

Balanced evaluation sets are convenient. They provide many positive examples, make class-specific error analysis easier, and reduce uncertainty for a fixed sample size.

They can also make observed precision look nothing like production.

If I intentionally sample equal numbers of fraud and legitimate transactions, ROC-AUC can still estimate pairwise ranking performance under the case-control sample. But the sample's 50% precision is not the precision users will see when fraud prevalence is 0.5%.

To estimate deployment alert volume and positive predictive value, I need either:

- An evaluation set with representative natural prevalence, or
- Correct statistical reweighting using a trustworthy deployment base rate.

The same warning applies to undersampling negatives during training. Resampling can help optimization, but raw probabilities and threshold metrics may require calibration or correction before deployment.

The test-set composition is part of the metric definition, not administrative metadata.

## ROC-AUC says nothing about probability calibration

Both datarust metrics accept any monotonic score. If I replace every score `s` with `s³`, the ordering remains the same, so ROC-AUC and average precision remain the same.

But `0.7³ = 0.343`, which is a very different probability claim.

AUC metrics measure ranking. They do not tell me whether cases scored `0.7` fail about 70% of the time. If the application consumes probabilities for expected cost, forecasting, or resource allocation, I also inspect calibration and a proper probability loss such as `log_loss`.

Good ranking, good calibration, and a useful operating threshold are three related but separate properties.

## Real prevalence shifts are usually less polite

This experiment changes only the number of negatives. The positive and negative score distributions remain identical by construction.

In production, prevalence often changes because the population changes:

- A new fraud campaign produces unfamiliar positive patterns.
- A marketing channel brings a different customer mix.
- A sensor firmware update shifts negative scores.
- Policy changes alter which cases receive labels.

Then ROC-AUC may move too. Stable AUC does not rule out drift, and changed prevalence does not explain every metric movement.

I monitor at least prevalence, score distributions by class when labels arrive, ROC-AUC, average precision, threshold confusion counts, alert volume, and delayed-label coverage. Counts matter because a stable rate applied to a growing population can still exceed operational capacity.

## What I would put on the model report

For a rare-event classifier, my compact report includes:

```text
positive prevalence
number of labeled positives and negatives
ROC-AUC with uncertainty
average precision and prevalence baseline
precision, recall, FPR, and alert count at the chosen threshold
the threshold-selection rule
the evaluation sampling scheme
```

Confidence intervals matter. This example has only 100 unique positive scores. A four-decimal metric can look more certain than the underlying evidence deserves.

I also keep the raw confusion counts beside ratios. “25% false-positive rate” and “475 false alerts” describe the same operating point from very different human perspectives.

## The metric did not fail

ROC-AUC stayed at `0.7945` because the ranking relationship stayed exactly the same. It answered its question correctly.

Average precision fell because positives had to compete with nineteen times as many negative rows in the ranked list. Threshold precision fell even more visibly because false-positive counts grew while true-positive counts stayed fixed. Accuracy rose because true negatives became abundant.

Every number was correct.

The mistake would be asking one of them to describe the whole system.

The lesson I keep is this:

> Ranking quality travels more easily across prevalence than alert quality does.

Before trusting an AUC headline, I now ask how many real positives exist, how many alerts the threshold creates, and what fraction of those alerts a person will discover were worth their time.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
