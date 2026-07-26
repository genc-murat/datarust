# Both Models Were 80% Accurate. One Was Confidently Wrong.

*A practical datarust guide to probability calibration, log-loss, Brier score, ECE, and the predictions whose class labels stayed right while their meaning fell apart.*

---

Both probability models made exactly the same class decisions.

```text
accuracy: 0.800
ROC-AUC:  0.850
```

One said a high-risk group had probability `0.9000`. Ninety of its 100 observations were positive.

The other said the same group had probability `0.9986`. It ranked every observation in the same place and still predicted the positive class. Ten of those near-certain predictions were wrong.

Accuracy did not move. ROC-AUC did not move. Log-loss jumped from `0.468` to `0.749`.

The second model was not worse at choosing a side of `0.5`. It was worse at saying what its number meant.

This distinction matters whenever a probability becomes more than an intermediate score. If `0.9` drives expected cost, staffing, pricing, manual review priority, or a statement shown to a user, then “usually on the correct side” is not enough. Cases assigned 90% should occur about 90% of the time among comparable predictions.

Let's hold ranking and hard decisions constant, change only confidence, and measure the result with [datarust](https://crates.io/crates/datarust).

## Four risk groups with known outcomes

The controlled dataset has 400 observations divided into four equal groups.

The calibrated probabilities and outcomes are:

```text
predicted 0.1 -> 10 positives out of 100
predicted 0.3 -> 30 positives out of 100
predicted 0.7 -> 70 positives out of 100
predicted 0.9 -> 90 positives out of 100
```

The predictions are perfectly calibrated at this group resolution.

I create a second probability vector by multiplying every logit by three:

```text
overconfident_probability = sigmoid(3 × logit(probability))
```

This is a strictly increasing transformation. It preserves ordering. It also leaves every value on the same side of `0.5`:

```text
0.1 -> 0.0014
0.3 -> 0.0730
0.7 -> 0.9270
0.9 -> 0.9986
```

Nothing crosses the classification threshold. The second set merely speaks with much more confidence than the outcomes justify.

The program uses datarust for accuracy, ROC-AUC, and log-loss. Brier score and a ten-bin expected calibration error are short local helpers because the current metrics module does not expose dedicated functions for them.

Here is the complete Rust program:

```rust
use datarust::metrics::classification::{
    accuracy_score, log_loss, roc_auc_score,
};

fn sharpen(probability: f64, factor: f64) -> f64 {
    let logit =
        (probability / (1.0 - probability)).ln();
    1.0 / (1.0 + (-factor * logit).exp())
}

fn hard_labels(probabilities: &[f64]) -> Vec<f64> {
    probabilities
        .iter()
        .map(|&p| if p >= 0.5 { 1.0 } else { 0.0 })
        .collect()
}

fn brier_score(
    labels: &[f64],
    probabilities: &[f64],
) -> f64 {
    labels
        .iter()
        .zip(probabilities)
        .map(|(y, p)| (y - p).powi(2))
        .sum::<f64>()
        / labels.len() as f64
}

fn expected_calibration_error(
    labels: &[f64],
    probabilities: &[f64],
    n_bins: usize,
) -> f64 {
    let mut counts = vec![0usize; n_bins];
    let mut label_sums = vec![0.0; n_bins];
    let mut probability_sums = vec![0.0; n_bins];

    for (&label, &probability) in
        labels.iter().zip(probabilities)
    {
        let bin = ((probability * n_bins as f64)
            .floor() as usize)
            .min(n_bins - 1);
        counts[bin] += 1;
        label_sums[bin] += label;
        probability_sums[bin] += probability;
    }

    (0..n_bins)
        .filter(|&bin| counts[bin] > 0)
        .map(|bin| {
            let observed =
                label_sums[bin] / counts[bin] as f64;
            let predicted = probability_sums[bin]
                / counts[bin] as f64;
            counts[bin] as f64 / labels.len() as f64
                * (observed - predicted).abs()
        })
        .sum()
}

fn print_metrics(
    name: &str,
    labels: &[f64],
    probabilities: &[f64],
) {
    let hard = hard_labels(probabilities);
    println!(
        "{name:<12} accuracy {:.3} | ROC-AUC {:.3} | \
         log-loss {:.3} | Brier {:.3} | ECE {:.3}",
        accuracy_score(labels, &hard).unwrap(),
        roc_auc_score(labels, probabilities).unwrap(),
        log_loss(labels, probabilities, 1e-15).unwrap(),
        brier_score(labels, probabilities),
        expected_calibration_error(
            labels,
            probabilities,
            10,
        ),
    );
}

fn main() {
    let risks = [0.1_f64, 0.3, 0.7, 0.9];
    let mut labels = Vec::new();
    let mut calibrated = Vec::new();

    for risk in risks {
        let positives =
            (risk * 100.0).round() as usize;

        for i in 0..100 {
            labels.push(
                if i < positives { 1.0 } else { 0.0 },
            );
            calibrated.push(risk);
        }
    }

    let overconfident: Vec<f64> = calibrated
        .iter()
        .map(|&p| sharpen(p, 3.0))
        .collect();

    print_metrics("calibrated", &labels, &calibrated);
    print_metrics(
        "overconfident",
        &labels,
        &overconfident,
    );

    println!(
        "\nrisk group   observed   calibrated   \
         overconfident   actual positives   \
         predicted positives"
    );

    for (group, risk) in risks.iter().enumerate() {
        let start = group * 100;
        let end = start + 100;
        let observed =
            labels[start..end].iter().sum::<f64>() / 100.0;
        let calibrated_mean = calibrated[start..end]
            .iter()
            .sum::<f64>()
            / 100.0;
        let overconfident_mean = overconfident[start..end]
            .iter()
            .sum::<f64>()
            / 100.0;

        println!(
            "{:>10.1}   {:>8.3}   {:>10.3}   \
             {:>13.4}   {:>16.0}   {:>7.2} / {:>6.2}",
            risk,
            observed,
            calibrated_mean,
            overconfident_mean,
            observed * 100.0,
            calibrated_mean * 100.0,
            overconfident_mean * 100.0,
        );
    }

    let same_decisions =
        hard_labels(&calibrated)
            == hard_labels(&overconfident);
    let same_order = (0..calibrated.len()).all(|i| {
        (0..calibrated.len()).all(|j| {
            calibrated[i].total_cmp(&calibrated[j])
                == overconfident[i]
                    .total_cmp(&overconfident[j])
        })
    });

    println!(
        "\nsame hard decisions: {same_decisions}"
    );
    println!("same pairwise order:  {same_order}");
    println!(
        "total actual / calibrated expected / \
         overconfident expected: {:.0} / {:.2} / {:.2}",
        labels.iter().sum::<f64>(),
        calibrated.iter().sum::<f64>(),
        overconfident.iter().sum::<f64>(),
    );
}
```

This is the output I measured:

```text
calibrated   accuracy 0.800 | ROC-AUC 0.850 | log-loss 0.468 | Brier 0.150 | ECE 0.000
overconfident accuracy 0.800 | ROC-AUC 0.850 | log-loss 0.749 | Brier 0.181 | ECE 0.163

risk group   observed   calibrated   overconfident   actual positives   predicted positives
       0.1      0.100        0.100          0.0014                 10     10.00 /   0.14
       0.3      0.300        0.300          0.0730                 30     30.00 /   7.30
       0.7      0.700        0.700          0.9270                 70     70.00 /  92.70
       0.9      0.900        0.900          0.9986                 90     90.00 /  99.86

same hard decisions: true
same pairwise order:  true
total actual / calibrated expected / overconfident expected: 200 / 200.00 / 200.00
```

The totals agree. The risk groups do not.

## Accuracy discarded the confidence

Accuracy sees only the hard labels produced at the threshold:

```text
probability >= 0.5 -> class 1
probability <  0.5 -> class 0
```

Both probability vectors put the 0.1 and 0.3 groups below the threshold and the 0.7 and 0.9 groups above it. They therefore make the same 400 class decisions.

The 0.9986 model can be catastrophically overconfident about a case and receive exactly the same accuracy credit as the 0.9 model when both choose class one. When they are wrong, each simply contributes one error.

That is appropriate if the only product question is “Was the final class correct?” It is insufficient if confidence changes the action or its size.

A lender allocating reserves, a hospital prioritizing follow-up, or an operations team forecasting failure volume needs more than threshold correctness.

## ROC-AUC discarded the probability scale

ROC-AUC asks how often a randomly selected positive receives a higher score than a randomly selected negative. It depends on ranking, including tie handling—not on whether `0.9` literally means 90%.

Multiplying logits by a positive factor is strictly monotonic. Every pair keeps the same order, which the program confirms:

```text
same pairwise order: true
```

So both score sets receive ROC-AUC `0.850`.

This is not a weakness in the metric. Ranking is exactly what ROC-AUC is designed to evaluate. A search system can care primarily about order. A triage queue may only need the riskiest cases first.

The mistake is interpreting a ranking metric as probability validation.

Good discrimination, good threshold decisions, and good calibration are separate properties.

## Log-loss made confidence accountable

Binary log-loss is:

```text
-mean(y × log(p) + (1-y) × log(1-p))
```

It rewards high probability on outcomes that occur and heavily penalizes high confidence on outcomes that do not.

In the highest-risk group, ten observations are negative. For each one, the overconfident prediction contributes approximately:

```text
-log(1 - 0.9986) ≈ 6.59
```

The calibrated prediction contributes:

```text
-log(1 - 0.9) ≈ 2.30
```

Both make the same wrong class decision. Log-loss distinguishes how implausible each model claimed that outcome was.

Across the dataset:

```text
calibrated log-loss:    0.468
overconfident log-loss: 0.749
```

Lower is better. Unlike accuracy, log-loss evaluates the complete probability, not only which side of a threshold it occupies.

## Brier score told a similar story in squared error

Brier score is the mean squared difference between probability and binary outcome:

```text
mean((probability - label)²)
```

It also prefers the calibrated vector:

```text
calibrated:    0.150
overconfident: 0.181
```

Log-loss reacts especially strongly near zero and one because confident contradictions approach infinite loss before clipping. Brier score remains bounded between zero and one for binary outcomes and has a more direct squared-error interpretation.

Both are proper scoring rules: in expectation, reporting the true probability minimizes the score. They make honest probability estimates the optimal strategy rather than rewarding confidence for its own sake.

I often report at least one proper probability loss alongside discrimination and threshold metrics.

## ECE summarized the reliability gap

The helper divides `[0, 1]` into ten equal-width bins. For each occupied bin, it compares:

```text
mean predicted probability
vs.
observed positive frequency
```

It then weights the absolute gaps by bin population.

The calibrated groups land exactly on their observed frequencies, producing ECE `0.000`. The sharpened scores produce ECE `0.163`.

ECE is intuitive but not absolute truth. Its value changes with bin boundaries, bin count, sample size, and the distribution of predictions. A small dataset can make reliability estimates noisy, especially near the extremes.

I use a reliability table or diagram with counts alongside ECE. A single summary number can hide whether the error lives in a critical high-risk region or an unimportant part of the range.

## Aggregate calibration hid four errors that canceled

Both probability vectors predict 200 positives in expectation:

```text
actual positives:                 200
calibrated expected positives:    200
overconfident expected positives: 200
```

An aggregate forecast check would declare both perfect.

The group table exposes the cancellation:

```text
observed 10, overconfident expected  0.14
observed 30, overconfident expected  7.30
observed 70, overconfident expected 92.70
observed 90, overconfident expected 99.86
```

Underprediction below 0.5 and overprediction above 0.5 balance in the total. The model gets the overall event count right while badly misrepresenting individual risk levels.

This is why I inspect calibration across probability ranges and relevant cohorts. Aggregate rate, geography, customer type, time period, or hardware version can each hide offsetting errors inside smaller groups.

Calibration is conditional, and every practical check approximates which conditions matter.

## The model output is a score until validated

After fitting binary `LogisticRegression`, datarust can return positive-class probabilities:

```rust
let probabilities =
    model.predict_positive_proba(&x_validation)?;
```

For a generic binary/multiclass interface:

```rust
let matrix = model.predict_proba(&x_validation)?;
```

Those values are produced by a sigmoid or softmax, so they lie between zero and one and sum appropriately across classes. That mathematical form does not prove empirical calibration on future data.

Regularization, class weighting, oversampling, undersampling, dataset shift, model misspecification, and limited sample size can all change how numerical confidence relates to observed frequency.

I call the output a probability only after evaluating that interpretation on held-out data from the deployment boundary. Before then, it is a model score with probability-shaped syntax.

## Calibration needs its own data boundary

If probabilities are miscalibrated but ranking is useful, a calibration mapping can be learned.

Common approaches include:

- Platt scaling: fit a sigmoid to model scores.
- Temperature scaling: rescale logits with a learned temperature.
- Isotonic regression: learn a monotonic piecewise mapping.

The sharpened vector in this article was created with a temperature-like logit multiplier. Applying the correct inverse factor would recover the calibrated values exactly, but real data does not reveal that factor in advance.

The mapping must be fitted on data not used to fit the underlying model. A typical sequence is:

1. Fit the classifier on a training partition.
2. Generate probabilities on a calibration partition.
3. Fit the calibrator using calibration labels.
4. Evaluate model plus calibrator once on untouched test data.

With limited data, out-of-fold model predictions can supply calibration inputs without wasting one large fixed partition. The fold construction still has to respect time and entities.

Fitting a calibrator on the final test set converts evaluation labels into model parameters. The reported result then describes the data it already studied.

The current datarust codebase provides probability outputs and `log_loss`, but not a dedicated Platt, temperature, or isotonic calibration estimator. I would implement the calibration layer explicitly, validate it separately, and persist it with the preprocessing and classifier rather than imply that `predict_proba` performs post-hoc calibration automatically.

## Calibration can expire

A model may remain well ranked while its probabilities drift.

Suppose the base failure rate doubles after a manufacturing change. High-risk cases may still rank above low-risk cases, preserving ROC-AUC, while every original probability becomes too low.

I monitor:

- log-loss and Brier score after labels mature,
- reliability tables with counts,
- expected versus observed events by score band,
- base rate and score distribution over time,
- calibration by important operational cohort,
- the fraction of predictions near zero or one,
- threshold performance at the actual action points.

Delayed outcomes matter. A 30-day churn probability cannot be evaluated honestly two days after prediction. The monitoring pipeline must wait for the label window or account for censoring.

Recalibration can repair a changed probability mapping while retaining useful ranking. It should also trigger investigation: sometimes calibration drift is the first visible symptom of a deeper feature or population shift.

## Sometimes ranking really is enough

Not every application needs literal probabilities.

If a team reviews exactly the top 100 alerts each day, stable ranking around that cutoff may matter more than whether the top score is `0.8` or `0.95`. ROC-AUC, average precision, precision at capacity, and operational yield may be the primary metrics.

Even then, I avoid presenting the raw score as “95% likely” without calibration evidence. A monotonic score can serve ranking perfectly while having no defensible frequency interpretation.

When expected cost determines the threshold, calibration matters directly. If an intervention costs 10 units and a prevented failure saves 100, a risk threshold derived from those quantities assumes the probability scale is meaningful. Overconfidence changes the economic decision even when ranking is unchanged.

The metric set should follow how the output is consumed.

## The class labels concealed the broken promise

The two prediction vectors in this experiment agreed on every class and every pairwise ordering.

That preserved:

```text
accuracy: 0.800
ROC-AUC:  0.850
```

It did not preserve the promise attached to a number such as `0.9986`.

The calibrated 0.9 group contained 90 positives. The sharpened model looked at those same outcomes and claimed almost 100. Its ten negative cases became extremely expensive under log-loss, and its group forecast overshot by nearly ten events.

Even the overall expected event count failed to reveal the problem because errors above and below the threshold canceled exactly.

So when a model returns a probability, I now ask three separate questions:

1. Does it rank the right cases higher?
2. Does its chosen threshold produce useful decisions?
3. Do groups assigned probability `p` experience the event about `p` of the time?

Accuracy measured one part of the second question—the class hit rate at one threshold. ROC-AUC answered the first across thresholds.

Log-loss, Brier score, and the reliability table finally addressed the third.

The model was not wrong about which side to choose.

It was wrong about how certain it had any right to be.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
