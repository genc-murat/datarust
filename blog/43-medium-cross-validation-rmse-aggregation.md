# Five Fold RMSEs Picked Model A. The Same 25 Predictions Picked Model B.

*A practical datarust guide to mean fold RMSE, pooled out-of-fold error, nonlinear metric aggregation, unequal fold sizes, and the innocent average that reversed a model decision.*

---

Model A won cross-validation.

Its five fold RMSE values were:

```text
1, 1, 1, 1, 9
```

The dashboard averaged them:

```text
(1 + 1 + 1 + 1 + 9) / 5 = 2.6
```

Model B returned:

```text
3, 3, 3, 3, 3
```

Its average fold RMSE was `3.0`, so the release report selected Model A.

Then I gathered the exact same held-out predictions from all five folds and calculated one RMSE across the 25 rows.

The winner changed:

```text
Model A pooled RMSE: 4.123
Model B pooled RMSE: 3.000
```

No prediction changed. No fold changed. No data was added or removed.

I had changed only the order of two operations:

```text
average the fold square roots
```

versus:

```text
average all squared errors, then take one square root
```

Square root is nonlinear, so those expressions are not interchangeable.

Let's reproduce the reversal with [datarust](https://crates.io/crates/datarust), then build a metric-reporting contract that says exactly what is being averaged and why.

## Two models with different failure shapes

Our controlled evaluation has 25 rows and five equal test folds. The target is zero on every row, which makes the errors easy to see directly.

Model A has an absolute error of `1` on the first 20 rows and an error of `9` on the final five:

```text
fold 1: 1, 1, 1, 1, 1
fold 2: 1, 1, 1, 1, 1
fold 3: 1, 1, 1, 1, 1
fold 4: 1, 1, 1, 1, 1
fold 5: 9, 9, 9, 9, 9
```

Model B has an error of `3` on every row.

This is not a training experiment. The prediction vectors are fixed deliberately so no estimator, random seed, or optimization detail can distract from the aggregation question. `KFold` defines the five test partitions, and datarust's `mean_squared_error` calculates both MSE and RMSE.

In a real cross-validation run, these vectors would be out-of-fold predictions: each row would receive one prediction from a model that did not train on that row.

## The complete Rust experiment

Create a small application and add datarust:

```sh
cargo new rmse_aggregation
cd rmse_aggregation
cargo add datarust
```

Replace `src/main.rs` with this:

```rust
use datarust::metrics::regression::mean_squared_error;
use datarust::model_selection::KFold;

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn evaluate(
    name: &str,
    truth: &[f64],
    predictions: &[f64],
    cv: &KFold,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut fold_rmse = Vec::new();
    let mut fold_mse = Vec::new();

    println!("{name}");
    for (fold, (_, test_indices)) in
        cv.split(truth.len())?.enumerate()
    {
        let fold_truth: Vec<f64> = test_indices
            .iter()
            .map(|&i| truth[i])
            .collect();
        let fold_predictions: Vec<f64> = test_indices
            .iter()
            .map(|&i| predictions[i])
            .collect();

        let rmse = mean_squared_error(
            &fold_truth,
            &fold_predictions,
            false,
        )?;
        let mse = mean_squared_error(
            &fold_truth,
            &fold_predictions,
            true,
        )?;

        fold_rmse.push(rmse);
        fold_mse.push(mse);
        println!(
            "  fold {} | rows {} | RMSE {:.3} | MSE {:.3}",
            fold + 1,
            test_indices.len(),
            rmse,
            mse,
        );
    }

    let mean_fold_rmse = mean(&fold_rmse);
    let pooled_rmse = mean_squared_error(
        truth,
        predictions,
        false,
    )?;
    let rmse_from_mean_mse = mean(&fold_mse).sqrt();

    println!(
        "  mean fold RMSE:        {mean_fold_rmse:.3}"
    );
    println!(
        "  pooled OOF RMSE:       {pooled_rmse:.3}"
    );
    println!(
        "  sqrt(mean fold MSE):   {rmse_from_mean_mse:.3}"
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let truth = vec![0.0; 25];

    let mut model_a = vec![1.0; 20];
    model_a.extend(vec![9.0; 5]);
    let model_b = vec![3.0; 25];

    let cv = KFold::new().with_n_splits(5);
    evaluate(
        "Model A: four quiet folds, one collapse",
        &truth,
        &model_a,
        &cv,
    )?;
    println!();
    evaluate(
        "Model B: the same error everywhere",
        &truth,
        &model_b,
        &cv,
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
Model A: four quiet folds, one collapse
  fold 1 | rows 5 | RMSE 1.000 | MSE 1.000
  fold 2 | rows 5 | RMSE 1.000 | MSE 1.000
  fold 3 | rows 5 | RMSE 1.000 | MSE 1.000
  fold 4 | rows 5 | RMSE 1.000 | MSE 1.000
  fold 5 | rows 5 | RMSE 9.000 | MSE 81.000
  mean fold RMSE:        2.600
  pooled OOF RMSE:       4.123
  sqrt(mean fold MSE):   4.123

Model B: the same error everywhere
  fold 1 | rows 5 | RMSE 3.000 | MSE 9.000
  fold 2 | rows 5 | RMSE 3.000 | MSE 9.000
  fold 3 | rows 5 | RMSE 3.000 | MSE 9.000
  fold 4 | rows 5 | RMSE 3.000 | MSE 9.000
  fold 5 | rows 5 | RMSE 3.000 | MSE 9.000
  mean fold RMSE:        3.000
  pooled OOF RMSE:       3.000
  sqrt(mean fold MSE):   3.000
```

Model A wins one summary and loses the other.

The arithmetic is correct in both cases.

## RMSE contains two operations

Root mean squared error is:

```text
RMSE = sqrt(mean(error²))
```

It first squares each residual, then averages the squared values, then takes one square root so the result returns to the target's units.

For Model A, the complete out-of-fold error vector contains twenty `1`s and five `9`s. Its pooled MSE is:

```text
(20 × 1² + 5 × 9²) / 25
= (20 + 405) / 25
= 17
```

Therefore:

```text
pooled RMSE = sqrt(17) = 4.123
```

The mean-fold calculation takes five separate roots first:

```text
sqrt(1), sqrt(1), sqrt(1), sqrt(1), sqrt(81)
```

and averages them:

```text
(1 + 1 + 1 + 1 + 9) / 5 = 2.6
```

The difference is not floating-point noise. It comes from moving the square root across an average.

In general:

```text
mean(sqrt(fold MSE)) != sqrt(mean(fold MSE))
```

They are equal only in special cases, including when all fold MSE values are identical—as they are for Model B.

## The square root softens fold-to-fold variation

Square root is concave. As its input becomes large, each additional unit of MSE produces a smaller increase in RMSE.

Model A's collapsed fold has:

```text
MSE = 81
RMSE = 9
```

When fold RMSE values are averaged, that fold contributes `9` beside four contributions of `1`.

When squared errors are pooled first, it contributes five errors of `81` to the common numerator. The final root is applied only after every row has contributed its squared loss.

This is why averaging already-rooted values can make uneven fold performance look gentler than one pooled root of all squared errors.

The mean fold RMSE is not fraudulent. It simply treats “RMSE of a fold” as the quantity being averaged. The pooled RMSE treats squared error per held-out row as the quantity being averaged.

The dashboard needs to say which one it means.

## What question does mean fold RMSE answer?

Mean fold RMSE gives each fold one RMSE value and then gives those fold values equal weight:

```text
What was the average RMSE across our validation folds?
```

That can be useful when folds are intentional evaluation units, such as sites, time periods, or repeated split replications. It retains the fact that model performance varied across those units.

But the mean alone hides the shape we actually saw:

```text
mean: 2.6
folds: 1, 1, 1, 1, 9
```

A user reading only `2.6` may imagine five moderately noisy folds. The real model had four excellent folds and one collapse.

Whenever I report mean fold RMSE, I also keep at least:

- every fold value
- standard deviation or another spread summary
- minimum and maximum
- fold sizes
- the split identity or time window

For a release gate, the worst fold may matter more than the mean if one fold represents a real region, device family, or month that the product must support.

## What question does pooled out-of-fold RMSE answer?

Pooled OOF RMSE concatenates one held-out prediction per row and computes the metric once:

```text
What is the root mean squared error across all held-out row predictions?
```

Every row contributes its squared residual to one common total. With equal business importance per row, this is often the most direct cross-validated estimate of per-row squared-error magnitude.

There is a subtlety: out-of-fold predictions come from different fitted models. The fold-one rows were predicted by a model trained without fold one; fold-two rows were predicted by another model, and so on.

Pooled OOF RMSE describes that complete cross-fitting procedure. It is not literally the test RMSE of the one final model that will later be fitted on all training rows.

That is normal. Cross-validation estimates a training procedure by repeatedly refitting it. A final untouched test set is still valuable after model selection.

## Equal folds made the example kind

Our 25 rows divide evenly into five folds, so each fold contains five rows.

That is why this shortcut works:

```text
pooled RMSE = sqrt(mean(fold MSE))
```

Real datasets are often not divisible by the number of folds. datarust's `KFold` distributes remainder rows across the first folds. For 23 rows and five folds, the sizes are:

```text
5, 5, 5, 4, 4
```

An unweighted mean of the five fold MSE values gives a four-row fold the same influence as a five-row fold.

The correct pooled calculation is size-weighted:

```text
pooled MSE = sum(fold_size × fold_MSE) / sum(fold_size)
pooled RMSE = sqrt(pooled MSE)
```

Equivalently, retain every out-of-fold prediction and let datarust calculate RMSE once:

```rust
let pooled_rmse = mean_squared_error(
    &all_oof_truth,
    &all_oof_predictions,
    false,
)?;
```

That avoids reconstructing the weighting algebra by hand.

## A weighted mean of fold RMSE is still not pooled RMSE

Weighting each fold RMSE by its row count repairs unequal representation at the fold-value level:

```text
sum(fold_size × fold_RMSE) / total_rows
```

But it still averages after each square root. It is not generally equal to taking one root after pooling squared errors.

The correct sequence for pooled RMSE is:

```text
1. preserve or reconstruct squared-error totals
2. add those totals
3. divide by the total row count
4. take one square root
```

This ordering rule is the heart of the article. Weighting alone does not make a nonlinear metric additive.

## `cross_val_score` returns evidence, not an aggregation policy

datarust's `cross_val_score` returns one score per fold:

```rust
let scores: Vec<f64> = cross_val_score(
    &model,
    &x,
    &y,
    &cv,
    scorer,
)?;
```

That design is useful because the library does not silently decide what those values mean to the product. The caller can inspect the vector, calculate a mean, compare spread, or use a domain-specific release rule.

If the scorer returns RMSE and I write:

```rust
let average = mean(&scores);
```

I have explicitly chosen mean fold RMSE.

If I need pooled OOF RMSE, fold scores alone are not sufficient unless I also retained fold MSE values and sizes. I usually run an explicit fold loop, write each prediction back to its original row position, and compute the final metric from the complete OOF vectors.

That also gives me row-level residuals for segment analysis instead of five numbers that cannot be unpacked later.

## The aggregation rule can reverse model selection

Our two candidates create a real decision conflict:

| Aggregation | Model A | Model B | Winner |
|---|---:|---:|---|
| Mean fold RMSE | 2.600 | 3.000 | A |
| Pooled OOF RMSE | 4.123 | 3.000 | B |

Model A is usually excellent and occasionally disastrous. Model B is consistently mediocre.

Neither metric can decide whether that tradeoff is acceptable without product context.

For a recommendation system, an occasional weak random fold may be tolerable. For dosage, credit limits, inventory commitments, or delivery promises in a specific market, one collapse may dominate the decision.

I choose the aggregation policy before comparing candidates whenever possible. Choosing it after seeing which model wins turns metric definition into another tuning knob.

## Additive metrics have simpler pooling rules

Some metrics are built from per-row losses that can be added before the final summary.

For out-of-fold predictions:

- MAE pools by adding absolute errors and dividing by total rows.
- MSE pools by adding squared errors and dividing by total rows.
- Log-loss pools by adding per-row log losses and dividing by total rows.
- Accuracy pools by adding correct decisions and dividing by total rows.

With unequal fold sizes, their pooled values equal a fold-size-weighted mean of the corresponding fold metrics.

RMSE adds one nonlinear outer operation. Pool MSE first, then take the root.

This gives me a practical implementation pattern: retain numerator totals and denominators for additive losses, not only already-normalized display metrics.

## Other metrics need their own aggregation contract

It is dangerous to generalize the RMSE repair into “always take a weighted average.”

For F1, precision, and recall, I can pool the underlying confusion counts and recompute the metric. Averaging fold F1 values answers a different question because F1 is a ratio and harmonic mean.

For R², fold scores use each fold's own target mean in the denominator. A pooled R² uses one global target mean. Averaging or pooling can therefore tell very different stories when folds have different target levels.

For ROC-AUC, pooling out-of-fold scores creates comparisons between probabilities emitted by different fitted fold models. If their score scales are not aligned, those cross-fold rankings can be misleading. Mean fold AUC avoids cross-model pair comparisons but weights folds according to the chosen averaging rule.

For calibration metrics, probability bins can also shift when predictions are pooled.

There is no universal “cross-validation average” that preserves every metric's meaning. I document aggregation per metric, ideally alongside its sufficient statistics and intended decision.

## Fold construction still matters more than arithmetic

A perfectly aggregated metric cannot rescue an unrealistic split.

Our fifth fold was designed to collapse. In a real project, a bad fold may reveal:

- a later time period with drift
- a site absent from training
- a rare customer segment
- a device generation with different sensors
- simple random bad luck in a small sample

Those causes imply different actions.

If time and groups matter in production, I build folds around them. Then I report per-fold behavior because the fold identity carries real meaning. If folds are random replications, I may focus more on the distribution and uncertainty across repeated runs.

Changing from mean RMSE to pooled RMSE should not become a way to hide a structurally bad fold. The fold table remains part of the evidence.

## The reporting contract I use now

For regression cross-validation, my report records:

```text
split strategy and seed
fold IDs and row counts
per-fold MAE, MSE, RMSE, and relevant business metrics
mean and spread across folds
pooled OOF metrics from row-aligned predictions
worst-fold behavior
segment and time-window residuals
```

The headline includes the aggregation in its name:

```text
mean_fold_rmse
pooled_oof_rmse
worst_fold_rmse
```

I avoid a field simply called `cv_rmse`. It invites every consumer to assume the version they expected.

I also persist the OOF prediction vector with row IDs and fold IDs. That makes the metric reproducible and lets reviewers inspect why two summaries disagree.

## One average hid a collapse

Model A's fold RMSE values were `1, 1, 1, 1, 9`. Averaging those five roots produced `2.600`, enough to beat Model B's steady `3.000`.

Pooling the same 25 held-out predictions told a different per-row squared-error story. Model A rose to `4.123`; Model B stayed at `3.000` and became the winner.

Neither answer came from a bug in datarust. `mean_squared_error(..., false)` calculated every requested RMSE correctly. The decision changed in the code that summarized those correct values.

The durable lessons are simple:

- preserve fold-level metrics instead of reporting only a mean
- retain row-aligned out-of-fold predictions
- pool MSE before taking one RMSE square root
- weight by fold size when folds are unequal
- define whether folds or rows are the primary evaluation unit
- choose aggregation before looking at the winner
- give every reported metric a name that exposes its contract

A metric formula does not end at the fold boundary. How we combine the folds is part of the model evaluation too.
