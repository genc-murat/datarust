# Accuracy Hit 100% at Iteration 5. The Coefficient Kept Growing for 95 More.

*A practical datarust guide to complete separation, unregularized logistic regression, exhausted iteration budgets, unstable probabilities, and the classifier that never found a finite reason to stop becoming more confident.*

---

The classifier was perfect after five iterations.

```text
training accuracy: 1.000
```

I increased the iteration budget anyway. The predicted class labels did not change. The mistakes could not decrease because there were none.

The coefficient kept moving:

```text
 5 iterations ->  2.9294
10 iterations ->  7.8262
20 iterations -> 17.8239
40 iterations -> 26.2049
```

On the training rows, every one of those models returned exactly the same hard labels.

Near the decision boundary, they did not tell the same story at all.

For two new points at `-0.1` and `+0.1`, the five-iteration model predicted positive-class probabilities of roughly 43% and 57%. The forty-iteration model returned 7% and 93%.

Nothing new had been learned about that empty region. The optimizer had only become more certain because the training data gave it no contradiction.

This is complete separation: a line can divide every negative training row from every positive one. For ordinary unregularized logistic regression, that seemingly ideal dataset creates an awkward mathematical result. The likelihood improves as the separating coefficient grows, so there is no finite maximum-likelihood coefficient to find.

Let's make that behavior visible with [datarust](https://crates.io/crates/datarust), compare it with an overlapping dataset, and separate three ideas that a perfect training score tends to blur together:

- correct class decisions
- stable fitted parameters
- trustworthy probabilities

They are not the same promise.

## Ten rows with a perfect gap

The first dataset has one numeric feature:

```text
x = -5, -4, -3, -2, -1 -> class 0
x =  1,  2,  3,  4,  5 -> class 1
```

There is no row at zero and no label overlap. Any decision boundary between `-1` and `1` classifies all ten rows correctly.

We fit the same datarust `LogisticRegression` four times with iteration budgets of 5, 10, 20, and 40. For this diagnostic, the tolerance is set to zero so the loop uses each requested budget and exposes the optimization path.

Then we fit once with datarust's ordinary stopping defaults. Finally, we modify two near-boundary labels:

```text
x = -1 -> class 1
x = +1 -> class 0
```

Those two contradictions remove perfect separation. The same model now has a finite compromise to learn.

## The complete Rust experiment

Create a small application and add datarust:

```sh
cargo new separation_audit
cd separation_audit
cargo add datarust
```

Replace `src/main.rs` with this:

```rust
use datarust::linear_model::{
    LogisticRegression, LogisticSolver,
};
use datarust::metrics::classification::{
    accuracy_score, log_loss,
};
use datarust::traits::Predictor;
use datarust::Matrix;

fn dataset(overlap: bool) -> (Matrix, Vec<f64>) {
    let values = [
        -5.0, -4.0, -3.0, -2.0, -1.0,
         1.0,  2.0,  3.0,  4.0,  5.0,
    ];
    let mut labels = vec![0.0; 5];
    labels.extend(vec![1.0; 5]);

    if overlap {
        labels[4] = 1.0;
        labels[5] = 0.0;
    }

    let rows = values
        .into_iter()
        .map(|value| vec![value])
        .collect();
    (Matrix::new(rows).unwrap(), labels)
}

fn report(
    name: &str,
    model: &LogisticRegression,
    x: &Matrix,
    y: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let predictions = model.predict(x)?;
    let probabilities =
        model.predict_positive_proba(x)?;
    let near = Matrix::new(vec![
        vec![-0.1],
        vec![0.1],
    ])?;
    let near_probabilities =
        model.predict_positive_proba(&near)?;

    println!(
        "{name:<18} iter {:>3} | coef {:>10.4} | \
         acc {:.3} | log-loss {:.6} | \
         P(-0.1) {:.6} | P(+0.1) {:.6}",
        model.n_iter(),
        model.coef()[0][0],
        accuracy_score(y, &predictions)?,
        log_loss(y, &probabilities, 1e-15)?,
        near_probabilities[0],
        near_probabilities[1],
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (separable_x, separable_y) = dataset(false);

    println!("Perfectly separable data");
    for max_iter in [5, 10, 20, 40] {
        let mut model = LogisticRegression::new()
            .with_solver(LogisticSolver::Svd)
            .with_max_iter(max_iter)
            .with_tol(0.0);
        model.fit(&separable_x, &separable_y)?;
        report(
            &format!("max_iter={max_iter}"),
            &model,
            &separable_x,
            &separable_y,
        )?;
    }

    let mut default_stopping = LogisticRegression::new()
        .with_solver(LogisticSolver::Svd);
    default_stopping.fit(&separable_x, &separable_y)?;
    report(
        "default stopping",
        &default_stopping,
        &separable_x,
        &separable_y,
    )?;

    let (overlap_x, overlap_y) = dataset(true);
    let mut overlap_model = LogisticRegression::new()
        .with_solver(LogisticSolver::Svd)
        .with_max_iter(100)
        .with_tol(1e-8);
    overlap_model.fit(&overlap_x, &overlap_y)?;

    println!();
    println!("Two contradictory near-boundary labels");
    report(
        "overlapping",
        &overlap_model,
        &overlap_x,
        &overlap_y,
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
Perfectly separable data
max_iter=5         iter   5 | coef     2.9294 | acc 1.000 | log-loss 0.011012 | P(-0.1) 0.427284 | P(+0.1) 0.572716
max_iter=10        iter  10 | coef     7.8262 | acc 1.000 | log-loss 0.000080 | P(-0.1) 0.313755 | P(+0.1) 0.686245
max_iter=20        iter  20 | coef    17.8239 | acc 1.000 | log-loss 0.000000 | P(-0.1) 0.144009 | P(+0.1) 0.855991
max_iter=40        iter  40 | coef    26.2049 | acc 1.000 | log-loss 0.000000 | P(-0.1) 0.067831 | P(+0.1) 0.932169
default stopping   iter 100 | coef    27.9194 | acc 1.000 | log-loss 0.000000 | P(-0.1) 0.057761 | P(+0.1) 0.942239

Two contradictory near-boundary labels
overlapping        iter   7 | coef     0.8617 | acc 0.800 | log-loss 0.299136 | P(-0.1) 0.478471 | P(+0.1) 0.521529
```

The separable models agree completely on training accuracy. Their coefficients and near-boundary probabilities do not agree.

The overlapping model behaves differently: it stops after seven iterations under a tight tolerance and returns a modest finite coefficient.

## Logistic regression models log-odds

For one feature, binary logistic regression predicts:

```text
P(class 1 | x) = sigmoid(intercept + coefficient × x)
```

where:

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

Our separable data is symmetric, so the fitted boundary remains at zero. A positive coefficient gives values below zero probabilities under 50% and values above zero probabilities over 50%.

Increasing the coefficient does not move the hard decision boundary:

```text
x < 0 -> class 0
x > 0 -> class 1
```

It steepens the probability curve around that boundary.

With coefficient `2.9294`:

```text
P(class 1 | -0.1) = 0.427284
P(class 1 | +0.1) = 0.572716
```

With coefficient `27.9194`:

```text
P(class 1 | -0.1) = 0.057761
P(class 1 | +0.1) = 0.942239
```

Both models classify the ten training rows perfectly. One describes nearby unseen points as uncertain; the other treats a two-tenths-wide gap as an almost decisive transition.

The training labels contain no evidence for choosing between those probability stories.

## Why there is no finite optimum

Consider any coefficient that separates all ten rows correctly. Now multiply it by a larger positive number.

The predicted probabilities for negative rows move closer to zero. The probabilities for positive rows move closer to one. Every training log-likelihood term improves. No row pushes back in the opposite direction.

Conceptually:

```text
coefficient -> infinity
training log-loss -> 0
```

The optimizer can always make the training likelihood a little better by increasing the magnitude again. The best value exists only as a limit, not as a finite coefficient.

This is why complete separation is not ordinary overfitting in the casual sense of “the model has too many parameters.” Our experiment has one feature and one slope. The maximum-likelihood problem itself has no finite solution for this sample.

The numerical routine must stop for a practical reason:

- it reaches `max_iter`
- coefficient changes fall below a tolerance because of finite precision or safeguards
- probabilities and weights saturate
- the linear solve becomes numerically difficult

None of those events creates the missing finite statistical optimum.

## Accuracy stopped being informative first

Hard-label accuracy only asks which side of `0.5` each probability occupies.

By iteration five, all negative training values are below the boundary and all positive values are above it. Accuracy reaches `1.000` and stays there.

The optimization objective is log-loss, not accuracy. Log-loss continues rewarding probabilities closer to the observed labels:

```text
iteration 5:  0.011012
iteration 10: 0.000080
iteration 20: below 0.0000005 when printed to six decimals
```

That improvement sounds attractive, but it is measured on the same perfectly separated rows that caused the problem. The lower training loss comes from increasing confidence, not correcting another class decision.

This is why I do not use a perfect accuracy number as evidence that an unregularized probability model has converged sensibly.

## The default fit exhausted its iteration budget

The four diagnostic fits use `tol = 0.0` deliberately. That prevents tolerance-based stopping and lets us compare fixed points along the optimization path.

The default-stopping fit is more revealing operationally. It uses datarust's ordinary `max_iter = 100` and `tol = 1e-4`. Its output says:

```text
iter 100
```

The fit returned a usable model, but it consumed the complete iteration budget. The API exposes `n_iter()`, so this condition is visible if we look for it.

I treat:

```rust
model.n_iter() == configured_max_iter
```

as a diagnostic that deserves investigation. It does not prove separation by itself—poor scaling, difficult geometry, or an overly strict tolerance can also exhaust the budget. But paired with perfect separation and growing coefficients, it is strong evidence.

The current fit call does not return a separate “converged” flag. Successful return means the numerical procedure produced fitted state, not that a finite maximum-likelihood estimate was established.

## Numerical safeguards are not regularization

datarust keeps the binary IRLS calculation finite in several practical ways. The sigmoid is evaluated stably, and probability weights are clamped away from an exact zero during fitting.

Those safeguards prevent overflow and completely degenerate weighted systems. They do not add a statistical preference for smaller coefficients.

The distinction matters:

```text
numerical safeguard -> keeps arithmetic executable
regularization      -> changes the optimization objective
```

An L2-regularized logistic objective would include a penalty such as:

```text
negative log-likelihood + alpha × ||coefficient||²
```

As the coefficient grows, the penalty grows too. That creates a finite tradeoff between fitting the separated labels and keeping the parameter magnitude controlled.

The current datarust `LogisticRegression` is explicitly unregularized and does not expose an L1 or L2 penalty parameter. I treat that as an important model-selection constraint rather than pretending `max_iter` or `tol` is a substitute for regularization.

## SVD solves a linear system; it does not cure separation

The experiment uses:

```rust
.with_solver(LogisticSolver::Svd)
```

SVD is a robust choice for the Newton step when the weighted design becomes close to rank-deficient. It can keep the internal linear solve working when Cholesky would struggle.

But solver robustness and statistical identification are different problems.

SVD can answer:

```text
What Newton update should I take from this local quadratic system?
```

It cannot answer:

```text
Which finite coefficient maximizes a likelihood that improves toward infinity?
```

Changing solvers may change the numerical path or failure mode. It does not create overlap, a prior, or a regularization penalty.

## Two contradictory labels create a finite compromise

In the second dataset, the points closest to zero contradict the simple sign rule:

```text
-1 belongs to class 1
+1 belongs to class 0
```

Now increasing the positive coefficient helps the eight outer rows but hurts those two inner rows. The likelihood has opposing evidence and settles at a finite balance:

```text
coefficient: 0.8617
iterations:  7
accuracy:    0.800
log-loss:    0.299136
```

Near zero, the probability curve remains modest:

```text
P(class 1 | -0.1) = 0.478471
P(class 1 | +0.1) = 0.521529
```

The lower accuracy is not a defect in the optimizer. It reflects the fact that one straight boundary cannot satisfy every label.

Real classification data often contains this kind of overlap because outcomes are probabilistic, features are incomplete, and two people with similar observed characteristics can have different results.

That messiness can make the probability model better identified than an unnaturally perfect sample.

## Do not manufacture overlap by corrupting labels

The comparison is a mathematical demonstration, not advice to flip labels until a model converges.

If the training data is separated, I first ask why:

- Is there genuine deterministic structure?
- Is a post-outcome field leaking the answer?
- Did preprocessing encode the target into a feature?
- Does a rare category appear only in one class?
- Is the sample too small to contain realistic overlap?
- Did filtering remove all difficult cases?

A feature named `refund_approved_at` can perfectly predict whether a refund was approved because it is recorded after the decision. Regularizing that feature may make the coefficient finite, but it does not make the feature valid at prediction time.

Separation is often a useful leak detector precisely because real outcomes are rarely that polite.

If the rule is genuinely deterministic and known in advance, it may belong in explicit application logic rather than a probability model trained to approximate it.

## Quasi-separation is easier to miss

Complete separation gives a clean demonstration: every row is on the correct side.

Quasi-separation is less obvious. A feature or combination may separate almost all rows, while a few observations sit exactly on a boundary or provide weak contradiction. The fitted coefficient can be finite but enormous, highly sensitive to small data changes, and slow to converge.

Warning signs include:

- coefficient norms much larger than neighboring model versions
- probabilities clustered extremely close to zero and one
- folds that consume the full iteration budget
- dramatic coefficient changes when one row is removed
- rare one-hot categories with only positive or only negative labels
- excellent training loss paired with unstable validation log-loss

Accuracy alone may remain calm through all of this because the hard decisions do not move.

## Scaling helps numerics, not existence

Standardizing features is still good practice for many iterative models. It can improve conditioning and make coefficient changes more comparable across columns.

It does not remove separation.

If a hyperplane perfectly separates the rows before an invertible scaling, a corresponding hyperplane separates them afterward. The coefficient values change because the units change, but the unregularized likelihood can still improve without reaching a finite maximum.

I use scaling to help the solver. I use regularization, priors, better data, or a different problem formulation to address unbounded coefficients.

Those jobs should not be confused.

## Cross-validation can hide fold-specific separation

The complete dataset may contain overlap while one training fold does not.

Imagine a rare contradictory case. When that case lands in the validation fold, the remaining training rows become perfectly separable. One fold model can then exhaust its iteration budget and produce extreme probabilities, while the other folds fit normally.

A vector of validation accuracies may not reveal the cause. To diagnose it, I inspect each fitted fold model and record:

- `n_iter()`
- coefficient norm and maximum absolute coefficient
- validation log-loss, not only accuracy
- minimum and maximum predicted probabilities
- class and rare-category counts
- whether the fold is linearly separable under the selected representation

The convenience `cross_val_score` returns fold scores, not the fitted models. For this audit I use an explicit fold loop so those diagnostics are not discarded after each score is computed.

## Probability use raises the stakes

If the application consumes only a hard label and the boundary remains stable, exploding confidence can look harmless.

Many applications use the probability itself:

- prioritize a review queue
- estimate expected loss
- price risk
- allocate intervention capacity
- combine multiple model outputs
- trigger different actions at several thresholds

Our iteration-5 and default models agree at the `0.5` threshold. They can disagree at `0.1`, `0.2`, `0.8`, or `0.9` for near-boundary cases.

An extreme sigmoid can also make modest feature movement look like a sudden jump from almost impossible to almost certain. That confidence is then multiplied by costs and business rules as if it were measured evidence.

If probabilities drive decisions, I evaluate log-loss and calibration on genuinely held-out, representative data. Training log-loss approaching zero on separated rows is not calibration evidence.

## What I would do in production

When I detect separation or repeated iteration-budget exhaustion, my response depends on the cause:

1. Remove post-outcome or target-derived leakage immediately.
2. Verify grouped and chronological splitting so repeated entities or future fields do not create artificial perfection.
3. Inspect rare categorical levels and consider principled pooling where the domain supports it.
4. Collect more representative data, especially near the operational boundary.
5. Use a logistic implementation with explicit regularization or a suitable prior when finite, stable coefficients are required.
6. Tune that regularization inside cross-validation and preserve an untouched final test set.
7. Evaluate probability calibration separately from class accuracy.
8. Monitor iteration counts, coefficient norms, and probability saturation after deployment.

I do not simply increase `max_iter` until the warning disappears. In a separated problem, a larger budget gives the coefficient more room to grow.

## Perfect labels, unfinished parameters

The first model reached 100% accuracy after five iterations with a coefficient of `2.9294`. Ninety-five iterations later, the default fit still had 100% accuracy and a coefficient of `27.9194`.

The hard predictions looked finished. The probability curve was still becoming steeper.

Adding two contradictory near-boundary labels changed the geometry of the problem. The overlapping fit stopped after seven iterations with a finite coefficient of `0.8617` and probabilities close to 50% around zero.

The lesson is not that imperfect data is desirable. It is that a perfect split can make unregularized logistic maximum likelihood mathematically unfinished.

When a classifier looks impossibly good, I now check more than the score:

- Did the optimizer use its entire budget?
- Do coefficients grow when the budget grows?
- Are probabilities saturating without held-out evidence?
- Is the separation real, leaked, or a small-sample accident?
- Does the implementation include an explicit penalty?

Accuracy tells me where the probabilities landed relative to one threshold. It does not tell me whether the fitted probability model found a finite home.
