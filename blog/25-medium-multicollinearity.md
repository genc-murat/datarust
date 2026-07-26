# My Regression Coefficients Were +1002 and -998. The Signal Only Needed 4.

*A practical datarust guide to multicollinearity, solver choice, coefficient instability, and the model that predicted well for the wrong geometric reason.*

---

The regression fit was perfect.

```text
training R²: 1.000000000
```

Its two coefficients were less reassuring:

```text
+1002.000143
 -998.000143
```

The underlying relationship only needed a combined slope of four.

Then I changed every training target by at most `0.01` and fitted the model again. The coefficients moved to approximately `+1052` and `-1048`. Their change had an L2 norm of `70.711`.

Predictions on the training rows still looked excellent. A coefficient report would have told a much more dramatic story: the first measurement had become strongly positive, the second strongly negative, and both effects were hundreds of times larger than the real signal.

This is what multicollinearity can do. When two features carry nearly the same information, the model can trade a large positive weight on one against a large negative weight on the other. Their contributions cancel for ordinary rows, so predictions remain calm while the individual coefficients become unstable.

Changing the linear solver can help when the matrix is exactly singular. It does not automatically make every nearly singular explanation sensible.

Let's watch Cholesky, SVD, and Ridge respond to the same controlled data with [datarust](https://crates.io/crates/datarust).

## Two sensors measuring almost the same thing

Imagine two sensors observing the same latent process value.

Their readings are:

```text
sensor_0 = latent + 0.0001 × artifact
sensor_1 = latent - 0.0001 × artifact
```

The shared process signal is large relative to their disagreement. Add the readings and the artifact disappears:

```text
sensor_0 + sensor_1 = 2 × latent
```

Subtract them and only a tiny artifact remains:

```text
sensor_0 - sensor_1 = 0.0002 × artifact
```

The first training snapshot contains a target artifact with amplitude `0.20`:

```text
target_A = 10 + 4 × latent + 0.20 × artifact
```

The second snapshot changes that amplitude from `0.20` to `0.21`:

```text
target_B = 10 + 4 × latent + 0.21 × artifact
```

No target moves by more than `0.01`. The difference is tiny in prediction space. It is enormous in the nearly redundant coefficient direction.

I fit ordinary least squares with both SVD and Cholesky, then Ridge with `alpha = 1`. A separate clean test set contains the shared latent signal but a different sensor disagreement that should not affect the target.

Finally, I make the two feature columns exactly identical to show the boundary between near-collinearity and rank deficiency.

Here is the complete Rust program:

```rust
use datarust::linear_model::{
    LinearRegression, LinearSolver, Ridge,
};
use datarust::metrics::regression::{
    mean_squared_error, r2_score,
};
use datarust::traits::Predictor;
use datarust::Matrix;

const EPSILON: f64 = 1e-4;

fn training_snapshot(
    artifact_amplitude: f64,
) -> (Matrix, Vec<f64>) {
    let mut rows = Vec::new();
    let mut targets = Vec::new();

    for i in 0..120 {
        let latent = (i as f64 - 59.5) / 20.0;
        let artifact = (i as f64 * 0.73).sin();

        rows.push(vec![
            latent + EPSILON * artifact,
            latent - EPSILON * artifact,
        ]);
        targets.push(
            10.0
                + 4.0 * latent
                + artifact_amplitude * artifact,
        );
    }

    (Matrix::new(rows).unwrap(), targets)
}

fn clean_test_set() -> (Matrix, Vec<f64>) {
    let mut rows = Vec::new();
    let mut targets = Vec::new();

    for i in 0..80 {
        let latent = (i as f64 - 39.5) / 15.0;
        let new_artifact =
            (i as f64 * 1.17 + 0.4).cos();

        rows.push(vec![
            latent + EPSILON * new_artifact,
            latent - EPSILON * new_artifact,
        ]);
        targets.push(10.0 + 4.0 * latent);
    }

    (Matrix::new(rows).unwrap(), targets)
}

fn rmse(expected: &[f64], predicted: &[f64]) -> f64 {
    mean_squared_error(expected, predicted, false)
        .unwrap()
}

fn l2_delta(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (x_a, y_a) = training_snapshot(0.20);
    let (x_b, y_b) = training_snapshot(0.21);
    let (x_test, y_test) = clean_test_set();

    let max_target_change = y_a
        .iter()
        .zip(&y_b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    let mut ols_a = LinearRegression::new()
        .with_solver(LinearSolver::Svd);
    ols_a.fit(&x_a, &y_a)?;

    let mut ols_b = LinearRegression::new()
        .with_solver(LinearSolver::Svd);
    ols_b.fit(&x_b, &y_b)?;

    let mut chol_a = LinearRegression::new()
        .with_solver(LinearSolver::Cholesky);
    chol_a.fit(&x_a, &y_a)?;

    let mut ridge_a =
        Ridge::new().with_alpha(1.0);
    ridge_a.fit(&x_a, &y_a)?;

    let mut ridge_b =
        Ridge::new().with_alpha(1.0);
    ridge_b.fit(&x_b, &y_b)?;

    println!(
        "maximum target change: {:.6}\n",
        max_target_change,
    );
    println!(
        "model                 coefficient 0    \
         coefficient 1    L2 norm"
    );

    for (name, coef) in [
        ("OLS SVD snapshot A", ols_a.coef()),
        ("OLS SVD snapshot B", ols_b.coef()),
        ("OLS Cholesky A", chol_a.coef()),
        ("Ridge snapshot A", ridge_a.coef()),
        ("Ridge snapshot B", ridge_b.coef()),
    ] {
        let norm =
            (coef[0].powi(2) + coef[1].powi(2)).sqrt();
        println!(
            "{name:<22} {:+14.6} {:+16.6}   {:10.3}",
            coef[0], coef[1], norm,
        );
    }

    println!(
        "\ncoefficient change A -> B — OLS: {:.3}, \
         Ridge: {:.6}",
        l2_delta(ols_a.coef(), ols_b.coef()),
        l2_delta(ridge_a.coef(), ridge_b.coef()),
    );

    let ols_train = ols_a.predict(&x_a)?;
    let ridge_train = ridge_a.predict(&x_a)?;
    let ols_test = ols_a.predict(&x_test)?;
    let ridge_test = ridge_a.predict(&x_test)?;

    println!(
        "\nOLS   — training R2: {:.9}, \
         clean-test RMSE: {:.6}",
        r2_score(&y_a, &ols_train)?,
        rmse(&y_test, &ols_test),
    );
    println!(
        "Ridge — training R2: {:.9}, \
         clean-test RMSE: {:.6}",
        r2_score(&y_a, &ridge_train)?,
        rmse(&y_test, &ridge_test),
    );

    let exact_rows: Vec<Vec<f64>> = (0..20)
        .map(|i| {
            let value = i as f64 - 9.5;
            vec![value, value]
        })
        .collect();
    let exact_y: Vec<f64> = exact_rows
        .iter()
        .map(|row| 10.0 + 4.0 * row[0])
        .collect();
    let exact_x = Matrix::new(exact_rows)?;

    let mut exact_cholesky = LinearRegression::new()
        .with_solver(LinearSolver::Cholesky);
    let cholesky_result =
        exact_cholesky.fit(&exact_x, &exact_y);

    let mut exact_svd = LinearRegression::new()
        .with_solver(LinearSolver::Svd);
    exact_svd.fit(&exact_x, &exact_y)?;

    println!("\nexact duplicate columns:");
    println!(
        "Cholesky fit succeeded: {}",
        cholesky_result.is_ok(),
    );
    println!(
        "SVD coefficients: {:?}",
        exact_svd.coef(),
    );

    Ok(())
}
```

This is the output I measured:

```text
maximum target change: 0.010000

model                 coefficient 0    coefficient 1    L2 norm
OLS SVD snapshot A       +1002.000143      -998.000143     1414.217
OLS SVD snapshot B       +1052.000151     -1048.000151     1484.927
OLS Cholesky A           +1002.000194      -998.000194     1414.217
Ridge snapshot A            +1.996456        +1.994064        2.822
Ridge snapshot B            +1.996417        +1.993906        2.822

coefficient change A -> B — OLS: 70.711, Ridge: 0.000163

OLS   — training R2: 1.000000000, clean-test RMSE: 0.141117
Ridge — training R2: 0.999580857, clean-test RMSE: 0.014601

exact duplicate columns:
Cholesky fit succeeded: false
SVD coefficients: [2.0, 2.0]
```

The ordinary model won the training metric by fitting a direction that barely existed in the feature matrix.

Ridge declined most of that bargain.

## Why +1002 and -998 add up to a sensible prediction

Write the two OLS coefficients as `b0` and `b1`.

Their contribution is:

```text
b0 × (latent + epsilon × artifact)
+ b1 × (latent - epsilon × artifact)
```

Collect the two directions:

```text
(b0 + b1) × latent
+ epsilon × (b0 - b1) × artifact
```

To recover the first snapshot exactly, the model needs:

```text
b0 + b1 = 4
epsilon × (b0 - b1) = 0.20
```

Because `epsilon` is `0.0001`:

```text
b0 - b1 = 2000
```

Solving the sum and difference gives:

```text
b0 =  1002
b1 =  -998
```

The shared direction sees a calm combined slope of four. The tiny disagreement direction needs a coefficient difference of two thousand to reproduce an artifact with amplitude `0.20`.

Ordinary training rows make the cancellation look safe because both sensor values move together. A new disagreement pattern exposes the huge opposing weights.

## A tiny label change became a large coefficient change

Snapshot B raises the artifact amplitude by only `0.01`.

The shared requirement remains:

```text
b0 + b1 = 4
```

The disagreement requirement becomes:

```text
0.0001 × (b0 - b1) = 0.21
b0 - b1 = 2100
```

The new solution is:

```text
b0 =  1052
b1 = -1048
```

Each coefficient moves by about 50, even though no target changed by more than one hundredth.

This is coefficient variance in a very literal form. The data strongly determines the *sum* of the weights because it contains plenty of variation along the shared sensor direction. It weakly determines their *difference* because the sensor disagreement is only two ten-thousandths wide.

The least-squares objective sees both as solvable directions. It does not know that one is operational signal and the other is a fragile measurement artifact.

## SVD solved the system; it did not regularize it

The SVD solver is the robust choice when the design matrix is rank-deficient or close to it. In the exact duplicate-column example, Cholesky cannot factor the singular normal-equation matrix:

```text
Cholesky fit succeeded: false
```

SVD succeeds and returns the minimum-norm solution:

```text
[2.0, 2.0]
```

Their sum is four, so predictions are correct. Among infinitely many pairs whose sum is four, `[2, 2]` has the smallest Euclidean norm.

Our main dataset is different. Its two columns are *nearly* identical, not exactly identical. The tiny disagreement direction is still numerically present and above the solver's cutoff, so SVD fits it. Cholesky reaches almost the same coefficients:

```text
SVD:       +1002.000143, -998.000143
Cholesky:  +1002.000194, -998.000194
```

Choosing SVD protects the fitting process from outright rank deficiency. It does not add an L2 penalty, decide that small-variance directions are noise, or guarantee stable individual coefficients.

Numerical robustness and statistical regularization solve related but different problems.

## Ridge made the unstable direction expensive

Ridge minimizes:

```text
squared prediction error + alpha × sum(coefficient²)
```

The OLS pair has a coefficient norm of about `1414`. The Ridge pair has a norm of `2.822`:

```text
OLS:    +1002.000143, -998.000143
Ridge:     +1.996456,   +1.994064
```

Ridge gives both correlated sensors nearly half of the shared slope. Fitting the tiny artifact would require enormous opposing weights, so the small reduction in training residual is not worth the penalty.

The result gives up a sliver of training fit:

```text
OLS training R²:   1.000000000
Ridge training R²: 0.999580857
```

On the clean test target, that restraint helps:

```text
OLS RMSE:   0.141117
Ridge RMSE: 0.014601
```

This controlled example uses `alpha = 1` to make the mechanism visible. In a real project, I choose alpha inside cross-validation using the actual deployment metric. I do not infer a universal Ridge strength from one synthetic result.

## Good predictions do not make every coefficient interpretable

The OLS model's training predictions are exact, but neither coefficient deserves the story “sensor zero increases the outcome by 1002 while sensor one decreases it by 998.”

Those are conditional slopes:

> Change one sensor by one unit while holding the nearly identical other sensor fixed.

That intervention is far outside the relationship seen in ordinary rows. The sensors move together; a one-unit disagreement would be about ten thousand times the small artifact scale used in training.

The coefficient question is asking the model to imagine data in a direction with almost no support.

For prediction, cancellation may work as long as the correlation structure remains stable. For explanation, policy, or causal interpretation, unstable conditional coefficients are a serious warning.

I report correlated features as a group, inspect derived directions such as their mean and difference, or choose one measurement based on domain quality rather than attaching independent causal meaning to both fitted weights.

## Scaling alone cannot remove redundancy

The two sensors already use the same unit and nearly the same spread. Standardizing them would give each zero mean and unit variance, but their correlation would remain almost perfect.

Scaling is essential before Ridge when feature units differ because the penalty acts on coefficient magnitude. It makes the regularization cost comparable across columns.

It does not make duplicate information independent.

This distinction matters:

- Scaling repairs unit comparability.
- Feature selection can remove a redundant column.
- PCA can rotate correlated features into high- and low-variance directions.
- Ridge stabilizes estimation by penalizing large weights.
- None of them decides which sensor is causally meaningful.

I choose among those operations based on the task rather than applying a scaler and declaring multicollinearity solved.

## What I test when coefficients matter

A single train/test score can miss this problem because predictions may remain stable within the same correlation structure.

When coefficient interpretation or model stability matters, I also check:

1. Pairwise correlations and known duplicate measurements.
2. Coefficient signs and magnitudes across cross-validation folds.
3. Coefficient movement across nearby time windows or bootstrap samples.
4. Prediction behavior when correlated features disagree more than usual.
5. Whether removing either feature changes validation performance materially.
6. Whether a grouped or domain-derived feature tells the story more honestly.

The perturbation test in this article is especially simple: refit after a tiny, plausible data change and measure the coefficient delta. Moving from snapshot A to B changed OLS weights by `70.711` and Ridge weights by only `0.000163`.

Coefficient stability is not proof of causality, but spectacular instability is valuable evidence against a confident coefficient narrative.

## Sometimes one feature should simply leave

Regularization is not the only response.

If two columns are duplicate exports from the same source, I remove the accidental copy. If one sensor is cheaper, more reliable, or available earlier at prediction time, I may keep that one. If their average reduces measurement noise, I can create a domain-reviewed aggregate. If their disagreement is itself meaningful, I model it explicitly and validate whether it generalizes.

Writing the intended features directly makes the geometry clearer:

```text
sensor_mean       = (sensor_0 + sensor_1) / 2
sensor_difference = sensor_0 - sensor_1
```

Now a large coefficient on `sensor_difference` is visible as a dependence on a tiny, unstable direction rather than hidden inside two canceling raw coefficients.

Ridge is useful when many correlated variables all carry partial information. It should not replace basic schema cleanup or domain judgment.

## The coefficients exposed a direction the score hid

The most flattering number in this experiment was the perfect training R².

It concealed a model that spent coefficients of roughly `+1002` and `-998` to fit a small artifact living in the difference between two almost identical measurements. Change the artifact by at most `0.01`, and both weights move by about 50.

SVD solved that full-rank least-squares problem correctly. Cholesky found almost the same answer. Only when the columns became exactly identical did the solver distinction become decisive: Cholesky failed, while SVD returned the minimum-norm `[2, 2]` solution.

Ridge asked a different question. It traded a tiny amount of training fit for a much smaller, more stable coefficient pair and a lower error when the sensor disagreement stopped matching the training artifact.

So when two large coefficients nearly cancel, I no longer congratulate the model for discovering two powerful opposing effects. I ask:

> Which low-variance combination of features required those weights, and will that combination behave the same way after deployment?

In this experiment, the answer lived in a difference only `0.0002` wide.

The prediction score barely mentioned it. The coefficients shouted it.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
