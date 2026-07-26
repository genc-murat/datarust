# Linear Regression Learned a Curve. Then Degree 15 Left the Planet.

*A practical datarust guide to polynomial features, cross-validating model complexity, and why a good in-range score says almost nothing about extrapolation.*

---

“Linear regression” sounds like a promise to draw a straight line.

That is usually how it is introduced: one input, one coefficient, one intercept, and a line passing through a cloud of points. Useful, interpretable, a little boring.

Then you meet a relationship that is obviously curved.

Energy use is a good example. A building may consume more energy on very cold days because of heating and on very hot days because of cooling. Somewhere in the comfortable middle, consumption is lowest. Draw energy against outdoor temperature and you get something closer to a bowl than a line.

We could abandon linear regression for a more complicated model. Or we could change the features.

In this article, we'll use [**datarust**](https://crates.io/crates/datarust) to give linear regression polynomial inputs:

```text
temperature → temperature, temperature², temperature³, ...
```

The estimator remains linear in its coefficients. The prediction becomes nonlinear in temperature.

We will compare polynomial degrees 1 through 15, choose the degree with cross-validation, test it on untouched data, and then ask two models to predict just outside the temperature range they saw during training.

That last step is where the polite degree-15 model becomes a science-fiction author.

## The experiment: a noisy U-shaped energy curve

We generate 200 temperatures between `-10°C` and `40°C`. Energy use follows a quadratic relationship plus noise:

```text
energy = 60 + 0.35 × (temperature − 18)² + noise
```

The true minimum is near `18°C`. Cold and hot conditions both raise consumption.

We reserve 75% of the rows for the final test set, leaving only 50 training observations. That is intentionally uncomfortable: a high-degree polynomial has enough flexibility to chase noise in a small sample.

Create a Rust project:

```sh
cargo new polynomial_energy
cd polynomial_energy
cargo add datarust
```

Replace `src/main.rs` with this:

```rust
use datarust::linear_model::{LinearRegression, LinearSolver};
use datarust::metrics::regression::{
    mean_squared_error, r2_score,
};
use datarust::model_selection::{
    cross_val_score, KFold, TrainTestSplit,
};
use datarust::pipeline::Pipeline;
use datarust::polynomial::PolynomialFeatures;
use datarust::scaler::StandardScaler;
use datarust::traits::Predictor;
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

    fn normal(&mut self, sigma: f64) -> f64 {
        let u = self.next_f64().max(f64::MIN_POSITIVE);
        let v = self.next_f64();
        sigma
            * (-2.0 * u.ln()).sqrt()
            * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn build_model(
    degree: usize,
) -> datarust::pipeline::SupervisedPipeline<LinearRegression> {
    Pipeline::new()
        .push(
            "scale",
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        .push(
            "polynomial",
            TransformerKind::PolynomialFeatures(
                PolynomialFeatures::new(degree).include_bias(false),
            ),
        )
        .with_estimator(
            LinearRegression::new().with_solver(LinearSolver::Svd),
        )
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Rng::new(2026);
    let mut rows = Vec::new();
    let mut targets = Vec::new();

    for _ in 0..200 {
        let temperature = -10.0 + 50.0 * rng.next_f64();
        let energy = 60.0
            + 0.35 * (temperature - 18.0).powi(2)
            + rng.normal(15.0);

        rows.push(vec![temperature]);
        targets.push(energy);
    }

    let x = Matrix::new(rows)?;
    let (x_train, x_test, y_train, y_test) =
        TrainTestSplit::new()
            .with_test_size(0.75)
            .with_random_state(42)
            .split(&x, &targets)?;

    let cv = KFold::new()
        .with_n_splits(5)
        .with_shuffle(true)
        .with_random_state(7);

    let mut best_degree = 1;
    let mut best_cv = f64::NEG_INFINITY;

    println!("degree  features  mean CV R²");
    for degree in 1..=15 {
        let model = build_model(degree);
        let scores = cross_val_score(
            &model,
            &x_train,
            &y_train,
            &cv,
            r2_score,
        )?;
        let score = mean(&scores);

        println!("{degree:>6}  {degree:>8}  {score:>10.3}");

        if score > best_cv {
            best_degree = degree;
            best_cv = score;
        }
    }

    println!(
        "\nSelected degree: {best_degree} (CV R² {best_cv:.3})"
    );
    println!("\nHeld-out test set");
    println!("degree  test R²  test RMSE");

    for degree in [1, best_degree, 15] {
        let mut model = build_model(degree);
        model.fit(&x_train, &y_train)?;
        let predictions = model.predict(&x_test)?;

        println!(
            "{degree:>6}  {:>7.3}  {:>9.2}",
            r2_score(&y_test, &predictions)?,
            mean_squared_error(&y_test, &predictions, false)?,
        );
    }

    let mut selected = build_model(best_degree);
    selected.fit(&x_train, &y_train)?;

    let scenarios =
        Matrix::new(vec![vec![-5.0], vec![18.0], vec![35.0]])?;
    let predicted = selected.predict(&scenarios)?;

    println!("\nPredicted energy use");
    for row in 0..scenarios.nrows() {
        println!(
            "temperature={:>5.1} -> energy={:>7.2}",
            scenarios.get(row, 0),
            predicted[row],
        );
    }

    // These temperatures are outside the generated training range.
    let outside = Matrix::new(vec![vec![-20.0], vec![50.0]])?;
    println!(
        "\nExtrapolation outside the training range (-10 to 40)"
    );

    for degree in [2, 15] {
        let mut model = build_model(degree);
        model.fit(&x_train, &y_train)?;
        let predicted = model.predict(&outside)?;

        println!(
            "degree {degree:>2}: at -20 -> {:>10.2}, \
             at 50 -> {:>10.2}",
            predicted[0],
            predicted[1]
        );
    }

    Ok(())
}
```

Run it:

```sh
cargo run --release
```

With datarust v0.6 and the fixed seeds above, cross-validation chooses degree 2:

```text
degree  features  mean CV R²
     1         1      -0.107
     2         2       0.954
     3         3       0.948
     4         4       0.949
     5         5       0.943
     6         6       0.928
     7         7       0.935
     8         8       0.944
     9         9       0.944
    10        10       0.917
    11        11       0.853
    12        12       0.775
    13        13       0.839
    14        14       0.754
    15        15       0.902

Selected degree: 2 (CV R² 0.954)
```

The untouched test set confirms the choice:

```text
Held-out test set
degree  test R²  test RMSE
     1    0.192      69.18
     2    0.964      14.55
    15    0.952      16.82

Predicted energy use
temperature= -5.0 -> energy= 237.23
temperature= 18.0 -> energy=  62.34
temperature= 35.0 -> energy= 162.95
```

Degree 15 is worse, but not humiliating. Its in-range test `R²` is still `0.952`.

Then we move ten degrees beyond either side of the training range:

```text
Extrapolation outside the training range (-10 to 40)
degree  2: at -20 ->     543.97, at 50 ->     413.98
degree 15: at -20 ->  426449.48, at 50 -> -391763.26
```

The same degree-15 model that looked respectable on the test set predicts negative 391,763 units of energy at `50°C`.

That is not a typo. It is a polynomial doing exactly what polynomials do outside familiar territory.

## Linear refers to coefficients, not raw inputs

Ordinary one-feature linear regression predicts:

```text
y = β₀ + β₁x
```

Every prediction lies on a straight line in `x`.

Polynomial feature expansion changes the design matrix:

```text
[x] → [x, x², x³]
```

Linear regression now predicts:

```text
y = β₀ + β₁x + β₂x² + β₃x³
```

The equation is curved in `x`, but it remains linear in the learned parameters `β`. Fitting is still a linear least-squares problem. We changed the representation, not the estimator family.

This idea reaches beyond powers. Interaction terms, splines, Fourier features, and basis expansions all let a linear estimator describe nonlinear input relationships by making the relevant shapes explicit in the feature space.

The model can only learn the curves the feature map makes available.

## Degree 1 failed for the right reason

The degree-1 model gets test `R² = 0.192` and RMSE `69.18`.

It is trying to place one line across a U-shaped relationship. If the line slopes upward, it misses cold-weather demand. If it slopes downward, it misses hot-weather demand. A nearly flat line compromises badly on both sides.

This is underfitting. The model has too much bias: its feature space cannot express the true structure even with perfect coefficient estimates.

Adding `temperature²` fixes the structural problem:

```text
degree 2 test R²   = 0.964
degree 2 test RMSE = 14.55
```

The target was generated by a quadratic, so this result is unusually neat. Real energy demand may have separate heating and cooling slopes, occupancy effects, delayed thermal response, holidays, and control-system thresholds. A polynomial may approximate those patterns, but we should not expect nature to reveal its degree in the source code.

That is why the selection workflow matters more than this particular winner.

## Scaling before powers is numerical hygiene

The pipeline starts with `StandardScaler`:

```rust
Pipeline::new()
    .push(
        "scale",
        TransformerKind::StandardScaler(StandardScaler::new()),
    )
    .push(
        "polynomial",
        TransformerKind::PolynomialFeatures(
            PolynomialFeatures::new(degree).include_bias(false),
        ),
    )
```

Without scaling, `40¹⁵` is an enormous number. Columns such as `x`, `x²`, and `x¹⁵` can differ by many orders of magnitude, making the least-squares system poorly conditioned and coefficients difficult to estimate reliably.

After standardization, typical training values are closer to `-2` through `2`. Their powers are still capable of growing, but the numeric problem is much more manageable.

Scaling does not prevent extrapolation failure. A value beyond the training range still becomes a standardized value beyond the training range, and high powers still amplify it. Scaling helps the solver; it does not make the polynomial wise.

We set `include_bias(false)` because `LinearRegression` already fits an intercept. Keeping a constant `1` feature as well would duplicate the bias term and introduce unnecessary collinearity.

## Complexity grows faster with multiple inputs

With one input feature, degree 15 creates 15 output features. That is why the example prints the same number in the degree and feature columns.

With several inputs, `PolynomialFeatures` also generates interactions:

```text
[temperature, humidity]
    ↓ degree 2
[temperature, humidity, temperature²,
 temperature×humidity, humidity²]
```

The interaction term lets the effect of temperature depend on humidity. That can be valuable and makes the feature count grow combinatorially.

Ten original features at degree 3 produce far more than 30 columns because every pair and triple combination becomes a candidate. Memory, fitting time, collinearity, and overfitting risk all rise.

If only cross-feature interactions matter and repeated powers do not, datarust supports:

```rust
PolynomialFeatures::new(3)
    .interaction_only(true)
    .include_bias(false)
```

Feature generation should reflect a hypothesis, not a desire to make the matrix impressive.

## Cross-validation chose complexity without asking the test set

We use only the 50 training rows during degree selection:

```rust
let scores = cross_val_score(
    &model,
    &x_train,
    &y_train,
    &cv,
    r2_score,
)?;
```

For each degree, five-fold cross-validation repeatedly fits on four training folds and evaluates on the fifth. We compare mean validation `R²` and select degree 2.

The test set remains untouched until that choice is complete.

If we tried degrees 1 through 15 on the test rows and selected the best test score, the test set would become part of model tuning. Reporting the same score as final performance would be optimistic.

Putting scaling and polynomial generation inside the pipeline is important too. Each cross-validation clone fits its own scaler using that fold's training rows. Validation statistics do not leak into feature construction.

The degree itself is a hyperparameter. “Feature engineering” does not place it outside the rules of model selection.

## Why degree 15 still looked decent in range

A high-degree polynomial contains lower-degree polynomials inside its hypothesis space. Degree 15 can represent a quadratic by setting coefficients for powers 3 through 15 near zero.

It can also bend around noise.

Our degree-15 model has 15 coefficients and only 50 training rows. It still performs reasonably within `-10°C` to `40°C` because the underlying relationship is strong, the SVD solver is robust, and the test rows come from the same range as training.

Cross-validation sees the additional variance: the mean score falls from `0.954` at degree 2 to `0.902` at degree 15. The held-out test difference is smaller but points the same way.

This is a useful reminder that overfitting is not always a spectacular test failure. Sometimes it is a modest degradation, unstable coefficients, or extreme sensitivity that appears only near boundaries.

A good average in-range metric does not inspect every behavior the application cares about.

## Extrapolation revealed the monster

Inside the observed interval, many different curves can pass close to the data. Outside it, their highest powers dominate.

For degree 15:

```text
y ≈ ... + β₁₄x¹⁴ + β₁₅x¹⁵
```

Even small high-order coefficients become powerful when `|x|` grows beyond training values. Opposing terms can cancel inside the fitted interval and diverge violently outside it.

That is how a model with test `R² = 0.952` produces `-391,763.26` at `50°C`.

The degree-2 model extrapolates much more sensibly here because the true data-generating relationship is quadratic. In a real system, even its prediction outside the observed domain would be an assumption. The building might change control modes, equipment may saturate, or the quadratic trend may stop.

I would put domain checks around prediction:

- Record training ranges for every input.
- Flag or reject values outside an accepted envelope.
- Monitor how often inference leaves the fitted domain.
- Evaluate boundary scenarios explicitly, not only random test averages.
- Prefer a model with domain-appropriate shape constraints when possible.

No regression model earns extrapolation rights from an interpolation score.

## Regularization can help without granting wisdom

Ridge or Lasso regression can penalize unstable high-degree coefficients:

```rust
.with_estimator(Ridge::new().with_alpha(1.0))
```

Regularization is often useful when polynomial features are numerous or correlated. Ridge shrinks coefficients; Lasso may drive some terms to zero.

The penalty strength must be cross-validated together with polynomial degree. A large degree plus a carefully chosen Ridge penalty can work better than unregularized least squares.

It still does not make arbitrary extrapolation trustworthy. Regularization reduces coefficient variance. It does not tell the model what physics happens outside the observed range.

If monotonicity, non-negativity, saturation, or known asymptotes matter, those constraints should appear in the model design or application boundary rather than being wished onto an unconstrained polynomial.

## The practical result

The comparison is compact:

| Model | CV R² | Test R² | Test RMSE | At 50°C |
|---|---:|---:|---:|---:|
| Degree 1 | -0.107 | 0.192 | 69.18 | — |
| Degree 2 | 0.954 | 0.964 | 14.55 | 413.98 |
| Degree 15 | 0.902 | 0.952 | 16.82 | -391,763.26 |

Degree 2 wins because it captures the curve without acquiring fifteen ways to misbehave.

I would carry five habits from this experiment:

1. Treat polynomial expansion as part of the model, not harmless preprocessing.
2. Scale before generating high powers.
3. Select degree using validation or cross-validation.
4. Measure feature growth when multiple inputs create interactions.
5. Test outside and near the edges of the operating domain.

Linear regression was never limited to a straight line. It was limited to the features we handed it.

That flexibility is useful.

And, at degree 15, apparently imaginative.

```sh
cargo add datarust
```

---

*datarust is MIT-licensed and available on [crates.io](https://crates.io/crates/datarust). Documentation lives at [genc-murat.github.io/datarust](https://genc-murat.github.io/datarust/), with `PolynomialFeatures` covered in the [API reference](https://genc-murat.github.io/datarust/api.html).*
