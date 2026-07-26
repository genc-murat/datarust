# StandardScaler Made My Revenue Smaller. It Didn't Make It Linear.

*A practical datarust experiment with skewed data, Box–Cox, Yeo–Johnson, and a regression model that was asking the wrong feature to behave.*

---

There is a particular kind of histogram that appears in almost every commercial dataset I've worked with.

Most customers spend a modest amount. Some spend a lot. A tiny number spend an amount that makes the x-axis stretch across the room.

Revenue, account balance, session duration, order size, claim value — the column name changes, but the shape is familiar. It has a crowded left side and a long right tail.

The first time I met one of these columns, I reached for `StandardScaler`. The values were huge and awkward, so making them zero-mean and unit-variance felt like the responsible thing to do.

The numbers became smaller. The shape did not change.

That distinction matters. Standard scaling changes the units of a feature. It does not turn a curved relationship into a straight one, and it does not make a skewed distribution symmetric.

To make this concrete, I built a small regression experiment in [datarust](https://crates.io/crates/datarust). The result is a useful reminder that “scaled” and “transformed” are not interchangeable words.

## A deliberately long-tailed revenue column

The synthetic dataset has 500 accounts. Their revenue follows a log-normal distribution, producing values from cents to thousands of dollars. The target is approximately linear in the *logarithm* of revenue, with some random noise added.

That setup is deliberate. I want to isolate one question:

> What happens when the model is linear, but the useful representation of the feature is not?

The first 400 rows are training data and the remaining 100 are held out for testing. I compare three workflows:

1. Linear regression on raw revenue
2. `StandardScaler` followed by linear regression
3. A Box–Cox `PowerTransformer` followed by linear regression

Here is the complete program:

```rust
use datarust::linear_model::LinearRegression;
use datarust::metrics::regression::{mean_squared_error, r2_score};
use datarust::pipeline::Pipeline;
use datarust::scaler::{PowerMethod, PowerTransformer, StandardScaler};
use datarust::traits::{Predictor, Transformer};
use datarust::transformer_kind::TransformerKind;
use datarust::Matrix;

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn uniform(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        ((x >> 11) as f64 / (1_u64 << 53) as f64)
            .max(f64::MIN_POSITIVE)
    }

    fn normal(&mut self, sigma: f64) -> f64 {
        let u = self.uniform();
        let v = self.uniform();
        sigma
            * (-2.0 * u.ln()).sqrt()
            * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn skewness(x: &Matrix) -> f64 {
    let values = x.col(0);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let m2 = values
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let m3 = values
        .iter()
        .map(|v| (v - mean).powi(3))
        .sum::<f64>()
        / values.len() as f64;
    m3 / m2.powf(1.5)
}

fn report(
    name: &str,
    y: &[f64],
    predictions: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{name:20} R² = {:8.4}    RMSE = {:7.4}",
        r2_score(y, predictions)?,
        mean_squared_error(y, predictions, false)?,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Rng::new(42);
    let mut train_rows = Vec::new();
    let mut train_y = Vec::new();
    let mut test_rows = Vec::new();
    let mut test_y = Vec::new();

    for i in 0..500 {
        let log_revenue = 4.0 + rng.normal(1.3);
        let revenue = log_revenue.exp();
        let target = 25.0 + 8.0 * log_revenue + rng.normal(2.0);

        if i < 400 {
            train_rows.push(vec![revenue]);
            train_y.push(target);
        } else {
            test_rows.push(vec![revenue]);
            test_y.push(target);
        }
    }

    let x_train = Matrix::new(train_rows)?;
    let x_test = Matrix::new(test_rows)?;

    // Fit diagnostic transformers on training data only.
    let mut standard = StandardScaler::new();
    let x_standard = standard.fit_transform(&x_train)?;

    let mut power = PowerTransformer::new()
        .method(PowerMethod::BoxCox)
        .standardize(true);
    let x_power = power.fit_transform(&x_train)?;

    let min_revenue = x_train
        .col(0)
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_revenue = x_train
        .col(0)
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    println!(
        "Training revenue range: ${min_revenue:.2} to ${max_revenue:.2}"
    );
    println!("Raw skewness:      {:8.4}", skewness(&x_train));
    println!("Standard skewness: {:8.4}", skewness(&x_standard));
    println!("Box-Cox skewness:  {:8.4}", skewness(&x_power));
    println!("Learned lambda:    {:8.4}\n", power.lambdas()[0]);

    let mut raw_model = LinearRegression::new();
    raw_model.fit(&x_train, &train_y)?;
    let raw_pred = raw_model.predict(&x_test)?;

    let mut standard_model = Pipeline::new()
        .push(
            "scale",
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        .with_estimator(LinearRegression::new());
    standard_model.fit(&x_train, &train_y)?;
    let standard_pred = standard_model.predict(&x_test)?;

    let mut power_model = Pipeline::new()
        .push(
            "power",
            TransformerKind::PowerTransformer(
                PowerTransformer::new().method(PowerMethod::BoxCox),
            ),
        )
        .with_estimator(LinearRegression::new());
    power_model.fit(&x_train, &train_y)?;
    let power_pred = power_model.predict(&x_test)?;

    report("Raw revenue", &test_y, &raw_pred)?;
    report("StandardScaler", &test_y, &standard_pred)?;
    report("Box-Cox pipeline", &test_y, &power_pred)?;

    // The fitted transform is reversible.
    let x_test_power = power.transform(&x_test)?;
    let restored = power.inverse_transform(&x_test_power)?;
    let max_round_trip_error = x_test
        .col(0)
        .iter()
        .zip(restored.col(0))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "\nMax inverse-transform error: {max_round_trip_error:.2e}"
    );

    Ok(())
}
```

This is the output I got:

```text
Training revenue range: $0.17 to $2187.39
Raw skewness:        4.9941
Standard skewness:   4.9941
Box-Cox skewness:    0.0032
Learned lambda:      0.0233

Raw revenue          R² =   0.4323    RMSE =  7.7213
StandardScaler       R² =   0.4323    RMSE =  7.7213
Box-Cox pipeline     R² =   0.9566    RMSE =  2.1352

Max inverse-transform error: 3.41e-12
```

The two most revealing lines are the ones that did not move.

## StandardScaler changed the units, not the shape

Raw revenue has a skewness of `4.9941`. A symmetric distribution has skewness near zero, while a large positive value indicates a long right tail.

After standard scaling, the skewness is still `4.9941`.

That is not a bug. `StandardScaler` applies an affine transformation:

```text
z = (x - mean) / standard_deviation
```

Subtracting a constant and dividing by a positive constant can move and resize a distribution, but it cannot change its basic shape. The smallest and largest values are still separated by the same relative geometry. The long tail remains a long tail.

The regression result makes the same point from another angle. Raw revenue and standardized revenue both produce:

```text
R²   = 0.4323
RMSE = 7.7213
```

With one feature and an intercept, ordinary least squares can absorb the change of units into its coefficient. The coefficient will look different, but the fitted line and predictions are effectively the same.

Standard scaling can be essential for regularization, numerical optimization, and models that compare distances across features. It simply does not give a linear model a new shape to fit.

If the relationship curves, a smaller x-axis still curves.

## Box–Cox learned something very close to a logarithm

The Box–Cox family transforms a strictly positive value using a parameter called lambda:

```text
               x^λ - 1
BoxCox(x) =    --------    when λ is not zero
                   λ

BoxCox(x) =    ln(x)       when λ is zero
```

datarust estimates lambda from the training feature by maximum likelihood. In this experiment, it learned `0.0233` — very close to zero and therefore very close to a logarithmic transform.

That makes sense because the revenue values were generated from an exponentiated normal variable. Taking their logarithm approximately recovers the symmetric variable underneath.

The measured skewness falls from `4.9941` to `0.0032`. More importantly for this particular task, the target is linear in that hidden log-revenue value. Once the feature is represented in compatible coordinates, a plain linear regression becomes a good model:

```text
R²   = 0.9566
RMSE = 2.1352
```

Nothing about the estimator became more sophisticated. It is still fitting one straight line. The preprocessing made the useful relationship straight enough for that line to see.

## The transformer did not optimize R²

This distinction is easy to miss.

`PowerTransformer` never saw `train_y` while estimating lambda. It tried to make the feature more Gaussian-like; it did not search for the transformation that maximized predictive performance.

The large improvement happened because I deliberately generated a target that depends on log-revenue. The data-generating process and the learned transformation agree.

Real data is not usually this polite.

A more symmetric feature can help a linear model, stabilize variance, or reduce the leverage of a long tail. It can also do nothing useful. Tree models, for example, often gain little from a monotonic power transform because they split by order rather than fit one global slope.

So I would treat “the histogram looks Gaussian now” as a diagnostic, not a victory condition. The transformation still has to earn its place through cross-validation and held-out metrics that match the product objective.

## Why keep the transform inside the pipeline?

The Box–Cox lambda, transformed mean, and transformed standard deviation are learned parameters. They belong to the fitted model just as much as the regression coefficient does.

This is why the example constructs one supervised pipeline:

```rust
let mut model = Pipeline::new()
    .push(
        "power",
        TransformerKind::PowerTransformer(
            PowerTransformer::new().method(PowerMethod::BoxCox),
        ),
    )
    .with_estimator(LinearRegression::new());

model.fit(&x_train, &train_y)?;
let predictions = model.predict(&x_test)?;
```

The pipeline creates one path through training and prediction. It prevents an easy production mistake: learning a transform during training, then recreating it slightly differently in the serving code.

It also helps during cross-validation. Each fold must learn its own lambda from that fold's training rows. Fitting the transformer once on the full dataset would let validation values influence the representation, which is a quiet form of leakage.

## Box–Cox has a hard boundary at zero

Box–Cox requires every value to be strictly positive. Not “mostly positive.” Not “positive except for refunds.” Strictly greater than zero during both fitting and future transformation.

datarust rejects invalid input explicitly:

```rust
let revenue = Matrix::new(vec![
    vec![120.0],
    vec![0.0],
    vec![-15.0], // a refund
])?;

let mut box_cox = PowerTransformer::new()
    .method(PowerMethod::BoxCox);

box_cox.fit(&revenue)?; // returns an InvalidInput error
```

In the test program, the error was:

```text
invalid input: Box-Cox requires strictly positive data;
non-positive at col 0 row 1
```

Adding an arbitrary constant to make the column positive is possible, but it changes the meaning of the transformation and makes production assumptions easy to forget. If zero and negative values are legitimate, I would first consider Yeo–Johnson.

```rust
let mut power = PowerTransformer::new()
    .method(PowerMethod::YeoJohnson)
    .standardize(true);

let transformed = power.fit_transform(&revenue)?;
```

Yeo–Johnson is designed to handle positive, zero, and negative observations. It still learns a lambda, but uses different formulas on either side of zero.

That does not mean refunds and purchases automatically belong in one numeric feature. A negative transaction may have different business meaning, and a separate indicator or model may be clearer. The transform can handle the number; it cannot understand the accounting policy.

## Standardization is still happening

`PowerTransformer::new()` standardizes by default after applying the power function. In other words, the default workflow has two stages:

1. Apply the learned Box–Cox or Yeo–Johnson transformation
2. Center and scale the transformed values to zero mean and unit variance

You can disable the second stage with `.standardize(false)` if you want the raw power-transformed values.

This is also why I did not add a separate `StandardScaler` after `PowerTransformer` in the pipeline. With the default settings, that would be redundant.

The fitted transformation is reversible as well. Our round trip from original values to transformed values and back had a maximum floating-point error of only `3.41e-12`:

```rust
let z = power.transform(&x_test)?;
let original_units = power.inverse_transform(&z)?;
```

That is useful when transformed predictions, diagnostics, or stored features need to be interpreted in their original units. It does not recover information discarded elsewhere in a pipeline, but the power transformation itself is invertible within floating-point precision.

## When I would try a power transform

I reach for one when a feature is strongly skewed and there is a plausible multiplicative process underneath it — growth rates, monetary values, durations, sizes, concentrations, and similar quantities.

I am especially interested when residual variance grows with the feature, a few large values dominate a linear fit, or domain knowledge suggests that ratios matter more than absolute differences. Moving from $10 to $20 may be more comparable to moving from $100 to $200 than to moving from $100 to $110. A log-like transform can express that.

I am more cautious when zero has special meaning, positive and negative values represent different processes, absolute differences are the actual business quantity of interest, or the downstream model already handles nonlinear monotonic relationships well.

And I never choose it from the training histogram alone. The decision belongs inside a validation process.

## What this experiment actually proved

It did not prove that Box–Cox is always better than standard scaling. It proved something narrower and more useful:

- StandardScaler changed the magnitude of a skewed feature without changing its skewness.
- Scaling alone did not add nonlinear capacity to linear regression.
- Box–Cox learned a near-log transform from this training distribution.
- That representation matched the synthetic target's underlying relationship and sharply improved held-out performance.
- The learned transformation remained part of the model and had to be fitted on training data only.

The lesson I keep is simple: before asking a model to become more complicated, look at the coordinate system you gave it.

Sometimes the model is not failing to learn the relationship.

Sometimes we handed it revenue when the relationship was written in log-revenue all along.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the project on [crates.io](https://crates.io/crates/datarust).*
