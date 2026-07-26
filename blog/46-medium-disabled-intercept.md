# The True Slope Was 2. I Disabled the Intercept and Learned 9.14.

*A practical datarust guide to regression through the origin, fixed baselines, misleading slopes, negative R², target centering, extrapolation, and the configuration flag that changed the question my model was allowed to answer.*

---

The relationship in my training data was exact:

```text
y = 50 + 2x
```

Every extra unit of `x` added two units to the target. Even when `x` was zero, the process still had a fixed baseline of 50.

The ordinary regression model recovered both facts:

```text
slope:     2.000000
intercept: 50.000000
```

Then I disabled the intercept.

I expected a slightly simpler model. What I got was a different model with a slope of `9.142857`.

At `x = 0`, it predicted `0` instead of `50`. At `x = 20`, it predicted `182.86` instead of `90`. Its training `R²` was `-15.233766` even though it was still an ordinary least-squares fit.

Nothing was wrong with the optimizer. I had forced the fitted line through `(0, 0)`, so the slope tried to do two jobs at once: represent the real two-unit change and compensate for a 50-unit baseline it was no longer allowed to express.

Let's reproduce that failure with [datarust](https://crates.io/crates/datarust), repair it when the baseline is genuinely known, and decide when `fit_intercept = false` is a valid model contract rather than an optimistic shortcut.

## An intercept is part of the question

A one-feature linear model with an intercept asks for two quantities:

```text
prediction = intercept + slope × x
```

The intercept is the predicted target when `x = 0`. The slope is the predicted change in the target for one additional unit of `x`.

Disabling the intercept changes the equation to:

```text
prediction = slope × x
```

This is not merely the same model with one less number in its report. It imposes an exact claim:

```text
when every feature is zero, the target must be zero
```

If that claim is false, the slope has to absorb as much of the missing offset as it can.

Our example could describe a system with a fixed 50-unit idle cost and a two-unit variable cost:

```text
total cost = fixed cost + variable cost × usage
```

Removing the intercept does not remove the fixed cost from reality. It only removes the model's ability to represent it separately.

## The complete Rust experiment

Create a small application and add datarust:

```sh
cargo new intercept_contract
cd intercept_contract
cargo add datarust
```

Replace `src/main.rs` with this:

```rust
use datarust::linear_model::LinearRegression;
use datarust::metrics::regression::{
    mean_absolute_error, r2_score,
};
use datarust::traits::Predictor;
use datarust::Matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Matrix::new(
        (1..=10).map(|value| vec![value as f64]).collect(),
    )?;
    let y: Vec<f64> = (1..=10)
        .map(|value| 50.0 + 2.0 * value as f64)
        .collect();

    let mut ordinary = LinearRegression::new();
    ordinary.fit(&x, &y)?;
    let ordinary_train = ordinary.predict(&x)?;

    let mut through_origin = LinearRegression::new()
        .with_fit_intercept(false);
    through_origin.fit(&x, &y)?;
    let origin_train = through_origin.predict(&x)?;

    // If 50 is a known physical or contractual baseline,
    // remove it before fitting through the origin.
    let y_after_baseline: Vec<f64> = y
        .iter()
        .map(|value| value - 50.0)
        .collect();
    let mut known_baseline = LinearRegression::new()
        .with_fit_intercept(false);
    known_baseline.fit(&x, &y_after_baseline)?;

    println!("training relationship: y = 50 + 2x");
    println!(
        "ordinary:       coef={:.6}, intercept={:.6}, \
         train R2={:.6}, train MAE={:.6}",
        ordinary.coef()[0],
        ordinary.intercept(),
        r2_score(&y, &ordinary_train)?,
        mean_absolute_error(&y, &ordinary_train)?,
    );
    println!(
        "through origin: coef={:.6}, intercept={:.6}, \
         train R2={:.6}, train MAE={:.6}",
        through_origin.coef()[0],
        through_origin.intercept(),
        r2_score(&y, &origin_train)?,
        mean_absolute_error(&y, &origin_train)?,
    );
    println!(
        "known baseline: coef={:.6} on y - 50",
        known_baseline.coef()[0],
    );

    let test_x = Matrix::new(vec![
        vec![0.0],
        vec![5.0],
        vec![10.0],
        vec![20.0],
    ])?;
    let expected = vec![50.0, 60.0, 70.0, 90.0];
    let ordinary_pred = ordinary.predict(&test_x)?;
    let origin_pred = through_origin.predict(&test_x)?;
    let restored_pred: Vec<f64> = known_baseline
        .predict(&test_x)?
        .into_iter()
        .map(|value| value + 50.0)
        .collect();

    println!();
    println!(
        "x    expected    ordinary    no intercept    \
         known baseline"
    );
    for i in 0..expected.len() {
        println!(
            "{:<4.0} {:>8.2} {:>11.2} {:>15.2} {:>17.2}",
            test_x.get(i, 0),
            expected[i],
            ordinary_pred[i],
            origin_pred[i],
            restored_pred[i],
        );
    }
    println!(
        "test R2: ordinary={:.6}, no intercept={:.6}, \
         known baseline={:.6}",
        r2_score(&expected, &ordinary_pred)?,
        r2_score(&expected, &origin_pred)?,
        r2_score(&expected, &restored_pred)?,
    );
    Ok(())
}
```

Run it:

```sh
cargo run --release
```

The output is:

```text
training relationship: y = 50 + 2x
ordinary:       coef=2.000000, intercept=50.000000, train R2=1.000000, train MAE=0.000000
through origin: coef=9.142857, intercept=0.000000, train R2=-15.233766, train MAE=19.285714
known baseline: coef=2.000000 on y - 50

x    expected    ordinary    no intercept    known baseline
0       50.00       50.00            0.00             50.00
5       60.00       60.00           45.71             60.00
10      70.00       70.00           91.43             70.00
20      90.00       90.00          182.86             90.00
test R2: ordinary=1.000000, no intercept=-12.469388, known baseline=1.000000
```

The same observations produce completely different fitted relationships because the two estimators are solving different constrained problems.

## Where 9.142857 came from

With an intercept, least squares can place the line at the correct height and estimate its change independently:

```text
y = 50 + 2x
```

Without an intercept, datarust finds the slope that minimizes:

```text
sum((yᵢ - slope × xᵢ)²)
```

The solution for a regression through the origin is:

```text
slope = sum(xᵢyᵢ) / sum(xᵢ²)
```

For `x = 1, 2, ..., 10` and `y = 50 + 2x`:

```text
slope
= sum(x(50 + 2x)) / sum(x²)
= 2 + 50 × sum(x) / sum(x²)
= 2 + 50 × 55 / 385
= 9.142857
```

The extra `7.142857` is not a newly discovered effect of `x`. It is the model's compromise for a missing constant.

This also reveals a dangerous property: the wrong slope depends on the distribution of the training feature. If the training range or usage mix changes, `sum(x) / sum(x²)` changes, so the fitted slope changes even when the real relationship remains `50 + 2x`.

A coefficient monitoring system might report slope drift. The underlying process has not drifted; the constrained model is reacting to a new distribution while trying to impersonate the missing intercept.

## The residuals tell a simple story

For the through-origin model, the residual is:

```text
actual - prediction
= (50 + 2x) - 9.142857x
= 50 - 7.142857x
```

That produces a strong pattern:

```text
small x  -> large positive residuals -> model predicts too low
large x  -> negative residuals       -> model predicts too high
```

The errors are not random noise around a sensible line. They rotate systematically across the feature range because the forced origin is wrong.

This is why I plot residuals against both predictions and original features. A single aggregate metric tells me the model is poor. The residual pattern tells me which structural assumption is poor.

## Negative R² is possible on the training data

It can feel contradictory that a least-squares model fitted on the training set has a negative training `R²`.

There is no contradiction.

The usual `R²` compares the model's squared error with a baseline that always predicts the mean target. A negative value means:

```text
model squared error > mean-prediction squared error
```

Ordinary least squares with a freely fitted intercept cannot do worse than that constant baseline on its own training data, because the model class contains the constant prediction. A regression constrained through the origin does not necessarily contain it.

In our example, the target mean is 61. The horizontal prediction `61` is imperfect but much better than a line forced to start at zero and swing past the data. The through-origin model is the best line inside its restricted model class; the restricted class itself is badly chosen.

Optimization success and model adequacy are different questions.

## The zero row is a contract test

The most revealing test case is often the simplest one:

```text
x = 0
```

The no-intercept prediction is necessarily zero:

```text
prediction = slope × 0 = 0
```

That remains true regardless of the training data, the learned slope, or the solver.

Before disabling an intercept, I write down what an all-zero feature row means after every preprocessing step:

- Does it represent a real idle system?
- Does it mean an average observation because the features were centered?
- Is it a padding or missing-data artifact?
- Can it occur in production?
- Should its target genuinely be zero?

If the row has no domain interpretation, the zero-origin constraint probably has no domain justification either.

Our controlled model predicts `0` for an idle system whose actual fixed cost is `50`. That one boundary test exposes the mismatch immediately.

## Standardizing X does not remove the need for an intercept

A common shortcut sounds like this:

```text
The features are standardized, so I can disable the intercept.
```

Feature centering and target centering are not the same operation.

If every feature column has mean zero, a no-intercept linear prediction also has mean zero on that fitted sample:

```text
mean(Xβ) = mean(X)β = 0
```

If the target mean is 61, the model still needs a way to represent that 61-unit level. Centering only `X` actually makes the missing target offset especially obvious.

datarust's default `LinearRegression::new()` handles this correctly. When intercept fitting is enabled, it centers both `X` and `y` internally, fits the slopes on the centered relationship, and recovers the intercept as:

```text
intercept = mean(y) - mean(X) · coefficients
```

You do not need to add a column of ones or manually center the data for ordinary use.

## When centering both X and y can work

There is a valid centered formulation:

```text
X_centered = X - training_X_mean
y_centered = y - training_y_mean
```

Fit a no-intercept model on those centered values, then reconstruct predictions:

```text
prediction = predicted_centered_y + training_y_mean
```

That is algebraically related to an intercept model. It is not the same as fitting `y` directly with the intercept disabled.

The training means also become fitted state. They must be learned only from training data, persisted with the model, and reused unchanged at inference. Recomputing them from a production request batch makes predictions depend on the other rows in that batch.

For ordinary linear regression, I prefer the built-in intercept because it keeps this bookkeeping inside one estimator. I use explicit centering only when another part of the workflow genuinely requires it.

## A known baseline is different from an unknown intercept

The third model in the experiment does use `with_fit_intercept(false)`, but it asks a defensible question.

Suppose the 50-unit baseline is fixed by a physical calibration or a contractual tariff rather than estimated from this dataset. We transform the target:

```text
target_above_baseline = target - 50
```

The relationship becomes:

```text
target_above_baseline = 2x
```

That relationship genuinely passes through the origin. The no-intercept model learns a slope of `2`, and inference restores the known baseline:

```text
final prediction = 50 + predicted_above_baseline
```

This is not a trick for improving the metric. It is an explicit decomposition of known and learned behavior.

The value `50` is now part of the production artifact and needs the same discipline as any model parameter:

- document where it came from
- version it
- use the same units during training and inference
- test the restoration step
- update it through a reviewed release when the tariff or calibration changes

If the baseline is not truly known, let the model estimate an intercept instead of hard-coding a convenient number.

## An explicit constant column needs care

Another way to represent a bias is to add a feature whose value is always one:

```text
[x] -> [1, x]
```

Then a model with `fit_intercept = false` can learn the coefficient of that constant column as a bias.

For unregularized regression, this can be mathematically equivalent to fitting an intercept. It is usually less clear than using the estimator's built-in option, and it becomes more delicate with regularization.

Ridge and Lasso commonly treat fitted intercepts differently from ordinary feature coefficients. If I turn the intercept into a normal constant feature, its coefficient may be penalized along with everything else. That shrinks the baseline and can again push compensating error into other coefficients.

I only use an explicit bias feature when I have checked the exact estimator and penalty semantics. A column of ones is an implementation choice, not universally interchangeable with an unpenalized intercept.

## Sometimes zero is outside the observed range

Our training inputs run from `1` to `10`. The value `x = 0` is just outside that range, so the intercept has a fairly direct interpretation.

In other datasets, zero may be far away from every observed feature value. A temperature measured in Kelvin, a timestamp measured from Unix epoch, or a property size in square meters can make the raw intercept an extreme extrapolation.

That can make the intercept numerically awkward or hard to interpret, but it does not justify forcing it to zero.

A better response is often to use meaningful reference values:

```text
temperature_above_20C = temperature - 20
days_since_launch = date - launch_date
size_above_100m2 = size - 100
```

With an intercept still enabled, the intercept now describes the prediction at a meaningful reference point. The fitted predictions remain equivalent under an exact linear reparameterization, while the reported numbers become easier to reason about.

Reference centering improves interpretation. It does not require a false origin constraint.

## Units can hide the bad assumption

The equation `prediction = 0` at an all-zero row applies in the units presented to the model.

If a preprocessing step converts Celsius to Kelvin, subtracts a reference value, or scales a feature, the semantic meaning of numeric zero changes. A no-intercept model that was defensible before a transformation may not be defensible after it, or vice versa.

The contract therefore belongs to the complete pipeline:

```text
raw input
-> parsing and units
-> feature transformations
-> numeric zero presented to estimator
-> target transformation
-> prediction restoration
```

I test the pipeline's zero and reference rows, not only the raw model's zero vector.

## Cross-validation will not repair the constraint

Cross-validation can reveal that a no-intercept model performs badly. It cannot make an incorrect origin assumption correct.

There is another subtle failure mode: if `x` stays within a narrow positive range, the forced line may approximate the data well enough inside that range to look acceptable. Its slope is still mixing baseline and marginal effect, and predictions can fail badly when the production range moves toward zero or beyond the training maximum.

My validation set therefore reflects the deployment envelope:

- low-usage and idle cases
- ordinary in-range cases
- expected high-usage cases
- important reference points
- future periods if the feature distribution changes over time

Random folds drawn from the same narrow cloud may miss the exact boundary where the false constraint becomes operationally expensive.

## A good metric does not make the coefficient causal

Even with the intercept configured correctly, the coefficient `2` is a fitted association unless the design and assumptions justify a causal interpretation.

The intercept mistake adds a more immediate problem: it can make the association itself hard to interpret. Our no-intercept slope `9.142857` is partly the real local change and partly a distribution-dependent attempt to pay the fixed baseline.

I do not describe that coefficient as “each unit of usage adds 9.14 units of cost.” The model's own residuals contradict that story.

When coefficient interpretation matters, I check:

- whether an intercept or known offset was represented correctly
- units and transformations
- confounding and omitted variables
- collinearity
- regularization
- the feature range supporting the estimate
- stability across relevant data slices

The intercept is one assumption in a much larger interpretation contract, but it is an unusually easy one to test.

## Persist the configuration, not only the coefficients

Two artifacts can contain a similar-looking coefficient vector and still implement different equations because one adds an intercept and the other does not.

I record at least:

```text
model family and library version
fit_intercept setting
coefficient order
intercept value
feature transformations and learned state
target transformation and inverse transformation
units and reference values
training data version
```

For the known-baseline version, I also store the fixed `50` and an explicit instruction to add it back after prediction.

A serving implementation that loads only `coef()` but forgets `intercept()` silently turns an ordinary regression into a through-origin regression. A deployment parity test should compare complete predictions from the training and serving paths, including a zero or reference row.

## Tests I keep beside an intercept decision

When the intercept setting is intentional, I make that intent executable.

Useful tests include:

```text
default model:
    prediction(x) = intercept + dot(coefficients, x)

no-intercept model:
    intercept == 0
    prediction(all-zero row) == 0

known-baseline model:
    final prediction(all-zero row) == known baseline
    target restoration is applied exactly once

all models:
    feature units match training
    reference cases stay within tolerance
    serialized and in-memory predictions match
```

I also compare residuals by feature range. If low values are consistently underestimated and high values consistently overestimated, I revisit the origin constraint before tuning more elaborate hyperparameters.

## When I would actually disable the intercept

I use `with_fit_intercept(false)` only when at least one of these statements is deliberately true:

- domain knowledge requires the relationship to pass through the origin in the model's exact units
- a known baseline has been removed from the target and will be restored after prediction
- both features and target were centered with persisted training statistics, and reconstruction is part of the pipeline
- an explicit bias representation is being managed intentionally and its penalty behavior is understood

“The model is simpler,” “the features were scaled,” and “the intercept was not statistically significant” are not sufficient reasons by themselves.

A false hard constraint can add far more error than the single removed parameter could ever save.

## The flag changed the meaning of the slope

The original data said:

```text
fixed baseline = 50
marginal change = 2 per unit of x
```

With its default intercept, datarust recovered exactly that relationship.

After I called `.with_fit_intercept(false)`, the model was required to predict zero at zero. It responded by moving the slope from `2` to `9.142857`. That coefficient changed with the training distribution, produced a negative training `R²`, missed the true zero-input target by 50, and overshot the target at `x = 20` by more than 92.

The repair was not a better solver. It was a correct model contract:

```text
unknown baseline -> fit an intercept
known baseline   -> subtract it, fit the remainder, restore it
true zero origin -> disable the intercept and test that claim
```

My safeguards are now simple:

- treat `fit_intercept` as a modeling assumption, not a cosmetic switch
- define what the all-zero row means after preprocessing
- inspect residuals across the feature range
- include boundary and reference rows in validation
- distinguish feature centering from target centering
- persist offsets, transformations, and intercept settings together
- compare full serving predictions, not only coefficient arrays

An intercept is just one number. Removing it can change the meaning of every other number in the model.
