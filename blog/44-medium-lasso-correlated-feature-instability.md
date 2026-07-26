# I Changed One Sensor Reading by 0.05. Lasso Replaced the Sensor.

*A practical datarust guide to correlated features, unstable Lasso selection, coordinate-descent ties, identical predictions, and the sparse coefficient report that looked more certain than the data.*

---

The first model selected `sensor_a`.

```text
sensor_a coefficient: 1.5299
sensor_b coefficient: 0.0000
```

Then one training value was corrected:

```text
sensor_a, row 0: -0.95 -> -1.00
```

Nothing else changed. The target values stayed the same. The other 39 sensor cells stayed the same. The correction was `0.05` across a feature range of roughly `1.9`.

I refitted the same Lasso configuration.

This time it selected `sensor_b`:

```text
sensor_a coefficient: 0.0000
sensor_b coefficient: 1.5299
```

Both models had the same training `R²`: `0.986633`.

On rows where the sensors agreed, their predictions were identical to twelve decimal places.

The sparse coefficient table had changed its explanation without changing its demonstrated predictive behavior.

That was the uncomfortable part. I had been reading an exact zero as a confident scientific conclusion: sensor B does not matter. The model was saying something narrower: under this penalty, this sample, this column order, and this optimization path, one of two interchangeable columns was enough.

Let's reproduce the switch with [datarust](https://crates.io/crates/datarust), then separate predictive stability from feature-selection stability before a zero coefficient becomes a hardware decision.

## Two sensors measuring the same signal

Our controlled dataset contains 20 rows. A hidden signal runs from `-0.95` to `0.95` in steps of `0.1`, and the target is:

```text
target = 3 × signal
```

In the original snapshot, both sensor columns equal the signal exactly:

```text
sensor_a = signal
sensor_b = signal
```

They are duplicate features. Either one contains all available predictive information.

The revised snapshot changes only the first value of `sensor_a`:

```text
original: [-0.95, -0.95]
revised:  [-1.00, -0.95]
```

Now `sensor_b` remains a perfect copy of the hidden signal while `sensor_a` contains one small disagreement.

Both snapshots are standardized before Lasso fitting. We use `alpha = 0.2` to make the sparse choice visible. This alpha is a controlled experimental setting, not a claim that `0.2` is universally correct.

## The complete Rust experiment

Create a small application and add datarust:

```sh
cargo new lasso_stability
cd lasso_stability
cargo add datarust
```

Replace `src/main.rs` with this:

```rust
use datarust::linear_model::Lasso;
use datarust::metrics::regression::r2_score;
use datarust::scaler::StandardScaler;
use datarust::traits::{Predictor, Transformer};
use datarust::Matrix;

struct Fitted {
    scaler: StandardScaler,
    model: Lasso,
}

fn snapshot(
    revise_sensor_a: bool,
) -> (Matrix, Vec<f64>) {
    let mut rows = Vec::new();
    let mut targets = Vec::new();

    for i in 0..20 {
        let signal = (i as f64 - 9.5) / 10.0;
        let mut sensor_a = signal;
        if revise_sensor_a && i == 0 {
            sensor_a -= 0.05;
        }
        rows.push(vec![sensor_a, signal]);
        targets.push(3.0 * signal);
    }
    (Matrix::new(rows).unwrap(), targets)
}

fn fit(
    x: &Matrix,
    y: &[f64],
) -> Result<Fitted, Box<dyn std::error::Error>> {
    let mut scaler = StandardScaler::new();
    let scaled = scaler.fit_transform(x)?;

    let mut model = Lasso::new()
        .with_alpha(0.2)
        .with_max_iter(20_000)
        .with_tol(1e-10);
    model.fit(&scaled, y)?;

    Ok(Fitted { scaler, model })
}

fn predict(
    fitted: &Fitted,
    x: &Matrix,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let scaled = fitted.scaler.transform(x)?;
    Ok(fitted.model.predict(&scaled)?)
}

fn max_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (original_x, y) = snapshot(false);
    let (revised_x, _) = snapshot(true);

    let original = fit(&original_x, &y)?;
    let revised = fit(&revised_x, &y)?;

    let original_train =
        predict(&original, &original_x)?;
    let revised_train =
        predict(&revised, &revised_x)?;

    println!(
        "One-cell revision: sensor_a row 0, -0.95 -> -1.00"
    );
    println!(
        "Original coefficients: {:?}",
        original.model.coef(),
    );
    println!(
        "Revised coefficients:  {:?}",
        revised.model.coef(),
    );
    println!(
        "Original train R2: {:.6}",
        r2_score(&y, &original_train)?,
    );
    println!(
        "Revised train R2:  {:.6}",
        r2_score(&y, &revised_train)?,
    );

    let original_on_clean =
        predict(&original, &original_x)?;
    let revised_on_clean =
        predict(&revised, &original_x)?;
    println!(
        "Max prediction difference on clean duplicate rows: \
         {:.12}",
        max_difference(
            &original_on_clean,
            &revised_on_clean,
        ),
    );

    let disagreement = Matrix::new(vec![
        vec![0.5, 0.0],
        vec![0.0, 0.5],
    ])?;
    let original_disagreement =
        predict(&original, &disagreement)?;
    let revised_disagreement =
        predict(&revised, &disagreement)?;

    println!();
    println!("When the sensors disagree");
    println!(
        "  input [0.5, 0.0] -> original {:.6}, \
         revised {:.6}",
        original_disagreement[0],
        revised_disagreement[0],
    );
    println!(
        "  input [0.0, 0.5] -> original {:.6}, \
         revised {:.6}",
        original_disagreement[1],
        revised_disagreement[1],
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
One-cell revision: sensor_a row 0, -0.95 -> -1.00
Original coefficients: [1.529884389200619, 0.0]
Revised coefficients:  [0.0, 1.529884389200619]
Original train R2: 0.986633
Revised train R2:  0.986633
Max prediction difference on clean duplicate rows: 0.000000000000

When the sensors disagree
  input [0.5, 0.0] -> original 1.326578, revised 0.000000
  input [0.0, 0.5] -> original 0.000000, revised 1.326578
```

The coefficient report flips completely. The headline score does not move at all.

## Lasso pays for coefficient magnitude

Lasso minimizes an objective of the form:

```text
(1 / 2n) × squared prediction error
+ alpha × sum(abs(coefficients))
```

The second term is the L1 penalty. It encourages the model to set some coefficients exactly to zero, producing a sparse solution.

That is genuinely useful. When hundreds of weak or irrelevant features compete, sparsity can reduce complexity and create a compact model.

But sparsity answers an optimization question:

```text
Which coefficient vector gives a good fit at this L1 cost?
```

It does not automatically answer a scientific one:

```text
Which physical sensor is uniquely responsible for the outcome?
```

When two columns carry nearly the same information, the first question may have several almost equivalent answers.

## Exact duplicates create a flat choice

In the original snapshot:

```text
sensor_a = sensor_b
```

Predictions depend on the sum of their coefficients:

```text
prediction contribution
= beta_a × signal + beta_b × signal
=(beta_a + beta_b) × signal
```

If both coefficients are non-negative, any redistribution that preserves their sum also preserves the L1 penalty:

```text
abs(beta_a) + abs(beta_b)
= beta_a + beta_b
```

These examples therefore have the same prediction and the same total penalty:

```text
[1.5299, 0.0000]
[1.0000, 0.5299]
[0.7649, 0.7650]
[0.0000, 1.5299]
```

The data cannot identify separate effects for duplicate columns. Sparsity does not create that missing information.

## Coordinate order resolves a tie the data cannot

datarust solves Lasso by coordinate descent. During each sweep, it updates one coefficient at a time in column order.

With exact duplicates and zero-initialized coefficients, `sensor_a` is visited first. It can absorb the useful signal before `sensor_b` is updated. A valid sparse solution emerges:

```text
[1.529884..., 0.0]
```

That does not mean the first sensor is more predictive. In the original snapshot the columns are byte-for-byte equivalent.

Column order is acting as a deterministic tie resolver for information the sample cannot distinguish.

If I swap the physical column order while preserving the names correctly, the optimizer can select the other duplicate. That makes column-order perturbation a useful stability test for sparse models with highly correlated inputs.

I do not call this randomness. The algorithm is deterministic. The interpretation is unstable because the objective has equivalent or near-equivalent sparse explanations.

## One correction tilted the flat surface

The revised snapshot changes only this row:

```text
sensor_a: -0.95 -> -1.00
sensor_b: -0.95
target:   -2.85
```

`sensor_b` remains perfectly aligned with the hidden signal. `sensor_a` now contains a small measurement disagreement.

That tiny asymmetry is enough for Lasso to move to the alternative sparse solution:

```text
[0.0, 1.529884...]
```

The coefficient magnitude is the same because the selected `sensor_b` column and target are unchanged. Only the identity of the active feature changes.

This is what near-collinearity often looks like in production. The columns do not need to be exactly equal. They only need to be similar enough that sampling noise, cleaning rules, one corrected row, or one validation fold can make either one look marginally preferable.

## Prediction stability and selection stability are different

On the original duplicate rows, the models produce exactly the same predictions:

```text
maximum difference = 0.000000000000
```

Both also return:

```text
training R² = 0.986633
```

The score is below one because L1 regularization intentionally shrinks the active coefficient. Both snapshots make the same bias-for-sparsity tradeoff.

If my only question is prediction on rows where the sensors continue to agree, these models are behaviorally equivalent.

If my question is “Which sensor should we keep?”, they violently disagree.

That distinction belongs in every feature-selection report:

```text
predictive stability -> do scores and predictions persist?
selection stability  -> do the same features and signs persist?
```

A model can be stable on the first axis and unstable on the second.

## The disagreement rows reveal the hidden dependency

Duplicate training data cannot tell us what happens when the sensors diverge, so we create two diagnostic inputs:

```text
[sensor_a = 0.5, sensor_b = 0.0]
[sensor_a = 0.0, sensor_b = 0.5]
```

The original model follows only `sensor_a`:

```text
[0.5, 0.0] -> 1.326578
[0.0, 0.5] -> 0.000000
```

The revised model follows only `sensor_b`:

```text
[0.5, 0.0] -> 0.000000
[0.0, 0.5] -> 1.326578
```

These rows are not evidence that either model is right. We do not have labeled disagreement cases in training.

They are contract tests. They expose which source each artifact depends on when the redundancy assumption breaks.

That matters during outages, calibration drift, delayed feeds, sensor replacement, and schema migrations—the exact moments when duplicated features stop being duplicated.

## A zero coefficient does not prove irrelevance

In the original model, `sensor_b` receives an exact zero. Yet replacing `sensor_a` with `sensor_b` leaves all clean predictions unchanged.

So the defensible interpretation is:

```text
sensor_b adds little unique predictive information beyond sensor_a
```

It is not:

```text
sensor_b contains no predictive information
```

Those sentences differ by the word “unique.” Lasso evaluates a feature conditional on the other columns and the chosen penalty. Redundant information can be extremely predictive in isolation while receiving zero in a joint sparse model.

The same issue appears with:

- two vendors providing the same credit attribute
- gross and net amounts that move together
- temperature in Celsius and Fahrenheit
- multiple windows of the same rolling signal
- one-hot levels tied by data construction
- embeddings or principal components with duplicated downstream meaning

Sparse coefficients describe one fitted allocation of shared signal, not a universal ranking of standalone value.

## Cross-validation can hide the coefficient churn

Suppose five folds produce nearly identical validation RMSE values. If each fold selects a different member of a correlated group, the mean score can look reassuring while the deployed feature dependency remains unresolved.

A conventional hyperparameter search may record only:

```text
alpha -> mean validation score
```

For sparse models, I also record per fold:

- selected feature names
- coefficient values and signs
- selection count for each feature
- selection count for each correlated group
- prediction agreement on a shared diagnostic set

For example:

```text
sensor_a selected in 2/5 folds
sensor_b selected in 3/5 folds
at least one of the pair selected in 5/5 folds
```

The group signal is stable. The individual identity is not.

That is a much more faithful report than choosing the final all-data fit and declaring its winner to be the discovered truth.

## Bootstrap and perturbation tests make instability visible

Fold selection frequencies are one form of stability analysis. I also use controlled perturbations:

1. Refit on bootstrap or subsampled rows.
2. Repeat with plausible data-cleaning alternatives.
3. Reorder highly correlated columns.
4. Add measurement noise within known sensor tolerance.
5. Remove one row at a time for small, high-stakes datasets.
6. Compare multiple nearby alpha values selected by validation.

Then I inspect how often each feature, sign, and correlated group survives.

If a feature disappears after a harmless row reorder or tiny measurement perturbation while predictions remain stable, its individual selection should not drive an irreversible business decision.

Stability analysis does not prove causality either. It tells me how much the selected explanation depends on accidents of this sample and fitting path.

## Alpha controls more than the number of zeros

Our example fixes `alpha = 0.2` to isolate correlated-feature behavior.

In real work, alpha must be chosen inside cross-validation. A smaller value may distribute coefficients across both noisy copies. A larger value may keep only one or shrink the whole group too aggressively.

Selection frequency should therefore be evaluated across plausible alpha values, not only at one point estimate.

There is often a range of configurations whose validation scores are statistically indistinguishable. If their selected features differ, choosing the numerically best fourth decimal place can create false interpretive certainty.

I prefer the simplest configuration inside an explicitly defined tolerance only after checking that its dependencies are operationally acceptable.

## Scaling remains part of the fitted artifact

We standardize before Lasso because the L1 penalty acts on coefficient magnitude. Without comparable feature scales, units influence which coefficient is cheapest to keep.

Each snapshot learns its own `StandardScaler`, so the printed coefficients live in standardized feature units. They cannot be multiplied directly by raw sensor values without applying the fitted transformation.

That is why the example stores the scaler and Lasso model together in `Fitted` and always predicts through both:

```rust
let scaled = fitted.scaler.transform(x)?;
fitted.model.predict(&scaled)?
```

Persisting only the sparse coefficients would lose the means and standard deviations that define their input space.

Scaling solves the unit-comparability problem. It does not make correlated columns independently identifiable.

## Ridge changes the allocation, not the available information

Ridge uses an L2 penalty rather than L1. For duplicate or strongly correlated features, it tends to share weight rather than choosing one exact zero.

That can make individual coefficients less brittle and is often a better predictive default when many related measurements carry the same signal.

But shared Ridge coefficients still do not prove two independent causal effects. The data contains one direction of information represented twice. Ridge chooses a smooth allocation across that direction; Lasso chooses a sparse one.

If I need group-level sparsity—keep or remove a family of related features together—I need a method whose penalty expresses that goal. Ordinary Lasso does not know that two columns belong to one sensor family.

The current datarust model set provides Lasso and Ridge separately, not group Lasso or elastic net. I choose between the available objectives with that limitation visible.

## Domain knowledge can resolve what optimization cannot

If two features are duplicate exports of the same source, I remove the accidental copy before modeling.

If they are independent sensors measuring the same physical quantity, I ask operational questions:

- Which sensor is available earlier?
- Which one has better calibration and uptime?
- Are their errors independent enough that averaging helps?
- Does disagreement carry diagnostic value?
- Which source will exist in the production contract?
- What should happen if one becomes missing?

Possible feature designs include:

```text
average level
sensor difference
absolute disagreement
missingness indicators
one reviewed primary source
```

Those choices encode knowledge Lasso cannot infer from two columns that agree on every training row.

The model objective can tell me one column is sufficient for fitting this sample. It cannot tell me which hardware contract is safer.

## What I publish with a sparse model now

A coefficient table is still useful, but I publish it with context:

```text
alpha and scaling configuration
validation score distribution
selected features per fold
selection frequency
correlated-feature groups
coefficient ranges and sign stability
diagnostic predictions when redundant inputs disagree
data snapshot and schema version
```

I reserve words such as “important,” “necessary,” and “irrelevant” for claims supported by more than one sparse fit.

For the current example, the honest summary is:

```text
The pair carries stable predictive signal.
The individual selected sensor is unstable.
```

That sentence is slightly less exciting than “Lasso discovered sensor A.” It is much closer to what the experiment demonstrated.

## The prediction stayed; the explanation moved

One sensor reading changed by `0.05`. Lasso moved from:

```text
[1.529884, 0.0]
```

to:

```text
[0.0, 1.529884]
```

Training `R²` stayed at `0.986633`. Predictions on clean duplicate rows remained identical. Only when the sensors disagreed did the two artifacts reveal opposite dependencies.

Nothing about this behavior makes Lasso broken. Sparse optimization did exactly what we asked: keep a compact coefficient vector while fitting two interchangeable inputs.

The mistake would be reading the chosen member of a correlated pair as uniquely proven.

The safeguards are practical:

- scale features before comparing L1 penalties
- tune alpha inside validation folds
- inspect correlated groups before interpreting zeros
- report selection frequency, not only final selection
- perturb rows, columns, and cleaning choices
- test predictions where redundant features disagree
- use domain knowledge to define the production source
- distinguish stable predictions from stable explanations

Lasso can remove a coefficient exactly. The certainty of that zero is rarely exact.
