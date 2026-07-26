# My New Enterprise Plan Looked Exactly Like Basic.

*A practical datarust guide to dropped one-hot categories, unknown values, all-zero collisions, and the model that could not distinguish “reference” from “never seen.”*

---

The new plan was called `enterprise`.

The model had never seen it during training, so the one-hot encoder ignored it and returned an all-zero row:

```text
enterprise -> [0.0, 0.0]
```

That seemed reasonable until I inspected the cheapest known plan:

```text
basic      -> [0.0, 0.0]
```

I had configured the encoder to drop the first category. `basic` was the sorted first category, so its all-zero representation was intentional. I had also configured unknown categories to be ignored, so `enterprise` received the same all-zero representation.

Two different meanings collapsed into one vector:

```text
known reference category -> all zeros
unknown category         -> all zeros
```

The downstream Ridge model had no remaining information with which to distinguish them. It predicted `10.055` support minutes for both.

Even `inverse_transform` decoded the unknown plan as `basic`.

Nothing crashed. The matrix shape was correct. The encoder and model both behaved according to their configuration. The ambiguity came from combining two individually valid policies.

Let's reproduce the collision with [datarust](https://crates.io/crates/datarust), then compare it with keeping all one-hot columns and with rejecting unknown categories explicitly.

## Three known plans and one new arrival

The training data contains three subscription plans:

```text
basic
pro
team
```

Each appears 20 times. The regression target represents expected monthly support minutes:

```text
basic -> 10
pro   -> 40
team  -> 90
```

The target values are deliberately simple. They make the information loss visible without asking the model to discover a complicated relationship.

At prediction time, four plans arrive:

```text
basic
enterprise
pro
team
```

`enterprise` did not exist during fitting.

We compare two tolerant encoders:

```text
DropStrategy::First + HandleUnknown::Ignore
DropStrategy::None  + HandleUnknown::Ignore
```

Then we try the strict `HandleUnknown::Error` policy.

Here is the complete Rust program:

```rust
use datarust::encoder::{
    DropStrategy, HandleUnknown, OneHotEncoder,
};
use datarust::linear_model::Ridge;
use datarust::traits::{FeatureNames, Predictor};
use datarust::StrMatrix;

fn training_data() -> (StrMatrix, Vec<f64>) {
    let mut plans = Vec::new();
    let mut targets = Vec::new();

    for (plan, support_minutes) in [
        ("basic", 10.0),
        ("pro", 40.0),
        ("team", 90.0),
    ] {
        for _ in 0..20 {
            plans.push(plan);
            targets.push(support_minutes);
        }
    }

    (StrMatrix::from_column(plans).unwrap(), targets)
}

fn run(
    name: &str,
    mut encoder: OneHotEncoder,
    train_plans: &StrMatrix,
    targets: &[f64],
    incoming: &StrMatrix,
) -> Result<(), Box<dyn std::error::Error>> {
    let x_train = encoder.fit_transform(train_plans)?;
    let x_incoming = encoder.transform(incoming)?;

    let mut model = Ridge::new().with_alpha(0.01);
    model.fit(&x_train, targets)?;
    let predictions = model.predict(&x_incoming)?;
    let decoded = encoder.inverse_transform(&x_incoming)?;
    let feature_names = encoder.feature_names_out(
        Some(&["plan".to_string()]),
    );

    println!("{name}");
    println!("encoded features: {feature_names:?}");

    for row in 0..incoming.nrows() {
        println!(
            "{:<10} -> {:?} -> decoded {:<7} \
             -> prediction {:6.3}",
            incoming.get(row, 0),
            x_incoming.row(row),
            decoded.get(row, 0),
            predictions[row],
        );
    }
    println!();

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (train_plans, targets) = training_data();
    let incoming = StrMatrix::from_column([
        "basic",
        "enterprise",
        "pro",
        "team",
    ])?;

    run(
        "drop first + ignore unknown",
        OneHotEncoder::new()
            .drop(DropStrategy::First)
            .handle_unknown(HandleUnknown::Ignore),
        &train_plans,
        &targets,
        &incoming,
    )?;

    run(
        "keep all + ignore unknown",
        OneHotEncoder::new()
            .handle_unknown(HandleUnknown::Ignore),
        &train_plans,
        &targets,
        &incoming,
    )?;

    let mut strict = OneHotEncoder::new()
        .handle_unknown(HandleUnknown::Error);
    strict.fit(&train_plans)?;
    let strict_result = strict.transform(
        &StrMatrix::from_column(["enterprise"])?
    );
    println!(
        "strict unknown policy accepts enterprise: {}",
        strict_result.is_ok()
    );

    Ok(())
}
```

Run it:

```sh
cargo run --release
```

This is the output I measured against the current datarust codebase:

```text
drop first + ignore unknown
encoded features: ["plan_pro", "plan_team"]
basic      -> [0.0, 0.0] -> decoded basic   -> prediction 10.055
enterprise -> [0.0, 0.0] -> decoded basic   -> prediction 10.055
pro        -> [1.0, 0.0] -> decoded pro     -> prediction 39.985
team       -> [0.0, 1.0] -> decoded team    -> prediction 89.960

keep all + ignore unknown
encoded features: ["plan_basic", "plan_pro", "plan_team"]
basic      -> [1.0, 0.0, 0.0] -> decoded basic   -> prediction 10.018
enterprise -> [0.0, 0.0, 0.0] -> decoded         -> prediction 46.667
pro        -> [0.0, 1.0, 0.0] -> decoded pro     -> prediction 40.003
team       -> [0.0, 0.0, 1.0] -> decoded team    -> prediction 89.978

strict unknown policy accepts enterprise: false
```

Dropping one category saved one column. It also removed the only representational difference between the known reference and the unknown value.

## “Drop first” creates a reference category

datarust sorts learned categories lexicographically. For our plan column, the order is:

```text
basic, pro, team
```

With `DropStrategy::First`, the encoder omits `basic` and emits two columns:

```text
plan_pro
plan_team
```

Known rows become:

```text
basic -> [0, 0]
pro   -> [1, 0]
team  -> [0, 1]
```

The all-zero vector does not mean “no plan.” It means “the known category chosen as the reference.”

For a linear model with an intercept, the reference category is absorbed into that intercept. The other coefficients describe differences relative to it. In our fitted Ridge model, the intercept sits near the `basic` target, while the `pro` and `team` columns add the remaining differences.

This is a standard and useful encoding. The problem begins when another semantic state is allowed to use the same zero vector.

## Ignoring an unknown also creates all zeros

We configured:

```rust
.handle_unknown(HandleUnknown::Ignore)
```

When `enterprise` is not found in the fitted vocabulary, datarust activates no known category column.

With the dropped encoder, that produces:

```text
enterprise -> [0, 0]
```

The exact same vector already belongs to `basic`.

By the time Ridge receives the matrix, the original string values are gone. It sees two identical rows and must return the same prediction:

```text
basic      -> 10.055
enterprise -> 10.055
```

This is not the model making an assumption that enterprise customers behave like basic customers. The encoder made the two cases indistinguishable before the model was called.

The prediction is inherited from the reference-category intercept because that is all the zero vector can express.

## Inverse transform proved the information was gone

The decoded values are even more direct:

```text
basic      -> decoded basic
enterprise -> decoded basic
```

For `DropStrategy::First`, an all-zero feature block is how `inverse_transform` recognizes the dropped category. It has no extra bit recording whether the original input was unknown.

Therefore decoding the unknown as `basic` is the only reconstruction consistent with the encoded matrix.

Inverse transformation is not a historical lookup. It can only recover distinctions preserved by the forward representation.

This matters beyond debugging. If an application encodes values, moves only the numeric matrix across a boundary, and later tries to reconstruct categories, unknown-vs-reference identity has already been lost.

No inverse API can restore information that was never stored.

## Keeping every column preserved the distinction

Without dropping a category, the encoder emits:

```text
plan_basic
plan_pro
plan_team
```

The same two rows now differ:

```text
basic      -> [1, 0, 0]
enterprise -> [0, 0, 0]
```

`inverse_transform` decodes `basic` correctly and returns an empty string for the unknown all-zero block. The empty value is not a prediction about the plan. It signals that none of the known categories was active.

The model predictions also separate:

```text
basic      -> 10.018
enterprise -> 46.667
```

That does not make `46.667` a trustworthy enterprise estimate.

The Ridge model never trained on an all-zero unknown row. With all three one-hot columns and an intercept, the zero vector receives only the fitted intercept. In this balanced example, regularization places it near the overall target mean.

Keeping all columns fixed the collision. It did not teach the model what a new category means.

That distinction is important:

> A unique representation for “unknown” prevents accidental equivalence. It does not create evidence about unknown behavior.

## Strict handling made uncertainty loud

The third configuration uses:

```rust
.handle_unknown(HandleUnknown::Error)
```

Transforming `enterprise` fails:

```text
strict unknown policy accepts enterprise: false
```

An error may be exactly right when an unseen value means:

- an upstream contract changed unexpectedly,
- a spelling or casing normalization broke,
- a deployment uses the wrong encoder artifact,
- a supposedly closed vocabulary was violated,
- or scoring the row would be unsafe.

It may be operationally unacceptable when new categories are a normal part of the product. A marketplace launches new sellers; a catalog gains products; countries, campaigns, and device models appear over time.

Unknown handling is not merely an encoder preference. It is a service policy balancing availability, observability, and semantic honesty.

## Why people drop a one-hot column

With three categories, keeping all three indicator columns produces this identity:

```text
plan_basic + plan_pro + plan_team = 1
```

If a separate intercept column is also present, the design matrix contains an exact linear dependency. This is commonly called the dummy-variable trap.

Dropping one category removes that redundancy and makes ordinary coefficient interpretation relative to a reference straightforward.

But the remedy depends on the estimator:

- Ridge regularization makes the linear system well behaved even with redundant indicator columns.
- An SVD-based least-squares solver can return a valid minimum-norm solution for rank-deficient input.
- Some estimators and coefficient-interpretation workflows still benefit from an explicit reference category.

I do not drop a category automatically because “one-hot encoding always needs it.” I decide whether the downstream estimator requires or benefits from the reference representation, then account for the unknown collision if tolerant handling is enabled.

Saving one column is rarely the decisive memory optimization. With high-cardinality data, sparse storage changes memory much more than dropping one category per original field.

## An explicit unknown indicator preserves both meanings

If I need a dropped reference category and tolerant unknown handling, I add a separate signal:

```text
plan_pro
plan_team
plan_is_unknown
```

Then the two cases can be represented as:

```text
basic      -> [0, 0, 0]
enterprise -> [0, 0, 1]
```

The model can distinguish them structurally.

Training still needs examples that make the unknown indicator learnable. If `plan_is_unknown` is always zero in training, a fitted coefficient cannot estimate what unknown plans will do. I may group rare training categories into an `OTHER` bucket, validate on deliberately held-out categories, route unknowns to a fallback model, or use a domain hierarchy such as plan family.

The indicator is valuable for monitoring even when the prediction falls back to a conservative baseline. It lets the service report how often the model is operating outside its fitted vocabulary.

## Refitting the encoder during serving is not a repair

It may be tempting to fit the encoder again after `enterprise` appears. That would create a new category column and eliminate the unknown state.

It would also change the feature schema expected by the existing model.

The old Ridge coefficients correspond to:

```text
[plan_pro, plan_team]
```

A refitted encoder might emit:

```text
[plan_enterprise, plan_pro, plan_team]
```

The width changes, or worse, positions shift while a downstream interface still accepts the matrix. The model and encoder are one fitted system. Their schemas cannot be revised independently.

The safe choices are:

- keep the fitted encoder frozen and apply the declared unknown policy,
- or retrain and validate a new encoder-plus-model artifact together.

Serving code calls `transform`. Training code calls `fit`.

## What I test for categorical production paths

For every categorical feature, I keep a small contract test containing:

- each known category,
- the dropped category if one exists,
- an unseen category,
- spelling and casing variants that should be normalized or rejected,
- missing or empty input according to the schema,
- output feature names and exact order,
- expected inverse behavior where inverse transformation is used.

When combining `DropStrategy::First` and `HandleUnknown::Ignore`, I assert the collision explicitly. If the application cannot tolerate it, that configuration should fail review rather than surprise us after launch.

In production, I monitor:

- unknown-category rate by feature,
- most common unknown values,
- prediction distribution for unknown rows,
- performance on known versus newly introduced categories when labels arrive,
- fitted vocabulary and normalization version,
- encoder and model artifact checksums.

A stable matrix width does not imply a stable categorical world.

## The all-zero row had two stories

For the dropped encoder, the numeric output looked simple:

```text
[0.0, 0.0]
```

It told two incompatible stories:

```text
This is the known basic plan.
This plan was never seen during fitting.
```

Ridge could not choose between them because the distinction had already disappeared. It returned the same `10.055` prediction. The inverse transform also returned `basic` for both.

Keeping all one-hot columns made the unknown vector distinct, although its intercept-only prediction remained an unsupported fallback. Strict handling refused to score it at all. An explicit unknown indicator could preserve availability and identity if the larger pipeline was designed around it.

So whenever I see an all-zero categorical block, I no longer assume it has one obvious meaning. I ask:

> Is this the dropped reference, an ignored unknown, or another state the representation failed to preserve?

In this experiment, `enterprise` arrived as a new product decision.

The matrix quietly introduced it to the model as `basic`.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
