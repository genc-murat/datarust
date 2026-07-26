# My Support Channels Became 0, 1, and 2. The Model Invented a Distance Between Them.

*A practical datarust comparison of OrdinalEncoder and OneHotEncoder — and why turning categories into numbers can quietly change what a feature means.*

---

Machine-learning models want numbers.

My data often has words.

At first glance, the bridge between those two facts looks almost embarrassingly simple. Give every category an integer:

```text
chat  -> 0
email -> 1
phone -> 2
```

The strings are gone. The matrix is numeric. The model fits. No error appears.

And that is exactly why this mistake survives so easily.

The numbers do more than identify the categories. They introduce an order and a distance. `email` is now one unit above `chat`. `phone` is one unit above `email` and two units above `chat`. A model may treat those relationships as real even though we created them by sorting words.

I wanted to see the damage rather than merely warn about it, so I built a small controlled experiment with [datarust](https://crates.io/crates/datarust).

## Three support channels with no natural order

Imagine that we are predicting how many minutes a support request will take to resolve. The only feature in this deliberately small example is the channel where the conversation began.

The average resolution times are:

```text
chat   ->  8 minutes
email  -> 30 minutes
phone  -> 12 minutes
```

There is no sensible numeric order here. Email happens to be slowest, phone is in the middle, and chat is fastest. The category names are identifiers, not measurements.

I generate 30 training observations and 10 test observations per channel, adding a small repeating noise pattern around each mean. Then I compare two encodings followed by the same `LinearRegression` model:

1. `OrdinalEncoder` with automatically inferred categories
2. `OneHotEncoder` with the first category dropped

Here is the complete Rust program:

```rust
use datarust::encoder::{
    DropStrategy, OneHotEncoder, OrdinalCategories, OrdinalEncoder,
};
use datarust::linear_model::LinearRegression;
use datarust::metrics::regression::{mean_squared_error, r2_score};
use datarust::traits::{FeatureNames, Predictor};
use datarust::StrMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let channels = ["chat", "email", "phone"];
    let means = [8.0, 30.0, 12.0];
    let noise = [-2.0, -1.0, 0.0, 1.0, 2.0];

    let mut train_channel = Vec::new();
    let mut train_y = Vec::new();
    let mut test_channel = Vec::new();
    let mut test_y = Vec::new();

    for (channel, mean) in channels.iter().zip(means) {
        for i in 0..30 {
            train_channel.push(*channel);
            train_y.push(mean + noise[i % noise.len()]);
        }
        for i in 0..10 {
            test_channel.push(*channel);
            test_y.push(mean + noise[i % noise.len()]);
        }
    }

    let train = StrMatrix::from_column(train_channel)?;
    let test = StrMatrix::from_column(test_channel)?;
    let checkpoints = StrMatrix::from_column(channels)?;

    let mut ordinal = OrdinalEncoder::new(OrdinalCategories::Auto);
    let x_train_ordinal = ordinal.fit_transform(&train)?;
    let x_test_ordinal = ordinal.transform(&test)?;
    let checkpoint_ordinal = ordinal.transform(&checkpoints)?;

    let mut ordinal_model = LinearRegression::new();
    ordinal_model.fit(&x_train_ordinal, &train_y)?;
    let ordinal_test_pred = ordinal_model.predict(&x_test_ordinal)?;
    let ordinal_checkpoint_pred =
        ordinal_model.predict(&checkpoint_ordinal)?;

    let mut one_hot = OneHotEncoder::new()
        .drop(DropStrategy::First);
    let x_train_one_hot = one_hot.fit_transform(&train)?;
    let x_test_one_hot = one_hot.transform(&test)?;
    let checkpoint_one_hot = one_hot.transform(&checkpoints)?;

    let mut one_hot_model = LinearRegression::new();
    one_hot_model.fit(&x_train_one_hot, &train_y)?;
    let one_hot_test_pred = one_hot_model.predict(&x_test_one_hot)?;
    let one_hot_checkpoint_pred =
        one_hot_model.predict(&checkpoint_one_hot)?;

    println!("Ordinal categories: {:?}", ordinal.categories()[0]);
    println!(
        "One-hot features:   {:?}\n",
        one_hot.feature_names_out(Some(&["channel".to_string()]))
    );

    println!(
        "channel   true mean   ordinal code   ordinal pred   one-hot pred"
    );
    for i in 0..channels.len() {
        println!(
            "{:<8}  {:9.2}  {:12.0}  {:13.2}  {:12.2}",
            channels[i],
            means[i],
            checkpoint_ordinal.get(i, 0),
            ordinal_checkpoint_pred[i],
            one_hot_checkpoint_pred[i],
        );
    }

    println!("\nOrdinalEncoder + LinearRegression");
    println!("R²:   {:.4}", r2_score(&test_y, &ordinal_test_pred)?);
    println!(
        "RMSE: {:.4}",
        mean_squared_error(&test_y, &ordinal_test_pred, false)?
    );

    println!("\nOneHotEncoder + LinearRegression");
    println!("R²:   {:.4}", r2_score(&test_y, &one_hot_test_pred)?);
    println!(
        "RMSE: {:.4}",
        mean_squared_error(&test_y, &one_hot_test_pred, false)?
    );

    Ok(())
}
```

This is the output from the run:

```text
Ordinal categories: ["chat", "email", "phone"]
One-hot features:   ["channel_email", "channel_phone"]

channel   true mean   ordinal code   ordinal pred   one-hot pred
chat           8.00             0          14.67          8.00
email         30.00             1          16.67         30.00
phone         12.00             2          18.67         12.00

OrdinalEncoder + LinearRegression
R²:   0.0285
RMSE: 9.5336

OneHotEncoder + LinearRegression
R²:   0.9786
RMSE: 1.4142
```

Both encoders produced valid numeric matrices. Only one preserved the meaning of the feature.

## Auto ordinal encoding sorted words, not business meaning

With `OrdinalCategories::Auto`, datarust learns the distinct values in each column and sorts them lexicographically. The learned order was:

```text
chat  = 0
email = 1
phone = 2
```

That behavior is deterministic and useful when I simply need stable codes. But alphabetical order does not become domain knowledge because it came from an encoder.

The resolution-time model sees one numeric feature and fits an equation of the form:

```text
predicted_minutes = intercept + coefficient * channel_code
```

One coefficient must explain every channel. Moving from code 0 to code 1 has exactly the same predicted effect as moving from code 1 to code 2.

Our fitted predictions reveal the constraint:

```text
14.67 -> 16.67 -> 18.67
```

They rise by exactly two minutes at each step. The model is unable to place email at 30 minutes while keeping chat at 8 and phone at 12, because that pattern is not a straight line through the codes `0, 1, 2`.

It compromises instead. The code runs perfectly while the representation prevents the model from expressing the answer.

## One-hot encoding gave every channel its own effect

`OneHotEncoder` asks a different question. It does not say that one channel is above or below another. It creates indicator columns:

```text
channel_email
channel_phone
```

Because the example uses `DropStrategy::First`, the lexicographically first category — `chat` — becomes the baseline. Its row is represented by zero in both columns:

```text
chat   -> [0, 0]
email  -> [1, 0]
phone  -> [0, 1]
```

The regression intercept learns the chat mean. The email coefficient learns how much email differs from chat, and the phone coefficient learns how much phone differs from chat.

Conceptually, the fitted equation becomes:

```text
minutes = 8 + 22 * is_email + 4 * is_phone
```

That representation can express `8`, `30`, and `12` without pretending the categories have equal spacing. On the noisy test rows, RMSE falls from `9.5336` to `1.4142`, which is exactly the scale of the noise we added.

The encoder did not discover that email is slow. It simply gave the model enough independent degrees of freedom to learn it.

## Why drop the first one-hot column?

With three categories, a full one-hot representation contains three columns. Every row has exactly one active value, so the columns always sum to one.

If a linear model also fits an intercept, one column is an exact combination of the others and the intercept. This is often called the dummy-variable trap: the design matrix is perfectly collinear.

Dropping one category avoids that redundancy:

```rust
let mut encoder = OneHotEncoder::new()
    .drop(DropStrategy::First);
```

It does not delete the baseline category's information. An all-zero block means the row belongs to that baseline, and every remaining coefficient is interpreted relative to it.

Not every estimator requires this. Regularized or pseudoinverse-based solvers can often work with all categories, and tree models are not fitting the same linear equation. datarust therefore keeps all categories by default. In this unregularized linear regression example, dropping the first category makes the parameterization explicit and full-rank.

## OrdinalEncoder is not the villain

The problem is not ordinal encoding. The problem is using ordinal encoding for a nominal feature.

Some categories really do have an order:

```text
low < medium < high < critical
```

For that feature, I can provide the order directly instead of trusting alphabetical sorting:

```rust
use datarust::encoder::{OrdinalCategories, OrdinalEncoder};

let mut severity_encoder = OrdinalEncoder::new(
    OrdinalCategories::Manual(vec![vec![
        "low".to_string(),
        "medium".to_string(),
        "high".to_string(),
        "critical".to_string(),
    ]]),
);
```

The resulting codes are meaningful in one sense:

```text
low      -> 0
medium   -> 1
high     -> 2
critical -> 3
```

But there is still a second assumption hiding in those values. A linear model treats the distance from low to medium as equal to the distance from high to critical.

The categories may be ordered without being equally spaced.

If that equal-step assumption is questionable, one-hot encoding may still be safer. Alternatively, domain experts can define meaningful scores, or the model can use a representation designed specifically for ordered effects. “Has an order” and “is a ruler” are not the same statement.

## The choice changes with the estimator

The failure in this experiment is especially sharp because linear regression turns one ordinal code into one global slope.

A decision tree can split an encoded feature at thresholds. With codes `0, 1, 2`, it might isolate the middle category through multiple splits, so it is not trapped by one straight line in the same way. Even then, the arbitrary ordering affects which categories are adjacent and which groupings are easy to express. A native categorical split or one-hot representation may better match the task.

Distance-based methods have another problem. With ordinal codes, phone is twice as far from chat as email is, purely because of the assigned integers. K-nearest neighbors and K-means can consume that invented geometry directly.

This is why I choose an encoder together with the estimator. There is no context-free answer called “convert strings to numbers.”

## Unknown categories also inherit the representation

By default, both encoders can report an error when production data contains a category that was absent during fitting. That may sound inconvenient, but an explicit failure is often safer than a quiet semantic guess.

`OrdinalEncoder` can encode unknown values as `-1` with `OrdinalHandleUnknown::UseNegOne`. A linear model will then interpret the unknown category as one numeric step below code 0. That is extrapolation, not neutrality.

`OneHotEncoder` can use `HandleUnknown::Ignore`, producing an all-zero block for an unseen value. When `DropStrategy::First` is also active, that representation is indistinguishable from the dropped baseline category.

Neither behavior can decide what a new support channel means. Should it inherit chat's expected time, receive a global fallback, trigger retraining, or be rejected upstream? That is a product and operations decision that belongs next to the model contract.

Whatever policy I choose, the encoder must be fitted on the training data and reused for validation, testing, and production. Re-fitting it later can change category indices or output columns while leaving the code syntactically valid.

## LabelEncoder is for the label

There is one naming trap worth calling out. `LabelEncoder` is intended for the target labels a model predicts, such as:

```text
fraud     -> 0
legitimate -> 1
```

It is not a shortcut for encoding input feature columns. For features, `OneHotEncoder`, `OrdinalEncoder`, `FrequencyEncoder`, and other feature-aware strategies make their assumptions explicit.

Using the right class name does not guarantee the right modeling choice, but it prevents two different roles from being blurred together.

## What the numbers made impossible to ignore

The ordinal workflow scored an R² of only `0.0285`. The one-hot workflow reached `0.9786` using the same rows, targets, estimator, and test split.

That difference did not come from a more powerful regression algorithm. It came from removing a relationship that never existed:

- Chat was not “zero support channel.”
- Phone was not twice as much channel as email.
- Alphabetical neighbors were not operational neighbors.
- Equal code differences were not equal time differences.

Encoding is part of the model, not clerical cleanup before the model begins.

Every time I replace a category with a number, I now ask one question before fitting anything:

> Is this number only an identifier, or am I asking the model to believe it is a measurement?

If I cannot defend the order and the distance, I do not give them to the model for free.

---

*The complete example and its reported metrics were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
