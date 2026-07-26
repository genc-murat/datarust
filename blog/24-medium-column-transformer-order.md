# I Reordered Two ColumnTransformer Blocks. The Model Still Ran.

*A practical datarust guide to transformed feature order, remainder columns, and the four-column matrix whose shape proved almost nothing.*

---

The request was valid.

```text
numeric columns: 4
missing values:  0
all values:      finite
```

The model also expected four columns. Prediction completed without an error.

It returned `219.752` instead of `384.400`.

I had not changed the raw request. I had not changed the model weights. I had only rebuilt a `ColumnTransformer` with two block declarations in the opposite order.

That changed the transformed schema from:

```text
[area_sq_ft, rooms, age_years, tax_rate]
```

to:

```text
[age_years, area_sq_ft, rooms, tax_rate]
```

Both outputs contained one row and four floating-point values. The estimator had no way to know that the first three meanings had moved.

`ColumnTransformer` is often introduced as a convenient way to apply different preprocessing to different columns. It is also a schema compiler. The order of its blocks, the order of indices inside each block, every categorical expansion, and the remainder policy together define the positional interface consumed by the model.

Let's make that interface fail in a controlled example with [datarust](https://crates.io/crates/datarust).

## One raw schema, two transformed schemas

The raw property table has four numeric columns:

```text
column 0: area_sq_ft
column 1: age_years
column 2: rooms
column 3: tax_rate
```

The training preprocessor contains two explicit blocks:

1. Standardize area and rooms together.
2. Standardize age separately.

`tax_rate` is not selected by either block, so `Remainder::Passthrough` appends it unchanged at the end.

The training output therefore becomes:

```text
block 1                block 2      remainder
[area_z, rooms_z]  +   [age_z]  +   [tax_rate]
```

I train linear regression on that matrix. Then I create a second, independently fitted preprocessor with the age block first:

```text
block 1      block 2                remainder
[age_z]  +   [area_z, rooms_z]  +   [tax_rate]
```

This second fit is deliberately wrong for serving. It isolates the effect of configuration drift while keeping learned means and scales based on the same training rows.

Here is the complete Rust program:

```rust
use datarust::compose::{
    ColumnTransformer, Remainder, Table,
};
use datarust::linear_model::LinearRegression;
use datarust::metrics::regression::r2_score;
use datarust::scaler::StandardScaler;
use datarust::traits::{FeatureNames, Predictor};
use datarust::transformer_kind::TransformerKind;
use datarust::Matrix;

fn make_preprocessor(age_first: bool) -> ColumnTransformer {
    let base = ColumnTransformer::new()
        .remainder(Remainder::Passthrough);

    if age_first {
        base.add_numeric(
            "age",
            vec![1],
            TransformerKind::StandardScaler(
                StandardScaler::new(),
            ),
        )
        .add_numeric(
            "size_and_layout",
            vec![0, 2],
            TransformerKind::StandardScaler(
                StandardScaler::new(),
            ),
        )
    } else {
        base.add_numeric(
            "size_and_layout",
            vec![0, 2],
            TransformerKind::StandardScaler(
                StandardScaler::new(),
            ),
        )
        .add_numeric(
            "age",
            vec![1],
            TransformerKind::StandardScaler(
                StandardScaler::new(),
            ),
        )
    }
}

fn row(x: &Matrix, i: usize) -> Vec<f64> {
    (0..x.ncols()).map(|j| x.get(i, j)).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    let mut targets = Vec::new();

    for i in 0..240 {
        let area =
            600.0 + ((i * 37) % 2_400) as f64;
        let age = ((i * 17) % 60) as f64;
        let rooms =
            1.0 + ((i * 11) % 5) as f64;
        let tax_rate =
            0.010 + ((i * 7) % 21) as f64 * 0.001;
        let price = 50.0
            + 0.20 * area
            - 2.5 * age
            + 18.0 * rooms
            + 300.0 * tax_rate;

        rows.push(vec![area, age, rooms, tax_rate]);
        targets.push(price);
    }

    let train =
        Table::from_numeric(Matrix::new(rows)?);
    let input_names: Vec<String> = [
        "area_sq_ft",
        "age_years",
        "rooms",
        "tax_rate",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let mut training_preprocessor =
        make_preprocessor(false);
    let x_train =
        training_preprocessor.fit_transform(&train)?;

    let mut model = LinearRegression::new();
    model.fit(&x_train, &targets)?;
    let train_predictions = model.predict(&x_train)?;

    let request = Table::from_numeric(Matrix::new(vec![
        vec![1_500.0, 10.0, 3.0, 0.018],
    ])?);
    let expected = 50.0
        + 0.20 * 1_500.0
        - 2.5 * 10.0
        + 18.0 * 3.0
        + 300.0 * 0.018;

    let correct_features =
        training_preprocessor.transform(&request)?;
    let correct_prediction =
        model.predict(&correct_features)?[0];

    let mut reordered_preprocessor =
        make_preprocessor(true);
    reordered_preprocessor.fit(&train)?;
    let reordered_features =
        reordered_preprocessor.transform(&request)?;
    let reordered_prediction =
        model.predict(&reordered_features)?[0];

    println!(
        "training R2: {:.12}\n",
        r2_score(&targets, &train_predictions)?,
    );
    println!(
        "training output schema:  {:?}",
        training_preprocessor
            .feature_names_out(Some(&input_names)),
    );
    println!(
        "reordered output schema: {:?}\n",
        reordered_preprocessor
            .feature_names_out(Some(&input_names)),
    );
    println!(
        "same raw request: [1500.0, 10.0, 3.0, 0.018]"
    );
    println!(
        "training-order features:  {:?}",
        row(&correct_features, 0),
    );
    println!(
        "reordered features:       {:?}\n",
        row(&reordered_features, 0),
    );
    println!(
        "expected prediction:            {:9.3}",
        expected,
    );
    println!(
        "training-order prediction:      {:9.3}",
        correct_prediction,
    );
    println!(
        "reordered-schema prediction:    {:9.3}",
        reordered_prediction,
    );
    println!(
        "absolute error after reorder: {:9.3}",
        (reordered_prediction - expected).abs(),
    );

    Ok(())
}
```

This is the output I measured:

```text
training R2: 1.000000000000

training output schema:  ["area_sq_ft", "rooms", "age_years", "tax_rate"]
reordered output schema: ["age_years", "area_sq_ft", "rooms", "tax_rate"]

same raw request: [1500.0, 10.0, 3.0, 0.018]
training-order features:  [-0.32852297129381003, -1.25607396694702e-15, -1.125989423201406, 0.018]
reordered features:       [-1.125989423201406, -0.32852297129381003, -1.25607396694702e-15, 0.018]

expected prediction:              384.400
training-order prediction:        384.400
reordered-schema prediction:      219.752
absolute error after reorder:     164.648
```

The first prediction path used the preprocessor that trained the model. The second used a perfectly valid preprocessor with a different output contract.

Only one of them was compatible with the weights.

## ColumnTransformer follows declaration order

The raw input positions do not determine the final output order by themselves.

For datarust's `ColumnTransformer`, concatenation follows this sequence:

1. Transform each declared spec in insertion order.
2. Preserve the column-index order written inside that spec.
3. Expand a categorical spec into its learned encoded columns at that position.
4. Append unused numeric remainder columns in their original order when passthrough is enabled.
5. Append encoded unused categorical remainder columns afterward.

The training declaration starts with:

```rust
.add_numeric("size_and_layout", vec![0, 2], ...)
```

That is why area and rooms become output columns zero and one even though age sat between them in the raw table.

The next declaration selects age:

```rust
.add_numeric("age", vec![1], ...)
```

Age becomes output column two. Finally, unselected tax rate becomes column three through `Remainder::Passthrough`.

`ColumnTransformer` did exactly what its configuration requested. The failure came from assuming it would restore original source order after applying the blocks.

## The model learned positions, not names

Linear regression fit one coefficient for each transformed position:

```text
coefficient 0 belongs to area_z
coefficient 1 belongs to rooms_z
coefficient 2 belongs to age_z
coefficient 3 belongs to tax_rate
```

The reordered matrix placed `age_z` in position zero. The model multiplied it by the area coefficient. It interpreted `area_z` as rooms and `rooms_z` as age.

All the Rust types remained correct:

- both values were `Matrix`,
- both had one row,
- both had four columns,
- every cell was a finite `f64`.

The semantic type was wrong, but that information no longer existed inside the matrix.

This is a recurring boundary in numeric machine learning. Once named fields become a flat row, compatibility depends on a positional contract that the estimator cannot rediscover.

## Scaling did not cause the mismatch

Both preprocessors were fitted on the same 240 training rows. Each `StandardScaler` learned the correct mean and standard deviation for its selected source columns.

That is why the same three standardized values appear in both outputs:

```text
-0.3285229713
-0.0000000000
-1.1259894232
```

They are merely permuted.

The problem would remain if the blocks used min-max scaling, imputation, PCA, polynomial features, or identity transformations. Any block that emits numeric columns participates in output ordering.

Learned transformation state and transformed schema are two parts of the same artifact.

## Remainder means appended, not untouched-in-place

`Remainder::Passthrough` can sound as if unused columns stay where they were. In the combined output matrix, they do not.

They are passed through in value, then appended after explicit specs.

Our raw tax rate started in position three and also ended in position three by coincidence. If I selected only age explicitly, the result would begin with transformed age and then append unused numeric columns in original order:

```text
[age_z, area_sq_ft, rooms, tax_rate]
```

Area would move from raw position zero to output position one even though its value was untouched.

The default remainder policy is `Remainder::Drop`, which creates a different risk: an unselected feature disappears completely. I choose `Drop` or `Passthrough` explicitly and test the resulting names and width. I do not let “remainder” stand in for a schema decision I have not made.

## Categorical blocks make drift wider

The example stays numeric so the permutation is easy to see. Categorical encoding makes the contract even more important.

A one-hot block may expand one source column into many outputs:

```text
city -> city_Ankara, city_Istanbul, city_Izmir
```

Moving that block moves the entire group. Refitting it on a different category vocabulary can change the group's width or category order as well. An unknown-category policy may preserve width while producing an all-zero encoded row, which is another semantic decision the final estimator cannot infer.

For a fitted production model, I reuse the fitted encoder. I do not reconstruct its category list from a new batch, even if the source code contains the same apparent `add_categorical` call.

The vocabulary, output names, order, and model coefficients must travel together.

## feature_names_out made the invisible contract visible

The most useful diagnostic lines in the example are not the matrix shapes. They are:

```rust
training_preprocessor
    .feature_names_out(Some(&input_names))
```

and:

```rust
reordered_preprocessor
    .feature_names_out(Some(&input_names))
```

They reveal the exact transformed order before a model consumes it.

I use those names in three ways:

1. Log them during training alongside the model version.
2. Save a canonical schema list or hash with the artifact.
3. Assert in tests that the fitted output schema matches the reviewed contract.

A test can be deliberately boring:

```rust
assert_eq!(
    preprocessor.feature_names_out(Some(&input_names)),
    vec![
        "area_sq_ft",
        "rooms",
        "age_years",
        "tax_rate",
    ],
);
```

That test protects something a four-column shape assertion cannot.

Feature names are still metadata; the `Matrix` does not enforce them at prediction time. The application must compare the contract before flattening named input into positional values.

## I deploy the fitted preprocessor, not its recipe

The reordered preprocessor in this experiment was fitted on the same data, yet it was already incompatible. Refitting on a new production batch would add changed means, standard deviations, category vocabularies, imputation values, and quantile boundaries to the order problem.

My deployment unit therefore contains:

- the fitted `ColumnTransformer`,
- the fitted estimator,
- the raw input schema and units,
- the transformed feature names in order,
- the library and artifact version,
- a small golden input with its expected transformed row and prediction.

At startup, a service can run the golden request and refuse to become ready if the result differs. This catches accidental block reordering, a mismatched preprocessor file, and many dependency or configuration changes before real requests arrive.

Recreating a preprocessing recipe from source is not equivalent to loading the fitted training artifact. The recipe omits the learned state, and a refactor can alter output order while looking cosmetically harmless.

## The safe edit is a model migration

Changing block order is not forbidden. It may make the schema clearer, group related features, or improve performance.

But once an estimator has been trained, the change is a model-interface migration:

1. Update the preprocessor configuration.
2. Recompute the transformed feature schema.
3. Retrain the estimator against that schema.
4. Re-run offline evaluation and golden tests.
5. Version and deploy both artifacts together.

An existing model can also be adapted by permuting its coefficients exactly, but that becomes difficult as soon as blocks expand, select, or combine features. Retraining through the reviewed pipeline is usually safer and easier to audit.

Code review should treat changes to spec order and column-index vectors as behavior changes, not formatting.

## Four columns were not enough evidence

The raw request in this experiment never moved:

```text
[1500.0, 10.0, 3.0, 0.018]
```

The training preprocessor compiled it into:

```text
[area_z, rooms_z, age_z, tax_rate]
```

The reordered preprocessor compiled it into:

```text
[age_z, area_z, rooms_z, tax_rate]
```

Both outputs passed every structural check the estimator could perform. One produced the exact expected result. The other missed it by `164.648`.

The lesson is not that `ColumnTransformer` has surprising behavior. Its behavior is deterministic: declared blocks first, remainder afterward.

The lesson is that preprocessing configuration creates a model ABI. After training, insertion order is no longer an implementation detail.

So before I trust a transformed matrix, I no longer ask only how many columns it has. I ask the question that survived this experiment:

> Which meaning occupies each position, and is that exactly the order the model learned?

If I cannot answer from a stored schema, the model is not ready to serve.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
