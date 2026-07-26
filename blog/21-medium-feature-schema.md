# The Matrix Had Three Columns. They Were the Wrong Three Columns.

*A practical datarust guide to feature order, units, schema contracts, and the production errors that pass every shape check.*

---

The request passed validation.

```text
rows:    1
columns: 3
values:  finite
```

The model expected three features, and the matrix contained three features. Prediction succeeded.

It returned `-3655.466` instead of `323.932`.

The problem was not the number of columns. Two columns had traded places. A second request was less dramatic but equally wrong: area arrived in square meters while the model expected square feet. It also had the correct shape.

Machine-learning interfaces often become dangerously narrow at the final boundary. A rich application object — names, units, optional fields, validation rules — turns into `Vec<f64>`. Once that happens, `[1500, 10, 3]` and `[10, 1500, 3]` are both perfectly respectable rows.

Rust can prove that each element is an `f64`. It cannot infer whether the first one means area, age, or room count.

Let's reproduce the failure with [datarust](https://crates.io/crates/datarust), then build the semantic contract that the matrix shape cannot provide.

## A three-feature property model

The training schema is:

```text
column 0: area_sq_ft
column 1: age_years
column 2: rooms
```

I train a `StandardScaler` plus `Ridge` pipeline on 240 synthetic properties. Then I send the same logical property in four representations:

1. Correct order and units
2. Area and age swapped
3. Area expressed in square meters
4. Age expressed in months

Every matrix has one row and three columns.

Here is the complete Rust program:

```rust
use datarust::linear_model::Ridge;
use datarust::pipeline::Pipeline;
use datarust::scaler::StandardScaler;
use datarust::traits::{FeatureNames, Predictor};
use datarust::transformer_kind::TransformerKind;
use datarust::Matrix;

const MODEL_SCHEMA: [&str; 3] = [
    "area_sq_ft",
    "age_years",
    "rooms",
];

struct PropertyInput {
    area_sq_ft: f64,
    age_years: f64,
    rooms: f64,
}

impl PropertyInput {
    fn to_model_row(&self) -> Vec<f64> {
        vec![self.area_sq_ft, self.age_years, self.rooms]
    }
}

fn validate_schema(received: &[&str]) -> Result<(), String> {
    if received == MODEL_SCHEMA {
        Ok(())
    } else {
        Err(format!(
            "schema mismatch: expected {:?}, received {:?}",
            MODEL_SCHEMA, received,
        ))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    let mut targets = Vec::new();

    for i in 0..240 {
        let area = 600.0 + ((i * 37) % 2400) as f64;
        let age = ((i * 17) % 60) as f64;
        let rooms = 1.0 + ((i * 11) % 5) as f64;
        let noise = (((i * 13) % 11) as f64 - 5.0) * 2.0;
        let price = 25.0
            + 0.18 * area
            - 2.5 * age
            + 18.0 * rooms
            + noise;

        rows.push(vec![area, age, rooms]);
        targets.push(price);
    }

    let x_train = Matrix::new(rows)?;
    let mut model = Pipeline::new()
        .push(
            "scale",
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        .with_estimator(Ridge::new().with_alpha(1.0));
    model.fit(&x_train, &targets)?;

    let request = PropertyInput {
        area_sq_ft: 1_500.0,
        age_years: 10.0,
        rooms: 3.0,
    };
    let correct = Matrix::new(vec![request.to_model_row()])?;
    let swapped =
        Matrix::new(vec![vec![10.0, 1_500.0, 3.0]])?;
    let area_in_square_meters =
        Matrix::new(vec![vec![139.3546, 10.0, 3.0]])?;
    let age_in_months =
        Matrix::new(vec![vec![1_500.0, 120.0, 3.0]])?;

    println!("Declared model schema: {:?}", MODEL_SCHEMA);
    let input_names: Vec<String> = MODEL_SCHEMA
        .iter()
        .map(|name| name.to_string())
        .collect();
    println!(
        "Names after preprocessing: {:?}\n",
        model
            .transformers()
            .feature_names_out(Some(&input_names)),
    );

    for (name, matrix) in [
        ("correct units/order", &correct),
        ("area and age swapped", &swapped),
        ("area sent in m²", &area_in_square_meters),
        ("age sent in months", &age_in_months),
    ] {
        println!(
            "{name:<22} -> prediction {:10.3}",
            model.predict(matrix)?[0],
        );
    }

    println!("\nAll four matrices have shape 1 x 3.");
    println!(
        "Schema check: {}",
        validate_schema(&[
            "age_years",
            "area_sq_ft",
            "rooms",
        ])
        .unwrap_err(),
    );

    Ok(())
}
```

This is the output I measured:

```text
Declared model schema: ["area_sq_ft", "age_years", "rooms"]
Names after preprocessing: ["area_sq_ft", "age_years", "rooms"]

correct units/order    -> prediction    323.932
area and age swapped   -> prediction  -3655.466
area sent in m²        -> prediction     79.937
age sent in months     -> prediction     49.877

All four matrices have shape 1 x 3.
Schema check: schema mismatch: expected ["area_sq_ft", "age_years", "rooms"],
received ["age_years", "area_sq_ft", "rooms"]
```

The model rejected nothing because every numeric interface promise was kept.

## Shape validation protects structure

`Matrix::new` validates that rows are rectangular and contain a consistent number of columns. The fitted scaler and estimator validate that incoming feature count matches training.

Those checks catch important errors:

- A row has two values while another has three.
- A new feature was added without retraining.
- A required feature disappeared.
- The request body produced an empty matrix.

They cannot catch a permutation. If a model expects three features, every permutation still has three features.

They also cannot catch a unit conversion. `1500` square feet and `139.3546` square meters describe approximately the same property, but the model sees two different floating-point values in the same column.

Shape is part of a feature schema. It is not the whole schema.

## StandardScaler amplified the semantic contract

The pipeline's scaler learned a separate mean and standard deviation for each position:

```text
position 0 statistics belong to area_sq_ft
position 1 statistics belong to age_years
position 2 statistics belong to rooms
```

When age value `10` arrives in the area position, it is transformed using area statistics. When area value `1500` arrives in the age position, it is interpreted as a property 1,500 years old and standardized accordingly.

The resulting coordinates are extreme, and Ridge produces `-3655.466`.

Scaling did not cause the schema error. It faithfully applied fitted position-specific state to the wrong semantics. Any learned column transformer — imputation, encoding, feature selection, PCA — has the same dependency on stable feature identity.

Preprocessing makes the ordered schema part of the model even before the final estimator sees a row.

## Plausible wrong values are more dangerous

The negative prediction is obviously suspicious. The unit mistakes are quieter:

```text
correct:       323.932
area in m²:     79.937
age in months:  49.877
```

Both wrong values may still fit inside a broad output range. A downstream system could accept them, store them, and make decisions without raising an exception.

Range checks help but cannot prove semantic correctness. An area of `139.3546` square feet is small but possible. An age of `120` years is unusual but not numerically impossible. A loose validator may let both through.

The best time to catch the error is before units and field names disappear into a numeric row.

## Deserialize names, then build positions once

I avoid constructing model rows from map iteration order, JSON property order, database `SELECT *`, or a mutable list of upstream fields.

Instead, the service deserializes a named application type:

```rust
struct PropertyInput {
    area_sq_ft: f64,
    age_years: f64,
    rooms: f64,
}
```

One conversion method owns the positional contract:

```rust
impl PropertyInput {
    fn to_model_row(&self) -> Vec<f64> {
        vec![self.area_sq_ft, self.age_years, self.rooms]
    }
}
```

That method is deliberately boring. It is also one of the most important lines in the serving system.

If the external API accepts square meters, I make the conversion explicit before constructing the row:

```rust
let area_sq_ft = area_m2 * 10.7639;
```

I do not rename an `area_m2` value to `area_sq_ft` and hope a later layer remembers what happened.

## Units can become types

Field names document units. Newtypes can make some unit mistakes harder to compile:

```rust
struct SquareFeet(f64);
struct Years(f64);
struct RoomCount(f64);

struct PropertyFeatures {
    area: SquareFeet,
    age: Years,
    rooms: RoomCount,
}
```

A constructor can accept `SquareMeters`, perform the conversion, and return `SquareFeet`. The final matrix still contains `f64`, but unit correctness is enforced while the data remains in the typed application layer.

This does not solve bad upstream metadata or values entered in the wrong field. It shrinks the number of places where units can be confused and makes conversion code searchable.

For high-stakes systems, that reduction is worth far more than the extra wrapper types cost.

## Feature names are lineage, not automatic validation

datarust transformers implement the `FeatureNames` trait. Given input feature names, a fitted pipeline can trace the output names:

```rust
let output_names = model
    .transformers()
    .feature_names_out(Some(&input_names));
```

`StandardScaler` preserves names, so this example returns:

```text
["area_sq_ft", "age_years", "rooms"]
```

Other transformers change them. One-hot encoding appends category names, feature selection filters the list, and PCA produces component names. This is valuable for model inspection, logging, coefficient reports, and checking the fitted transformation graph.

It does not attach names to an incoming `Matrix`. The caller supplied the name list used for lineage. A swapped numeric matrix does not carry metadata that lets `predict` compare its columns against that list.

I use `feature_names_out` to understand and verify the fitted pipeline, then enforce incoming schema in the application boundary that still has named fields.

## A schema is more than a list of names

The compact constant in the example catches ordering mistakes:

```rust
const MODEL_SCHEMA: [&str; 3] = [
    "area_sq_ft",
    "age_years",
    "rooms",
];
```

A production schema manifest usually needs more:

```text
feature name
exact position
numeric or categorical type
unit and conversion version
missing-value policy
valid domain and invariants
categorical vocabulary policy
aggregation window
event-time cutoff
default behavior
model/schema version
```

`customer_spend` is incomplete if training meant “settled USD spend in the trailing 30 days” and serving supplies “local-currency cart value since account creation.” The column name and numeric type can match while the feature definition changes completely.

Windows and timestamps are units too.

## Validate values without pretending ranges are enough

After schema identity and units, I validate domain rules:

```text
area_sq_ft is finite and positive
age_years is within an approved range
rooms is an allowed count
required values are present
```

I distinguish rejection rules from monitoring rules. A negative room count may be invalid and rejected. A 15,000-square-foot property may be rare but legitimate, so it should perhaps be scored with an out-of-distribution warning rather than silently dropped.

Training ranges are useful telemetry but dangerous hard limits. Production will eventually contain valid values beyond them.

The schema decides what a value means. Distribution monitoring asks whether the values with that meaning have drifted.

## Database and API evolution need explicit mapping

A model input should not depend on incidental storage order.

This query is fragile:

```sql
SELECT * FROM properties
```

Adding or reordering a database column can alter the extraction layer. An explicit projection is safer:

```sql
SELECT area_sq_ft, age_years, rooms
FROM properties
```

The same principle applies to CSV headers, message schemas, and API payloads. I map by field identity into a versioned struct, validate, convert units, and only then produce the model's ordered row.

If schema version 2 adds a feature, I do not append a zero to satisfy shape and reuse the old model. The old artifact still expects version 1 semantics. A new feature graph requires training and validating a matching artifact.

## Golden requests should begin before the matrix

The most useful serving test starts with a realistic named request, not a prebuilt numeric matrix:

```text
JSON or domain object
  -> validation
  -> unit conversion
  -> ordered feature row
  -> fitted preprocessing
  -> prediction
```

I keep a small golden set with expected rows and predictions. Tests deliberately swap fields, change units, omit values, add unknown categories, and use boundary values.

For this example, a regression test can assert:

```text
area_sq_ft=1500, age_years=10, rooms=3
    -> model row [1500, 10, 3]
    -> prediction approximately 323.932
```

That one end-to-end assertion covers semantic mapping and model behavior together. A unit test beginning at `[1500, 10, 3]` would never notice that the API adapter produced `[10, 1500, 3]`.

## What the compiler can and cannot promise

Rust gives this system meaningful protection:

- Numeric and string matrices are distinct types.
- Domain newtypes can encode units.
- Named structs make field mapping explicit.
- Invalid shapes return errors.
- Fitted model types and method calls are checked.

At the boundary where everything becomes `f64`, semantic distinctions intentionally collapse. The model needs a numeric matrix, but that conversion is an information-losing operation.

The answer is not to expect `Matrix` to rediscover lost meaning. It is to delay the loss, centralize it, and test the conversion.

## Three columns were never the contract

All four requests in this experiment satisfied `1 × 3`:

```text
correct order and units ->    323.932
swapped columns         ->  -3655.466
square meters           ->     79.937
age in months           ->     49.877
```

Only the first satisfied the feature schema.

The model artifact contains learned numeric state. It still needs a versioned agreement with the application that produces its inputs: names, order, units, transformations, and time windows.

The lesson I keep is blunt:

> The right number of columns can still describe the wrong world.

Shape validation tells me whether the matrix fits the model.

Schema validation tells me whether the matrix means what the model learned.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
