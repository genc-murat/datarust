# Examples

Runnable examples from the `examples/` directory. Each can be run with `cargo run --example <name>`.

## Basic preprocessing

`cargo run --example basic_preprocessing`

Standardize, scale, and encode a small mixed-type dataset. Demonstrates `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, and the `Transformer` trait.

## Pipeline workflow

`cargo run --example pipeline_workflow`

Build a 3-step pipeline (StandardScaler → MinMaxScaler → RobustScaler), fit and transform, then inspect individual steps via the runtime ergonomics API.

## Target encoding

`cargo run --example target_encoding`

Use `TargetEncoder` with smoothed mean encoding on a high-cardinality categorical feature, including the `TargetTransformer` trait and `fit_with_target`.

## Benchmark comparison

`cargo run --release --example bench_compare_rust`

The deterministic benchmark harness that produces the numbers on the [Performance](./performance.md) page. Uses a xorshift64 PRNG (seed 42) so Rust and Python generate identical data. Pass a repetition count as an argument:

```sh
cargo run --release --features matrixmultiply --example bench_compare_rust 15
```

## End-to-end: preprocess, train, evaluate

A complete workflow tying together preprocessing, model fitting, and evaluation:

```rust
use datarust::prelude::*;
use datarust::model_selection::{train_test_split, KFold, cross_val_score};
use datarust::linear_model::Ridge;
use datarust::metrics::regression::r2_score;

// 1. Prepare data
let x = /* your Matrix */;
let y = /* your targets */;

// 2. Split train/test
let (x_tr, x_te, y_tr, y_te) = train_test_split(&x, &y)?;

// 3. Preprocess: fit scaler on train only
let mut scaler = StandardScaler::new();
scaler.fit(&x_tr)?;
let x_tr_s = scaler.transform(&x_tr)?;
let x_te_s = scaler.transform(&x_te)?;

// 4. Train a Ridge regression
let mut model = Ridge::new().with_alpha(1.0);
model.fit(&x_tr_s, &y_tr)?;

// 5. Evaluate
let r2 = model.score(&x_te_s, &y_te)?;
println!("Test R² = {r2:.4}");

// 6. Cross-validate for a robust estimate
let cv = KFold::new().with_n_splits(5);
let scores = cross_val_score(&Ridge::new().with_alpha(1.0), &x, &y, &cv, r2_score)?;
let mean_r2 = scores.iter().sum::<f64>() / scores.len() as f64;
println!("CV R² = {mean_r2:.4}");
```

> **Note:** the `prelude::*` import is illustrative — import the specific types you need (`use datarust::{Matrix, scaler::StandardScaler, ...};`).
