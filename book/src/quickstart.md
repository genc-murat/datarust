# Quick Start

Get from `cargo add` to a fitted, transformed pipeline in five minutes.

## 1. Add the dependency

```toml
[dependencies]
datarust = "0.3"
```

Or with the most useful feature flags:

```toml
[dependencies]
datarust = { version = "0.3", features = ["serde", "rayon"] }
```

See [Installation](./installation.md) for what each feature does.

## 2. Create a matrix

All numeric data flows through `Matrix` — a row-major dense matrix backed by a single contiguous `Vec<f64>`:

```rust
use datarust::Matrix;

let x = Matrix::new(vec![
    vec![1.0, 10.0],
    vec![2.0, 20.0],
    vec![3.0, 30.0],
    vec![4.0, 40.0],
])?;
```

## 3. Fit and transform

Every numeric transformer implements the [`Transformer`](https://docs.rs/datarust/latest/datarust/trait.Transformer.html) trait: call `fit` to learn parameters, then `transform` to apply them.

```rust
use datarust::scaler::StandardScaler;
use datarust::traits::Transformer;

let mut scaler = StandardScaler::new();
let standardized = scaler.fit_transform(&x)?;  // fit + transform in one call

// mean=0, variance=1 per column
```

## 4. Train a model

Regression and classification models implement [`Regressor`](https://docs.rs/datarust/latest/datarust/trait.Regressor.html): call `fit(X, y)` then `predict(X)`.

```rust
use datarust::linear_model::LinearRegression;
use datarust::traits::Regressor;

let y = vec![3.0, 5.0, 7.0, 9.0]; // y = 2x + 1 (for feature 0)
let features = x.select_columns(&[0])?; // pick one feature
let mut model = LinearRegression::new();
model.fit(&features, &y)?;
let pred = model.predict(&features)?;
```

## 5. Evaluate

```rust
use datarust::metrics::regression::r2_score;

let r2 = r2_score(&y, &pred)?;
println!("R² = {r2:.4}"); // ≈ 1.0 for a clean linear signal
```

## 6. Split and cross-validate

```rust
use datarust::model_selection::{train_test_split, cross_val_score, KFold};
use datarust::metrics::regression::r2_score;

let (x_tr, x_te, y_tr, y_te) = train_test_split(&x, &y)?;

let cv = KFold::new().with_n_splits(3);
let scores = cross_val_score(&LinearRegression::new(), &x, &y, &cv, r2_score)?;
// scores.len() == 3, one R² per fold
```

## Where to go next

- **All transformers at a glance**: [Module Guide](./guide/scalers.md)
- **Mixing numeric + categorical columns**: [Compose](./guide/compose.md)
- **Why datarust is fast**: [Performance](./performance.md)
