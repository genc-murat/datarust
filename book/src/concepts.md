# Core Concepts

Understanding five concepts unlocks the entire crate.

## 1. `Matrix` — the data container

All numeric data flows through [`Matrix`](https://docs.rs/datarust/latest/datarust/struct.Matrix.html): a row-major dense matrix backed by a single contiguous `Vec<f64>` buffer. The flat layout keeps every numeric hot loop cache-friendly and auto-vectorizable.

```rust
use datarust::Matrix;

let m = Matrix::new(vec![
    vec![1.0, 2.0, 3.0],
    vec![4.0, 5.0, 6.0],
])?;
assert_eq!(m.nrows(), 2);
assert_eq!(m.ncols(), 3);
assert_eq!(m.get(0, 1), 2.0);
```

Companion types:
- [`StrMatrix`](https://docs.rs/datarust/latest/datarust/struct.StrMatrix.html) — categorical string input for encoders.
- [`SparseMatrix`](https://docs.rs/datarust/latest/datarust/struct.SparseMatrix.html) — CSR format for memory-efficient one-hot output.

## 2. The `Transformer` trait

All **numeric** transformers implement [`Transformer`](https://docs.rs/datarust/latest/datarust/trait.Transformer.html):

```rust
pub trait Transformer {
    fn name(&self) -> &'static str;
    fn fit(&mut self, x: &Matrix) -> Result<()>;
    fn transform(&self, x: &Matrix) -> Result<Matrix>;
    fn fit_transform(&mut self, x: &Matrix) -> Result<Matrix>; // default: fit + transform
    fn inverse_transform(&self, _x: &Matrix) -> Result<Matrix>; // optional
    fn is_fitted(&self) -> bool;
}
```

- `fit` learns parameters from training data (takes `&mut self`).
- `transform` applies the learned transformation (takes `&self`).
- `fit_transform` is a convenience that calls both.
- `inverse_transform` reverses the transformation where supported.

## 3. The `Regressor` trait

Regression and classification estimators implement [`Regressor`](https://docs.rs/datarust/latest/datarust/trait.Regressor.html):

```rust
pub trait Regressor {
    fn name(&self) -> &'static str;
    fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()>;
    fn predict(&self, x: &Matrix) -> Result<Vec<f64>>;
    fn fit_predict(&mut self, x: &Matrix, y: &[f64]) -> Result<Vec<f64>>; // default
    fn is_fitted(&self) -> bool;
}
```

- `LinearRegression`, `Ridge`, `Lasso` — `predict` returns continuous predictions.
- `LogisticRegression` — `predict` returns `P(y=1|x)` probabilities; use `predict_class` for hard 0/1 labels.

## 4. The categorical traits

Categorical data is kept separate at the type level:

- [`CategoricalTransformer`](https://docs.rs/datarust/latest/datarust/trait.CategoricalTransformer.html) — `StrMatrix → Matrix` (OneHot, Ordinal, Frequency encoders).
- [`TargetTransformer`](https://docs.rs/datarust/latest/datarust/trait.TargetTransformer.html) — supervised encoders needing `y` during fit (TargetEncoder).
- [`LabelTransformer`](https://docs.rs/datarust/latest/datarust/trait.LabelTransformer.html) — 1-D `&[String] ↔ Vec<usize>` (LabelEncoder).

This separation means the **compiler** prevents you from accidentally passing strings to a numeric scaler.

## 5. Error handling

Every fallible public method returns `Result<T, DatarustError>`. No hidden panics on bad input. The error variants are ML-domain-specific:

```rust
pub enum DatarustError {
    NotFitted(String),
    InvalidInput(String),
    ShapeMismatch { expected: String, actual: String },
    EmptyInput(String),
    InvalidConfig(String),
    Singular(String),       // e.g. rank-deficient matrix in Cholesky
    UnknownCategory(String),
    // ...
}
```

Typical usage:

```rust
match scaler.transform(&x) {
    Ok(out) => { /* use out */ }
    Err(DatarustError::NotFitted(name)) => eprintln!("call fit() first on {name}"),
    Err(DatarustError::ShapeMismatch { expected, actual }) => eprintln!("{expected} vs {actual}"),
    Err(e) => eprintln!("error: {e}"),
}
```
