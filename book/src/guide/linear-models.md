# Linear Models

Regression and classification estimators. Live in [`datarust::linear_model`](https://docs.rs/datarust/latest/datarust/linear_model/index.html). All implement [`Predictor`](https://docs.rs/datarust/latest/datarust/trait.Predictor.html); regression models also implement `Regressor` and classifiers implement `Classifier`.

## LinearRegression

Ordinary least squares. Estimates `y ≈ Xβ + b` by minimising `‖Xβ − y‖²`.

```rust
use datarust::linear_model::{LinearRegression, LinearSolver};
use datarust::traits::Predictor;

let mut model = LinearRegression::new()
    .with_fit_intercept(true)             // default
    .with_solver(LinearSolver::Cholesky); // default; or LinearSolver::Svd

model.fit(&x, &y)?;
let pred = model.predict(&new_x)?;

model.coef();          // coefficients β
model.intercept();     // intercept b
model.n_features_in(); // feature count
let r2 = model.score(&x, &y)?;  // R²
```

**Solvers:**
- `Cholesky` (default) — solves `XᵀX β = Xᵀy` via pure-Rust Cholesky. Fast; requires full column rank.
- `Svd` — eigendecomposition pseudo-inverse. Stable for rank-deficient / collinear inputs.

## Ridge

L2-regularized regression. Minimises `‖Xβ − y‖² + α‖β‖²`. The penalty guarantees the system matrix is positive-definite — Ridge **succeeds on collinear inputs** where `LinearRegression` fails.

```rust
use datarust::linear_model::{Ridge, RidgeSolver};

let mut model = Ridge::new()
    .with_alpha(1.0)                      // regularization strength (default 1.0)
    .with_solver(RidgeSolver::Cholesky);  // or Svd
model.fit(&x, &y)?;
```

Larger `alpha` → more shrinkage (coefficients shrink toward zero).

## Lasso

L1-regularized regression. Minimises `(1/(2n))‖Xβ − y‖² + α‖β‖₁`. Solved by **coordinate descent** with soft-thresholding.

The L1 penalty drives irrelevant coefficients to **exactly zero**, producing a sparse model that performs implicit feature selection.

```rust
use datarust::linear_model::Lasso;

let mut model = Lasso::new()
    .with_alpha(0.1)          // larger alpha → more sparsity
    .with_max_iter(1000)      // default 1000
    .with_tol(1e-4);          // convergence tolerance
model.fit(&x, &y)?;

model.coef();   // some entries may be exactly 0.0 (sparsity)
model.n_iter(); // iterations actually run
```

## LogisticRegression

Binary classification via IRLS (Iteratively Reweighted Least Squares / Newton-Raphson). The crate's first classifier.

```rust
use datarust::linear_model::{LogisticRegression, LogisticSolver};
use datarust::traits::Predictor;

let mut model = LogisticRegression::new()
    .with_solver(LogisticSolver::Cholesky) // or Svd
    .with_max_iter(100)                    // default 100
    .with_tol(1e-4);

model.fit(&x, &y)?;             // y must be 0.0 / 1.0
let classes = model.predict(&x)?; // 0.0 / 1.0
let probabilities = model.predict_proba(&x)?; // P(class=0), P(class=1)
let acc = model.score(&x, &y)?; // mean accuracy
```

## Choosing a model

| Goal | Model |
|---|---|
| Simple baseline regression | `LinearRegression` |
| Collinear features, regularization | `Ridge` (L2) |
| Feature selection via sparsity | `Lasso` (L1) |
| Binary classification | `LogisticRegression` |
