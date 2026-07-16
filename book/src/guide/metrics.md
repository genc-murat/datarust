# Metrics

Model evaluation. Mirrors `sklearn.metrics`. Live in [`datarust::metrics`](https://docs.rs/datarust/latest/datarust/metrics/index.html).

## Regression metrics

In [`datarust::metrics::regression`](https://docs.rs/datarust/latest/datarust/metrics/regression/index.html). Each takes `y_true` and `y_pred` as `&[f64]`.

```rust
use datarust::metrics::regression::*;

let mse  = mean_squared_error(&y_true, &y_pred, true)?;   // squared=true → MSE
let rmse = mean_squared_error(&y_true, &y_pred, false)?;  // squared=false → RMSE
let mae  = mean_absolute_error(&y_true, &y_pred)?;
let r2   = r2_score(&y_true, &y_pred)?;
let me   = max_error(&y_true, &y_pred)?;
let ev   = explained_variance_score(&y_true, &y_pred)?;
```

| Metric | Range | Best | Notes |
|---|---|---|---|
| MSE | `[0, ∞)` | 0 | Mean of squared errors |
| RMSE | `[0, ∞)` | 0 | Root MSE (same units as `y`) |
| MAE | `[0, ∞)` | 0 | Mean of absolute errors |
| R² | `(-∞, 1]` | 1 | 1.0 = perfect; 0.0 = predicting the mean |
| max_error | `[0, ∞)` | 0 | Worst single prediction |
| explained_variance | `(-∞, 1]` | 1 | Variance of residuals explained |

## Classification metrics

In [`datarust::metrics::classification`](https://docs.rs/datarust/latest/datarust/metrics/classification/index.html). Labels are `0.0` / `1.0` floats.

```rust
use datarust::metrics::classification::*;

let acc  = accuracy_score(&y_true, &y_pred)?;
let prec = precision_score(&y_true, &y_pred)?;
let rec  = recall_score(&y_true, &y_pred)?;
let f1   = f1_score(&y_true, &y_pred)?;
let cm   = confusion_matrix(&y_true, &y_pred)?; // [[tn, fp], [fn, tp]]
let ll   = log_loss(&y_true, &y_proba, 1e-15)?;  // cross-entropy (needs probabilities)
```

| Metric | Range | Best | Notes |
|---|---|---|---|
| accuracy | `[0, 1]` | 1 | Fraction correctly classified |
| precision | `[0, 1]` | 1 | TP / (TP + FP) |
| recall | `[0, 1]` | 1 | TP / (TP + FN) |
| F1 | `[0, 1]` | 1 | Harmonic mean of precision & recall |
| log_loss | `[0, ∞)` | 0 | Cross-entropy; needs probabilities, not hard labels |

## Estimator `.score()` shorthand

Every estimator has a built-in `score` method:
- Regression models (`LinearRegression`, `Ridge`, `Lasso`) → R².
- `LogisticRegression` → accuracy.

```rust
let r2 = ridge.score(&x, &y)?;       // R²
let acc = logistic.score(&x, &y)?;   // accuracy
```
