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

In [`datarust::metrics::classification`](https://docs.rs/datarust/latest/datarust/metrics/classification/index.html). All metrics auto-detect binary `{0, 1}` vs multiclass `{0, 1, 2, …}` integer labels. Precision, recall, and F1 apply **macro-averaging** (mean of per-class scores) for multiclass input.

```rust
use datarust::metrics::classification::*;

// Hard-label metrics (work for binary and multiclass):
let acc  = accuracy_score(&y_true, &y_pred)?;
let prec = precision_score(&y_true, &y_pred)?;     // macro-avg for multiclass
let rec  = recall_score(&y_true, &y_pred)?;
let f1   = f1_score(&y_true, &y_pred)?;
let cm   = confusion_matrix(&y_true, &y_pred)?;    // Vec<Vec<usize>>, n×n
let ll   = log_loss(&y_true, &y_proba, 1e-15)?;     // binary cross-entropy

// Ranking metrics (binary, consume predict_proba output):
let auc  = roc_auc_score(&y_true, &y_score)?;        // ROC-AUC (Mann–Whitney U)
let ap   = average_precision_score(&y_true, &y_score)?; // PR-AUC

// Agreement & correlation (binary + multiclass):
let kap  = cohen_kappa_score(&y_true, &y_pred)?;     // chance-corrected agreement
let mcc  = matthews_corrcoef(&y_true, &y_pred)?;     // Matthews correlation
```

| Metric | Range | Best | Binary | Multiclass | Notes |
|---|---|---|---|---|---|
| accuracy | `[0, 1]` | 1 | ✓ | ✓ | Fraction correctly classified |
| precision | `[0, 1]` | 1 | ✓ | ✓ (macro) | TP / (TP + FP) |
| recall | `[0, 1]` | 1 | ✓ | ✓ (macro) | TP / (TP + FN) |
| F1 | `[0, 1]` | 1 | ✓ | ✓ (macro) | Harmonic mean of precision & recall |
| confusion_matrix | `n×n` | diagonal | ✓ (2×2) | ✓ (n×n) | `cm[true][pred]` counts |
| log_loss | `[0, ∞)` | 0 | ✓ | — | Cross-entropy; needs probabilities |
| roc_auc_score | `[0, 1]` | 1 | ✓ | — | Ranking quality; needs scores |
| average_precision_score | `[0, 1]` | 1 | ✓ | — | PR-curve area; needs scores |
| cohen_kappa_score | `[-1, 1]` | 1 | ✓ | ✓ | Chance-corrected agreement |
| matthews_corrcoef | `[-1, 1]` | 1 | ✓ | ✓ | Balanced; robust to imbalance |

## Clustering metrics

In [`datarust::cluster::metrics`](https://docs.rs/datarust/latest/datarust/cluster/metrics/index.html). Evaluate clustering quality without ground-truth labels.

```rust
use datarust::cluster::metrics::silhouette_score;

let s = silhouette_score(&x, &labels)?;  // [-1, 1], higher is better
```

| Metric | Range | Best | Notes |
|---|---|---|---|
| silhouette_score | `[-1, 1]` | 1 | `(b−a)/max(a,b)` averaged over samples |

## Estimator `.score()` shorthand

Every estimator has a built-in `score` method:
- Regression models (`LinearRegression`, `Ridge`, `Lasso`) → R².
- `LogisticRegression` → accuracy.

```rust
let r2 = ridge.score(&x, &y)?;       // R²
let acc = logistic.score(&x, &y)?;   // accuracy
```
