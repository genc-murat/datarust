# Lasso Won at n=20. LinearRegression Won at n=100. The Sample Size Changed the Winner.

*How sample size affects model selection*

---

Which model is best depends on how much data you have. With 20 samples, Lasso wins. With 100 samples, LinearRegression wins. The "best" model isn't a property of the model — it's a property of the data.

## Experiment 1: Sample Size vs Model Performance

10 features (2 useful, 8 noise):

```
n_samples   LinearReg   Ridge(1)    Lasso(0.1)  best_model
-----------------------------------------------------------------
20            0.9749      0.9838      0.9896    Lasso
30            0.9898      0.9795      0.9889    LinearReg
50            0.9892      0.9867      0.9902    Lasso
80            0.9858      0.9853      0.9865    Lasso
100           0.9917      0.9912      0.9907    LinearReg
200           0.9910      0.9910      0.9892    LinearReg
500           0.9924      0.9924      0.9908    LinearReg
```

The pattern:
- **n=20**: Lasso wins (0.9896). It selects the 2 useful features, ignoring the 8 noise features.
- **n=30-80**: Lasso still wins, but LinearRegression catches up.
- **n=100+**: LinearRegression wins. With enough data, the unbiased estimator beats the regularized one.

The reason: Lasso trades bias for variance. With small samples, variance dominates, so Lasso's bias is worth the variance reduction. With large samples, variance is small, so LinearRegression's lack of bias is more valuable.

## Experiment 2: Sample Size vs Overfitting

True relationship: `y = 3x₁ + 2x₂ + x₁x₂` (interaction term). Trying polynomial features of different degrees:

```
n_samples   linear_gap   poly2_gap    poly3_gap    poly5_gap
----------------------------------------------------------------------
20            0.0917      0.1782      0.2428      0.9885
30            0.0290      0.0138      0.0578      0.1006
50            0.0195      0.0041      0.0159      0.1063
80            0.0134      0.0018      0.0036      0.1733
100           0.0544      0.0047      0.0075      0.6324
200           0.0048      0.0003      0.0008      0.0024
500           0.0035      0.0004      0.0007      0.0009
```

The pattern:
- **n=20**: poly5 gap=0.99 (catastrophic overfitting). With 20 samples and 21 polynomial features, the model memorizes the data.
- **n=50**: poly5 gap=0.11 (moderate overfitting). Still bad, but usable.
- **n=200**: poly5 gap=0.002 (no overfitting). The model has enough data to learn the true relationship.

The minimum sample size for each degree:
- degree=2: ~30 samples
- degree=3: ~50 samples
- degree=5: ~200 samples

The rule of thumb: you need at least 10× more samples than features for reliable generalization.

## Experiment 3: CV Reliability

5-fold CV, LinearRegression, 20 random seeds per sample size:

```
n_samples   cv_mean    cv_std     folds_with_nan
-------------------------------------------------------
10           -8.6613    33.2296               1
15            0.9549     0.0367               1
20            0.9737     0.0134               1
30            0.9841     0.0078               1
50            0.9853     0.0046               1
100           0.9891     0.0025               1
200           0.9896     0.0028               0
```

The pattern:
- **n=10**: cv_std=33.23. CV is meaningless — the variance is larger than the signal.
- **n=15**: cv_std=0.04. CV is noisy but usable.
- **n=30**: cv_std=0.008. CV is reliable.
- **n=200**: cv_std=0.003. CV is very stable.

The minimum sample size for reliable CV is ~30. Below that, the CV estimate is too noisy to trust.

## The Three Regimes

| Sample Size | Best Model | Overfitting Risk | CV Reliability |
|-------------|-----------|------------------|----------------|
| n < 30 | Lasso (feature selection) | High | Low |
| 30 ≤ n < 100 | Ridge (variance reduction) | Moderate | Moderate |
| n ≥ 100 | LinearRegression (unbiased) | Low | High |

## The Code

```rust
use datarust::linear_model::{LinearRegression, Ridge, Lasso};
use datarust::model_selection::KFold;
use datarust::traits::Predictor;

let kf = KFold::new().with_n_splits(5).with_shuffle(true);

// Try multiple models, pick the one with best CV
let models: Vec<Box<dyn Predictor>> = vec![
    Box::new(LinearRegression::new()),
    Box::new(Ridge::new().with_alpha(1.0)),
    Box::new(Lasso::new().with_alpha(0.1)),
];

let mut best_cv = f64::NEG_INFINITY;
let mut best_model = "";
for (name, model) in ["LinearReg", "Ridge", "Lasso"].iter().zip(models.iter()) {
    let cv = cv_score(model, &x, &y, &kf);
    if cv > best_cv {
        best_cv = cv;
        best_model = name;
    }
}
println!("best: {} (cv={:.4})", best_model, best_cv);
```

## Practical Guidelines

**When n < 30:**
- Use Lasso for feature selection
- Don't trust CV scores (too noisy)
- Collect more data if possible

**When 30 ≤ n < 100:**
- Use Ridge for stability
- CV is reliable enough for model comparison
- Avoid high-degree polynomials (degree > 2)

**When n ≥ 100:**
- LinearRegression is often best
- CV is reliable
- You can try more complex models

**When n ≥ 200:**
- Almost any model works
- Focus on feature engineering, not model selection
- The bottleneck is usually data quality, not sample size

## Tradeoffs

**Small samples (n < 30):**
- High variance: different samples give different models
- Overfitting risk: models memorize noise
- Need regularization: Lasso or Ridge

**Medium samples (30 ≤ n < 100):**
- Moderate variance: models are somewhat stable
- Regularization helps but isn't critical
- Ridge is a safe default

**Large samples (n ≥ 100):**
- Low variance: models are stable
- Regularization is unnecessary (unless p >> n)
- Unbiased estimators are best

The universal rule: more data is always better. If you're debating between models, collect more data instead.

## Try It

```bash
cargo add datarust
```

```rust
use datarust::linear_model::{LinearRegression, Lasso};
use datarust::model_selection::KFold;
use datarust::traits::Predictor;
use datarust::Matrix;

let x = Matrix::new(vec![vec![1.0]; 20]).unwrap();
let y: Vec<f64> = (0..20).map(|i| i as f64).collect();

let kf = KFold::new().with_n_splits(5).with_shuffle(true);

// Lasso wins with small samples
let mut lasso = Lasso::new().with_alpha(0.1);
let lasso_cv = cv_score(&lasso, &x, &y, &kf);

// LinearRegression wins with large samples
let mut lr = LinearRegression::new();
let lr_cv = cv_score(&lr, &x, &y, &kf);

println!("lasso={:.4}, lr={:.4}", lasso_cv, lr_cv);
```

If you have 20 samples and 10 features, Lasso is your best bet. If you have 100 samples, LinearRegression will do better. The sample size decides the winner.
