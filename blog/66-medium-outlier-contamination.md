---
title: "5% Outliers Broke LinearRegression. 20% Outliers Made It Worse. Here's Why."
subtitle: "Outlier contamination kills performance — and RobustScaler can't save you"
author: "Murat Genc"
date: "2026-07-27"
tags: ["machine-learning", "rust", "datarust", "outliers", "robust-scaling", "contamination"]
series: "datarust-v06"
---

Outliers are not all the same. Some are extreme values from the same distribution. Others follow a completely different relationship. RobustScaler handles the first type. It does nothing for the second type. And the second type is what kills your model.

## Experiment 1: Outlier Contamination

True relationship: `y = 3x₁ + 2x₂ + noise`. Outliers follow a different relationship: `y = -1x₁ + 0.5x₂ + extreme noise`.

200 samples, 5-fold CV:

```
contam%   LinearReg   Ridge(1)    Lasso(0.1)  RobustScaler+LR
----------------------------------------------------------------------
0           0.9924      0.9924      0.9910      0.9924
0.01        0.8940      0.8944      0.8934      0.8940
0.05        0.2725      0.2731      0.2759      0.2725
0.1         0.1031      0.1036      0.1089      0.1031
0.2         0.0675      0.0676      0.0664      0.0675
0.3         0.3353      0.3353      0.3355      0.3353
0.5         0.1861      0.1861      0.1866      0.1861
```

The pattern:
- **0%**: R²=0.99. Perfect.
- **1%**: R²=0.89. Noticeable drop.
- **5%**: R²=0.27. Catastrophic.
- **10%**: R²=0.10. Near zero.
- **50%**: R²=0.19. Slightly better (model is fitting the majority class now).

Key observation: **RobustScaler+LR gives the same scores as raw LR.** RobustScaler cannot help when outliers follow a different relationship.

## Experiment 2: Coefficient Bias

True coefficients: `[3.0, 2.0]`. Outliers follow `y = -1x₁ + 0.5x₂`.

```
contam%   LR_coef1   LR_coef2   Ridge_c1   Ridge_c2   Lasso_c1   Lasso_c2
--------------------------------------------------------------------------------
0           2.9927     2.0200     2.9779     2.0094     2.8940     1.9141
0.05        0.8277     0.9358     0.8258     0.9331     0.7796     0.8789
0.1         0.9710     1.7337     0.9675     1.7276     0.9097     1.6574
0.2        -0.2521     0.7986    -0.2518     0.7978    -0.2338     0.7788
0.5        -0.7337     0.7103    -0.7334     0.7101    -0.7249     0.7022
```

The pattern:
- **0%**: `[2.99, 2.02]`. Near perfect.
- **5%**: `[0.83, 0.94]`. Drastically different.
- **20%**: `[-0.25, 0.80]`. **Completely wrong direction!**
- **50%**: `[-0.73, 0.71]`. Completely wrong.

The model is fitting a compromise between the true relationship and the outlier relationship. With 20% outliers, the compromise is worse than random.

## Experiment 3: RobustScaler vs StandardScaler

```
contam%   LR_raw     LR_std     LR_robust
--------------------------------------------------
0           0.9912      0.9912      0.9912
0.05        0.3553      0.3553      0.3553
0.1        -0.1866     -0.1866     -0.1866
0.2        -0.0950     -0.0950     -0.0950
0.5         0.3346      0.3346      0.3346
```

All three scalers give **identical scores**. Why? Because RobustScaler only handles outliers that are extreme values from the same distribution. When outliers follow a different relationship, scaling doesn't help — the data points themselves are wrong, not their scale.

## The Two Types of Outliers

| Type | Example | RobustScaler Effect |
|------|---------|-------------------|
| Extreme values from same distribution | x=100 when most x are 1-10 | Works |
| Different relationship | y=-x when true y=3x | Doesn't help |
| Measurement errors | y=999 when true y=3 | Works |
| Systematic bias | y=2x+10 when true y=3x | Doesn't help |

## What Actually Works

**For Type 1 outliers (extreme values):**
```rust
use datarust::scaler::RobustScaler;

let mut scaler = RobustScaler::new(); // median/IQR
let x_clean = scaler.fit_transform(&x)?;
```

**For Type 2 outliers (different relationship):**
You need to detect and remove them before training.

```rust
// Simple approach: remove points with large residuals
let mut lr = LinearRegression::new();
lr.fit(&x, &y)?;
let pred = lr.predict(&x)?;
let residuals: Vec<f64> = y.iter().zip(pred.iter()).map(|(yi, pi)| (yi - pi).abs()).collect();

// Remove points with residuals > 3 * median
let median = residuals.iter().cloned().collect::<Vec<f64>>().into_iter().sum::<f64>() / residuals.len() as f64;
let threshold = 3.0 * median;
let mask: Vec<usize> = residuals.iter().enumerate()
    .filter(|(_, r)| **r < threshold)
    .map(|(i, _)| i)
    .collect();
```

**For high contamination (>20%):**
- Use robust estimators (RANSAC, Theil-Sen)
- Collect more data
- Use domain knowledge to identify outliers

## Practical Guidelines

**contamination < 1%:**
- RobustScaler helps
- LinearRegression is still usable
- CV scores are trustworthy

**1% < contamination < 10%:**
- RobustScaler may not be enough
- Consider outlier removal
- Ridge is safer than LinearRegression

**10% < contamination < 30%:**
- Models are severely compromised
- Outlier removal is critical
- Consider domain-specific approaches

**contamination > 30%:**
- The "outliers" are the majority
- You need domain knowledge to identify the true signal
- Standard models are unreliable

## The Code

```rust
use datarust::linear_model::LinearRegression;
use datarust::scaler::RobustScaler;
use datarust::traits::Predictor;

// Type 1 outliers: RobustScaler helps
let mut scaler = RobustScaler::new();
let x_robust = scaler.fit_transform(&x)?;
let mut lr = LinearRegression::new();
lr.fit(&x_robust, &y)?;

// Type 2 outliers: RobustScaler doesn't help
// You need to remove outliers first
let pred = lr.predict(&x)?;
let residuals: Vec<f64> = y.iter().zip(pred.iter())
    .map(|(yi, pi)| (yi - pi).abs())
    .collect();
// ... filter outliers ...
```

## Tradeoffs

**RobustScaler:**
- ✅ Handles extreme values
- ✅ Simple to use
- ❌ Doesn't help with different relationships
- ❌ Reduces effective sample size

**Outlier removal:**
- ✅ Works for both types
- ❌ Requires a model to compute residuals
- ❌ Risk of removing valid data

**Domain knowledge:**
- ✅ Most reliable
- ✅ Understands the data
- ❌ Requires expertise
- ❌ Doesn't scale

The universal rule: **identify the type of outlier before choosing the solution.** RobustScaler is a tool, not a silver bullet.

## Try It

```bash
cargo add datarust
```

```rust
use datarust::linear_model::LinearRegression;
use datarust::scaler::RobustScaler;
use datarust::traits::Predictor;
use datarust::Matrix;

let x = Matrix::new(vec![vec![1.0, 2.0]; 100]).unwrap();
let y: Vec<f64> = (0..100).map(|i| 3.0 * 1.0 + 2.0 * 2.0 + (i as f64) * 0.01).collect();

// RobustScaler
let mut scaler = RobustScaler::new();
let x_robust = scaler.fit_transform(&x).unwrap();
let mut lr = LinearRegression::new();
lr.fit(&x_robust, &y).unwrap();
println!("coef: {:?}", lr.coef());
```

5% outliers with a different relationship will tank your R² from 0.99 to 0.27. RobustScaler won't help. Identify the type of outlier first.
