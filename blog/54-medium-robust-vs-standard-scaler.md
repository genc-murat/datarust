---
title: "One Outlier Pushed StandardScaler's Coefficients to 21.3. RobustScaler Stayed at 3.6."
subtitle: "When median beats mean for feature scaling"
author: "Murat Genc"
date: "2026-07-26"
tags: ["machine-learning", "rust", "datarust", "preprocessing", "outliers", "scaling"]
series: "datarust-v06"
---

You have 200 data points and one is a measurement error. StandardScaler uses the mean and standard deviation — both are sensitive to outliers. The outlier inflates the standard deviation, compressing all other points toward zero. RobustScaler uses the median and IQR, which are resistant to outliers. The result: coefficients that actually reflect the data, not the error.

## Experiment 1: Clean Data (No Outliers)

LinearRegression on `3x₁ + 2x₂ + noise`:

```
No scaling:    R² = 0.9954, coef = [2.99, 2.00]
StandardScaler: R² = 0.9954, coef = [3.51, 1.99]
RobustScaler:  R² = 0.9954, coef = [5.08, 2.61]
```

All three give the same R². The coefficients differ because the scalers produce different scales — but the model learns the correct relationship either way. With clean data, it doesn't matter which scaler you use.

## Experiment 2: 5% Outliers

10 out of 200 points are outliers (10× magnitude):

```
StandardScaler stats: mean=[-0.24, -0.26], std=[2.37, 1.54]
RobustScaler stats:  center=[-0.18, -0.18], scale=[1.53, 1.47]
```

StandardScaler's mean is pulled toward the outliers (-0.24 instead of ~0). Its standard deviation is inflated (2.37 instead of ~1.0). RobustScaler's median (-0.18) and IQR (1.53) are barely affected.

The coefficients tell the real story:

```
No scaling:        R² = 0.9894, coef = [3.03, 1.96]
StandardScaler:    R² = 0.9894, coef = [7.17, 3.01]
RobustScaler:      R² = 0.9894, coef = [4.62, 2.89]
```

The R² is identical (it's scale-invariant), but StandardScaler's coefficient for feature 0 is 7.17 — more than double the true value (3.0). RobustScaler's is 4.62, closer to truth but still affected because the outlier affects the *target* as well.

## Experiment 3: Single Extreme Outlier

One point at (100, 100) in otherwise standard-normal data:

```
StandardScaler stats: mean=[0.53, 0.46], std=[7.12, 7.12]
RobustScaler stats:  center=[0.04, -0.02], scale=[1.20, 1.18]
```

One outlier inflates StandardScaler's standard deviation from ~1.0 to 7.12 — a 7× distortion. RobustScaler's IQR stays at 1.20, barely affected.

```
StandardScaler:    R² = 0.9999, coef = [21.30, 14.28]
RobustScaler:      R² = 0.9999, coef = [3.60, 2.37]
```

StandardScaler's coefficients are 21.3 and 14.3 — wildly wrong. RobustScaler's are 3.6 and 2.4, close to the true 3.0 and 2.0. The R² looks perfect (0.9999) because the outlier fits its own prediction — the model learned to predict the outlier perfectly at the cost of everything else.

## The Math

StandardScaler computes:
```
z = (x - mean) / std
```

One outlier at x=100 with n=200:
- mean shifts from ~0 to 0.53
- std inflates from ~1.0 to 7.12

RobustScaler computes:
```
z = (x - median) / IQR
```

The same outlier:
- median stays at 0.04 (barely moved)
- IQR stays at 1.20 (barely moved)

The median is the 50th percentile. The IQR is the 75th minus 25th percentile. Both are order statistics — they depend on rank, not magnitude. A single extreme value changes the rank of a few points but doesn't affect the median or IQR.

## When to Use Each

| Scenario | StandardScaler | RobustScaler |
|----------|---------------|--------------|
| Clean data, no outliers | ✓ | ✓ (but unnecessary) |
| Data with outliers | ✗ (distorted) | ✓ |
| Sparse data (many zeros) | ✓ | ✗ (median may be 0) |
| Production with unknown data quality | ✗ | ✓ (safe default) |

The practical rule: if you don't know whether your data has outliers, use RobustScaler. It's never worse than StandardScaler on clean data, and it's dramatically better on dirty data.

## The Code

```rust
use datarust::scaler::{StandardScaler, RobustScaler};
use datarust::traits::Transformer;

// StandardScaler — sensitive to outliers
let mut std_scaler = StandardScaler::new();
let x_std = std_scaler.fit_transform(&x)?;

// RobustScaler — resistant to outliers
let mut robust_scaler = RobustScaler::new();
let x_robust = robust_scaler.fit_transform(&x)?;
```

Both implement `Transformer`, so they work identically in pipelines. The only difference is what statistics they compute during `fit`.

## Tradeoffs

**RobustScaler's downside**: it uses more data for the IQR calculation than StandardScaler needs for mean/std. With very small datasets (<20 samples), the IQR can be unstable. StandardScaler is more efficient with small, clean data.

**StandardScaler's upside**: when there are no outliers, it's the maximum-likelihood estimator for Gaussian data. RobustScaler throws away information by ignoring the tails.

**The real-world compromise**: use RobustScaler for exploration and initial modeling. Switch to StandardScaler only if you've verified there are no outliers and you need the efficiency gain.

## Try It

```bash
cargo add datarust
```

```rust
use datarust::scaler::{StandardScaler, RobustScaler};
use datarust::traits::Transformer;
use datarust::Matrix;

let x = Matrix::new(vec![
    vec![1.0, 2.0],
    vec![1.1, 2.1],
    vec![0.9, 1.9],
    vec![100.0, 100.0],  // outlier
]).unwrap();

let mut std_s = StandardScaler::new();
let x_std = std_s.fit_transform(&x)?;
println!("std mean={:?}, std={:?}", std_s.mean(), std_s.std());

let mut rob_s = RobustScaler::new();
let x_rob = rob_s.fit_transform(&x)?;
println!("robust center={:?}, scale={:?}", rob_s.center(), rob_s.scale());
```

The outlier at (100, 100) will inflate StandardScaler's std to ~50. RobustScaler's IQR will stay near 0.2. Your coefficients will thank you.
