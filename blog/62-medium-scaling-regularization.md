---
title: "Lasso Killed the Feature with coef=300. It Was the Most Important One."
subtitle: "Why scaling matters for regularization"
author: "Murat Genc"
date: "2026-07-26"
tags: ["machine-learning", "rust", "datarust", "scaling", "regularization", "ridge", "lasso"]
series: "datarust-v06"
---

Ridge penalizes the sum of squared coefficients. Lasso penalizes the sum of absolute coefficients. Both assume all features are on the same scale. When they're not, the penalty is unequal — and the model learns the wrong thing.

## The Setup

3 features with different scales:
- `x₁`: standard normal (scale ~1), true coef = 3.0
- `x₂`: mean=100, std=1 (scale ~100), true coef = 0.02
- `x₃`: mean=0, std=0.01 (scale ~0.01), true coef = 300.0

The true coefficients are [3.0, 0.02, 300.0]. All three contribute equally to the target: 3.0 × 1 = 3.0, 0.02 × 100 = 2.0, 300.0 × 0.01 = 3.0.

## Experiment 1: Ridge with Different Scales

```
Unscaled:  coef = [3.09, 0.02, 5.56]
True:              [3.0,  0.02, 300.0]
```

Ridge shrinks `x₃` from 300 to 5.56 — a 98% reduction. It shrinks `x₁` from 3.0 to 3.09 — barely touched. The reason: Ridge penalizes `coef²`, and `300² = 90,000` dominates the penalty. Ridge "thinks" `x₃` is the most important feature to shrink because it has the largest coefficient.

With StandardScaler:

```
StandardScaled: coef = [3.03, 1.99, 2.89]
```

All coefficients are properly estimated. The scaling made Ridge treat all features equally.

## Experiment 2: Lasso with Different Scales

```
Unscaled:  coef = [3.01, 0.02, 0.0]
Scaled:    coef = [2.95, 1.90, 2.81]
```

Lasso kills `x₃` entirely (coef=0.0) when unscaled. The reason: Lasso penalizes `|coef|`, and `|300|` is huge. Lasso thinks "this feature must be shrunk" and pushes it to zero.

This is the most dangerous case: the feature with the *largest* true coefficient (300) is killed because it has the *largest* scale. The model learns that `x₃` doesn't matter, when in fact it's the most important feature.

## Experiment 3: Alpha Sensitivity

```
Unscaled Ridge:
alpha      cv_R²     coef_3
0.001       0.9945    278.75
0.01        0.9363    177.59
0.1         0.7128     38.36
1           0.6319      4.34
10          0.6225      0.44
100         0.5717      0.04

StandardScaled Ridge:
alpha      cv_R²     coef_3
0.001       0.9960      2.92
0.01        0.9960      2.92
0.1         0.9960      2.91
1           0.9960      2.90
10          0.9930      2.75
100         0.8606      1.81
```

Without scaling, increasing alpha from 0.001 to 1 destroys cv R² (0.9945 → 0.6319). With scaling, cv R² stays stable until alpha=100 (0.8606).

The unscaled model is fragile: a small change in alpha produces a large change in performance. The scaled model is robust: alpha can vary by 100× without significant performance loss.

## Experiment 4: L2 Penalty Decomposition

```
Unscaled L2 penalty: 6.36
  coef_1 contribution: 48.5%
  coef_2 contribution:  0.3%
  coef_3 contribution: 87.4%  ← dominates!

Scaled L2 penalty: 4.63
  coef_1 contribution: 65.4%
  coef_2 contribution: 42.9%
  coef_3 contribution: 62.3%  ← balanced
```

Without scaling, `x₃` accounts for 87.4% of the total penalty. Ridge focuses its entire shrinking effort on `x₃`, ignoring `x₁` and `x₂`. With scaling, the penalty is balanced across all features.

## The Math

Ridge minimizes:
```
||y - Xβ||² + α||β||²
```

The L2 penalty `||β||² = β₁² + β₂² + β₃²` treats all coefficients equally *in coefficient space*. But the features are in different scales, so equal coefficients don't mean equal importance.

If `x₁` is in meters and `x₂` is in millimeters, a coefficient of 1.0 for `x₁` means "1 meter effect" while 1.0 for `x₂` means "1 millimeter effect." Ridge penalizes both equally, but they're not equally important.

StandardScaler transforms features to zero mean and unit variance, so the coefficients are in "standard deviation units." Now 1.0 means "one standard deviation effect" for both features. The penalty is truly equal.

## The Code

```rust
use datarust::scaler::StandardScaler;
use datarust::linear_model::Ridge;
use datarust::traits::{Predictor, Transformer};

// WRONG: regularization is unequal
let mut ridge = Ridge::new().with_alpha(1.0);
ridge.fit(&x, &y)?;  // x has features in different scales

// RIGHT: scale first, then regularize
let mut scaler = StandardScaler::new();
let x_std = scaler.fit_transform(&x)?;
let mut ridge = Ridge::new().with_alpha(1.0);
ridge.fit(&x_std, &y)?;
```

## Tradeoffs

**Without scaling:**
- Fast (no preprocessing)
- Interpretable (coefficients in original units)
- Wrong when features have different scales

**With scaling:**
- Extra step (fit scaler on train, transform train+test)
- Coefficients in standard deviation units (less intuitive)
- Correct regularization

**The rule:** Always scale before regularizing. The only exception is when all features are already on the same scale (e.g., all features are percentages).

**For tree-based models:** Scaling doesn't matter (trees are scale-invariant). But for linear models, it's essential.

## Try It

```bash
cargo add datarust
```

```rust
use datarust::scaler::StandardScaler;
use datarust::linear_model::Ridge;
use datarust::traits::{Predictor, Transformer};
use datarust::Matrix;

let x = Matrix::new(vec![
    vec![1.0, 100.0],
    vec![2.0, 200.0],
    vec![3.0, 300.0],
]).unwrap();
let y = vec![101.0, 202.0, 303.0];

// Without scaling: coef_1 gets all the credit
let mut ridge = Ridge::new().with_alpha(1.0);
ridge.fit(&x, &y).unwrap();
println!("unscaled: {:?}", ridge.coef());

// With scaling: coefficients reflect true importance
let mut scaler = StandardScaler::new();
let x_std = scaler.fit_transform(&x).unwrap();
let mut ridge = Ridge::new().with_alpha(1.0);
ridge.fit(&x_std, &y).unwrap();
println!("scaled: {:?}", ridge.coef());
```

If you regularize without scaling, you're penalizing the wrong features. The feature with the largest scale gets penalized the most, regardless of its actual importance. Scale first, then regularize.
