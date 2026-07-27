---
title: "Ridge Kept All 5 Features at Every Alpha. Lasso Killed 3 at alpha=0.05."
subtitle: "How regularization controls model complexity"
author: "Murat Genc"
date: "2026-07-26"
tags: ["machine-learning", "rust", "datarust", "regularization", "ridge", "lasso", "alpha"]
series: "datarust-v06"
---

Regularization is a dial, not a switch. Turning it too low gives you overfitting. Turning it too high gives you underfitting. The art is finding the sweet spot — and understanding what the dial actually does.

## Experiment 1: Ridge Alpha Path

5 features, 2 useful (`x₁`, `x₂`), 3 noise:

```
alpha      train_R²   cv_R²     gap     n_nonzero
-------------------------------------------------------
0.001        0.9952    0.9935   0.0017          5
0.01         0.9952    0.9935   0.0017          5
0.1          0.9952    0.9935   0.0017          5
1            0.9951    0.9935   0.0016          5
10           0.9856    0.9836   0.0020          5
100          0.7343    0.7133   0.0210          5
1000         0.1702    0.1307   0.0395          5
```

Ridge keeps all 5 features at every alpha. The coefficients shrink, but none reach zero. This is the L2 penalty: it penalizes the sum of squared coefficients, which shrinks everything proportionally.

The CV R² stays stable (0.9935) from alpha=0.001 to alpha=1, then drops at alpha=10 (0.9836) and collapses at alpha=100 (0.7133). The sweet spot is alpha ≤ 1.

## Experiment 2: Lasso Alpha Path

Same data:

```
alpha      train_R²   cv_R²     gap     n_nonzero
-------------------------------------------------------
0.001        0.9952    0.9935   0.0017          5
0.01         0.9952    0.9936   0.0016          5
0.05         0.9947    0.9934   0.0013          2  ← feature selection!
0.1          0.9938    0.9926   0.0011          2
0.5          0.9657    0.9643   0.0014          2
1            0.8781    0.8735   0.0046          2
5            0.0000   -0.0410   0.0410          0
```

Lasso does what Ridge can't: it selects features. At alpha=0.05, it zeros 3 noise features, keeping only `x₁` and `x₂`. The CV R² barely changes (0.9935 → 0.9934), but the model is now simpler.

At alpha=5, Lasso kills everything — even the useful features. The model predicts the mean (R²=0.000). This is the underfitting trap.

## Experiment 3: Ridge Coefficient Magnitudes

```
alpha      coef_1     coef_2     |coef|_sum
----------------------------------------------
0.001      3.0204     2.0236     5.1515
0.01       3.0201     2.0234     5.1510
0.1        3.0179     2.0215     5.1463
1          2.9957     2.0030     5.0995
10         2.7918     1.8351     4.7457
100        1.6761     0.9988     2.8948
```

Ridge shrinks coefficients proportionally. At alpha=0.001, the sum of absolute coefficients is 5.15. At alpha=100, it's 2.89 — a 44% reduction. But the ratio between `coef_1` and `coef_2` stays roughly constant (3.0/2.0 ≈ 1.5).

Ridge doesn't distinguish between useful and useless features. It treats them all equally.

## The Ridge vs Lasso Decision Tree

```
Do you want feature selection?
├── Yes → Use Lasso
│         ├── alpha too high? → All features zeroed
│         └── alpha too low?  → No selection
└── No  → Use Ridge
          ├── alpha too high? → Underfitting (coefficients too small)
          └── alpha too low?  → Overfitting (coefficients too large)
```

## The Alpha Selection Pattern

```rust
use datarust::linear_model::{Ridge, Lasso};
use datarust::model_selection::KFold;
use datarust::traits::Predictor;

let kf = KFold::new().with_n_splits(5).with_shuffle(true);
let mut best_alpha = 0.0;
let mut best_cv_r2 = f64::NEG_INFINITY;

for alpha in &[0.001, 0.01, 0.1, 1.0, 10.0, 100.0] {
    let mut cv_scores = Vec::new();
    for (train_idx, test_idx) in kf.split(n)? {
        let mut model = Ridge::new().with_alpha(*alpha);
        model.fit(&x_train, &y_train)?;
        let pred = model.predict(&x_test)?;
        cv_scores.push(r2_score(&y_test, &pred)?);
    }
    let mean_cv = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;
    if mean_cv > best_cv_r2 {
        best_cv_r2 = mean_cv;
        best_alpha = *alpha;
    }
}
```

The pattern: try a log-spaced grid of alphas, pick the one with highest CV R². For Ridge, the optimal alpha is usually in [0.1, 10]. For Lasso, it's usually in [0.01, 1].

## Tradeoffs

**Ridge pros:**
- Stable: small changes in data produce small changes in coefficients
- Never zeros features: good when all features are useful
- Closed-form solution: fast to compute

**Ridge cons:**
- No feature selection: keeps all features even if some are noise
- Harder to interpret: all features have non-zero coefficients

**Lasso pros:**
- Feature selection: zeros irrelevant features
- Sparse model: easier to interpret
- Implicit dimensionality reduction

**Lasso cons:**
- Unstable with correlated features: might pick either feature
- No closed-form: requires iterative optimization
- Can over-zero: kills useful features if alpha is too high

**ElasticNet** (not in datarust yet) combines both: L1 for selection, L2 for stability. It's the practical default when you have correlated features.

## When to Use Each

| Scenario | Ridge | Lasso |
|----------|-------|-------|
| All features are useful | ✓ | ✗ (might zero useful ones) |
| Many features are noise | ✗ (keeps them all) | ✓ |
| Features are correlated | ✓ (stable) | ✗ (unstable selection) |
| You need interpretability | ✗ (all features) | ✓ (sparse model) |
| You need speed | ✓ (closed-form) | ✗ (iterative) |

## Try It

```bash
cargo add datarust
```

```rust
use datarust::linear_model::{Ridge, Lasso};
use datarust::traits::Predictor;
use datarust::Matrix;

let x = Matrix::new(vec![
    vec![1.0, 0.0, 0.5],
    vec![0.0, 1.0, 0.3],
    vec![1.0, 1.0, 0.8],
]).unwrap();
let y = vec![2.0, 0.0, 2.0];

let mut ridge = Ridge::new().with_alpha(1.0);
ridge.fit(&x, &y).unwrap();
println!("Ridge: {:?}", ridge.coef()); // all non-zero

let mut lasso = Lasso::new().with_alpha(0.5);
lasso.fit(&x, &y).unwrap();
println!("Lasso: {:?}", lasso.coef()); // some zero
```

If you don't know whether your features are all useful, start with Lasso. If it zeros too many, switch to Ridge. If you need both selection and stability, wait for ElasticNet.
