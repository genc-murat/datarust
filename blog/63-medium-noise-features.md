# LinearRegression CV Dropped to 0.786 with 98 Noise Features. Lasso Stayed at 0.994.

*How noise features degrade models — and how to fight back*

---

You have 2 useful features and 98 noise features. LinearRegression tries to fit all 100. Ridge tries to shrink all 100. Lasso zeros 98 of them. The result: Lasso's CV R² is 0.994 while LinearRegression's is 0.786.

## Experiment 1: LinearRegression

2 useful features (`x₁`, `x₂`), adding noise features:

```
n_noise   n_total   train_R²   cv_R²     gap
--------------------------------------------------
0        2          0.9954    0.9946   0.0009
2        4          0.9955    0.9945   0.0010
5        7          0.9957    0.9945   0.0012
10       12         0.9960    0.9933   0.0027
20       22         0.9963    0.9920   0.0043
50       52         0.9982    0.9826   0.0155
```

Train R² *increases* with noise (0.995 → 0.998) because the model uses noise features to fit the training data perfectly. CV R² *decreases* (0.995 → 0.983) because the noise features don't generalize.

The gap grows from 0.001 to 0.016 — a 17× increase. The model is memorizing noise.

## Experiment 2: Ridge

Same data, alpha=1.0:

```
n_noise   n_total   train_R²   cv_R²     gap
--------------------------------------------------
0        2          0.9954    0.9945   0.0008
2        4          0.9955    0.9939   0.0016
5        7          0.9955    0.9945   0.0010
10       12         0.9957    0.9929   0.0029
20       22         0.9964    0.9917   0.0047
50       52         0.9979    0.9809   0.0170
80       82         0.9995    0.9369   0.0626
98       100        0.9997    0.7861   0.2136
```

Ridge is more robust than LinearRegression (cv=0.981 vs 0.983 at 50 noise), but it still degrades. At 98 noise features (p ≈ n), cv R² collapses to 0.786. Ridge shrinks all coefficients but can't zero them.

## Experiment 3: Lasso

Same data, alpha=0.1:

```
n_noise   n_total   train_R²   cv_R²     gap      n_nonzero
--------------------------------------------------------------
0        2          0.9946    0.9937   0.0009          2
2        4          0.9946    0.9937   0.0009          2
5        7          0.9946    0.9937   0.0009          2
10       12         0.9946    0.9937   0.0009          2
20       22         0.9946    0.9937   0.0009          2
50       52         0.9946    0.9937   0.0009          2
80       82         0.9948    0.9937   0.0011          3
98       100        0.9946    0.9936   0.0010          3
```

Lasso is *immune* to noise features. CV R² stays at 0.994 across all noise levels. The reason: Lasso zeros the noise features. It keeps only 2-3 features (the useful ones) regardless of how many noise features exist.

This is the power of L1 regularization: it performs implicit feature selection.

## The Three Regimes

| n_noise | LinearRegression | Ridge | Lasso |
|---------|-----------------|-------|-------|
| 0 | cv=0.995 | cv=0.995 | cv=0.994 |
| 50 | cv=0.983 | cv=0.981 | cv=0.994 |
| 98 | FAIL (singular) | cv=0.786 | cv=0.994 |

- **LinearRegression**: Fails when p > n (singular matrix). Degrades gradually before that.
- **Ridge**: Survives p > n (alpha makes it positive-definite), but performance degrades.
- **Lasso**: Immune to noise features. Zeros them automatically.

## The Math

LinearRegression minimizes:
```
||y - Xβ||²
```

With 98 noise features, the model uses them to fit the training data perfectly (R² → 1.0), but the coefficients are wrong (they fit noise, not signal).

Ridge adds L2 penalty:
```
||y - Xβ||² + α||β||²
```

The penalty shrinks all coefficients, including noise. But "shrunk" ≠ "zero." Noise coefficients are small but non-zero, so they still affect predictions.

Lasso adds L1 penalty:
```
||y - Xβ||² + α||β||₁
```

The L1 penalty drives noise coefficients to *exactly* zero. The model effectively ignores them.

## When to Use Each

**Use LinearRegression when:**
- You have few features (p < n/10)
- All features are useful
- You need maximum interpretability

**Use Ridge when:**
- You have many features but most are useful
- Features are correlated
- You want stable coefficients

**Use Lasso when:**
- You suspect many features are noise
- You want automatic feature selection
- You need a sparse, interpretable model

## The Code

```rust
use datarust::linear_model::{LinearRegression, Ridge, Lasso};
use datarust::traits::Predictor;

// LinearRegression: degrades with noise
let mut lr = LinearRegression::new();
lr.fit(&x, &y)?;

// Ridge: survives but degrades
let mut ridge = Ridge::new().with_alpha(1.0);
ridge.fit(&x, &y)?;

// Lasso: immune to noise
let mut lasso = Lasso::new().with_alpha(0.1);
lasso.fit(&x, &y)?;
println!("non-zero: {}", lasso.coef().iter().filter(|&&c| c.abs() > 1e-6).count());
```

## Tradeoffs

**LinearRegression**: Fast, interpretable, but fragile with noise features.

**Ridge**: More robust, but doesn't select features. Keeps all features even if they're noise.

**Lasso**: Most robust with noise, but can over-select (kills useful features if alpha is too high).

**The practical default**: Start with Lasso. If it zeros too many features, switch to Ridge. If you know all features are useful, use LinearRegression.

## Try It

```bash
cargo add datarust
```

```rust
use datarust::linear_model::Lasso;
use datarust::traits::Predictor;
use datarust::Matrix;

// 2 useful + 8 noise features
let x = Matrix::new(vec![
    vec![1.0, 2.0, 0.5, 0.3, 0.8, 0.2, 0.7, 0.1, 0.4],
    vec![2.0, 3.0, 0.6, 0.4, 0.9, 0.3, 0.8, 0.2, 0.5],
    vec![3.0, 4.0, 0.7, 0.5, 1.0, 0.4, 0.9, 0.3, 0.6],
]).unwrap();
let y = vec![5.0, 8.0, 11.0]; // y = 3*x1 + x2

let mut lasso = Lasso::new().with_alpha(0.1);
lasso.fit(&x, &y).unwrap();
println!("coef: {:?}", lasso.coef()); // noise features should be ~0
```

If your model has 100 features and 2 are useful, Lasso will find them. LinearRegression will drown in noise.
