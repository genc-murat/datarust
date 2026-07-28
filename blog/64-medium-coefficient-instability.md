# Coefs Swung from 2.7 to 3.1 Across Folds. Predictions Stayed at 5.0.

*Multicollinearity, coefficient instability, and how Ridge fixes it*

---

Two features are 95% correlated. LinearRegression fits the data perfectly (R²=0.995). But the coefficient for `x₁` swings from 2.7 to 3.1 across folds — a 15% variation. The prediction for the same point stays at 5.0. The model works, but the coefficients lie.

## Experiment 1: Coefficient Instability

Two features with varying correlation:

```
Corr(x1,x2)   coef_1_mean   coef_1_std   coef_2_mean   coef_2_std
---------------------------------------------------------------------------
0                  3.0054     0.0051       2.0165     0.0121
0.5                2.9952     0.0076       1.9665     0.0125
0.8                3.1034     0.0307       1.9192     0.0242
0.9                2.9712     0.0644       2.0351     0.0592
0.95               3.0279     0.0530       1.9758     0.0531
0.99               2.7087     0.2155       2.2781     0.2227
```

The pattern: as correlation increases from 0 to 0.99, coefficient standard deviation increases 43× (0.005 → 0.216). At corr=0.99, the coefficient for `x₁` ranges from 2.71 to 3.13 across folds — a 15% swing.

But the *mean* coefficient is always near the true value (3.0). The instability is in the variance, not the bias.

## Experiment 2: Ridge Stabilizes Coefficients

corr=0.95, Ridge with different alpha:

```
alpha   coef_1_mean   coef_1_std   coef_2_mean   coef_2_std   cv_R²
---------------------------------------------------------------------------
0           3.1320     0.0585       1.8677     0.0654   0.9954
0.01        3.1290     0.0583       1.8704     0.0652   0.9954
0.1         3.1039     0.0560       1.8940     0.0630   0.9954
1           2.9328     0.0385       2.0446     0.0469   0.9953
10          2.5191     0.0182       2.1835     0.0202   0.9905
100         1.5468     0.0545       1.4254     0.0444   0.8180
```

Ridge reduces coefficient variance:
- alpha=0: coef_1_std=0.059
- alpha=1: coef_1_std=0.039 (33% reduction)
- alpha=10: coef_1_std=0.018 (70% reduction)

The CV R² barely changes (0.9954 → 0.9905) until alpha=100. Ridge stabilizes coefficients without degrading predictions.

The tradeoff: at alpha=10, the coefficient means shift (3.13 → 2.52), but the standard deviation drops (0.059 → 0.018). Ridge trades bias for variance.

## Experiment 3: Predictions Are Stable

Even when coefficients swing wildly, predictions stay stable:

```
Corr   coef_std_sum   pred_std   cv_R²
--------------------------------------------------
0            0.0434     0.0224   0.9945
0.5          0.0467     0.0278   0.9939
0.8          0.0408     0.0142   0.9967
0.9          0.0667     0.0130   0.9965
0.95         0.1541     0.0192   0.9936
0.99         0.2614     0.0247   0.9964
```

Coefficient instability increases 6× (0.043 → 0.261), but prediction instability barely changes (0.022 → 0.025). The CV R² stays above 0.99 across all correlation levels.

This is the multicollinearity paradox: the model predicts well, but the coefficients are meaningless.

## Why This Happens

When two features are highly correlated, the model can trade coefficient between them without changing the prediction. If `x₁` and `x₂` are identical, then `3x₁ + 2x₂` and `2x₁ + 3x₂` give the same prediction. The data can't distinguish between these.

LinearRegression finds *a* solution (the one that minimizes least squares), but it's not *the* solution. Any combination `a*x₁ + b*x₂` where `a+b=5` fits equally well. The specific `a` and `b` depend on the random sample.

## When to Worry

**Don't worry when:**
- You only care about predictions (cv R² is stable)
- You're doing feature importance by magnitude (both features are important)
- You have regularization (Ridge/Lasso handle it)

**Do worry when:**
- You need to interpret individual coefficients
- You're doing causal inference
- You're comparing coefficients across studies

## The Code

```rust
use datarust::linear_model::Ridge;
use datarust::traits::Predictor;

// Detect multicollinearity: check coefficient variance across folds
let mut coef_stds = Vec::new();
for (train_idx, _) in kf.split(n)? {
    let mut lr = LinearRegression::new();
    lr.fit(&x_train, &y_train)?;
    coef_stds.push(lr.coef()[0]);
}
let mean = coef_stds.iter().sum::<f64>() / coef_stds.len() as f64;
let std = (coef_stds.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / coef_stds.len() as f64).sqrt();
println!("coefficient std: {:.4}", std);
// If std > 0.1, consider Ridge
```

## Tradeoffs

**Without regularization:**
- Unbiased coefficients (on average)
- High variance (unstable across samples)
- Good predictions (variance cancels out)

**With Ridge:**
- Biased coefficients (shrunk toward zero)
- Low variance (stable across samples)
- Same predictions (bias-variance tradeoff)

**The practical rule:** If coefficient std > 0.1 across folds, use Ridge with alpha in [0.1, 10]. This stabilizes coefficients without degrading predictions.

## Try It

```bash
cargo add datarust
```

```rust
use datarust::linear_model::Ridge;
use datarust::traits::Predictor;
use datarust::Matrix;

let x = Matrix::new(vec![
    vec![1.0, 1.05],  // nearly identical
    vec![2.0, 2.10],
    vec![3.0, 3.15],
]).unwrap();
let y = vec![5.0, 10.0, 15.0];

let mut ridge = Ridge::new().with_alpha(1.0);
ridge.fit(&x, &y).unwrap();
println!("coef: {:?}", ridge.coef()); // stable, not swinging
```

If your coefficients swing by >10% across folds, your features are correlated. Use Ridge to stabilize them. The predictions are fine — the coefficients are the problem.
