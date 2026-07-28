# Ridge Spread Credit to All Four Features. Lasso Gave It to Two.

*Extracting and interpreting feature importance from linear models*

---

Feature importance is the most requested explanation from stakeholders. "Which features matter?" is a simple question with a nuanced answer. Linear models give you coefficients — but how you interpret them depends entirely on regularization.

## Experiment 1: Ridge vs Lasso on Correlated Features

8 features: `signal_1`, `signal_2` (true predictors), `copy_1` (correlated with signal_1), `copy_2` (correlated with signal_2), and 4 noise features. Target: `3 × signal_1 + 2 × signal_2 + noise`.

**Ridge (alpha=1.0):**
```
feature              coef        abs(coef)
----------------------------------------------
signal_1         2.199607       2.199607
signal_2         1.560398       1.560398
copy_1           0.967027       0.967027
copy_2           0.479101       0.479101
noise_1          0.030499       0.030499
noise_2         -0.027847       0.027847
noise_3         -0.011131       0.011131
noise_4          0.007113       0.007113
```

Ridge spreads credit across correlated features. `signal_1` gets 2.20, but `copy_1` (its correlated twin) gets 0.97. The noise features get near-zero but not exactly zero. Ridge tells you "these features are all somewhat useful."

**Lasso (alpha=0.1):**
```
feature              coef        abs(coef)
----------------------------------------------
signal_1         2.935254       2.935254
signal_2         1.944859       1.944859
copy_1           0.127835       0.127835
copy_2           0.000000       0.000000
noise_1          0.000000       0.000000
noise_2          0.000000       0.000000
noise_3          0.000000       0.000000
noise_4          0.000000       0.000000
```

Lasso selects: `signal_1` gets 2.94, `signal_2` gets 1.94, everything else is zero. The L1 penalty forces irrelevant coefficients to exactly zero, producing a sparse model. Lasso tells you "these two features are the only ones that matter."

The true coefficients are 3.0 and 2.0. Lasso's estimates (2.94, 1.94) are closer to truth than Ridge's (2.20, 1.56) because Ridge's L2 penalty shrinks coefficients toward zero.

## Experiment 2: The Lasso Alpha Path

How does alpha affect which features Lasso selects?

```
alpha      signal_1   copy_1   noise_1   n_nonzero
------------------------------------------------------
0.001      3.052933   0.1156   0.03135         7
0.010      3.017795   0.1404   0.02050         6
0.050      2.955073   0.1591   0.00000         3
0.100      2.935254   0.1278   0.00000         3
0.500      2.654464   0.0000   0.00000         2
1.000      2.144448   0.0000   0.00000         2
5.000      0.000000   0.0000   0.00000         0
```

The pattern:
- **alpha=0.001**: 7 features active. Noise features have tiny but non-zero coefficients.
- **alpha=0.05**: 3 features active. Noise is eliminated, but `copy_1` survives because it carries real signal (it's correlated with `signal_1`).
- **alpha=0.5**: 2 features active. `copy_1` is eliminated — its contribution is redundant with `signal_1`.
- **alpha=5.0**: 0 features. The penalty is so strong it kills everything, including the true signals.

The practical insight: alpha=0.1 gives the right answer (selects the true signals), but alpha=5.0 destroys the model. Regularization is not "more is better" — there's a sweet spot.

## Experiment 3: Logistic Regression Odds Ratios

For classification, coefficients have a probabilistic interpretation. The odds ratio is `exp(coef)`:

```
feature              coef     odds ratio
--------------------------------------------
important_1      1.915213       6.7884
important_2     -1.751675       0.1735
noise_4         -0.377393       0.6856
weak_3           0.007209       1.0072

Accuracy: 0.8350
```

The odds ratio tells you: "a one-unit increase in this feature multiplies the odds of class=1 by this factor."

- `important_1`: odds ratio 6.79 — one unit increases the odds by 6.8x. Strong positive predictor.
- `important_2`: odds ratio 0.17 — one unit decreases the odds by 5.8x. Strong negative predictor.
- `weak_3`: odds ratio 1.007 — negligible effect. The feature doesn't matter.
- `noise_4`: odds ratio 0.69 — this is a false positive. The noise feature happened to correlate with the target in this sample.

## The Interpretation Matrix

| Model | Coefficient means | Use for |
|-------|-------------------|---------|
| Ridge | "Direction and magnitude of effect, shared across correlated features" | When you have correlated features and want to keep all of them |
| Lasso | "Which features matter (non-zero) and by how much" | Feature selection, sparse models |
| LogisticRegression | "Log-odds change per unit feature" | Probability interpretation, odds ratios |

## Standardization Matters

The coefficients above are from standardized features (zero mean, unit variance). Without standardization, coefficient magnitudes reflect both the feature's importance *and* its scale. A feature measured in millimeters will have a smaller coefficient than the same feature measured in kilometers, even if it's more important.

Rule of thumb: always standardize before interpreting coefficient magnitudes as feature importance.

```rust
fn standardize(x: &Matrix) -> (Matrix, Vec<f64>, Vec<f64>) {
    // Compute mean and std per column
    // Return (standardized_x, means, stds)
}
```

## Tradeoffs

**Ridge** is stable: small changes in data produce small changes in coefficients. But it never zeroes anything, so you can't use it for feature selection.

**Lasso** is unstable with correlated features: it might pick `signal_1` in one run and `copy_1` in another, depending on the random sample. The coefficients are correct on average but vary across samples.

**ElasticNet** (not implemented in datarust yet) combines L1 and L2, getting Lasso's sparsity with Ridge's stability. It's the practical default for feature importance when you have correlated features.

For production use: don't trust a single model's coefficients. Bootstrap the data, fit 100 models, and report the median coefficient and its confidence interval. If a coefficient is consistently non-zero across bootstraps, it's important.

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
println!("Ridge: {:?}", ridge.coef());

let mut lasso = Lasso::new().with_alpha(0.1);
lasso.fit(&x, &y).unwrap();
println!("Lasso: {:?}", lasso.coef());
```

If someone asks "which features matter?" and you hand them a Ridge model with 50 non-zero coefficients, you haven't answered the question. Lasso gives you a sparse answer. But the real answer requires bootstrapping, standardization, and understanding that "importance" is model-dependent, not absolute.
