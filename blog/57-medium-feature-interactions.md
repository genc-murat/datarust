# Linear Model Got R²=0.07. Polynomial Features Got R²=0.99. The Data Had Interactions.

*When and how to detect feature interactions*

---

A linear model scores R²=0.07 on your data. You check the features, the target, the preprocessing — everything looks correct. The problem isn't the model or the data. It's that the relationship has interactions, and linear models can't see them.

## The Three Relationships

### Experiment 1: Pure Interaction (y = 3x₁x₂ + noise)

The target depends *only* on the product of two features:

```
Linear (2 features):     R² = 0.0685, coef = [-0.39, -0.78]
Polynomial (6 features): R² = 0.9937, coef = [x1*x2 = 2.98]
Interaction (3 features): R² = 0.9936, coef = [x1*x2 = 2.98]
```

Linear model fails completely (R²=0.07). It tries to fit `ax₁ + bx₂`, but the true relationship is `3x₁x₂`. No combination of linear terms can approximate a product.

PolynomialFeatures with degree=2 creates 5 features: `[x₁, x₂, x₁², x₁x₂, x₂²]`. The model learns that x₁x₂ has coefficient 2.98 — almost exactly the true value (3.0). The squared terms get near-zero coefficients.

Interaction-only mode creates 3 features: `[x₁, x₂, x₁x₂]`. Same R², same coefficient for the interaction term. When you know the relationship has no squared terms, interaction-only is more efficient.

### Experiment 2: Quadratic Without Interaction (y = x₁² + x₂² + noise)

The target depends on squared features but not their product:

```
Linear (2 features):     R² = 0.0364
Polynomial (6 features): R² = 0.9748, coef = [x1² = 0.996, x2² = 0.995]
Interaction (3 features): R² = 0.0485
```

Linear model fails (R²=0.04). PolynomialFeatures succeeds (R²=0.97) with the correct coefficients. Interaction-only fails (R²=0.05) because there are no interactions — only squared terms.

This is the diagnostic: if interaction-only gives low R² but polynomial gives high R², the relationship has squared terms but not interactions.

### Experiment 3: Linear (y = x₁ + x₂ + noise)

The target depends linearly on features:

```
Linear (2 features):     R² = 0.9612, coef = [0.99, 1.01]
Polynomial (6 features): R² = 0.9619, coef = [x1 = 0.99, x2 = 1.01, ...]
```

Both give the same R². Adding polynomial features doesn't hurt (R² stays at 0.96) but doesn't help either. The extra features get near-zero coefficients.

This is the important case: polynomial features are *safe* when the relationship is linear. They add computation but don't degrade performance.

## The Diagnostic Pattern

| True relationship | Linear R² | Polynomial R² | Interaction R² |
|-------------------|-----------|---------------|----------------|
| x₁ × x₂ | Low | High | High |
| x₁² + x₂² | Low | High | Low |
| x₁ + x₂ | High | High | High |
| x₁ × x₂ + x₁² | Low | High | Low |

The pattern:
1. If Polynomial R² >> Linear R²: there are non-linear terms
2. If Interaction R² ≈ Polynomial R²: only interactions matter
3. If Interaction R² << Polynomial R²: squared terms matter
4. If all R² are similar: the relationship is linear

## The Code

```rust
use datarust::linear_model::LinearRegression;
use datarust::polynomial::PolynomialFeatures;
use datarust::traits::{Predictor, Transformer};
use datarust::Matrix;

let x = Matrix::new(vec![
    vec![1.0, 2.0],
    vec![2.0, 3.0],
    vec![3.0, 4.0],
]).unwrap();
let y = vec![6.0, 18.0, 36.0]; // y = x1 * x2

// Linear: fails
let mut lr = LinearRegression::new();
lr.fit(&x, &y).unwrap();
println!("linear R²: {:.4}", r2_score(&y, &lr.predict(&x).unwrap()).unwrap());

// Polynomial: succeeds
let mut pf = PolynomialFeatures::new(2).include_bias(false);
let x_poly = pf.fit_transform(&x).unwrap();
lr.fit(&x_poly, &y).unwrap();
println!("polynomial R²: {:.4}", r2_score(&y, &lr.predict(&x_poly).unwrap()).unwrap());
```

## Feature Count Explosion

For `n` input features and degree `d`, PolynomialFeatures generates:

```
C(n + d, d) = (n + d)! / (n! × d!)
```

| Features | Degree 2 | Degree 3 | Degree 4 |
|----------|----------|----------|----------|
| 2 | 6 | 10 | 15 |
| 5 | 21 | 56 | 126 |
| 10 | 66 | 286 | 1001 |
| 50 | 1326 | 23426 | 270725 |

With 50 features and degree 3, you get 23,426 features. This is the "curse of dimensionality" in action — polynomial features cause combinatorial explosion.

The practical limit: degree=2 with <20 features, degree=3 with <10 features. Beyond that, use interaction-only or feature selection.

## Tradeoffs

**PolynomialFeatures的优点:**
- Captures non-linear relationships
- Works with any linear model (Ridge, Lasso, LinearRegression)
- Feature names are interpretable

**PolynomialFeatures的缺点:**
- Combinatorial explosion with many features
- Can cause overfitting (especially degree > 2)
- Sensitive to feature scaling (polynomial features amplify scale differences)

**The scaling trap:** If x₁ ranges [0, 1] and x₂ ranges [0, 1000], then x₁x₂ ranges [0, 1000] while x₁² ranges [0, 1]. The interaction term dominates. Always standardize features before creating polynomial features.

**When to use degree=2:**
- You suspect interactions or quadratic effects
- You have <20 features
- You're willing to accept 3× more features

**When to use interaction-only:**
- You specifically want to detect interactions
- You don't believe in squared effects
- You want fewer features than full polynomial

**When to skip polynomial features:**
- You have >50 features
- The relationship is known to be linear
- You need fast training/inference

## Try It

```bash
cargo add datarust
```

```rust
use datarust::polynomial::PolynomialFeatures;
use datarust::traits::Transformer;
use datarust::Matrix;

let x = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

let mut pf = PolynomialFeatures::new(2).include_bias(false);
let x_poly = pf.fit_transform(&x).unwrap();
println!("{} → {} features", x.ncols(), x_poly.ncols());
// 2 → 5 features: [x1, x2, x1², x1*x2, x2²]

let mut pf_int = PolynomialFeatures::new(2).include_bias(false).interaction_only(true);
let x_int = pf_int.fit_transform(&x).unwrap();
println!("{} → {} features (interaction only)", x.ncols(), x_int.ncols());
// 2 → 3 features: [x1, x2, x1*x2]
```

If your linear model scores R² < 0.5, don't blame the model. Check whether the relationship has interactions. PolynomialFeatures tells you in 3 lines of code.
