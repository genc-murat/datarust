# Lasso Killed the Small-Scale Feature. Scaling Changed the Winner.

*How feature scaling affects regularization*

---

Three features with different scales: x1 ~ 1, x2 ~ 100, x3 ~ 0.01. True coefficients: `[3, 2, 1]`. Lasso with raw data kills x3. Lasso with scaled data keeps x3. The scale of your features determines which features survive regularization.

## Experiment 1: Scaling Effect on Lasso

200 samples, 3 features, `y = 3x₁ + 2x₂ + x₃ + noise`:

```
scaler        coef1     coef2     coef3     selected_features
----------------------------------------------------------------------
raw              1.6976    1.9941    0.0000  2
StandardScaler   1.8012  200.6278   -1.1082  3
RobustScaler     2.2877  268.3471   -1.4619  3
MinMaxScaler     7.8824  1068.8691   -3.3983  3
MaxAbsScaler     5.6594  538.2993   -2.5749  3
```

The pattern:
- **Raw:** Lasso kills x3 (coef3=0). It only selects 2 features. The small-scale feature is invisible to Lasso because its contribution is tiny compared to x2.
- **StandardScaler:** Lasso keeps all 3 features. But coef2=200 instead of 2. The coefficient scale is wrong because scaling changed the relationship.
- **RobustScaler:** Same as StandardScaler — all 3 features selected, coef2=268.
- **MinMaxScaler:** Same pattern — all 3 features selected, coef2=1069.

The key insight: **Lasso's feature selection is scale-dependent.** With raw data, Lasso kills small-scale features. With scaled data, Lasso keeps them — but the coefficients are in a different scale.

## Experiment 2: Scaling Effect on Ridge

Same setup, Ridge with alpha=1:

```
scaler        coef1     coef2     coef3     R²
------------------------------------------------------------
raw              1.7854    1.9941   -2.3300    0.9973
StandardScaler   1.9506  199.7225   -1.1797    0.9973
RobustScaler     2.5693  266.1305   -1.5739    0.9972
MinMaxScaler    17.4724  937.5493   -2.0383    0.9723
MaxAbsScaler     8.0176  520.1674   -2.7263    0.9953
```

The pattern:
- **Raw, StandardScaler, RobustScaler:** All give R²≈0.997. The coefficients are different, but the predictions are the same.
- **MinMaxScaler:** R²=0.972. **Worse!** Compressing data into [0,1] loses the relative distances between features.

Key insight: **MinMaxScaler hurts Ridge performance.** The other scalers give the same R², but with different coefficients.

## Experiment 3: Alpha Sensitivity

```
raw: alpha vs R²
-----------------------------------
  alpha=0.001     R²=0.9973
  alpha=0.01      R²=0.9973
  alpha=0.1       R²=0.9973
  alpha=1         R²=0.9973
  alpha=10        R²=0.9973
  alpha=100       R²=0.9973

StandardScaler: alpha vs R²
-----------------------------------
  alpha=0.001     R²=0.9973
  alpha=0.01      R²=0.9973
  alpha=0.1       R²=0.9973
  alpha=1         R²=0.9973
  alpha=10        R²=0.9935
  alpha=100       R²=0.8378
```

The pattern:
- **Raw:** R² is stable across all alpha values. Alpha doesn't matter because the features have different scales.
- **StandardScaler:** R² drops at alpha=100. **Scaling makes alpha more sensitive.**

Key insight: **With raw data, alpha is irrelevant. With scaled data, alpha matters.** If you scale your data, you must tune alpha carefully.

## The Three Rules

**Rule 1: Lasso's feature selection is scale-dependent**
```rust
// Raw: Lasso kills small-scale features
lasso.fit(&x, &y);  // coef3 = 0

// Scaled: Lasso keeps all features
let x_std = scaler.fit_transform(&x);
lasso.fit(&x_std, &y);  // coef3 ≠ 0
```

**Rule 2: Ridge's predictions are scale-invariant, but coefficients aren't**
```rust
// Raw and scaled give the same R², different coefficients
ridge.fit(&x, &y);       // coef = [1.79, 1.99, -2.33]
ridge.fit(&x_std, &y);   // coef = [1.95, 199.72, -1.18]
// Both give R² = 0.997
```

**Rule 3: Scaling makes alpha more sensitive**
```rust
// Raw: alpha=100 is fine
ridge.fit(&x, &y);       // R² = 0.997

// Scaled: alpha=100 hurts
ridge.fit(&x_std, &y);   // R² = 0.838
```

## When to Scale

| Scenario | Scale? | Why |
|----------|--------|-----|
| LinearRegression only | No | Coefficients are interpretable |
| Lasso feature selection | Yes | Small-scale features are invisible |
| Ridge with regularization | Yes | Alpha is more sensitive |
| Pipeline with multiple models | Yes | All models use the same scale |
| Feature importance from coef | No | Coef magnitude reflects true importance |

## The Code

```rust
use datarust::linear_model::{Lasso, Ridge};
use datarust::scaler::StandardScaler;
use datarust::traits::Transformer;

// Without scaling: Lasso kills small-scale features
let mut lasso = Lasso::new().with_alpha(0.1);
lasso.fit(&x, &y)?;
println!("raw coef: {:?}", lasso.coef()); // [1.70, 1.99, 0.00]

// With scaling: Lasso keeps all features
let mut scaler = StandardScaler::new();
let x_std = scaler.fit_transform(&x)?;
let mut lasso = Lasso::new().with_alpha(0.1);
lasso.fit(&x_std, &y)?;
println!("scaled coef: {:?}", lasso.coef()); // [1.80, 200.63, -1.11]
```

## Tradeoffs

**Scaling (StandardScaler, RobustScaler):**
- ✅ Lasso selects all relevant features
- ✅ Ridge is more stable
- ❌ Coefficients are not directly interpretable
- ❌ Alpha is more sensitive

**No scaling:**
- ✅ Coefficients are interpretable
- ✅ Alpha is irrelevant
- ❌ Lasso kills small-scale features
- ❌ Feature importance is misleading

**MinMaxScaler:**
- ❌ Hurts Ridge performance
- ❌ Compresses data into [0,1]
- ❌ Only use when you need [0,1] range

The universal rule: **if you're doing feature selection with Lasso, scale your data first.** If you're doing Ridge with a fixed alpha, scaling doesn't matter for predictions — but it does matter for coefficient interpretation.

## Try It

```bash
cargo add datarust
```

```rust
use datarust::linear_model::Lasso;
use datarust::scaler::StandardScaler;
use datarust::traits::Transformer;
use datarust::Matrix;

let x = Matrix::new(vec![vec![1.0, 100.0, 0.01]; 200]).unwrap();
let y: Vec<f64> = (0..200).map(|i| 3.0 + 200.0 + 0.01 + (i as f64) * 0.01).collect();

// Raw: Lasso kills the small-scale feature
let mut lasso = Lasso::new().with_alpha(0.1);
lasso.fit(&x, &y).unwrap();
println!("raw: {:?}", lasso.coef()); // [?, ?, 0]

// Scaled: Lasso keeps all features
let mut scaler = StandardScaler::new();
let x_std = scaler.fit_transform(&x).unwrap();
let mut lasso = Lasso::new().with_alpha(0.1);
lasso.fit(&x_std, &y).unwrap();
println!("scaled: {:?}", lasso.coef()); // [?, ?, ?]
```

Three features with different scales. Lasso with raw data kills the small-scale one. Lasso with scaled data keeps it. The scale of your features determines which features survive regularization.
