# All Scalers Gave the Same R². But the Coefficients Were Completely Different.

*How feature scaling affects model interpretation*

---

Five scalers. Same R²=0.9977. But the coefficients are wildly different. Scaling doesn't change predictions — it changes interpretation. And if you misinterpret scaled coefficients, your conclusions are wrong.

## Experiment 1: Scaling Effect on LinearRegression

True relationship: `y = 3x₁ + 2x₂ + noise`. x1 ~ N(0,1), x2 ~ N(0,100).

```
scaler        R²      coef1     coef2
---------------------------------------------
raw              0.9977    2.9745    1.9996
StandardScaler   0.9977    3.4863   19.9163
RobustScaler     0.9977    5.0467   26.1526
MinMaxScaler     0.9977   19.3283  100.2924
MaxAbsScaler     0.9977   11.2900   53.4779
```

The pattern:
- **All scalers give the same R².** Predictions are identical.
- **Raw coefficients are closest to truth:** [2.97, 2.00] ≈ [3, 2].
- **StandardScaler:** [3.49, 19.92]. The second coefficient is 10x too large.
- **MinMaxScaler:** [19.33, 100.29]. The second coefficient is 50x too large.

Key insight: **Scaling changes coefficient interpretation.** Raw coefficients reflect true importance. Scaled coefficients reflect scale-adjusted importance.

## Experiment 2: Scaling Effect on Ridge

Same setup, Ridge with alpha=1:

```
scaler        R²      coef1     coef2
---------------------------------------------
raw              0.9977    2.9637    1.9995
StandardScaler   0.9977    3.4696   19.8173
RobustScaler     0.9976    4.9960   25.9293
MinMaxScaler     0.9777   16.8258   89.0207
MaxAbsScaler     0.9956   10.7418   51.6200
```

The pattern:
- **StandardScaler:** Same R²=0.9977. Safe.
- **MinMaxScaler:** R²=0.9777. **Worse!** Compressing data into [0,1] loses information.
- **MaxAbsScaler:** R²=0.9956. Slightly worse.

Key insight: **MinMaxScaler hurts Ridge performance.** The other scalers are safe.

## Experiment 3: Scaling Effect on Lasso

Same setup, Lasso with alpha=0.1:

```
scaler        R²      coef1     coef2     selected
-------------------------------------------------------
raw              0.9977    2.9017    1.9986  2/2
StandardScaler   0.9977    3.3870   19.8170  2/2
RobustScaler     0.9976    4.8384   25.9814  2/2
MinMaxScaler     0.9962   16.2733   97.7752  2/2
MaxAbsScaler     0.9972   10.2471   52.7627  2/2
```

The pattern:
- **All scalers select 2/2 features.** Feature selection is the same.
- **All scalers give the same R².** Predictions are the same.
- **Coefficients are different.** Interpretation is different.

Key insight: **Scaling doesn't change which features Lasso selects.** It changes how you interpret the coefficients.

## Experiment 4: Scaling with Outliers

10% outliers following a different relationship (`y = -x + 0.5x`):

```
scaler        R²
-------------------------
raw              0.5601
StandardScaler   0.5601
RobustScaler     0.5601
```

All scalers give the same R². **RobustScaler doesn't help** because the outliers follow a different relationship, not just extreme values.

Key insight: **RobustScaler only helps with Type 1 outliers** (extreme values from same distribution). It doesn't help with Type 2 outliers (different relationship).

## The Five Scalers

| Scaler | R² | Coefficient Interpretation | Use Case |
|--------|-----|---------------------------|----------|
| raw | Same | True importance | Default |
| StandardScaler | Same | Unit variance importance | When features have different units |
| RobustScaler | Same | Median/IQR importance | When outliers exist |
| MinMaxScaler | Worse | [0,1] range importance | When you need [0,1] |
| MaxAbsScaler | Same | [-1,1] range importance | When you need [-1,1] |

## The Code

```rust
use datarust::linear_model::LinearRegression;
use datarust::scaler::StandardScaler;
use datarust::traits::Transformer;

// Raw: coefficients reflect true importance
let mut lr = LinearRegression::new();
lr.fit(&x, &y)?;
println!("raw coef: {:?}", lr.coef()); // [2.97, 2.00]

// Scaled: coefficients reflect scale-adjusted importance
let mut scaler = StandardScaler::new();
let x_std = scaler.fit_transform(&x)?;
let mut lr = LinearRegression::new();
lr.fit(&x_std, &y)?;
println!("scaled coef: {:?}", lr.coef()); // [3.49, 19.92]
```

## When to Scale

**For prediction only:**
- Scaling doesn't matter
- All scalers give the same R²
- Use raw data for simplicity

**For interpretation:**
- Scaling matters a lot
- Raw coefficients reflect true importance
- Scaled coefficients reflect scale-adjusted importance

**For regularization:**
- Scaling affects which features Lasso selects
- Scaling affects alpha sensitivity
- StandardScaler is safe for Ridge

**For outliers:**
- RobustScaler only helps with Type 1 outliers
- Type 2 outliers need outlier removal
- Scaling doesn't help with different relationships

## Tradeoffs

**Raw data:**
- ✅ Coefficients are interpretable
- ✅ Simple
- ❌ Lasso may kill small-scale features
- ❌ Regularization is scale-dependent

**StandardScaler:**
- ✅ Safe for most models
- ✅ Lasso selects all relevant features
- ❌ Coefficients are not directly interpretable
- ❌ Alpha is more sensitive

**MinMaxScaler:**
- ✅ Data is in [0,1] range
- ❌ Hurts Ridge performance
- ❌ Loses relative distances
- ❌ Sensitive to outliers

The universal rule: **scaling changes interpretation, not predictions.** If you only care about predictions, skip scaling. If you care about coefficients, use raw data. If you're doing regularization, use StandardScaler.

## Try It

```bash
cargo add datarust
```

```rust
use datarust::linear_model::LinearRegression;
use datarust::scaler::StandardScaler;
use datarust::traits::Transformer;
use datarust::Matrix;

let x = Matrix::new(vec![vec![1.0, 100.0]; 200]).unwrap();
let y: Vec<f64> = (0..200).map(|i| 3.0 + 200.0 + (i as f64) * 0.01).collect();

// Raw
let mut lr = LinearRegression::new();
lr.fit(&x, &y).unwrap();
println!("raw: {:?}", lr.coef());

// Scaled
let mut scaler = StandardScaler::new();
let x_std = scaler.fit_transform(&x).unwrap();
let mut lr = LinearRegression::new();
lr.fit(&x_std, &y).unwrap();
println!("scaled: {:?}", lr.coef());
```

All scalers give the same R². But the coefficients are completely different. Scaling doesn't change predictions — it changes interpretation.
