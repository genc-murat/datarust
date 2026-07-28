# Train R² Stayed at 0.97. CV R² Dropped to -4.53. The Degree Was Too High.

*The overfitting trap in polynomial features*

---

PolynomialFeatures is seductive: add more features, get better training performance. But the training performance is a lie. The real story is in the gap between training and cross-validation R².

## Experiment 1: True Degree=2, Trying Degrees 1-8

Single feature, true relationship is `x² + noise`:

```
degree   n_features   train_R²   cv_R²     gap
-------------------------------------------------------
1       1              0.1579   -0.6442   0.8021
2       2              0.9739    0.9404   0.0336  ← optimal
3       3              0.9741    0.9393   0.0348
4       4              0.9746    0.8896   0.0850
5       5              0.9754    0.2424   0.7330
6       6              0.9767   -1.8731   2.8498
7       7              0.9771    0.7734   0.2037
8       8              0.9772   -4.5338   5.5110
```

The pattern:
- **degree=1**: Underfitting. Train R²=0.16, CV R²=-0.64. The model is too simple.
- **degree=2**: Optimal. Train R²=0.97, CV R²=0.94. Small gap (0.03).
- **degree=4**: Starting to overfit. Train R²=0.97, CV R²=0.89. Gap growing (0.08).
- **degree=6**: Severe overfitting. Train R²=0.98, CV R²=-1.87. Gap exploded (2.85).
- **degree=8**: Catastrophic overfitting. Train R²=0.98, CV R²=-4.53. Gap is 5.51.

Notice: train R² barely changes (0.97 → 0.98), but CV R² collapses (0.94 → -4.53). The model memorizes the training data but learns nothing generalizable.

## Experiment 2: True Degree=1, Trying Degrees 1-6

Two features, true relationship is `3x₁ + 2x₂ + noise`:

```
degree   n_features   train_R²   cv_R²     gap
-------------------------------------------------------
1       2              0.9926    0.9843   0.0082  ← optimal
2       5              0.9928    0.9830   0.0098
3       9              0.9934    0.9720   0.0214
4       14             0.9943    0.8168   0.1776
5       20             0.9957   -5.0431   6.0389
6       27             0.9977  -433.3087  434.3064
```

Same pattern, but more dramatic. With 27 features (degree=6), the CV R² is -433. The model is predicting wildly wrong values on test data while scoring 0.998 on training data.

The gap is the signal: when gap < 0.05, the model generalizes. When gap > 0.1, it's overfitting. When gap > 1.0, it's catastrophic.

## Experiment 3: Feature Count Explosion

```
input_features   degree=2   degree=3   degree=4
--------------------------------------------------
2                5         9         14
3                9         19        34
5                20        55        125
10               65        285       1000
15               135       815       3875
20               230       1770      10625
```

With 10 input features and degree=4, you get 1,000 features. With 20 features, you get 10,625. The model now has more parameters than training samples, guaranteeing overfitting.

The rule: if `n_features_after_poly > n_samples / 10`, you're at risk of overfitting.

## The Diagnostic

| Gap (train - cv) | Interpretation | Action |
|------------------|----------------|--------|
| < 0.02 | Good fit | Keep this degree |
| 0.02 - 0.05 | Slight overfitting | Consider lower degree or regularization |
| 0.05 - 0.1 | Moderate overfitting | Lower degree or add regularization |
| > 0.1 | Severe overfitting | Lower degree, add regularization, or get more data |

The gap is more informative than either R² alone. A high train R² with a high gap means overfitting. A low train R² with a small gap means underfitting.

## The Code

```rust
use datarust::linear_model::LinearRegression;
use datarust::polynomial::PolynomialFeatures;
use datarust::traits::{Predictor, Transformer};

let lr = LinearRegression::new();
let kf = KFold::new().with_n_splits(5).with_shuffle(true);

for degree in 1..=8 {
    let mut pf = PolynomialFeatures::new(degree).include_bias(false);
    let x_poly = pf.fit_transform(&x)?;

    let mut train_scores = Vec::new();
    let mut cv_scores = Vec::new();

    for (train_idx, test_idx) in kf.split(x_poly.nrows())? {
        let x_train = x_poly.select_rows(&train_idx)?;
        let x_test = x_poly.select_rows(&test_idx)?;

        let mut m = lr.clone();
        m.fit(&x_train, &y_train)?;
        train_scores.push(r2_score(&y_train, &m.predict(&x_train)?)?);
        cv_scores.push(r2_score(&y_test, &m.predict(&x_test)?)?);
    }

    let gap = train_mean - cv_mean;
    println!("degree={degree}: gap={gap:.4}");
}
```

## Tradeoffs

**degree=1**: Fast, interpretable, no overfitting risk. Use when you believe the relationship is linear.

**degree=2**: Captures interactions and quadratic effects. The sweet spot for most problems.

**degree=3+**: Rarely necessary. Only use when you have strong domain knowledge that higher-order terms matter.

**Regularization helps**: Ridge with alpha > 0 can tolerate higher degrees because it penalizes large coefficients. But it's a band-aid — the real fix is choosing the right degree.

**More data helps**: With 10x more samples, degree=4 might not overfit. But "get more data" is often not an option.

The practical rule: start with degree=1. If the gap is small but CV R² is low, try degree=2. If degree=2 doesn't help, the relationship probably isn't polynomial — try a different model.

## Try It

```bash
cargo add datarust
```

```rust
use datarust::linear_model::LinearRegression;
use datarust::polynomial::PolynomialFeatures;
use datarust::traits::{Predictor, Transformer};
use datarust::Matrix;

let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]).unwrap();
let y = vec![1.0, 4.0, 9.0, 16.0]; // y = x²

let mut pf = PolynomialFeatures::new(2).include_bias(false);
let x_poly = pf.fit_transform(&x).unwrap();
println!("features: {}", x_poly.ncols()); // 2: [x, x²]

let mut lr = LinearRegression::new();
lr.fit(&x_poly, &y).unwrap();
println!("coef: {:?}", lr.coef()); // should be ~[0, 1]
```

If your CV R² is negative, your model is worse than predicting the mean. Check the degree before checking the data.
