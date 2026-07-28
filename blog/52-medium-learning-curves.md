# Train R² Was 0.99. Test R² Was 0.06. More Data Fixed It.

*Diagnosing bias, variance, and data hunger with learning curves*

---

Your model scores 99% on training data. You feel good. Then you deploy it and it fails. The problem isn't the model — it's the relationship between model complexity, data size, and generalization. Learning curves reveal this relationship by plotting performance against training set size.

## The Three Patterns

### Pattern 1: Good Fit

LinearRegression on data generated as `3x₁ + 2x₂ + noise`:

```
LinearRegression
size         train R²      test R²
10         0.9942 ± 0.003  0.9940 ± 0.001
20         0.9959 ± 0.001  0.9946 ± 0.000
30         0.9952 ± 0.001  0.9949 ± 0.000
50         0.9946 ± 0.001  0.9952 ± 0.000
80         0.9943 ± 0.000  0.9958 ± 0.000
120        0.9950 ± 0.001  0.9956 ± 0.001
160        0.9950 ± 0.000  0.9962 ± 0.001
```

Train and test are both high (~0.995) and converge quickly. The small gap means the model isn't overfitting. The high values mean it isn't underfitting. This is the target pattern.

### Pattern 2: Underfitting

LinearRegression on data generated as `x₁² + noise` (quadratic relationship, linear model):

```
LinearRegression
size         train R²      test R²
10         0.1424 ± 0.139  -0.1830 ± 0.116
20         0.1177 ± 0.156  -0.1536 ± 0.139
30         0.0793 ± 0.071  -0.1018 ± 0.086
50         0.0306 ± 0.034  -0.1042 ± 0.068
80         0.0314 ± 0.031  -0.1114 ± 0.097
120        0.0183 ± 0.026  -0.0855 ± 0.116
160        0.0093 ± 0.011  -0.1891 ± 0.105
```

Both train and test are near zero. The gap is small (no overfitting), but the performance is terrible. More data won't help — the model is too simple for the relationship. The fix: add polynomial features, switch to a non-linear model, or engineer features.

Notice the variance decreasing as size increases (std drops from 0.139 to 0.011). The model is consistent — consistently wrong.

### Pattern 3: Data Hunger (Ridge vs LinearRegression on 40 Features)

40 features, only 5 useful. Ridge with alpha=10.0:

```
Ridge(alpha=10.0)
size         train R²      test R²
10         0.9400 ± 0.015  0.0575 ± 0.306
20         0.9457 ± 0.028  0.3916 ± 0.089
30         0.9523 ± 0.010  0.5519 ± 0.026
50         0.9670 ± 0.005  0.8247 ± 0.055
80         0.9809 ± 0.005  0.9429 ± 0.009
120        0.9903 ± 0.002  0.9772 ± 0.004
160        0.9941 ± 0.001  0.9896 ± 0.002
```

At size 10, the gap is enormous: train=0.94, test=0.06. The model memorizes the training data but can't generalize. As data increases, the gap closes. By size 80, test R² reaches 0.94. This is the "data hunger" pattern — the model needs enough data to distinguish signal from noise in high dimensions.

LinearRegression without regularization fails entirely at small sizes (singular matrix), then works once it has enough data:

```
LinearRegression (no reg)
size         train R²      test R²
10               FAIL         FAIL
20               FAIL         FAIL
30               FAIL         FAIL
50         0.9996 ± 0.000  0.9878 ± 0.004
80         0.9990 ± 0.000  0.9958 ± 0.001
120        0.9986 ± 0.000  0.9971 ± 0.001
160        0.9985 ± 0.000  0.9977 ± 0.001
```

Ridge's regularization is essential when n < p (samples < features). Once n >> p, the unregularized model catches up.

## How to Read the Gap

The gap between train and test curves tells you the bias-variance tradeoff:

**Large gap, high train score** → High variance (overfitting). The model memorizes training data. Fix: regularization, fewer features, more data.

**Small gap, low train score** → High bias (underfitting). The model is too simple. Fix: more features, non-linear model, polynomial features.

**Small gap, high train score** → Good fit. The model generalizes well.

**Large gap, low train score** → Rare. Usually means the model is complex enough to memorize but not to learn the pattern. Consider feature engineering.

## The Code

```rust
fn learning_curve(
    model: &(impl Predictor + Clone),
    x: &Matrix,
    y: &[f64],
    train_sizes: &[usize],
    n_repeats: usize,
) -> Vec<(f64, f64, f64, f64)> {
    let mut results = Vec::new();

    for &size in train_sizes {
        let mut train_scores = Vec::new();
        let mut test_scores = Vec::new();

        for rep in 0..n_repeats {
            // Shuffle and split
            let train_idx = &indices[..size];
            let test_idx = &indices[size..];

            let mut m = model.clone();
            m.fit(&x_train, &y_train)?;

            train_scores.push(r2_score(&y_train, &m.predict(&x_train)?)?);
            test_scores.push(r2_score(&y_test, &m.predict(&x_test)?)?);
        }

        let train_mean = train_scores.iter().sum::<f64>() / n_repeats as f64;
        let test_mean = test_scores.iter().sum::<f64>() / n_repeats as f64;
        results.push((train_mean, train_std, test_mean, test_std));
    }
    results
}
```

The `n_repeats` parameter controls variance estimation. With 5 repeats, you get stable mean and standard deviation for each size.

## Practical Decisions

**Should I collect more data?**
- If the test curve is still rising at the largest size: yes
- If the test curve has plateaued: no, it won't help
- If both curves are low and parallel: no, you need a different model

**Should I add regularization?**
- If the gap is large at small sizes but closes with data: yes, until you have enough data
- If the gap is small regardless: no, it won't help

**Should I add features?**
- If both curves are low: yes
- If the gap is large but train is high: no, you need regularization or data, not features

## Tradeoffs

Learning curves are expensive: you train the model `len(train_sizes) × n_repeats` times. For large models, this can take hours. The practical approach: use a coarse grid (5-7 sizes) with 3-5 repeats, then zoom in on interesting regions.

The computational cost scales linearly with `n_repeats`. More repeats give better variance estimates but cost more. For initial diagnostics, 3 repeats is enough. For publication-quality results, use 10+.

The `train_sizes` should span from very small (10-20 samples) to the full dataset. Logarithmic spacing works well: `[10, 20, 50, 100, 200, 500, 1000]`.

## Try It

```bash
cargo add datarust
```

```rust
use datarust::linear_model::LinearRegression;
use datarust::matrix::Matrix;

let x = Matrix::new(vec![vec![1.0, 2.0]; 100]).unwrap();
let y: Vec<f64> = (0..100).map(|i| i as f64).collect();

let lr = LinearRegression::new();
let sizes = vec![10, 20, 50, 80];
let curve = learning_curve(&lr, &x, &y, &sizes, 5);

for (size, (tm, _, vm, _)) in sizes.iter().zip(curve.iter()) {
    println!("n={size}: train={tm:.4}, test={vm:.4}");
}
```

If your train score is 0.99 and your test score is 0.06, you don't need a better model. You need more data, or regularization, or a different feature set. Learning curves tell you which one.
