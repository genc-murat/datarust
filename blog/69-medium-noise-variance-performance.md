# SNR=13: R²=0.91. SNR=0.5: R²=0.33. The Signal-to-Noise Ratio Decides Everything.

*How noise variance affects model performance*

---

Signal variance is 13. Noise variance is 1. R²=0.93. Now increase noise variance to 100. R²=0.14. The signal-to-noise ratio (SNR) is the single most important factor in model performance.

## Experiment 1: Noise Variance vs Model Performance

True relationship: `y = 3x1 + 2x2 + noise`. Signal variance = 13.0.

```
noise_sigma  noise_var  LinearReg  Ridge(1)   Lasso(0.1)
--------------------------------------------------------------
0            0          1.0000     1.0000     0.9988
0.1          0.01       0.9991     0.9991     0.9972
0.3          0.09       0.9936     0.9936     0.9919
0.5          0.25       0.9803     0.9802     0.9785
1            1          0.9320     0.9320     0.9304
2            4          0.7688     0.7688     0.7679
5            25         0.3431     0.3432     0.3431
10           100        0.1412     0.1413     0.1427
```

The pattern:
- **noise=0:** R²=1.00. Perfect prediction.
- **noise=1:** R²=0.93. Good.
- **noise=2:** R²=0.77. Moderate.
- **noise=5:** R²=0.34. Poor.
- **noise=10:** R²=0.14. Near zero.

Key insight: **All models perform similarly.** When noise dominates, regularization doesn't help -- the signal is buried.

## Experiment 2: Noise vs Coefficient Bias

True coefficients: `[3.0, 2.0]`.

```
noise_sigma  LR_coef1  LR_coef2  Ridge_c1  Ridge_c2
-----------------------------------------------------
0            3.0000    2.0000    2.9842    1.9902
0.5          3.0618    1.9750    3.0474    1.9651
1            2.9967    2.0267    2.9795    2.0128
2            2.8297    2.1434    2.8155    2.1334
5            3.1561    2.5283    3.1391    2.5176
10           2.0954    1.8343    2.0853    1.8249
```

The pattern:
- **noise=0:** Perfect coefficients.
- **noise=1:** Slight bias.
- **noise=5:** Noticeable bias.
- **noise=10:** Major bias. The model is fitting noise, not signal.

Key insight: **Coefficients become unreliable as noise increases.** The model "sees" patterns in the noise.

## Experiment 3: Noise vs Train-Test Gap

```
noise_sigma  LR_train  LR_cv    LR_gap   Lasso_train Lasso_cv Lasso_gap
------------------------------------------------------------------------
0            1.0000    1.0000   0.0000   0.9985      0.9984   0.0001
0.5          0.9766    0.9746   0.0020   0.9746      0.9727   0.0019
1            0.9215    0.9155   0.0061   0.9201      0.9145   0.0056
2            0.8028    0.7852   0.0176   0.8018      0.7838   0.0180
5            0.4080    0.3796   0.0285   0.4075      0.3805   0.0270
10           0.1318    0.0533   0.0786   0.1316      0.0534   0.0782
```

The pattern:
- **noise=0:** No gap. No overfitting.
- **noise=2:** Gap=0.018. Slight overfitting.
- **noise=10:** Gap=0.079. Moderate overfitting.

Key insight: **Overfitting increases with noise.** The model memorizes noise in the training set.

## Experiment 4: Signal-to-Noise Ratio

Signal variance = 13.0.

```
noise_sigma  noise_var  SNR      R²_LR    R²_Lasso
----------------------------------------------------
0            0          inf      1.0000   0.9977
0.1          0.01       1300.0   0.9994   0.9979
0.3          0.09       144.4    0.9932   0.9917
0.5          0.25       52.0     0.9810   0.9794
1            1          13.0     0.9103   0.9094
2            4          3.2      0.7363   0.7355
5            25         0.5      0.3273   0.3273
10           100        0.1      0.0602   0.0611
```

The pattern:
- **SNR > 100:** R² > 0.99. Excellent.
- **SNR = 13:** R² = 0.91. Good.
- **SNR = 3.2:** R² = 0.74. Moderate.
- **SNR = 0.5:** R² = 0.33. Poor.
- **SNR = 0.1:** R² = 0.06. Near zero.

Key insight: **When SNR < 1, noise dominates the signal.** The model is fitting noise, not signal.

## The Three SNR Regimes

| SNR | R² Range | Interpretation |
|-----|----------|---------------|
| SNR > 10 | R² > 0.9 | Signal dominates. Models work well. |
| 1 < SNR < 10 | 0.3 < R² < 0.9 | Moderate noise. Regularization helps. |
| SNR < 1 | R² < 0.3 | Noise dominates. Models are unreliable. |

## The Code

```rust
use datarust::linear_model::LinearRegression;
use datarust::metrics::regression::r2_score;
use datarust::model_selection::KFold;
use datarust::traits::Predictor;
use datarust::Matrix;

let n_samples = 200;
let noise_sigma = 2.0; // noise std
let signal_var = 13.0; // 3^2 + 2^2

let mut rows = Vec::new();
let mut y = Vec::new();
for _ in 0..n_samples {
    let x1 = rng.normal(1.0);
    let x2 = rng.normal(1.0);
    let noise = rng.normal(noise_sigma);
    rows.push(vec![x1, x2]);
    y.push(3.0 * x1 + 2.0 * x2 + noise);
}
let x = Matrix::new(rows).unwrap();

let kf = KFold::new().with_n_splits(5).with_shuffle(true);
let lr = LinearRegression::new();
let cv = cv_score(&lr, &x, &y, &kf);

let noise_var = noise_sigma * noise_sigma;
let snr = signal_var / noise_var;
println!("SNR={:.1}, R²={:.4}", snr, cv); // SNR=3.2, R²=0.769
```

## Practical Guidelines

**When SNR > 100:**
- Any model works
- Focus on feature engineering, not noise reduction
- CV is very stable

**When SNR = 10-100:**
- LinearRegression is fine
- Regularization helps slightly
- CV is stable

**When SNR = 1-10:**
- Use Ridge or Lasso
- Regularization is important
- CV is somewhat noisy

**When SNR < 1:**
- Models are unreliable
- Collect more data or reduce noise
- Domain knowledge is critical

## Tradeoffs

**High SNR (>100):**
- All models work well
- Coefficients are reliable
- CV is stable
- Focus on feature engineering

**Moderate SNR (1-100):**
- Regularization helps
- Coefficients are somewhat reliable
- CV is stable
- Tune alpha carefully

**Low SNR (<1):**
- Models are unreliable
- Coefficients are meaningless
- CV is noisy
- Collect more data

The universal rule: **SNR determines the ceiling of your model's performance.** No amount of tuning can overcome a low SNR. Measure your SNR first.

## Try It

```bash
cargo add datarust
```

```rust
use datarust::linear_model::LinearRegression;
use datarust::model_selection::KFold;
use datarust::traits::Predictor;
use datarust::Matrix;

let noise_sigma = 2.0;
let signal_var = 13.0;
let noise_var = noise_sigma * noise_sigma;
let snr = signal_var / noise_var;

println!("SNR: {:.1}", snr);

let x = Matrix::new(vec![vec![1.0, 2.0]; 200]).unwrap();
let y: Vec<f64> = (0..200).map(|i| 3.0 + 4.0 + (i as f64) * 0.01).collect();

let kf = KFold::new().with_n_splits(5).with_shuffle(true);
let lr = LinearRegression::new();
let cv = cv_score(&lr, &x, &y, &kf);
println!("R²: {:.4}", cv);
```

SNR=13: R²=0.91. SNR=0.5: R²=0.33. The signal-to-noise ratio decides everything.
