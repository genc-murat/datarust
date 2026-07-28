# Ridge R² Went from 0.99 to −3.75. The Scaler Leaked the Test Set.

*How preprocessing leakage destroys model performance with covariate shift*

---

I standardized my data on the full dataset, split into train/test, trained Ridge, and got R²=−3.75. The same model, with the scaler fitted only on the training set, gave R²=0.99. The difference? I leaked the test set statistics into my preprocessing.

## The Setup

Covariate shift — the test set has a different distribution from the training set:

- **Train:** x ~ N(0, 1), 100 samples
- **Test:** x ~ N(10, 1), 100 samples

True relationship: `y = 3x₁ + 2x₂ + noise`.

## Experiment 1: MinMaxScaler — Same R² for LinearRegression

```
  method           R²      note
  ----------------------------------------
  correct (fit/t)    0.9911  fit on train only
  leaky (fit/all)    0.9911  fit on all data
  difference         0.0000  leaky overestimates
```

LinearRegression is scale-invariant. Scaling the full data or just the training data gives identical predictions. The coefficients adjust.

## Experiment 2: StandardScaler — Same, for LinearRegression

```
  StandardScaler:
    correct: 0.9911
    leaky:   0.9911
```

Same story. LinearRegression doesn't care when you scale — predictions are the same.

## Experiment 3: Ridge — Catastrophic Difference

```
  Ridge (alpha=1):
    correct: 0.9917
    leaky:   −3.7521
    diff:    −4.7437
```

Ridge is NOT scale-invariant. The regularization penalty `alpha * sum(coef²)` changes when the scaling changes. Leaky scaling uses parameters from the full data (including test data), which changes the scaling of the training data, which changes the regularization penalty.

The result: Ridge over-regularizes, and the test set predictions are terrible.

## Experiment 4: What Actually Leaks?

```
  feature   train_min   all_min   train_max   all_max
  -------------------------------------------------------
  x1          -3.796     -3.796      2.702     12.506
  x2          -1.807     -1.807      2.674     11.840
```

The test set min/max leaked into the scaler. The training data min/max were [−3.80, 2.70] but the full data min/max were [−3.80, 12.51]. The scaler "saw" the test set range and shifted its parameters.

With correct scaling, test values around 10-11 get transformed using train min/max, mapping to approximately 3-4× the training range. The model was trained on data in [0, 1] range — it doesn't know how to handle inputs at 3-4× that range.

With leaky scaling, test values get mapped to [0, 1] alongside training data — but the training data now occupies only the lower half of [0, 1], compressing the training signal.

## Why Ridge Is More Vulnerable

LinearRegression finds coefficients that minimize `MSE`. Rescaling the input doesn't change the optimal coefficients in terms of prediction — the coefficients just get reweighted.

Ridge minimizes `MSE + alpha * sum(coef²)`. When the scaling changes, the coefficients change, and the regularization penalty changes. This makes Ridge (and Lasso) vulnerable to preprocessing leakage.

## The Correct Way: Fit Scaler Inside CV

```rust
for (train_idx, test_idx) in kf.split(n)? {
    let x_train = x.select_rows(&train_idx)?;
    let x_test = x.select_rows(&test_idx)?;

    let mut scaler = StandardScaler::new();
    let xs_tr = scaler.fit_transform(&x_train)?;
    let xs_te = scaler.transform(&x_test);  // fit on train only

    let mut model = Ridge::new().with_alpha(1.0);
    model.fit(&xs_tr, &y_train)?;
    let p = model.predict(&xs_te)?;
}
```

Each fold fits the scaler on that fold's training data. The test fold never influences the scaler parameters.

## The Leaky Way: Fit Scaler Before Split

```rust
// WRONG — fit on ALL data
let mut scaler = StandardScaler::new();
let x_scaled = scaler.fit_transform(&x);

// Then split
let (x_train, x_test) = (x_scaled.select_rows(&train_idx), x_scaled.select_rows(&test_idx));
// The scaler already saw the test data!
```

## When It Matters

| Scenario | Vulnerability | Why |
|----------|--------------|-----|
| LinearRegression | None | Scale-invariant |
| Ridge/Lasso | High | Regularization is scale-dependent |
| No covariate shift | Low | Test and train have same distribution |
| Covariate shift | High | Scaling parameters shift |

## Practical Guidelines

**Always fit preprocessing inside CV folds.** Never call `fit` or `fit_transform` on the full dataset before splitting. Use a Pipeline to enforce this automatically.

**Regularized models are more vulnerable.** Ridge and Lasso need correct scaling to apply the right regularization strength. Leaky scaling changes the effective alpha.

**Covariate shift amplifies leakage.** If your test set has a different distribution, leaky scaling doesn't just inflate your CV score — it can make your model catastrophically wrong.

**Pipeline prevents this mechanically:**

```rust
use datarust::pipeline::Pipeline;
use datarust::TransformerKind::StandardScaler;

let pipeline = Pipeline::new()
    .push("scale", StandardScaler(StandardScaler::new()))
    .with_estimator(LogisticRegression::new());

// fit_transform inside each CV fold automatically
pipeline.fit(&x_train, &y_train)?;
pipeline.predict(&x_test)?;
```

## Tradeoffs

**Fitting scaler inside CV:**
- ✅ Correct evaluation
- ✅ Prevents data leakage
- ✅ Works with any model
- ❌ Slightly more code (but Pipeline helps)

**Fitting scaler before validation:**
- ✅ Simpler code
- ❌ Leaks test information
- ❌ Inflates or deflates scores
- ❌ Can make Ridge/Lasso fail catastrophically

The universal rule: **fit everything inside CV.** A Pipeline enforces this automatically. One `fit_transform` on the full dataset is all it takes to turn Ridge from 0.99 to −3.75.

## Try It

```bash
cargo add datarust
```

```rust
use datarust::linear_model::Ridge;
use datarust::scaler::StandardScaler;
use datarust::traits::{Predictor, Transformer};
use datarust::Matrix;

let x = Matrix::new(vec![vec![1.0, 2.0]; 200]).unwrap();
let y: Vec<f64> = (0..200).map(|i| i as f64 * 0.01).collect();

// Correct: fit scaler on training data
let mut scaler = StandardScaler::new();
let x_train_scaled = scaler.fit_transform(&x)?;
let mut ridge = Ridge::new().with_alpha(1.0);
ridge.fit(&x_train_scaled, &y)?;

// Leaky: scaler already saw test data
// Ridge R² goes from 0.99 to −3.75
```

Ridge R² went from 0.99 to −3.75. The scaler leaked the test set. Fit your preprocessing inside CV — or use a Pipeline.
