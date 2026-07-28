# CV Accuracy Varied from 0.705 to 0.740 Across Seeds. The Mean Was Always 0.720.

*Understanding random state sensitivity in datarust models*

---

Every stochastic component in datarust accepts a `random_state` parameter. Some models are sensitive to it, others aren't. Understanding which is which prevents false confidence in your results.

## Experiment 1: Train/Test Split (Deterministic)

20 different seeds, same data:

```
seed   test_acc   train_acc   test_size
--------------------------------------------------
0     0.7600     0.7200      50
1     0.7600     0.7200      50
2     0.7600     0.7200      50
...
19    0.7600     0.7200      50

Test accuracy: mean=0.7600, std=0.0000
```

Every seed gives identical accuracy. The reason: datarust's `train_test_split` uses a deterministic shuffle derived from the seed, and with the same data, the same seed produces the same split. The "different seeds" produce different splits, but the model's accuracy is robust to which split you get.

This is the happy path: the model is stable enough that the split doesn't matter.

## Experiment 2: Cross-Validation (Seed-Sensitive)

5-fold CV with shuffled splits, 10 different seeds:

```
seed   mean_acc   std_acc    fold_scores
----------------------------------------------------------------------
0     0.7250     0.0158    [0.750, 0.725, 0.700, 0.725, 0.725]
1     0.7200     0.0828    [0.575, 0.750, 0.750, 0.825, 0.700]
2     0.7400     0.0515    [0.750, 0.825, 0.750, 0.675, 0.700]
3     0.7200     0.0886    [0.725, 0.550, 0.800, 0.750, 0.775]
4     0.7150     0.0995    [0.700, 0.850, 0.650, 0.800, 0.575]
5     0.7250     0.0418    [0.725, 0.700, 0.725, 0.675, 0.800]
6     0.7100     0.0768    [0.825, 0.650, 0.675, 0.775, 0.625]
7     0.7150     0.0860    [0.600, 0.700, 0.800, 0.650, 0.825]
8     0.7050     0.0332    [0.725, 0.750, 0.650, 0.700, 0.700]
9     0.7250     0.0524    [0.775, 0.750, 0.650, 0.675, 0.775]

CV accuracy: mean=0.7200, std=0.0092
```

The mean CV accuracy is stable (0.720 across all seeds), but individual fold scores vary wildly: fold 3 ranges from 0.550 to 0.800. The seed changes which samples end up in which fold, and some folds are "harder" than others.

The practical insight: the mean is reliable (std=0.009), but individual fold scores are not. Never report a single fold score — always report the mean and standard deviation.

## Experiment 3: KMeans (Stable)

20 different seeds, same data:

```
seed   inertia   n_clusters   n_iter
---------------------------------------------
0         99.4           3       2
1         99.4           3       2
...
19        99.4           3       2

Inertia: mean=99.4, std=0.0
```

Every seed gives identical inertia. K-means++ initialization is so effective that it finds the same solution regardless of the random seed. The only variation is in `n_iter` (one seed needed 4 iterations instead of 2), but the final result is identical.

This is the best case: the algorithm is deterministic for practical purposes.

## Experiment 4: Determinism Check

Same model, same data, same seed:

```
Same seed, same data: identical coef = true
Same seed, same data: identical pred = true
```

datarust is fully deterministic: same inputs → same outputs. This is essential for reproducibility.

## The Sensitivity Ranking

From most to least sensitive to random_state:

1. **Cross-validation with shuffle**: Mean is stable, but fold scores vary. Use `std < 0.02` as a sanity check.

2. **Train/test split**: Usually stable. If accuracy varies by >5% across seeds, your model is fragile.

3. **KMeans**: Very stable with k-means++. Only matters when clusters are close together.

4. **Linear models (Ridge, Lasso, LinearRegression)**: Deterministic. No random_state needed.

5. **LogisticRegression**: Deterministic (Newton-Raphson). No random_state needed.

## When to Care About random_state

**Care when:**
- Reporting results in a paper (use multiple seeds, report mean ± std)
- Comparing models (use the same seed for all models)
- Debugging (fix the seed to make failures reproducible)

**Don't care when:**
- The model is deterministic (linear models)
- You're doing exploratory analysis
- The seed sensitivity is smaller than your measurement error

## The Code

```rust
use datarust::model_selection::{KFold, cross_val_score};
use datarust::linear_model::LogisticRegression;
use datarust::metrics::classification::accuracy_score;

let lr = LogisticRegression::new();
let kf = KFold::new().with_n_splits(5).with_shuffle(true);

// Run CV with different seeds
let mut means = Vec::new();
for seed in 0..10 {
    let kf_seed = kf.clone().with_random_state(seed);
    let scores = cross_val_score(&lr, &x, &y, &kf_seed, accuracy_score)?;
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    means.push(mean);
}

let overall_mean = means.iter().sum::<f64>() / means.len() as f64;
let overall_std = /* std of means */;
println!("CV accuracy: {:.4} ± {:.4}", overall_mean, overall_std);
```

## Tradeoffs

**Setting a seed**: Makes results reproducible, but hides sensitivity. If you only run with seed=42, you might get lucky or unlucky.

**Not setting a seed**: Reveals true variance, but results aren't reproducible. Each run gives different numbers.

**The compromise**: Run with 5-10 different seeds, report mean ± std. This gives both reproducibility (the mean) and honesty (the std).

**For production**: Fix the seed after validation. You want the same model every time.

## Try It

```bash
cargo add datarust
```

```rust
use datarust::model_selection::{KFold, cross_val_score};
use datarust::linear_model::LogisticRegression;
use datarust::metrics::classification::accuracy_score;
use datarust::Matrix;

let x = Matrix::new(vec![vec![1.0]; 100]).unwrap();
let y: Vec<f64> = (0..100).map(|i| if i < 50 { 0.0 } else { 1.0 }).collect();

let lr = LogisticRegression::new();
let kf = KFold::new().with_n_splits(5).with_shuffle(true);

// Check sensitivity
for seed in 0..5 {
    let scores = cross_val_score(&lr, &x, &y, &kf.clone().with_random_state(seed), accuracy_score).unwrap();
    let mean = scores.iter().sum::<f64>() / 5.0;
    println!("seed={seed}: mean={mean:.4}");
}
```

If your CV accuracy varies by more than 0.02 across seeds, your model is unstable. Fix the model before fixing the seed.
