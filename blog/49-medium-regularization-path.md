# Lasso Zeroed Five Coefficients. Ridge Kept Them All. Only One Was Right.

*A practical datarust guide to L1 vs L2 regularization paths, why correlated features confuse Lasso, and how alpha controls the tradeoff between sparsity and stability.*

---

I had eight features. Three were real. Two were correlated copies of two of the real ones. Three were noise.

Ordinary least squares recovered all eight coefficients. The real ones were close to their true values. The correlated copies got small but nonzero weights. The noise features got tiny weights that could have been anything.

Then I fitted Lasso with `alpha = 0.1`. Five coefficients became exactly zero. The three surviving features were the true signals — `x1`, `x2`, `x3` — and nothing else.

That felt like magic. It was also fragile. At `alpha = 1.0`, Lasso zeroed `x3` too, leaving only two features. At `alpha = 5.0`, everything was zero. The same algorithm that found the right features at one penalty strength destroyed the model at another.

Ridge never zeroed anything. It shrank coefficients gracefully, kept all eight features, and produced nearly identical test performance across a wide range of alpha values.

Let's reproduce both paths with [**datarust**](https://crates.io/crates/datarust), watch the coefficients change as alpha increases, and understand why the "best" regularization depends on what you want the model to do.

## The experiment: three signals, two shadows, three noise

We generate 200 rows with eight features:

- **x1, x2, x3** — true signals with coefficients `3`, `2`, `1`.
- **x4** — correlated with x1 (r ≈ 0.8). A shadow of the first signal.
- **x5** — correlated with x2 (r ≈ 0.8). A shadow of the second signal.
- **x6, x7, x8** — pure noise.

The true relationship is:

```text
y = 3·x1 + 2·x2 + x3 + noise
```

Create a Rust project:

```sh
cargo new regularization_path
cd regularization_path
cargo add datarust
```

Replace `src/main.rs` with this:

```rust
use datarust::linear_model::{Lasso, LinearRegression, Ridge};
use datarust::metrics::regression::r2_score;
use datarust::model_selection::{cross_val_score, KFold, TrainTestSplit};
use datarust::traits::Predictor;
use datarust::Matrix;

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self, sigma: f64) -> f64 {
        let u = self.next_f64().max(f64::MIN_POSITIVE);
        let v = self.next_f64();
        sigma * (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn make_data(rng: &mut Rng) -> (Matrix, Vec<f64>) {
    let mut rows = Vec::new();
    let mut targets = Vec::new();

    for _ in 0..200 {
        let x1 = rng.normal(1.0);
        let x2 = rng.normal(1.0);
        let x3 = rng.normal(1.0);

        let x4 = 0.8 * x1 + 0.6 * rng.normal(1.0);
        let x5 = 0.8 * x2 + 0.6 * rng.normal(1.0);

        let x6 = rng.normal(1.0);
        let x7 = rng.normal(1.0);
        let x8 = rng.normal(1.0);

        let y = 3.0 * x1 + 2.0 * x2 + x3 + rng.normal(0.5);

        rows.push(vec![x1, x2, x3, x4, x5, x6, x7, x8]);
        targets.push(y);
    }

    (Matrix::new(rows).unwrap(), targets)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Rng::new(2026);
    let (x, y) = make_data(&mut rng);

    println!("Dataset: 200 rows, 8 features");
    println!("  x1, x2, x3  = true signals (coef: 3, 2, 1)");
    println!("  x4*         = correlated with x1 (r~0.8)");
    println!("  x5*         = correlated with x2 (r~0.8)");
    println!("  x6, x7, x8  = pure noise\n");

    let (x_train, x_test, y_train, y_test) = TrainTestSplit::new()
        .with_test_size(0.20)
        .with_random_state(42)
        .split(&x, &y)?;

    // Ordinary least squares
    let mut lr = LinearRegression::new();
    lr.fit(&x_train, &y_train)?;
    let lr_pred = lr.predict(&x_test)?;
    let lr_r2 = r2_score(&y_test, &lr_pred)?;
    let c = lr.coef();
    println!(
        "OLS          intercept={intercept:>7.3}  \
         x1={c0:>7.3}  x2={c1:>7.3}  x3={c2:>7.3}  \
         x4={c3:>7.3}  x5={c4:>7.3}",
        intercept = lr.intercept(),
        c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3], c4 = c[4]
    );
    println!("  test R2 = {lr_r2:.4}\n");

    // Ridge path
    println!("=== Ridge (L2) regularization path ===");
    println!("alpha      coef_x1  coef_x2  coef_x3  coef_x4  coef_x5  test_R2");
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] {
        let mut ridge = Ridge::new().with_alpha(alpha);
        ridge.fit(&x_train, &y_train)?;
        let pred = ridge.predict(&x_test)?;
        let r2 = r2_score(&y_test, &pred)?;
        let c = ridge.coef();
        println!(
            "{alpha:<10.3} {c0:>8.3}  {c1:>8.3}  {c2:>8.3}  \
             {c3:>8.3}  {c4:>8.3}  {r2:>8.4}",
            c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3], c4 = c[4]
        );
    }

    // Lasso path
    println!("\n=== Lasso (L1) regularization path ===");
    println!("alpha      coef_x1  coef_x2  coef_x3  coef_x4  coef_x5  nonzero  test_R2");
    for alpha in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0] {
        let mut lasso = Lasso::new().with_alpha(alpha).with_max_iter(5000);
        lasso.fit(&x_train, &y_train)?;
        let pred = lasso.predict(&x_test)?;
        let r2 = r2_score(&y_test, &pred)?;
        let c = lasso.coef();
        let nonzero = c.iter().filter(|&&v| v.abs() > 0.001).count();
        println!(
            "{alpha:<10.3} {c0:>8.3}  {c1:>8.3}  {c2:>8.3}  \
             {c3:>8.3}  {c4:>8.3}  {nonzero:>7}  {r2:>8.4}",
            c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3], c4 = c[4]
        );
    }

    // Cross-validation
    println!("\n=== 5-fold CV R2 for selected alpha values ===");
    let cv = KFold::new()
        .with_n_splits(5)
        .with_shuffle(true)
        .with_random_state(42);

    let alphas = [0.001, 0.01, 0.1, 1.0, 10.0];
    println!("alpha   Ridge    Lasso");
    for alpha in alphas {
        let ridge = Ridge::new().with_alpha(alpha);
        let ridge_scores =
            cross_val_score(&ridge, &x_train, &y_train, &cv, r2_score)?;
        let ridge_mean: f64 =
            ridge_scores.iter().sum::<f64>() / ridge_scores.len() as f64;

        let lasso =
            Lasso::new().with_alpha(alpha).with_max_iter(5000);
        let lasso_scores =
            cross_val_score(&lasso, &x_train, &y_train, &cv, r2_score)?;
        let lasso_mean: f64 =
            lasso_scores.iter().sum::<f64>() / lasso_scores.len() as f64;

        println!("{alpha:<7.3} {ridge_mean:>8.4}  {lasso_mean:>8.4}");
    }

    Ok(())
}
```

Run it:

```sh
cargo run --release
```

With datarust v0.6 and the fixed seed above, the output is:

```text
Dataset: 200 rows, 8 features
  x1, x2, x3  = true signals (coef: 3, 2, 1)
  x4*         = correlated with x1 (r~0.8)
  x5*         = correlated with x2 (r~0.8)
  x6, x7, x8  = pure noise

OLS          intercept=  0.008  x1=  3.011  x2=  1.993  x3=  1.052  x4=  0.025  x5=  0.077
  test R2 = 0.9829

=== Ridge (L2) regularization path ===
alpha      coef_x1  coef_x2  coef_x3  coef_x4  coef_x5  test_R2
0.001         3.011     1.993     1.052     0.025     0.077    0.9829
0.010         3.011     1.992     1.052     0.026     0.077    0.9829
0.100         3.007     1.989     1.051     0.029     0.080    0.9829
1.000         2.971     1.960     1.045     0.060     0.101    0.9829
10.000        2.675     1.729     0.987     0.299     0.256    0.9762
100.000       1.613     1.019     0.610     0.758     0.474    0.8561

=== Lasso (L1) regularization path ===
alpha      coef_x1  coef_x2  coef_x3  coef_x4  coef_x5  nonzero  test_R2
0.001         3.010     1.993     1.051     0.025     0.076        8    0.9829
0.010         3.005     1.993     1.042     0.019     0.068        7    0.9832
0.100         2.930     1.965     0.942     0.000     0.000        3    0.9838
0.500         2.580     1.571     0.473     0.000     0.000        3    0.9450
1.000         2.144     1.078     0.000     0.000     0.000        2    0.8105
5.000         0.000     0.000     0.000     0.000     0.000        0   -0.0087
10.000        0.000     0.000     0.000     0.000     0.000        0   -0.0087

=== 5-fold CV R2 for selected alpha values ===
alpha   Ridge    Lasso
0.001     0.9813    0.9813
0.010     0.9813    0.9814
0.100     0.9813    0.9799
1.000     0.9811    0.7997
10.000    0.9728   -0.0256
```

There is a lot happening in those tables. Let's unpack it.

## OLS got the right answer for the wrong reason

The OLS coefficients are close to the true values:

```text
x1: 3.011 (true: 3)
x2: 1.993 (true: 2)
x3: 1.052 (true: 1)
```

The correlated features x4 and x5 got small weights (`0.025` and `0.077`), and the noise features x6–x8 got even smaller weights. OLS correctly identified that x1 and x2 are the dominant predictors, and it assigned the correlated copies near-zero weights because they carry redundant information.

But "near-zero" is not "zero." Those coefficients are nonzero, which means the model still uses all eight features at prediction time. In a dataset with hundreds of features and many correlated groups, those small weights accumulate and make the model harder to interpret.

## Ridge shrinks everything, zeroes nothing

At `alpha = 0.001`, Ridge coefficients are identical to OLS. As alpha increases, every coefficient shrinks toward zero — but none of them reach it.

The key observation is in the last row of the Ridge table:

```text
alpha=100: x1=1.613, x2=1.019, x3=0.610, x4=0.758, x5=0.474
```

At this extreme penalty, the correlated features x4 and x5 have coefficients *larger* than x3. Ridge cannot distinguish between a feature that is truly useful and a feature that is merely correlated with a useful one. It treats them equally because the L2 penalty cares about the magnitude of coefficients, not their identity.

This is Ridge's fundamental limitation for feature selection: it can shrink, but it cannot discard.

## Lasso's path tells a different story

The Lasso table is where the interesting behavior appears. Watch what happens as alpha increases:

```text
alpha=0.001:  8 nonzero features (all of them)
alpha=0.010:  7 nonzero features (one noise feature zeroed)
alpha=0.100:  3 nonzero features (x1, x2, x3 — the true signals!)
alpha=0.500:  3 nonzero features (same, but coefficients shrinking)
alpha=1.000:  2 nonzero features (x3 zeroed — too aggressive)
alpha=5.000:  0 nonzero features (everything zeroed)
```

At `alpha = 0.1`, Lasso perfectly identifies the three true signals. The correlated features x4 and x5 are exactly zero. The noise features x6–x8 are exactly zero. The surviving coefficients (`2.930`, `1.965`, `0.942`) are close to their true values.

This is the L1 magic: the penalty drives some coefficients to exactly zero, not just near zero. The result is a sparse model that uses only the features it needs.

But the window is narrow. At `alpha = 1.0`, Lasso has already zeroed x3 — a true signal with coefficient `1.0`. At `alpha = 5.0`, everything is gone. The same algorithm that found the right features at one penalty strength destroys the model at another.

## Why Lasso zeros correlated features

When two features are highly correlated, they carry nearly the same information. A model can use either one to explain the target. Lasso, faced with two redundant features, tends to pick one and zero out the other.

This is not a bug. It is a consequence of the L1 geometry. The L1 penalty constrains the sum of absolute coefficients:

```text
|β₁| + |β₂| + ... + |βₚ| ≤ C
```

When two features are redundant, the optimal solution under this constraint is to put all the weight on one and zero out the other — because spreading weight across both increases the penalty without improving the fit.

Ridge uses L2 geometry:

```text
β₁² + β₂² + ... + βₚ² ≤ C
```

This penalizes large individual coefficients but does not favor sparsity. Spreading weight across two correlated features is equally cheap under L2, so Ridge keeps both.

## The test R² reveals the practical tradeoff

Lasso at `alpha = 0.1` achieves test R² = `0.9838` — slightly better than OLS (`0.9829`) because it removed noise features. But at `alpha = 1.0`, test R² drops to `0.8105` because x3 was zeroed too aggressively.

Ridge test R² is stable across alpha values:

```text
Ridge alpha=0.1:  0.9829
Ridge alpha=1.0:  0.9829
Ridge alpha=10:   0.9762
```

The cross-validation table confirms this:

```text
alpha=0.1:  Ridge=0.9813, Lasso=0.9799
alpha=1.0:  Ridge=0.9811, Lasso=0.7997
```

Ridge is forgiving. Lasso is not. If the penalty is too strong, Lasso removes features you need and performance collapses. If the penalty is too weak, Lasso keeps features you do not need and you lose the sparsity advantage.

## The alpha window matters more than the algorithm

The choice between Lasso and Ridge is less important than finding the right alpha for either one.

For Lasso, the useful alpha window is narrow:

```text
too small (0.001):  keeps everything — no feature selection
just right (0.1):   keeps only the true signals
too large (1.0):    removes true signals — model degrades
```

For Ridge, the window is wide:

```text
too small (0.001):  identical to OLS
just right (0.1):   nearly identical to OLS, slight shrinkage
too large (100):    everything shrunk, model underfits
```

This is why cross-validation is essential. The "best" alpha is not a property of the algorithm. It is a property of the data, the feature correlations, and the noise level.

## When I would choose each

**Use Lasso when:**

- You want automatic feature selection.
- You suspect many features are irrelevant.
- Interpretability matters — a sparse model is easier to explain.
- The correlated feature groups are small and you are comfortable with Lasso picking one from each group.

**Use Ridge when:**

- All features are potentially useful.
- Features are correlated and you want to keep all of them (e.g., in a prediction task where interpretability is less important).
- You want a stable model that is less sensitive to the exact alpha value.
- The number of features is close to or larger than the number of samples.

**Use Elastic Net (L1 + L2) when:**

- You want the feature selection of Lasso with the stability of Ridge.
- You have groups of correlated features and want to keep or drop them together.
- You are not sure which one to pick.

datarust does not currently include Elastic Net, but the pattern is clear: the penalty shape determines which features survive.

## One practical workflow

After running this experiment, here is what I would carry to a real project:

1. **Start with OLS** to see the unconstrained coefficients. Identify correlated feature groups.
2. **Fit Lasso at several alpha values** and watch the coefficient path. Which features disappear first? Those are the weakest.
3. **Fit Ridge at several alpha values** and compare. Does Ridge agree with Lasso on which features matter most? If not, investigate the disagreements.
4. **Cross-validate alpha** for both Lasso and Ridge. Compare the best CV scores.
5. **Inspect the surviving coefficients.** Are they close to what domain knowledge predicts? Do the signs make sense?
6. **Monitor the coefficient path** in production. If retraining on new data changes which features are selected, the model may be fragile.

The coefficient path is a diagnostic tool, not just a fitting procedure. It tells you which features the model trusts, which ones it tolerates, and which ones it cannot decide about.

## The five coefficients that told the truth

The most revealing number in the entire experiment is not a coefficient. It is the count of nonzero features at each alpha:

```text
alpha=0.001:  8 nonzero (everything)
alpha=0.010:  7 nonzero (one noise gone)
alpha=0.100:  3 nonzero (only the truth)
alpha=0.500:  3 nonzero (still the truth, shrinking)
alpha=1.000:  2 nonzero (truth lost a member)
alpha=5.000:  0 nonzero (everything gone)
```

The jump from 7 to 3 is where Lasso earned its reputation. The jump from 3 to 0 is where it earned its cautionary tales.

The model does not know which features are true. It knows which ones minimize the penalized loss. At the right alpha, those happen to be the same. At the wrong alpha, they are not.

That is the real lesson: regularization is not a setting. It is a question about how much complexity the data can support. The answer depends on the data, and it changes when the data changes.

```sh
cargo add datarust
```

---

*datarust is MIT-licensed and available on [crates.io](https://crates.io/crates/datarust). Documentation lives at [genc-murat.github.io/datarust](https://genc-murat.github.io/datarust/), including the [Ridge and Lasso guide](https://genc-murat.github.io/datarust/guide/linear-models.html).*
