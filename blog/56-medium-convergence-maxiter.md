# LogisticRegression Converged in 6 Iterations. You Gave It 100.

*Understanding convergence, max_iter, and tol in datarust models*

---

Every iterative model in datarust has `max_iter` and `tol` parameters. The defaults work for most problems, but understanding what they do helps you diagnose when they don't.

## The Three Convergence Patterns

### Pattern 1: LogisticRegression

LogisticRegression uses Newton-Raphson (IRLS) — a second-order method that converges quadratically:

```
max_iter   n_iter   converged?   accuracy
--------------------------------------------------
1              1   NO           0.8250
2              2   NO           0.8350
3              3   NO           0.8350
5              5   NO           0.8300
10             6   yes          0.8300
20             6   yes          0.8300
50             6   yes          0.8300
100            6   yes          0.8300
```

It converges in 6 iterations. The default max_iter=100 is 16× more than needed. Even max_iter=5 isn't enough — it needs exactly 6.

The accuracy doesn't improve after convergence: 0.8300 at iter 6 and 0.8300 at iter 100. Extra iterations don't help.

### Pattern 2: Lasso

Lasso uses coordinate descent — a first-order method that converges linearly:

```
alpha   max_iter   n_iter   converged?   R²
-------------------------------------------------------
0.001 10             4   yes          0.9938
0.01  10             3   yes          0.9937
0.1   10             3   yes          0.9919
1     10             3   yes          0.8187
```

Lasso converges in 3-4 iterations regardless of alpha. The R² decreases with alpha because stronger regularization shrinks coefficients more, but convergence speed doesn't change.

### Pattern 3: KMeans

KMeans uses Lloyd's algorithm — alternating between assignment and update:

```
max_iter   n_iter   converged?   inertia
--------------------------------------------------
1              1   NO           45.6
2              2   NO           45.6
3              2   yes          45.6
5              2   yes          45.6
10             2   yes          45.6
300            2   yes          45.6
```

KMeans converges in 2 iterations for well-separated clusters. The inertia is identical (45.6) whether it converges or not — because the clusters are so separated that the first assignment is already optimal.

## The tol Effect

How does convergence tolerance affect the number of iterations?

```
tol        n_iter   accuracy
---------------------------------------------
0.1            4   0.8300
0.01           5   0.8300
0.001          6   0.8300
0.0001         6   0.8300
0.00001        6   0.8300
0.000001       7   0.8300
```

Tighter tolerance (smaller tol) requires more iterations: 4 at tol=0.1, 7 at tol=0.000001. But accuracy doesn't change — it's 0.8300 across all tolerances.

The reason: once the model finds the right coefficients, finer tolerance just refines the decimal places. For practical purposes, tol=1e-4 (the default) is sufficient.

## How to Detect Non-Convergence

datarust doesn't emit warnings for non-convergence. You need to check manually:

```rust
let mut lr = LogisticRegression::new().with_max_iter(10);
lr.fit(&x, &y)?;

if lr.n_iter() >= 10 {
    eprintln!("WARNING: did not converge in {} iterations", lr.n_iter());
}
```

The pattern: `n_iter() >= max_iter` means non-convergence. When this happens, the model has found *a* solution, but not necessarily the *optimal* one.

## When to Increase max_iter

**Increase max_iter when:**
- The model is large (many features or classes)
- Features are highly correlated
- The learning rate is small (for gradient-based models)
- You see `n_iter() >= max_iter` in the output

**The default is fine when:**
- The dataset is small to medium (<10,000 samples)
- Features are not highly correlated
- You're using the default solver

## When to Decrease max_iter

**Decrease max_iter when:**
- You're doing exploratory analysis and speed matters
- The model converges in 5 iterations but you're waiting for 100
- You're doing hyperparameter search and need fast feedback

A model that converges in 6 iterations with max_iter=100 will also converge in 6 iterations with max_iter=10. The extra iterations are just wasted computation.

## Tradeoffs

**max_iter too low**: The model doesn't converge. The coefficients are suboptimal, predictions are less accurate, and you might not notice because the accuracy drop is often small.

**max_iter too high**: Wasted computation. If the model converges in 6 iterations, running 100 iterations costs 16× more with no benefit.

**tol too high**: The model converges quickly but to a less precise solution. For most applications, this doesn't matter.

**tol too low**: The model requires more iterations for marginal improvement. The accuracy gain from tol=1e-4 to tol=1e-6 is usually negligible.

The practical default: max_iter=100, tol=1e-4. This works for 99% of problems. Only change these if you have evidence that the default isn't sufficient.

## The Code

```rust
use datarust::linear_model::LogisticRegression;
use datarust::traits::Predictor;

let mut lr = LogisticRegression::new()
    .with_max_iter(50)    // reduce if converges fast
    .with_tol(1e-3);      // looser tolerance for faster convergence

lr.fit(&x, &y)?;
println!("converged in {} iterations", lr.n_iter());

// Check for non-convergence
assert!(lr.n_iter() < 50, "model did not converge");
```

## Try It

```bash
cargo add datarust
```

```rust
use datarust::linear_model::LogisticRegression;
use datarust::traits::Predictor;
use datarust::Matrix;

let x = Matrix::new(vec![
    vec![-1.0], vec![-0.5], vec![0.5], vec![1.0],
]).unwrap();
let y = vec![0.0, 0.0, 1.0, 1.0];

let mut lr = LogisticRegression::new().with_max_iter(10);
lr.fit(&x, &y).unwrap();
println!("converged in {} iterations (max=10)", lr.n_iter());
```

If your model converges in 6 iterations and you're giving it 100, you're paying for 94 iterations of nothing. Check `n_iter()`, adjust `max_iter`, and move on.
