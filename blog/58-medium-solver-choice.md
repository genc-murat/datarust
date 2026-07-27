---
title: "Cholesky Was 30x Faster Than SVD. Both Gave Identical R²."
subtitle: "When solver choice matters (and when it doesn't)"
author: "Murat Genc"
date: "2026-07-26"
tags: ["machine-learning", "rust", "datarust", "solver", "ridge", "linear-algebra"]
series: "datarust-v06"
---

datarust offers two solvers for Ridge and LogisticRegression: Cholesky (default) and SVD. Both solve the same problem, but they compute the solution differently. The choice matters for speed and numerical stability, not for accuracy.

## Experiment 1: Well-Conditioned Data

3 features, 200 samples, condition number ~1:

```
Cholesky: R² = 0.994309, time = 35.3µs
SVD:      R² = 0.994309, time = 20.3µs
Coef diff: [1.3e-15, 8.9e-16, 2.2e-16]
```

Both give identical R² and coefficients (difference at machine precision ~1e-15). SVD is actually faster here, but both are under 40 microseconds — irrelevant for practical purposes.

## Experiment 2: Collinear Features

Feature 2 is feature 1 plus noise (σ=0.001):

```
Cholesky: R² = 0.997059, time = 12.6µs
SVD:      R² = 0.997059, time = 12.5µs
Coef diff: [8.6e-14, 8.5e-14, 4.4e-16]
```

Collinearity doesn't affect either solver when alpha > 0. The regularization term `αI` makes the system matrix positive-definite, so Cholesky succeeds. The coefficient difference is still at machine precision.

## Experiment 3: High-Dimensional Data (p > n)

50 features, 20 samples:

```
Cholesky: R² = 0.999574, time = 65.0µs
SVD:      R² = 0.999574, time = 1.98ms
```

Cholesky is 30× faster. The reason: Cholesky solves the p×p system `(XᵀX + αI)β = Xᵀy`, which is 50×50. SVD decomposes the n×p matrix X directly, which is 20×50 — more expensive when p > n.

Both give identical R² and coefficients. The speed difference is the only practical consideration.

## Experiment 4: Alpha Stability

Different alpha values, same data:

```
alpha        Cholesky R²    SVD R²        coef diff
------------------------------------------------------
0.001        0.994328     0.994328     6.66e-16
0.01         0.994328     0.994328     4.44e-16
0.1          0.994328     0.994328     8.88e-16
1            0.994309     0.994309     1.33e-15
10           0.992515     0.992515     5.55e-16
100          0.899729     0.899729     1.33e-15
```

The coefficient difference is always ~1e-15 — machine precision. The solvers are numerically identical across all alpha values.

## The Math

**Cholesky** solves the normal equations:
```
(XᵀX + αI) β = Xᵀy
```
Decomposes `XᵀX + αI = LLᵀ` (lower triangular), then solves two triangular systems. Cost: O(p³) for the decomposition, O(p²) for the solve.

**SVD** decomposes X directly:
```
X = U Σ Vᵀ
β = V (Σ² + αI)⁻¹ Σ Uᵀ y
```
Cost: O(np²) for the decomposition, O(p²) for the solve.

When n ≈ p, both are O(p³). When n >> p, SVD is O(np²) which is more expensive. When p >> n, Cholesky is O(p³) which is cheaper than SVD's O(np²).

## When to Use Each

| Scenario | Cholesky | SVD |
|----------|----------|-----|
| Well-conditioned (cond < 100) | ✓ (faster) | ✓ (same result) |
| Collinear (cond > 1000) | ✓ (alpha fixes it) | ✓ (same result) |
| High-dimensional (p > n) | ✓ (faster) | ✓ (slower) |
| Near-singular (alpha ≈ 0) | ✗ (may fail) | ✓ (more stable) |
| Production (default) | ✓ | ✓ |

**Use Cholesky when:**
- Features are not extremely collinear
- alpha > 0 (default)
- You need maximum speed

**Use SVD when:**
- alpha is very small (near 0)
- You suspect numerical instability
- You want the most robust solution (even if slower)

**The practical default:** Cholesky. It's faster in most cases, and the regularization term (alpha > 0) prevents numerical issues.

## The Code

```rust
use datarust::linear_model::{Ridge, RidgeSolver};
use datarust::traits::Predictor;

// Default: Cholesky (fast)
let mut ridge = Ridge::new().with_alpha(1.0);
ridge.fit(&x, &y)?;

// Explicit SVD (robust)
let mut ridge_svd = Ridge::new()
    .with_alpha(1.0)
    .with_solver(RidgeSolver::Svd);
ridge_svd.fit(&x, &y)?;

// Both give identical coefficients
assert_eq!(ridge.coef(), ridge_svd.coef()); // within machine precision
```

## Tradeoffs

**Cholesky advantages:**
- Faster (O(p³) vs O(np²))
- Simpler implementation
- Cache-friendly (triangular solves are sequential)

**Cholesky disadvantages:**
- Requires positive-definite matrix (alpha > 0 guarantees this)
- Can fail if alpha is extremely small and X is rank-deficient

**SVD advantages:**
- Always works (even for rank-deficient systems)
- More numerically stable
- No requirement on alpha

**SVD disadvantages:**
- Slower (especially when p > n)
- More memory (stores U, Σ, V)
- Less cache-friendly

The speed difference matters for hyperparameter search (fitting the model thousands of times). For a single fit, both are under 2ms — choose based on robustness, not speed.

## Try It

```bash
cargo add datarust
```

```rust
use datarust::linear_model::{Ridge, RidgeSolver};
use datarust::traits::Predictor;
use datarust::Matrix;

let x = Matrix::new(vec![
    vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0],
]).unwrap();
let y = vec![1.0, 2.0, 3.0];

let mut ridge_c = Ridge::new().with_solver(RidgeSolver::Cholesky);
ridge_c.fit(&x, &y).unwrap();
println!("Cholesky coef: {:?}", ridge_c.coef());

let mut ridge_s = Ridge::new().with_solver(RidgeSolver::Svd);
ridge_s.fit(&x, &y).unwrap();
println!("SVD coef: {:?}", ridge_s.coef());
```

If your model works with Cholesky, keep using it. Switch to SVD only when you see numerical errors — and even then, increasing alpha usually fixes the problem faster than changing solvers.
