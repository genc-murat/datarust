# Decomposition

Dimensionality reduction. Live in [`datarust::decomposition`](https://docs.rs/datarust/latest/datarust/decomposition/index.html). Both implement [`Transformer`](https://docs.rs/datarust/latest/datarust/trait.Transformer.html).

## PCA

Principal Component Analysis via Jacobi eigenvalue decomposition of the covariance matrix.

```rust
use datarust::decomposition::{PCA, PCAComponents};

// Select components by count, variance ratio, or keep all:
let mut pca = PCA::new(PCAComponents::Variance(0.95)) // keep 95% of variance
    .whiten(true);                                    // optional: whiten components
let projected = pca.fit_transform(&x)?;

// After fit:
pca.components();                // [[f64; n_features]; n_components]
pca.explained_variance_ratio();  // how much variance each component captures
pca.noise_variance();            // estimated noise variance
let reconstructed = pca.inverse_transform(&projected)?; // approximate recovery
```

### Component selection

- `PCAComponents::Count(k)` — keep exactly `k` components.
- `PCAComponents::Variance(0.95)` — keep enough to explain 95% of variance.
- `PCAComponents::All` — keep all components.

### Solver

```rust
use datarust::decomposition::PCASolver;

let mut pca = PCA::new(PCAComponents::Count(10))
    .solver(PCASolver::Randomized); // Halko–Martinsson–Tropp randomized SVD
```

- `Auto` (default) — exact eigensolver.
- `Full` — full Jacobi eigendecomposition.
- `Randomized` — randomized SVD, much faster for tall-and-wide, low-rank data.

> **Performance tip:** enabling the `matrixmultiply` feature speeds up PCA significantly on large dense inputs by dispatching covariance and transform matmuls to a tuned pure-Rust GEMM.

## TruncatedSVD

Dimensionality reduction via truncated SVD. Does **not** center the data, making it suitable for sparse or TF-IDF inputs.

```rust
use datarust::decomposition::{TruncatedSVD, SVDComponents};

// By count, variance threshold, or all:
let mut svd = TruncatedSVD::new(5).unwrap();                 // 5 components
let mut svd = TruncatedSVD::new(0.95).unwrap();              // 95% variance
let mut svd = TruncatedSVD::new(SVDComponents::All).unwrap();
let out = svd.fit_transform(&x)?;
```

## PCA vs TruncatedSVD

| | PCA | TruncatedSVD |
|---|---|---|
| Centers data | Yes | No |
| Sparse input | No | Yes |
| Use case | Dense, find directions of max variance | Sparse/TF-IDF, latent semantic analysis |
