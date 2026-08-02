# The GEMM Feature Is a Matmul Story, Not a Magic Switch.

*Stable scalar-vs-`matrixmultiply` kernel numbers for datarust — where the optional GEMM pays 2–3×, and where it barely moves.*

---

datarust's default build has zero external dependencies. Every matrix product, every covariance, every eigendecomposition is pure Rust. But there's an optional `matrixmultiply` feature that swaps the scalar loops for a tuned GEMM kernel — still pure Rust, no system BLAS, but a much more carefully arranged inner loop.

I've mentioned it in the README and the changelogs with a few headline numbers. But I hadn't actually pinned down *which* kernels benefit and which don't. So I ran the full kernel benchmark suite twice — once scalar, once with `matrixmultiply` — with stable measurement settings, and here's what the numbers actually say.

## The methodology

Same deterministic xorshift64 data, same Apple M5 Pro (18 cores, arm64), Rust 1.96 release build. Each number is a criterion median with `--warm-up-time 0.5 --measurement-time 1.5 --sample-size 10`, single-threaded. `+GEMM` is the identical build with `features = ["matrixmultiply"]`. The `Speedup` column is `scalar / +GEMM`.

## The kernel table

| Benchmark | Size | scalar | +GEMM | Speedup |
|---|---|---:|---:|---:|
| `matrix_matmul` | 100 × 100 | 106 µs | **38.2 µs** | **2.8×** |
| `matrix_matmul` | 50 × 50 | 14.9 µs | **5.85 µs** | 2.5× |
| `matrix_matmul` | 10 × 10 | 290 ns | **162 ns** | 1.8× |
| `correlation_matrix_flat` (Pearson) | 10 000 × 100 | 6.07 ms | **4.12 ms** | 1.5× |
| `correlation_matrix_flat` | 10 000 × 50 | 1.98 ms | **1.25 ms** | 1.6× |
| `correlation_matrix_flat` | 100 000 × 20 | 5.32 ms | **2.93 ms** | **1.8×** |
| `linear_regression` fit | 100 000 × 100 | 147 ms | **54.3 ms** | **2.7×** |
| `linear_regression` fit | 10 000 × 50 | 4.33 ms | **1.75 ms** | 2.5× |
| `linear_regression` fit | 1 000 × 10 | 73.3 µs | **27.9 µs** | 2.6× |
| `linear_regression` predict | any | ≈ | ≈ | ~1.0× |
| `truncated_svd` | 1000 × 50 → 10 comps | 1.75 ms | **914 µs** | **1.9×** |
| `truncated_svd` | 500 × 30 → 5 comps | 356 µs | **183 µs** | 1.9× |
| `pca` | 200 × 20 → 5 comps | 109 µs | **92.8 µs** | 1.2× |
| `pca` | 50 × 10 → 3 comps | 21.3 µs | 19.9 µs | 1.1× |

## Where the GEMM pays off

The pattern is unmistakable: the feature is a **matmul story**. The three biggest wins are all places where a `p×p` or `n×p` product dominates the runtime:

- **`Matrix::matmul` — 2.5–2.8×.** This is the raw kernel, and it's the clearest win. 100 × 100 goes from 106 µs to 38 µs.
- **`LinearRegression` fit — 2.5–2.7×.** `fit` forms the normal equations `XᵀX` (a `p×p` product) and `Xᵀy` (a matvec) via `Matrix::matmul`, then solves with Cholesky. The GEMM slices the product part, and since that dominates, the whole fit accelerates almost as much as the raw kernel. At 100 000 × 100: **147 ms → 54 ms**.
- **`TruncatedSVD` — ~1.9×.** The `XᵀX` build is GEMM-backed here too. 1000 × 50 → 10 components drops from 1.75 ms to 914 µs.

The speedup persists down to surprisingly small sizes — even a 10 × 10 matmul is 1.8× faster, so there's no "only worth it on huge matrices" caveat for the *product* itself.

## Where it barely moves

Equally important: the places the feature does *not* help, because they're not matmul-bound.

- **`linear_regression` predict — ~1.0×.** Prediction is a memory-bound matvec (`X·β`). GEMM tuning helps repeated-multiply kernels, not a single pass over a wide row. If inference latency is your bottleneck, the feature is irrelevant.
- **PCA at small sizes — 1.1–1.2×.** At 200 × 20 the covariance is a 20 × 20 product — too small for GEMM's layout overhead to pay off. The eigensolver and transforms dominate instead. PCA only starts showing real gains on much larger matrices (the headline 50 000 × 200 case drops from 838 ms to 104 ms).
- **Pearson correlation — 1.5–1.8×.** This one deserves its own section.

## The Pearson surprise: the scalar path caught up

The most interesting result is Pearson. A few weeks ago `correlation_matrix_flat` at 10 000 × 100 took **38.7 ms scalar** and the GEMM feature looked essential. Then the scalar covariance switched to lower-triangle accumulation — building only half of `XcᵀXc` and mirroring it, with the same per-element accumulation order so the result is bit-identical.

That single change dropped the scalar kernel to **6.7 ms**, and the stable measurement settled at **6.07 ms scalar vs 4.12 ms +GEMM**. The GEMM still wins — by 1.5× — but the "must enable the feature" argument is gone. The same story plays out end-to-end in `datarust-profile`: wide `profile_matrix` at 10 000 × 100 is **31.6 ms scalar vs 30.5 ms +GEMM** — essentially tied, because Pearson is no longer the dominant cost of wide profiling. The per-column sorts and duplicate hashing took over that role.

The lesson: before reaching for a GEMM, check whether your scalar inner loop is doing wasted work. Halving the flops beat adding a tuned kernel.

## So should you enable it?

The decision rule is now fairly clean:

- **Enable `matrixmultiply`** if your workload is matmul-heavy — dense `Matrix::matmul`, `LinearRegression`/`Ridge`/`Lasso` fit, `TruncatedSVD`, or PCA on large matrices. Expect **2–3×** on those, and the cost is a pure-Rust dependency with no BLAS, no system libraries, no WASM breakage.
- **Skip it** if you're doing prediction, small PCA, or wide-table profiling — the numbers above show it buys little there, and keeping zero dependencies is simpler.
- **Measure on your own shapes.** The table is a map, not the territory. Run `cargo bench -p datarust --bench benchmarks` (with and without the feature) on your actual dimensions before you commit.

The feature was already documented as "faster for large dense inputs." Now it's measured: for the kernels that matter, it's a reliable 2–3×; for the kernels that don't, it's a no-op.

---

*All numbers are criterion medians (`--warm-up-time 0.5 --measurement-time 1.5 --sample-size 10`) on an Apple M5 Pro, Rust 1.96 release, deterministic xorshift64 data. Full tables live in the [datarust README](https://github.com/genc-murat/datarust/blob/main/crates/datarust/README.md). datarust is MIT-licensed and available on [crates.io](https://crates.io/crates/datarust).*
