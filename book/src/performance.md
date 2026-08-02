# Performance vs scikit-learn

The numbers below are **measured**, not estimated. The same deterministic synthetic dataset (xorshift64, seed 42, values in `[-100, 100)`) is fed to both libraries, and the median `fit_transform` time over 15 runs (after one warmup) is reported.

**Test setup:** Apple M5 Pro (18 cores, arm64), Rust 1.96.0 (release), Python 3.9.6, scikit-learn 1.6.1, numpy 2.0.2, scipy 1.13.1. Times are in **milliseconds**. The `Ratio` column is `sklearn_ms / datarust_ms` — values `> 1` mean **datarust is faster**.

## Benchmark table

| Workload | Size (rows × cols) | datarust default (ms) | datarust +rayon (ms) | sklearn (ms) | best ratio |
|---|---|---:|---:|---:|---:|
| StandardScaler | 1 000 × 10 | 0.023 | 0.016 | 0.280 | **17.5×** |
| StandardScaler | 10 000 × 100 | 1.24 | 1.20 | 2.46 | 2.0× |
| StandardScaler | 50 000 × 200 | 8.2 | 4.7 | 22.4 | **4.8×** |
| MinMaxScaler | 1 000 × 10 | 0.025 | 0.014 | 0.202 | **14.4×** |
| MinMaxScaler | 10 000 × 100 | 1.70 | 1.47 | 1.32 | 0.9× |
| MinMaxScaler | 50 000 × 200 | 10.8 | 7.5 | 11.6 | **1.5×** |
| RobustScaler | 1 000 × 10 | 0.17 | 0.13 | 0.768 | **5.8×** |
| RobustScaler | 10 000 × 100 | 11.2 | 2.09 | 21.4 | **10×** |
| RobustScaler | 50 000 × 200 | 123 | 14.0 | 193.5 | **13.8×** |
| PCA (k = min(10, cols/2)) | 1 000 × 10 | 0.18 | 0.10 | 0.226 | 2.2× |
| PCA | 10 000 × 100 | 45 | 41 | 1.39 | 0.03× |
| PCA | 50 000 × 200 | 838 | 819 | 12.2 | 0.01× |
| Pipeline (Standard→MinMax→Robust) | 1 000 × 10 | 0.20 | 0.21 | 1.02 | **4.9×** |
| Pipeline | 10 000 × 100 | 13.2 | 4.1 | 25.2 | 6.1× |
| Pipeline | 50 000 × 200 | 144 | 26.7 | 229.6 | **8.6×** |
| OneHotEncoder (string) | 1 000 × 5 | 0.38 | 0.55 | 0.800 | 1.5× |
| OneHotEncoder | 10 000 × 10 | 7.4 | 6.8 | 9.9 | 1.5× |
| OneHotEncoder | 50 000 × 20 | 89 | 80 | 205 | **2.6×** |
| ColumnTransformer (num + cat) | 1 000 × 5 | 0.026 | 0.026 | 4.6 | **179×** |
| ColumnTransformer | 10 000 × 10 | 0.23 | 0.24 | 79.8 | **347×** |
| ColumnTransformer | 50 000 × 20 | 1.31 | 1.32 | 812.8 | **620×** |
| LinearRegression (fit+predict) | 1 000 × 10 | 0.16 | 0.16 | — | — |
| LinearRegression | 10 000 × 100 | 14.4 | 14.4 | — | — |
| LinearRegression | 50 000 × 200 | 258 | 258 | — | — |

## Feature flags make a difference

**`matrixmultiply`.** Enabling this feature dispatches covariance and matmuls to a tuned pure-Rust GEMM (no system BLAS). On 50 000 × 200, PCA drops from **838 ms → 104 ms** (8× faster), and `LinearRegression` from **258 ms → 84 ms** (3× faster).

**`rayon`.** Parallel column/row processing. `RobustScaler` at 50 000 × 200 drops from 123 ms (default) to **14 ms** (8.8× faster with rayon).

## Scalar vs `matrixmultiply` kernels

The table below isolates what the optional `matrixmultiply` feature buys inside the
numeric hot paths — medians from the criterion suite
(`cargo bench -p datarust --bench benchmarks -- --warm-up-time 0.5 --measurement-time 1.5 --sample-size 10`),
single-threaded release build on the same Apple M5 Pro hardware as above. `+GEMM` is the
same build with `features = ["matrixmultiply"]`; `Speedup` is `scalar / +GEMM`. Each time
carries its own unit (ms / µs / ns).

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

The GEMM pays off most on matmul-heavy kernels — `Matrix::matmul` **2.5–2.8×**,
`LinearRegression` fit **2.5–2.7×**, `TruncatedSVD` **~1.9×** — while predict paths are
memory-bound matvecs and barely move (~1.0×). Pearson no longer needs GEMM for wide
tables (the lower-triangle scalar covariance puts the kernel at **6.1 ms** for
10 000 × 100); PCA barely moves at these small sizes and shines on larger matrices.

## Where datarust wins

- **Mixed numeric + categorical composition.** `ColumnTransformer` is **179–620×** faster than scikit-learn's on large inputs. This is the headline result — it reflects the cost of sklearn's per-column Python dispatch, dtype coercion, and object-array marshalling.
- **String / categorical encoding.** `OneHotEncoder` is ~1.5–2.6× faster because datarust operates on a native `StrMatrix` directly — no Python object-array overhead, no GIL.
- **Numeric scalers with `rayon`.** `StandardScaler`/`RobustScaler`/`Pipeline` beat sklearn by **4.8–13.8×** at 50 000 × 200.
- **Small data and startup latency.** At 1 000 × 10, datarust is faster on every workload — up to **17.5×** on `StandardScaler`. No Python interpreter to spin up, no numpy import cost.

## Where scikit-learn still wins

- **PCA on tall-and-wide data** (without the `matrixmultiply` feature). sklearn calls into LAPACK's full SVD via shared-library BLAS; datarust uses a from-scratch Jacobi sweep. With `matrixmultiply` the gap narrows from 85× to ~8×, and `PCASolver::Randomized` closes it further for low-rank inputs.

## Reproduce the benchmarks

The harness lives in `examples/bench_compare_rust.rs` (Rust side) and `benches/compare_sklearn.py` (Python side). Run on your own hardware:

```sh
# Rust (all feature combos)
cargo run --release --features matrixmultiply --example bench_compare_rust 15

# Python (requires numpy, scikit-learn)
python3 benches/compare_sklearn.py 15
```
