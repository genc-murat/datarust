# Performance vs scikit-learn

The numbers below are **measured**, not estimated. The same deterministic synthetic dataset (xorshift64, seed 42, values in `[-100, 100)`) is fed to both libraries, and the median `fit_transform` time over 15 runs (after one warmup) is reported.

**Test setup:** Apple M5 Pro (18 cores, arm64), Rust 1.96.0 (release), Python 3.9.6, scikit-learn 1.6.1, numpy 2.0.2, scipy 1.13.1. Times are in **milliseconds**. The `Ratio` column is `sklearn_ms / datarust_ms` — values `> 1` mean **datarust is faster**.

## Benchmark table

| Workload | Size (rows × cols) | datarust default (ms) | datarust +rayon (ms) | sklearn (ms) | best ratio |
|---|---|---:|---:|---:|---:|
| StandardScaler | 1 000 × 10 | 0.031 | 0.030 | 0.270 | **8.9×** |
| StandardScaler | 10 000 × 100 | 1.69 | 1.69 | 2.39 | 1.4× |
| StandardScaler | 50 000 × 200 | 14.2 | 10.4 | 21.5 | 2.1× |
| MinMaxScaler | 1 000 × 10 | 0.033 | 0.035 | 0.199 | **5.9×** |
| MinMaxScaler | 10 000 × 100 | 1.75 | 1.88 | 1.32 | 0.8× |
| MinMaxScaler | 50 000 × 200 | 17.7 | 13.4 | 11.4 | 0.8× |
| RobustScaler | 1 000 × 10 | 0.11 | 0.18 | 0.722 | **6.3×** |
| RobustScaler | 10 000 × 100 | 6.05 | 1.90 | 21.4 | **11×** |
| RobustScaler | 50 000 × 200 | 68.7 | 14.7 | 193.8 | **13×** |
| PCA (k = min(10, cols/2)) | 1 000 × 10 | 0.11 | 0.12 | 0.220 | 2.0× |
| PCA | 10 000 × 100 | 14.0 | 14.1 | 1.35 | 0.10× |
| PCA | 50 000 × 200 | 206 | 205 | 12.0 | 0.06× |
| Pipeline (Standard→MinMax→Robust) | 1 000 × 10 | 0.16 | 0.27 | 0.921 | **5.7×** |
| Pipeline | 10 000 × 100 | 9.57 | 5.04 | 28.0 | **5.6×** |
| Pipeline | 50 000 × 200 | 101.5 | 39.9 | 227.5 | **5.7×** |
| OneHotEncoder (string) | 1 000 × 5 | 0.21 | 0.40 | 0.780 | **3.8×** |
| OneHotEncoder | 10 000 × 10 | 4.25 | 3.57 | 10.2 | **2.9×** |
| OneHotEncoder | 50 000 × 20 | 54.7 | 45.0 | 179.8 | **4.0×** |
| ColumnTransformer (num + cat) | 1 000 × 5 | 0.029 | 0.031 | 4.41 | **153×** |
| ColumnTransformer | 10 000 × 10 | 0.31 | 0.31 | 77.9 | **255×** |
| ColumnTransformer | 50 000 × 20 | 2.10 | 2.11 | 796.7 | **380×** |
| LinearRegression (fit+predict) | 1 000 × 10 | 0.12 | 0.12 | 0.314 | 2.6× |
| LinearRegression | 10 000 × 100 | 15.1 | 15.1 | 15.1 | 1.0× |
| LinearRegression | 50 000 × 200 | 263 | 264 | 118 | 0.45× |

## Feature flags make a difference

**`matrixmultiply`.** Enabling this feature dispatches covariance and matmuls to a tuned pure-Rust GEMM (no system BLAS). On 50 000 × 200, PCA drops from **206 ms → 111 ms** (1.9× faster), and `LinearRegression` `fit+predict` from **263 ms → 91 ms** (2.9× faster).

**`rayon`.** Parallel column/row processing. `RobustScaler` at 50 000 × 200 drops from 68.7 ms (default) to **14.7 ms** (4.7× faster with rayon).

## Scalar vs `matrixmultiply` kernels

The table below isolates what the optional `matrixmultiply` feature buys inside the
numeric hot paths — medians from the criterion suite
(`cargo bench -p datarust --bench benchmarks -- --warm-up-time 0.5 --measurement-time 1.5 --sample-size 10`),
single-threaded release build on the same Apple M5 Pro hardware as above. `+GEMM` is the
same build with `features = ["matrixmultiply"]`; `Speedup` is `scalar / +GEMM`. Each time
carries its own unit (ms / µs / ns).

| Benchmark | Size | scalar | +GEMM | Speedup |
|---|---|---:|---:|---:|
| `matrix_matmul` | 100 × 100 | 104 µs | **37.6 µs** | **2.8×** |
| `matrix_matmul` | 50 × 50 | 20.2 µs | **5.48 µs** | **3.7×** |
| `matrix_matmul` | 10 × 10 | 280 ns | **158 ns** | 1.8× |
| `correlation_matrix_flat` (Pearson) | 10 000 × 100 | 5.98 ms | **4.05 ms** | 1.5× |
| `correlation_matrix_flat` | 10 000 × 50 | 1.91 ms | **1.25 ms** | 1.5× |
| `correlation_matrix_flat` | 100 000 × 20 | 5.22 ms | **2.89 ms** | **1.8×** |
| `linear_regression` fit | 100 000 × 100 | 147 ms | **54.1 ms** | **2.7×** |
| `linear_regression` fit | 10 000 × 50 | 5.87 ms | **1.74 ms** | **3.4×** |
| `linear_regression` fit | 1 000 × 10 | 74.1 µs | **27.5 µs** | 2.7× |
| `linear_regression` predict | any | ≈ | ≈ | ~1.0× |
| `truncated_svd` | 1000 × 50 → 10 comps | 1.80 ms | **949 µs** | **1.9×** |
| `truncated_svd` | 500 × 30 → 5 comps | 376 µs | **199 µs** | **1.9×** |
| `pca` | 200 × 20 → 5 comps | 109 µs | **93.2 µs** | 1.2× |
| `pca` | 50 × 10 → 3 comps | 21.1 µs | 19.8 µs | 1.1× |

The GEMM pays off most on matmul-heavy kernels — `Matrix::matmul` **2.8–3.7×**,
`LinearRegression` fit **2.7–3.4×**, `TruncatedSVD` **~1.9×** — while predict paths are
memory-bound matvecs and barely move (~1.0×). Pearson no longer needs GEMM for wide
tables (the lower-triangle scalar covariance puts the kernel at **6.0 ms** for
10 000 × 100); PCA barely moves at these small sizes and shines on larger matrices.

## Where datarust wins

- **Mixed numeric + categorical composition.** `ColumnTransformer` is **153–380×** faster than scikit-learn's on large inputs. This is the headline result — it reflects the cost of sklearn's per-column Python dispatch, dtype coercion, and object-array marshalling.
- **String / categorical encoding.** `OneHotEncoder` is ~2.9–4.0× faster because datarust operates on a native `StrMatrix` directly — no Python object-array overhead, no GIL.
- **Numeric scalers with `rayon`.** `StandardScaler`/`RobustScaler`/`Pipeline` beat sklearn by **2.1–13×** at 50 000 × 200.
- **Small data and startup latency.** At 1 000 × 10, datarust is faster on every workload — up to **8.9×** on `StandardScaler` and **153×** on `ColumnTransformer`. No Python interpreter to spin up, no numpy import cost.

## Where scikit-learn still wins

- **PCA on tall-and-wide data** (without the `matrixmultiply` feature). sklearn calls into LAPACK's full SVD via shared-library BLAS; datarust uses a from-scratch Jacobi sweep. With `matrixmultiply` the gap narrows from ~17× to ~9×, and `PCASolver::Randomized` closes it further for low-rank inputs.

## Criterion microbenchmark suite

Alongside the sklearn comparison, the criterion suite
(`cargo bench -p datarust --bench benchmarks` and
`cargo bench -p datarust-profile --bench benchmarks`) covers the whole API
surface. The medians below are from the **default** (zero-dependency) build with
`--warm-up-time 0.5 --measurement-time 1.5 --sample-size 10` on the same Apple M5
Pro, and document the operations added most recently. Each time carries its own
unit.

### datarust

| Benchmark | Size | median |
|---|---|---:|
| `label_encoder` fit_transform | 10 000 × 10 | 558 µs |
| `label_encoder` fit_transform | 100 000 × 10 | 5.67 ms |
| `label_encoder` fit_transform | 100 000 × 10 000 classes | 9.17 ms |
| `stats_nested` mean_var | 10 000 × 100 | 220 µs |
| `stats_nested` quantiles | 10 000 × 100 | 7.90 ms |
| `stats_nested` mode_column | 10 000 × 100 | 11.1 ms |
| `stats_nested` mean_var | 100 000 × 20 | 705 µs |
| `stats_nested` quantiles | 100 000 × 20 | 16.2 ms |
| `stats_nested` mode_column | 100 000 × 20 | 27.7 ms |
| `train_test_split` | 10 000 × 50 | 116 µs |
| `train_test_split` | 100 000 × 20 | 784 µs |
| `matrix_ops` from_flat | 10 000 × 50 | 49.0 µs |
| `matrix_ops` from_flat | 100 000 × 20 | 216 µs |

The `stats_nested` group mirrors the flat-storage kernels over `Vec<Vec<f64>>`
inputs; `mode_column` shares `mode`'s sort-then-scan, so a 100 000 × 20
mostly-distinct table modes in **27.7 ms**. `matrix_ops/from_flat` (49 µs at
10 000 × 50) shows why flat construction is the fast path over the nested
`from_nested` (~289 µs for the same shape).

The next pass added eight groups covering the **production-time paths** that
`fit_transform`-only benchmarks hid — `transform`/`inverse_transform` on fitted
transformers, `predict_proba`, fitted-encoder transforms, sparse one-hot, the
remaining classification metrics, the splitters, and string-column gathering:

| Benchmark | Size | median |
|---|---|---:|
| `metrics_more` confusion_matrix | 100 000 | 1.25 ms |
| `metrics_more` precision/recall/F1/kappa/MCC | 100 000 | ~1.24 ms |
| `metrics_more` log_loss | 100 000 | 714 µs |
| `metrics_more` average_precision_score | 100 000 | 1.70 ms |
| `logistic_predict_proba` predict_proba_binary | 50 000 × 100 | 6.14 ms |
| `logistic_predict_proba` predict_proba_multiclass | 10 000 × 50 | 715 µs |
| `encoder_transform` onehot_transform | 50 000 × 20 | 14.3 ms |
| `encoder_transform` ordinal/frequency/target transform | 50 000 × 20 | ~9.3 ms |
| `polynomial_transform` transform | 10 000 × 5, d3 | 876 µs |
| `scaler_transform` standard/minmax/robust transform | 100 000 × 20 | ~1.72 ms |
| `scaler_transform` standard/minmax/robust inverse | 100 000 × 20 | ~1.45 ms |
| `scaler_transform` quantile_transform | 100 000 × 20 | **14.6 ms** |
| `onehot_sparse` transform_sparse | 50 000 × 20 | 16.8 ms |
| `onehot_sparse` fit_transform_sparse | 50 000 × 20 | 32.9 ms |
| `model_selection` kfold_5_shuffled | 100 000 | 210 µs |
| `model_selection` stratified_kfold_5 | 100 000 | 1.86 ms |
| `strmatrix_column` column_clone | 100 000 × 20 | 72.0 ms |
| `strmatrix_column` column_refs | 100 000 × 20 | 8.12 ms |

Three rows are the current bottlenecks. **`QuantileTransformer.transform` is
still the per-value outlier — ~7.3 ns per value** (14.6 ms at 100 000 × 20,
down from 23.4 ms) against ~0.85 ns for the linear scalers: each value maps
into one of 512 pre-partitioned value spans over the 1 000-point reference and
interpolates inside that span's handful of entries — O(1) plus a few compares
instead of a full log₂(1000) binary search, with bit-identical output. The
remaining cost is the per-value interpolation itself (the nested-Vec transpose
round-trips are gone; the transform now streams the flat buffer row-major).
**`StrMatrix::column` (clone) is ~8× slower than the borrowing `column_refs`**
(72.0 ms vs 8.12 ms at 100 000 × 20) — all four categorical encoders used to
pay this clone tax in their fit paths, and switching them to `column_refs` made
`fit_transform` **60–83% faster** (ordinal/frequency fit at 10 000 × 20: ~21 ms
→ ~3.6–3.9 ms, with transform unchanged). **`StratifiedKFold.split`** used to
rebuild each fold's train set with a per-fold `HashSet` scan of every sample; a
reusable boolean mask (mark → scan → reset) cut it from 6.31 ms to **1.86 ms**
at 100 000 rows (**−71%**), leaving the stratification setup itself as the
remaining cost. `onehot_sparse`
`transform_sparse` was 2× slower than the dense transform because it
round-tripped through per-row triplet `Vec`s and `SparseMatrix::from_triplets`'
per-row re-sort; it now builds the CSR arrays directly — 16.8 ms vs 14.3 ms
dense at 50 000 × 20 (down from 27.6 ms), and `fit_transform_sparse` dropped
from 78.3 ms to 32.9 ms.

### datarust-profile

| Benchmark | Size | median |
|---|---|---:|
| `cramers_v_high_cardinality` | 10 000 × 10, 100 levels | 2.04 ms |
| `cramers_v_high_cardinality` | 10 000 × 10, 1 000 levels | 6.01 ms |
| `point_biserial` | 10 000 × 20 num + 20 bin | 12.0 ms |
| `point_biserial` | 50 000 × 10 num + 10 bin | 20.8 ms |
| `profile_str_high_cardinality` | 10 000 × 20 | 82.6 ms |
| `profile_str_high_cardinality` | 50 000 × 10 | 122 ms |
| `report_json` to_json | 10 000 × 20 | 36.1 µs |
| `report_json` to_json | 100 000 × 20 | 38.5 µs |

The high-cardinality groups pin down the worst cases of wide-table profiling:
near-unique identifier columns (10 000 × 20) take ~83 ms end-to-end — dominated
by duplicate detection and per-column categorical tallies — and Cramér's V over
1 000-level columns costs ~6 ms per 10-column table. `report_json` scales with
the **column** count, not the row count (the profile is column-summarized), so
10 000 × 20 and 100 000 × 20 serialize in ~36–39 µs.

## Reproduce the benchmarks

The harness lives in `examples/bench_compare_rust.rs` (Rust side) and `benches/compare_sklearn.py` (Python side). Run on your own hardware:

```sh
# Rust (all feature combos)
cargo run --release --features matrixmultiply --example bench_compare_rust 15

# Python (requires numpy, scikit-learn)
python3 benches/compare_sklearn.py 15
```
