# Four Allocation Taxes, One Reusable Mask — Then We Re-Measured Everything

*The optimization pass that started as a benchmark review, and the full table refresh (sklearn included) that followed.*

---

A benchmark suite is only as good as the last time you actually ran it. The datarust criterion suite had grown to cover the whole API surface — 47 groups in `datarust`, eleven more in `datarust-profile` (plus the serde-gated `report_json`) — and the README and book tables were full of numbers. But two things were stale: the numbers themselves (several predated fixes already in the tree), and one entire comparison table that had never been re-runnable on this machine.

So this pass had two halves. First, review the benchmarks, find the hot paths that were spending time on allocation instead of math, and fix them. Second, re-run *everything* — the full criterion suite, the `matrixmultiply` kernel subset, the datarust-profile suite, and finally the sklearn comparison harness with the exact documented Python stack — and refresh every table in one go.

Here's what the review found, what got fixed, and what the fresh numbers say.

## The methodology

Same protocol as the rest of the docs: criterion medians with `--warm-up-time 0.5 --measurement-time 1.5 --sample-size 10`, single-threaded release build on an Apple M5 Pro (18 cores, arm64), Rust 1.96. All `datarust` numbers are the **default** (zero-dependency) build on deterministic xorshift64 data. For the sklearn comparison, the mirrored Rust/Python harnesses ran with scikit-learn 1.6.1 / numpy 2.0.2 / scipy 1.13.1 in an isolated venv — the exact versions the README documents.

```sh
cargo bench -p datarust --bench benchmarks -- --warm-up-time 0.5 --measurement-time 1.5 --sample-size 10
cargo bench -p datarust-profile --bench benchmarks --features serde -- --warm-up-time 0.5 --measurement-time 1.5 --sample-size 10
cargo run --release --example bench_compare_rust 15        # default, +rayon, +matrixmultiply
python3 crates/datarust/benches/compare_sklearn.py 15
```

## Four bottlenecks the benchmarks exposed

None of these needed new algorithms. All four were constant-factor problems — allocation churn and needless passes — that only showed up because the benchmarks finally measured the right operations.

### 1. The categorical encoders were cloning every cell to read a column

`StrMatrix` has two ways to read a column: `column` (which clones every `String`) and `column_refs` (which borrows). `OneHotEncoder`, `OrdinalEncoder`, `FrequencyEncoder`, and `TargetEncoder` all used the cloning variant in their fit paths — one heap allocation per cell, purely to look at the values.

Switching all four to `column_refs`, materializing owned keys only for unique categories, removed the churn with identical results:

| Encoder `fit_transform` | Before | After | Improvement |
|---|---:|---:|---:|
| OrdinalEncoder 10 000 × 20 | ~20.7 ms | 3.60 ms | **5.7×** |
| FrequencyEncoder 10 000 × 20 | ~22.9 ms | 3.93 ms | **5.8×** |
| TargetEncoder 10 000 × 20 | ~11.8 ms | 4.16 ms | **2.8×** |
| OneHotEncoder 1 000 × 10 | ~0.64 ms | 0.23 ms | **2.8×** |

The same clone tax shows up end-to-end in the sklearn comparison: OneHotEncoder at 50 000 × 20 dropped from 89 ms to **54.7 ms** in the default build — that's almost entirely this fix.

### 2. `QuantileTransformer::transform` round-tripped through nested `Vec`s

The transform built a column-major nested `Vec`, transposed it into a row-major nested `Vec`, then flattened through `Matrix::new`. Two full copies of the buffer for no numerical reason. It now streams each value straight into a flat row-major output — sequential reads and writes, bit-identical values. 100 000 × 20: **28.3 ms → 22.9 ms**.

### 3. `transform_sparse` re-sorted rows that were already sorted

The sparse one-hot transform collected per-row triplets, then handed them to `SparseMatrix::from_triplets`, which re-sorted and deduplicated each row. But `encode_one` visits input columns in ascending order — the rows were already strictly increasing. Building the CSR `indptr`/`indices`/`data` arrays directly (values are always 1.0) skipped the per-row `Vec`s and the pointless sort:

| Sparse one-hot 50 000 × 20 | Before | After |
|---|---:|---:|
| `transform_sparse` | 27.6 ms | **18.3 ms** (1.5×) |
| `fit_transform_sparse` | 78.3 ms | **33.2 ms** (2.4×) |

The dense transform at the same size is 14.3 ms — sparse went from 2× slower than dense to within ~1.2×, which is the real target for a CSR path.

### 4. `StratifiedKFold::split` rebuilt each fold with a `HashSet`

Every fold built a `HashSet<usize>` of test indices, then scanned all `0..n` with hash lookups to build the training complement — k hash sets and k·n hash probes. A single reusable boolean mask (mark the test positions, scan once for the complement, reset) does the same job in k mask passes, with the complement `Vec` pre-sized so it never grows geometrically. The emitted folds are identical. `split` at 100 000 samples: **6.31 ms → 1.82 ms** (3.5×), bringing it from 31× slower than plain `KFold` to ~9× — the remainder being the stratification setup itself (class grouping + round-robin assignment).

## The full criterion refresh

With the fixes in, I ran the entire suite — 47 `datarust` groups plus the `datarust-profile` groups — and refreshed every table. Most rows moved a few percent (machine noise, ±2–6%); the interesting deltas:

| Benchmark | Before | After |
|---|---:|---:|
| `onehot_sparse` transform_sparse 50 000 × 20 | 18.3 ms | **16.8 ms** |
| `onehot_sparse` fit_transform_sparse 50 000 × 20 | 33.2 ms | **32.9 ms** |
| `strmatrix_column` column_refs 100 000 × 20 | 8.89 ms | **8.12 ms** |
| `stats_nested` quantiles 100 000 × 20 | 18.5 ms | **16.2 ms** |
| `label_encoder` fit_transform 10 000 × 10 | 578 µs | **558 µs** |
| `scaler_transform` quantile 100 000 × 20 | 23.4 ms | **14.6 ms** |

The `strmatrix_column` row is worth a second look: **`column` (clone) is 72.0 ms vs `column_refs` (borrow) 8.12 ms — a ~9× gap at 100 000 × 20.** That's the tax the encoders used to pay, now isolated as its own benchmark so it can't silently come back.

The `matrixmultiply` kernel table was also refreshed in the same session — scalar and `+GEMM` builds back-to-back so the ratios are apples-to-apples:

| Kernel | scalar | +GEMM | Speedup |
|---|---:|---:|---:|
| `matrix_matmul` 50 × 50 | 20.2 µs | **5.48 µs** | **3.7×** |
| `matrix_matmul` 100 × 100 | 104 µs | **37.6 µs** | **2.8×** |
| `linear_regression` fit 10 000 × 50 | 5.87 ms | **1.74 ms** | **3.4×** |
| `truncated_svd` 1000 × 50 → 10 comps | 1.80 ms | **949 µs** | **1.9×** |
| Pearson 10 000 × 100 | 5.98 ms | **4.05 ms** | 1.5× |

## The sklearn table, finally re-runnable

The one table that had been frozen: the "datarust vs scikit-learn" comparison needs Python, and this machine didn't have it. That's fixed — an isolated venv with the documented pins (scikit-learn 1.6.1, numpy 2.0.2, scipy 1.13.1) — and the whole comparison was re-run for the first time since several of the optimizations above landed.

The refresh replaced stale pre-optimisation numbers across the board:

- **OneHotEncoder 50 000 × 20: 89 → 54.7 ms** default (the encoder clone fix) — datarust now beats sklearn's 179.8 ms by **3.3×** in the default build (**4.0×** with `rayon`).
- **PCA 50 000 × 200: 838 → 206 ms.** Still behind sklearn (12.0 ms; ratio 0.06×), but the gap that used to look enormous is now ~17× default and **~9× with `matrixmultiply`** (206 → 111 ms).
- **`ColumnTransformer` 50 000 × 20: 2.10 ms vs sklearn's 796.7 ms — 380×.** The headline result holds up: sklearn pays per-column Python dispatch and object-array marshalling on mixed inputs.
- **The `LinearRegression` sklearn column is no longer empty.** It had been "—" because numpy/scipy weren't available. Now filled in: `fit+predict` at 50 000 × 200 is **263 ms default vs sklearn's 118 ms** (0.45×) — but **91 ms with `matrixmultiply`**, which edges out sklearn. That's the first row where datarust leads on a matmul-heavy workload.

The `datarust-profile` numbers moved too (cramér's V 2.12 → 2.04 ms, point-biserial 12.3 → 12.0 ms, high-cardinality profiling 135 → 122 ms, `report_json` 38.8 → 36.1 µs) — drift plus the shared mode/point-biserial fixes from the previous pass.

## What the fresh numbers say

The refresh confirmed the earlier analysis rather than overturning it. The four fixes this pass were the last of the *allocation* class of bottlenecks — the ones where the code was doing real work, plus a tax on top. What's left is algorithmic:

- **PCA is the remaining sklearn gap** — LAPACK's full SVD via shared-library BLAS is a hard target for a from-scratch Jacobi sweep. The `matrixmultiply` feature and the opt-in randomized SVD both close it further for the shapes where it matters.
- **The GEMM feature is confirmed as a matmul story** — 2.7–3.7× across matmul kernels and fits, ~1.0× on memory-bound predict paths. Nothing changed about that conclusion; it just has fresh numbers.

And the pass I thought was finished wasn't quite: the refresh flagged `QuantileTransformer::transform` as the last per-value outlier, and that turned out to be fixable too. Each value paid a full `log₂(1000)` binary search over the column's reference array; the transform now pre-partitions that array into 512 uniform value spans (built once per call) so a value lands in O(1) and only a handful of entries need scanning — **23.4 ms → 14.6 ms at 100 000 × 20** (1.6×, ~7.3 ns/value, bit-identical output). The table above already reflects it.

The takeaway is the same one every pass reinforces: the benchmarks are only trustworthy if they're current. Both harnesses are now one command each, the venv is reproducible, and every table in the README and book reflects this exact build. The next time something gets faster — or slower — the suite will show it.

---

*All numbers are criterion medians (`--warm-up-time 0.5 --measurement-time 1.5 --sample-size 10`) on an Apple M5 Pro, Rust 1.96 release, deterministic xorshift64 data; the sklearn comparison used scikit-learn 1.6.1 / numpy 2.0.2 / scipy 1.13.1. Full tables live in the [datarust README](https://github.com/genc-murat/datarust/blob/main/crates/datarust/README.md) and the [mdBook performance page](https://github.com/genc-murat/datarust/blob/main/book/src/performance.md). datarust is MIT-licensed and available on [crates.io](https://crates.io/crates/datarust).*
