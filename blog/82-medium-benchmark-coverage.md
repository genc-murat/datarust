# Eleven New Benchmarks, Two Fixed Bottlenecks

*What the datarust criterion suite now covers — stable medians, and the two hot paths that were spending time on allocation instead of math.*

---

The criterion suite is how datarust finds its slow spots. Every new feature lands with a benchmark group; every optimization lands with a before/after. But coverage has always lagged the API surface — a handful of public entry points had no benchmark at all, which means nobody could tell whether a change made them faster or slower.

So I went looking for the gaps. This pass added **11 benchmark groups** across `datarust` (36 → 40 groups) and `datarust-profile` (8 → 12 groups), measured them all with the same stable protocol the README and book use, and — as often happens when you actually measure — two real bottlenecks fell out that were fixable without touching any numerics.

## The methodology

Same protocol as the rest of the docs: criterion medians with `--warm-up-time 0.5 --measurement-time 1.5 --sample-size 10`, single-threaded release build on an Apple M5 Pro (18 cores, arm64), Rust 1.96. All `datarust` numbers are the **default** (zero-dependency) build on deterministic xorshift64 data. The `datarust-profile` bench was run with `--features serde` so the `report_json` group can measure the JSON path.

```sh
cargo bench -p datarust --bench benchmarks -- --warm-up-time 0.5 --measurement-time 1.5 --sample-size 10
cargo bench -p datarust-profile --bench benchmarks --features serde -- --warm-up-time 0.5 --measurement-time 1.5 --sample-size 10
```

## What the suite was missing

A quick inventory of the public API against the benchmark list turned up these blind spots:

| Missing coverage | Why it matters |
|---|---|
| `LabelEncoder` | The only 1-D label transformer; was never measured at all |
| Nested-`Vec` stats (`column_mean_var`, `column_quantiles_many`, `mode_column`) | Only the flat-storage variants were benched; callers using the nested API were unmeasured |
| `train_test_split` | The most common model-selection entry point, zero coverage |
| `Matrix::from_flat` | Flat construction is the fast path everywhere, yet only `from_nested` was benched |
| Cramér's V at high cardinality | The pairwise path was only exercised on 5-level columns; the hash-map contingency fallback was never stressed |
| Point-biserial (binary × numeric) | Only ran indirectly inside a mixed-table bench |
| Near-unique / identifier columns | The worst case for categorical tallies, duplicate detection, and Cramér's V was unmeasured |
| `report::to_json` | HTML rendering was benched; the serde JSON path was not |

One thing I wanted to add and couldn't: `RandomizedSVD` is `pub(crate)` (still being verified), so it can't be benchmarked from outside the crate. It'll get coverage when it ships as public API.

## The new numbers

**datarust** (default build):

| Benchmark | Size | median |
|---|---|---:|
| `label_encoder` fit_transform | 10 000 × 10 | 578 µs |
| `label_encoder` fit_transform | 100 000 × 10 | 5.79 ms |
| `label_encoder` fit_transform | 100 000 × 10 000 classes | 9.42 ms |
| `stats_nested` mean_var | 10 000 × 100 | 223 µs |
| `stats_nested` quantiles | 10 000 × 100 | 8.32 ms |
| `stats_nested` mode_column | 10 000 × 100 | 11.6 ms |
| `stats_nested` mean_var | 100 000 × 20 | 711 µs |
| `stats_nested` quantiles | 100 000 × 20 | 18.5 ms |
| `stats_nested` mode_column | 100 000 × 20 | 27.6 ms |
| `train_test_split` | 10 000 × 50 | 114 µs |
| `train_test_split` | 100 000 × 20 | 763 µs |
| `matrix_ops` from_flat | 10 000 × 50 | 48.3 µs |
| `matrix_ops` from_flat | 100 000 × 20 | 214 µs |

**datarust-profile** (`--features serde`):

| Benchmark | Size | median |
|---|---|---:|
| `cramers_v_high_cardinality` | 10 000 × 10, 100 levels | 2.12 ms |
| `cramers_v_high_cardinality` | 10 000 × 10, 1 000 levels | 6.19 ms |
| `point_biserial` | 10 000 × 20 num + 20 bin | 12.3 ms |
| `point_biserial` | 50 000 × 10 num + 10 bin | 21.8 ms |
| `profile_str_high_cardinality` | 10 000 × 20 | 84.9 ms |
| `profile_str_high_cardinality` | 50 000 × 10 | 135 ms |
| `report_json` to_json | 10 000 × 20 | 38.8 µs |
| `report_json` to_json | 100 000 × 20 | 38.7 µs |

## Where the time actually goes

Now that the suite covers the whole surface, the per-value cost ranking is clear. Convert everything to nanoseconds per value (rows × cols) and the outliers pop out immediately:

- **`PowerTransformer` fit — ~1.0 µs per value.** At 1 000 × 20 (20k values) the fit takes **20.0 ms**; `StandardScaler` on a comparable workload is ~1.6 ns per value. That's a ~600× gap. It's not a bug — every candidate λ re-transforms the whole column to score its log-likelihood (101 grid points + 40 golden-section refinements per column). It *is* the most expensive transformer per unit data in the crate, and the search is arguably over-tuned for the precision it needs.
- **Quantile machinery — sorts dominate.** `stats_bulk/quantiles_flat` at 100 000 × 100 is **83.5 ms** (10M values, ~8.4 ns/value, all in per-column order statistics), and `kbins_discretizer` quantile at 10 000 × 100 is **17.2 ms**. These are the same per-column sort pattern, so they scale with `n·log n` per column.
- **Imputers — the sort tax is visible.** At 50 000 × 20 (1M cells): `Mean` **3.43 ms**, `Constant` **3.05 ms**, but `Median` **12.6 ms** and `MostFrequent` **14.8 ms**. The ~10 ms gap between the strategies is exactly the 20 per-column sorts. It's algorithmic — sklearn pays the same — but it's the first thing to look at if imputation shows up in a profile.
- **Quadratic kernels are the floor for their algorithms.** `KnnImputer` 1 000 × 20: **17.3 ms**; `silhouette_score` 5 000 × 20 × 10 clusters: **85.6 ms**. Both are O(n²·d) by construction; the recently-removed inner-loop allocations already brought these down, and the remaining cost is distance math.
- **`SelectKBest` mutual information — 36.0 ms at 50 000 × 50.** Histogram-based MI over 50 features; the `f_classif` path is ~2× cheaper. Worth revisiting if feature selection becomes hot.

Two of these — `mode_column` and point-biserial — were not "algorithmic, accept it" cases. They were bad constant factors, and they're now fixed.

## Two fixes that fell out of the measurement

### `mode_column`: hash-map counting → sort-then-scan

The 1-D `stats::mode` had already been switched from a `HashMap<u64, …>` (one heap entry per distinct value) to "sort a copy, scan the longest run" — a huge win on continuous columns where every value is distinct. But its columnar sibling `mode_column` still used the hash map. The new `stats_nested/mode_column` benchmark exposed exactly that: 100 000 × 20 of mostly-distinct values.

The fix shares a single private `mode_sorted` helper between both functions: gather the column, `total_cmp` sort, walk runs. Tie-breaking is identical (smallest value wins), empty-column `NaN` semantics are unchanged, and the benchmark now reads **27.6 ms** for 100 000 × 20 — the sort is the whole story, and there's no per-value hash to allocate.

### Point-biserial: three allocations per pair → zero

`Relationships::compute` runs the point-biserial path for every (binary categorical, numeric) pair. The old `calculate_point_biserial` materialised three `Vec`s per pair — `group_0`, `group_1`, and a concatenated `all_y` — then ran means and a std over them. On 20 × 20 pairs of 10k rows, that's ~1 200 allocations and ~96 MB of churn, per profile.

The rewrite is a single allocation-free pass using the computational identity `SS = Σy² − (Σy)² / N` for pooled variance — same accumulation order, same ddof, same filtering of non-binary or non-finite pairs. A parity test against the previous two-pass reference confirms bit-level agreement. The new `point_biserial` benchmark puts 10 000 × (20 + 20) at **12.3 ms**, and wide-table profiling no longer spends its time on `malloc`.

## What I deliberately did not touch

Measuring surfaces more than you can fix safely in one pass:

- **`PowerTransformer`'s λ search** — the 1 µs/value outlier. It's over-tuned (40 golden-section iterations shrink a 0.4-wide bracket to ~5e-10; λ estimates don't need that), but shrinking it changes fitted lambdas at the last bits, so it deserves its own careful pass rather than a drive-by tweak.
- **`OrdinalEncoder` Auto-fit** — clones every cell into a `BTreeSet`; a `HashSet` + sort would be faster on large columns. Medium risk, real but modest payoff (10k × 20 ≈ 21 ms).
- **Duplicate-row detection** — builds a `Vec` per row into a `HashSet`; hashing rows incrementally would avoid the allocation, but the payoff is smaller than the churn the point-biserial fix removed.

Each of these is now *measurable*, which is the point. When they get fixed, the suite will show the before/after.

---

*All numbers are criterion medians (`--warm-up-time 0.5 --measurement-time 1.5 --sample-size 10`) on an Apple M5 Pro, Rust 1.96 release, deterministic xorshift64 data. The new groups and medians are documented in the [datarust README](https://github.com/genc-murat/datarust/blob/main/crates/datarust/README.md) and the [mdBook performance page](https://github.com/genc-murat/datarust/blob/main/book/src/performance.md). datarust is MIT-licensed and available on [crates.io](https://crates.io/crates/datarust).*
