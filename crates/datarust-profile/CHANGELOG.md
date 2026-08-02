# Changelog

All notable changes to `datarust-profile` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Hash-based row deduplication.** `count_duplicate_numeric`,
  `count_duplicate_str`, and `count_duplicate_table` use a `HashSet` instead of
  the previous O(n²) `Vec` scan. `-0.0`/`0.0` are canonicalised to the same
  bits (matching `f64` `==`) and NaN-containing rows are skipped (they never
  equal anything under `PartialEq`).
- **Borrow-based categorical access.** Categorical columns are gathered with
  `StrMatrix::column_refs` (`Vec<&str>`) and the infer/profile/relationships
  paths are generic over `T: AsRef<str>`, eliminating the per-cell `String`
  clone on the profile hot path.
- **Allocation-free missing detection.** `is_missing` replaces the
  `to_ascii_lowercase()` copy with an `eq_ignore_ascii_case` chain.
- **Cramér's V over encoded integer codes.** Each categorical column is
  pre-encoded into `usize` level codes once, so every pairwise contingency table
  is keyed by integers instead of hashing string tuples for every row of every
  column pair; small tables use a flat array and chi-squared is computed with
  the `ΣO²/E − N` identity.
- **Flat Pearson correlation.** `compute_pearson` builds a single flat
  row-major buffer and calls `datarust::stats::correlation_matrix_flat`, which
  streams the mean/center/covariance passes over contiguous memory instead of a
  nested `Vec<Vec<f64>>` — and automatically picks up the GEMM-backed
  covariance when `datarust` is built with the `matrixmultiply` feature. The
  scalar covariance path inside `datarust` now accumulates only the lower
  triangle (flop-halving, bit-identical), further accelerating wide-table
  Pearson.
- **Allocation-free point-biserial.** `calculate_point_biserial` replaces the
  per-(categorical, numeric)-pair `group_0` / `group_1` / `all_y` `Vec`s with
  two fused scalar passes (counts and sums, then the pooled variance),
  eliminating the allocation churn that dominated wide-table profiles with many
  binary columns. Results are unchanged — same accumulation order, ddof, and
  filtering of non-binary / non-finite pairs.

### Added

- **New benchmarks.** `relationships` (wide numeric Pearson plus a mixed
  Pearson/Cramér's V table), `profile_table_wide` (up to 60-column mixed
  tables), and `report_html` (HTML report rendering).
- **More benchmarks.** `cramers_v_high_cardinality` (Cramér's V over 100/1 000
  level columns, hitting the hash-map contingency path),
  `point_biserial` (binary × numeric pairwise), `profile_str_high_cardinality`
  (near-unique identifier columns), and serde-gated `report_json`
  (`JsonReport` → `to_json`).

### Docs

- **Performance documentation.** The datarust README "Scalar vs
  `matrixmultiply` kernels" table and the mdBook `performance` page now
  document the profile-crate wide-table numbers: `relationships` Pearson
  10 000 × 100 is ~6.7 ms scalar vs ~5.0 ms +GEMM, and wide `profile_matrix`
  10 000 × 100 is ~31.6 ms scalar vs ~30.5 ms +GEMM — Pearson is no longer the
  dominant cost of wide profiling, so the `matrixmultiply` feature adds little
  there.
- **Microbenchmark medians documented.** The mdBook `performance` page and the
  datarust README now list criterion medians for the new groups:
  `cramers_v_high_cardinality` 10 000 × 10 × 1 000 levels ≈ 6.2 ms,
  `point_biserial` 10 000 × (20 num + 20 bin) ≈ 12.3 ms,
  `profile_str_high_cardinality` 10 000 × 20 ≈ 85 ms, and `report_json`
  ≈ 39 µs regardless of row count (JSON output scales with the column count,
  since the profile is column-summarized).

### Performance

Measured on Apple M5 Pro (18 cores, arm64), Rust 1.96 release, criterion with
`--warm-up-time 0.5 --measurement-time 1.5 --sample-size 10`, median profile
time on deterministic xorshift64 data:

| Workload | Before | After | Improvement |
|---|---:|---:|---:|
| `profile_str_matrix` 100 000 × 20 | 269 ms | 132 ms | **2.0×** |
| `profile_str_matrix` 10 000 × 20 | 22.3 ms | 10.1 ms | **2.2×** |
| `relationships` Pearson 10 000 × 100 | ~45 ms | ~7 ms | **6.4×** |
| `relationships` Pearson 10 000 × 50 | ~8.9 ms | ~2.3 ms | **3.8×** |
| `profile_matrix` 10 000 × 100 (wide) | ~63 ms | ~31 ms | **2.0×** |

All figures are criterion `--quick`, single-threaded default build on the same
Apple Silicon machine. The `relationships` Pearson kernel dropped to ~7 ms
after the scalar covariance switched to lower-triangle accumulation; wide-table
`profile_matrix` halves alongside it.

## [0.3.0] - 2026-07-31

This release adds pairwise relationship analysis across numeric and categorical columns, target-leakage detection hints, correlation heatmaps in the HTML report, and two new data-quality checks.

### Added

- **`Relationships` block** on `DatasetProfile`:
  - **Pearson correlation matrix** over numeric columns using `datarust::stats::correlation_matrix`.
  - **Cramér's V association matrix** over categorical columns via pure-Rust contingency tables (zero dependencies).
  - **Point-biserial correlation** between binary categorical columns and numeric columns.
- **Target-leakage hints**:
  - `profile_matrix_with_target` and `profile_table_with_target` entry points.
  - `DatasetProfile::with_target` builder method.
- **Data-quality checks**:
  - `QualityKind::HighCorrelation` — flags pairs of numeric columns with `|r| >= 0.95`.
  - `QualityKind::TargetLeakage` — flags feature columns strongly correlated (`|r| >= 0.90` or `V >= 0.90`) with a designated target column.
- **Correlation heatmaps in HTML report**:
  - Diverging Red–White–Blue palette for Pearson correlation.
  - Sequential White–Purple palette for Cramér's V matrix.
  - Point-biserial summary table.
  - Full dark-mode CSS support.
- New runnable `correlation_analysis` example demonstrating relationship profiling and leakage hints.

## [0.2.0] - 2026-07-31

Distributional depth release: shape statistics, IQR outlier detection, categorical imbalance, and a `_flat` fast path.

### Added

- `skewness` and excess `kurtosis` (Fisher–Pearson) on `NumericStats`.
- `Histogram` (equal-width, Sturges bins) on `NumericStats`.
- IQR outlier detection (`outlier_count`, `outlier_fraction`) and `QualityKind::Outliers`.
- Categorical `imbalance_ratio` (`freq / present`) and `QualityKind::Imbalance`.
- `_flat` fast path in `from_matrix` via `column_mean_var_flat` and `column_quantiles_many_flat`.
- Responsive CSS card-grid layout in HTML report with inline mini-bar charts.

## [0.1.0] - 2026-07-31

Initial release of `datarust-profile`.

### Added

- Core data model: `DatasetProfile`, `ColumnProfile`, `NumericStats`, `CategoricalStats`.
- Entry points: `profile_matrix`, `profile_str_matrix`, `profile_table`.
- Basic statistics: count, missing_count, missing_fraction, mean, std (ddof=1), five-number summary, unique, top, freq.
- Automatic column-type inference (`infer_column`) and missing-value markers.
- Duplicate-row detection.
- Quality checks: `HighMissing`, `ConstantColumn`, `NearUnique`, `DuplicateRows`.
- Self-contained HTML report renderer (`report::to_html`).
- Serde-gated JSON report renderer (`report::to_json`).
