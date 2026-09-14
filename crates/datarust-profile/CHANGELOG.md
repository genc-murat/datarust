# Changelog

All notable changes to `datarust-profile` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-09-13

The data-loading & ergonomics release: meet the user at the file, not the
in-memory `Matrix`. Adds a pure-Rust CSV reader, one-call CSV profiling entry
points, a streaming single-pass profiler for large files, a terminal-table
renderer, and the `datarust-profile` CLI binary — all behind optional feature
flags, so the default build keeps **zero external dependencies**.

### Added

- **`csv` feature — pure-Rust RFC-4180 reader** (`src/csv.rs`, zero
  dependencies): quoted fields with `""` escapes, embedded
  delimiters/newlines in quoted fields, LF and CRLF line endings, UTF-8 BOM
  stripping, configurable delimiter/quote bytes, optional field trimming,
  header detection, and strict ragged-row rejection with file line numbers.
  `CsvTable` pairs the optional header with a `StrMatrix` of data rows, and
  the reader feeds the exact same path as `profile_str_matrix` — asserted by
  a bit-identity parity test.
- **One-call CSV profiling entry points** (csv feature):
  `DatasetProfile::from_csv(path)`, `from_csv_with(path, opts)`,
  `from_csv_with_types(path, schema)`, plus the `profile_csv`,
  `profile_csv_with`, and `profile_csv_with_types` free functions.
  Schema-forced types split columns into a numeric block and a categorical
  block (grouped via `from_table`); parse failures under a forced `Numeric`
  type become missing values, matching the crate's marker semantics. Column
  profiles are reordered back into original CSV order.
- **`streaming` feature — single-pass CSV profiler** (`src/streaming.rs`)
  with bounded memory: exact Welford mean/variance and incremental central
  moments (skewness/kurtosis), online min/max, reservoir-sampled quantiles
  and histogram (`reservoir_capacity`, default 10 000 — exact when the table
  fits), online Pearson co-moments, hashed Cramér's V contingency tables over
  interned level codes, point-biserial group sums, deterministic
  (`xorshift64*`) sampling, optional exact duplicate tracking, and a
  `distinct_cap` (default 100 000) beyond which categorical `unique` becomes
  a documented lower bound. Column types are decided online: a numeric column
  permanently converts to categorical at its first non-parseable cell
  (documented one-pass trade-off). Small tables match the in-memory profile
  bit-for-bit on exact statistics.
- **`pretty` feature — terminal-table renderer**
  (`report::to_terminal`/`to_terminal_auto`/`to_terminal_with`): fixed-width
  column table (type, missing %, μ/σ or unique/top), findings list with
  CRIT/WARN/INFO tags, hand-rolled ANSI styling, auto color on TTY and
  `NO_COLOR` support, zero dependencies.
- **`cli` feature — `datarust-profile` binary** (`src/bin/`):
  `datarust-profile <CSV> [-o report.html] [--json report.json] [--pretty]
  [--target COLUMN] [--streaming] [--delimiter CHAR] [--no-header]
  [--types ...]`. Implies `csv`, `streaming`, `pretty`, `serde`; clap pinned
  to `=4.4.18` (default features off except `derive`, keeping the workspace
  on the Rust 1.70 MSRV). Exit codes: 0 success, 1 runtime error, 2 usage.
  13 integration tests exercise the real binary via `CARGO_BIN_EXE`.
- **`profile_csv` runnable example** demonstrating the one-call workflow.

### Changed

- **Pearson correlation over flat buffers.** `compute_pearson` now
  interleaves columns into one flat row-major buffer and calls
  `datarust::stats::correlation_matrix_flat` (single allocation, contiguous
  streaming, GEMM-backed covariance under the `matrixmultiply` feature),
  replacing the nested `Vec<Vec<f64>>` transposition — the switch deferred
  since 0.3.1 until datarust 0.6.6 shipped.

## [0.3.1] - 2026-08-18

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
- **Pearson correlation.** `compute_pearson` calls `datarust::stats::correlation_matrix`.
  After datarust 0.6.6 is published, this will switch to `correlation_matrix_flat`
  (flat row-major buffer) for contiguous-memory streaming and GEMM-backed covariance.
- **Single-pass point-biserial.** `calculate_point_biserial` replaces the
  per-(categorical, numeric)-pair `group_0` / `group_1` / `all_y` `Vec`s with
  a single allocation-free pass using the computational identity
  `SS = Σy² − (Σy)² / N` for pooled variance, eliminating the allocation
  churn that dominated wide-table profiles with many binary columns. A parity
  test confirms bit-level agreement with the previous two-pass reference
  implementation.

### Added

- **New benchmarks.** `relationships` (wide numeric Pearson plus a mixed
  Pearson/Cramér's V table), `profile_table_wide` (up to 60-column mixed
  tables), and `report_html` (HTML report rendering).
- **More benchmarks.** `cramers_v_high_cardinality` (Cramér's V over 100/1 000
  level columns, hitting the hash-map contingency path),
  `point_biserial` (binary × numeric pairwise), `profile_str_high_cardinality`
  (near-unique identifier columns), and serde-gated `report_json`
  (`JsonReport` → `to_json`).
- **Point-biserial parity tests.** Two new tests verify the single-pass
  `calculate_point_biserial` against a two-pass reference implementation:
  50 deterministic pseudo-random cases with edge-case coverage (NaN, inf,
  non-binary indicators) and 6 hand-computed known-value cases.

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
