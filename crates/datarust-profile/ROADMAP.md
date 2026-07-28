# Roadmap

This document tracks the path from the initial release (v0.1.0) toward a
**complete one-line data-profiling & data-quality toolkit** for the datarust
ecosystem (v1.0). It is a living document — priorities may shift, but the
principles and the destination stay fixed.

> **Where we are today (v0.1.0):** one-call profiling of numeric `Matrix`,
> categorical `StrMatrix`, and mixed tables; per-column descriptive
> statistics (count, missing, mean, std, five-number summary, cardinality,
> top value); type inference; duplicate-row detection; four data-quality
> checks (HighMissing, ConstantColumn, NearUnique, DuplicateRows) with
> configurable thresholds; a self-contained HTML report (no dependencies) and
> a JSON report behind the `serde` feature. Built entirely on `datarust::stats`
> primitives, zero external dependencies by default.

## Guiding principles

These are non-negotiable. Every item on the roadmap respects them.

1. **Zero dependencies by default.** The default build compiles with no
   external crates beyond `std` and `datarust` itself. I/O formats (CSV, etc.)
   and richer output (pandas-compatible JSON, CLI) stay behind optional feature
   flags. This keeps `datarust-profile` embeddable in WASM, embedded targets,
   CLIs, and services without a toolchain headache — and consistent with
   `datarust`'s own zero-dep identity.
2. **Reuse, don't reimplement.** Statistics flow through `datarust::stats`
   (`mean`, `std`, `quantile`, `median_sorted`, `covariance_matrix`,
   `correlation_matrix`). The profiling crate owns the *interpretation* of
   those numbers (summaries, flags, reports), not the numerical kernels.
3. **A report, not a notebook.** The output is structured data
   (`DatasetProfile`) that renders to HTML or JSON. Interactive widgets and
   notebook integration are explicitly out of scope (see below) — they are a
   different layering problem solved better by tooling that consumes this
   crate's JSON output.
4. **Type safety and no panics.** Public APIs return `Result`; invalid input
   is a recoverable `ProfileError`, never a panic. `#![warn(missing_docs)]`
   and `#![warn(clippy::all)]` are enforced in CI — same contract as `datarust`.
5. **Opt-in weight.** Heavier capabilities stay behind feature flags (`serde`,
   a future `csv`, a future `rayon`). Users who don't need them pay nothing.
6. **Measured against the reference tools.** Every statistic ships with a test
   that asserts agreement with known reference values (pandas `describe`,
   ydata-profiling where applicable), not just self-consistency.

---

## Recently shipped (v0.1.0)

The initial release establishes the data model and the rendering pipeline.
Everything below is part of the stable v0.1 API:

- ✅ `DatasetProfile` / `ColumnProfile` data model (numeric + categorical).
- ✅ `profile_matrix` / `profile_str_matrix` / `profile_table` entry points.
- ✅ Per-column stats: count, missing count & fraction, mean, std (ddof=1),
      five-number summary (min/Q1/median/Q3/max), unique count, top value,
      top frequency.
- ✅ Column-type inference (`infer_column`): all-non-empty-parse-as-f64 ⇒
      numeric; else categorical.
- ✅ Missing-value markers recognised (`""`, `NA`, `N/A`, `null`, `NaN`,
      `None`, `-`, `?`).
- ✅ Exact-duplicate-row detection (numeric, string, and mixed tables).
- ✅ Data-quality checks with configurable `Thresholds`:
  - ✅ `HighMissing` (with Critical/Warning severity split at 0.9).
  - ✅ `ConstantColumn` (near-zero variance).
  - ✅ `NearUnique` (cardinality ≈ row count ⇒ likely identifier).
  - ✅ `DuplicateRows`.
- ✅ Self-contained HTML report (inline CSS, dark-mode aware, no JS, no deps).
- ✅ JSON report (`serde` feature) bundling profile + quality findings.
- ✅ `profile_table` runnable example.

---

## Release track

Progress on each item can be tracked by the checkboxes below.

### v0.2 — Distributional depth

> *Theme: turn the five-number summary into a real distributional picture.*

The current numeric stats cover the centre and the extremes but say nothing
about the *shape* of a column. v0.2 adds the distributional statistics that
every profiling tool is expected to report, plus a first pass at outlier
detection.

**Deliverables:**

- [ ] Higher moments: `skewness` and `kurtosis` (excess) on `NumericStats`,
      NaN-safe (computed on non-missing values).
- [ ] Histogram / equi-width binning: a `Histogram { edges: Vec<f64>, counts:
      Vec<usize> }` with a configurable bin count (Sturges + Freedman–Diaconis
      rules as defaults). Exposed on `NumericStats` and rendered inline in the
      HTML report as a CSS bar chart (no JS).
- [ ] Outlier flagging: IQR rule (`< Q1 − 1.5·IQR` or `> Q3 + 1.5·IQR`) as a new
      `QualityKind::Outliers` finding, with the count and fraction of flagged
      cells in the message.
- [ ] `Imbalance` / low-cardinality flag for categorical columns whose top
      value dominates (e.g. ≥ 95% of rows) — a data-quality analogue of
      `ConstantColumn` for categoricals.
- [ ] The `_flat` fast path: route numeric column stats through
      `Matrix::as_slice()` + `column_quantiles_many_flat` /
      `column_mean_var_flat` instead of per-column `Vec` gathering, to cut
      allocation on wide tables.

### v0.3 — Relationships & interaction

> *Theme: how columns relate to each other.*

A profile that only looks at columns in isolation misses collinearity,
redundant features, and leakage risk. v0.3 introduces pairwise and
dataset-wide relationship analysis, reusing `datarust::stats`'s
`correlation_matrix` and `covariance_matrix`.

**Deliverables:**

- [ ] `Relationships` block on `DatasetProfile`: a Pearson correlation matrix
      over numeric columns (already available in `datarust::stats`), plus a
      Cramér's V matrix over categorical columns (pure-Rust χ², no deps).
- [ ] Redundancy detection: a `QualityKind::HighCorrelation` finding when any
      pair of numeric columns exceeds a configurable `|ρ|` threshold (default
      0.95), naming both columns in the message.
- [ ] Target-leakage hint: an opt-in `profile_table_with_target(..., target:
      &str)` that flags features highly correlated (or high Cramér's V) with a
      designated target column — a common cause of over-optimistic CV scores.
- [ ] Correlation heat-map tile in the HTML report (coloured `<td>` cells, no
      JS).
- [ ] Point-biserial correlation between a binary categorical column and each
      numeric column (detects strong numeric ⇄ binary relationships).

### v0.4 — Data loading & ergonomics

> *Theme: meet the user at the file, not the in-memory `Matrix`.*

Today a caller must construct a `Matrix` / `StrMatrix` by hand before
profiling. v0.4 adds optional I/O so a CSV path can be profiled in one call,
plus a CLI for ad-hoc use. Both live behind feature flags to preserve the
zero-dep default.

**Deliverables:**

- [ ] `csv` feature: a pure-Rust CSV reader (or a gated wrapper over the
      `csv` crate) returning a typed table ready for `profile_table`.
- [ ] `from_csv(path)` / `from_csv_with_types(path, schema)` entry points
      behind the `csv` feature, with automatic type inference per column.
- [ ] `cli` feature + binary: `datarust-profile path/to/data.csv -o
      report.html` (and `--json`). Arg parsing via a gated `clap` dependency
      or a hand-rolled parser to stay dep-light.
- [ ] Streaming-friendly profiling for large files: compute column stats in a
      single forward pass (Welford mean/variance, reservoir-sampled quantiles
      or t-digest) so the whole file never needs to reside in memory.
- [ ] `pretty` feature: a terminal-table summary printer (think `bat` /
      `term-table`) for quick inspection without leaving the shell.

### v0.5 — Missingness & comparison

> *Theme: missing data is first-class, and datasets are compared, not just
> profiled.*

Missing values are profiled only as a count today. v0.4 promotes them to a
structured analysis, and adds the dataset-comparison capability that
ydata-profiling users reach for when tracking data drift between releases.

**Deliverables:**

- [ ] `Missingness` analysis: per-column missing fraction (present), plus a
      missing-pattern summary (which columns tend to be missing *together*) —
      computed from the co-occurrence matrix of missingness indicators.
- [ ] `QualityKind::MissingPattern` finding when a stable block of columns is
      missing together in a way that suggests an upstream join/filter bug.
- [ ] `compare_profiles(a: &DatasetProfile, b: &DatasetProfile) ->
      DatasetDiff`: schema diff (added/removed/renamed columns), per-column
      distributional drift (e.g. population stability index, PSI, for
      numerics; frequency-table distance for categoricals).
- [ ] `compare_reports` HTML/JSON renderer highlighting drifted columns.
- [ ] `QualityKind::Drift` for columns whose PSI exceeds a threshold — the
      backbone of a production data-monitoring use case.

### v0.6 — Text & semantics

> *Theme: make categorical profiling as rich as numeric profiling.*

Categorical columns currently report only cardinality and the top value. v0.6
adds the text-aware summaries that distinguish a profiling tool from a
`describe()` call: length distributions, casing, and basic lexical structure.

**Deliverables:**

- [ ] String-length statistics on `CategoricalStats` (min/mean/max char
      length, distribution histogram).
- [ ] Casing / format flags: detected all-uppercase, all-lowercase,
      title-case, mixed; leading/trailing whitespace; embedded digits.
- [ ] `QualityKind::InconsistentCasing` and `QualityKind::WhitespaceNoise`
      findings (the silent duplicates that break group-by: `"USA"` vs `"usa "`
      vs `" U.S.A"`).
- [ ] Cardinality-vs-row ratio as a *continuous* feature-quality signal
      (already flagged as `NearUnique` at the extreme; here generalised into a
      "looks like an identifier" confidence score).
- [ ] Optional Unicode block / script summary behind a `unicode` feature
      (detects mixed Latin/Cyrillic/etc. in the same column).

### v0.7 — Performance

> *Theme: profile million-row tables in seconds, not minutes.*

The duplicate-row and per-column gather loops are O(n²) or allocation-heavy
today. v0.7 makes the hot paths competitive with vectorised tooling, without
adding a hard dependency.

**Deliverables:**

- [ ] `rayon` feature: parallelise per-column statistics (mirror
      `datarust`'s existing `rayon` feature wiring in `stats.rs`).
- [ ] Replace the O(n²) duplicate-row scan with a hash-based dedup
      (canonicalise each row to a `u64`/`u128` hash, sort, count runs).
- [ ] Cache-friendly column-major gather: transpose once, then stride-1 over
      each column for the quantile/sort passes.
- [ ] Benchmark suite (`benches/`) with `criterion`, tracking throughput on
      synthetic 10⁵ / 10⁶ / 10⁷ row tables — added to CI as a no-regression
      guard.
- [ ] Memory budget option: cap peak allocation for streaming profiles (ties
      into the v0.4 t-digest work).

### v0.8 — Time series

> *Theme: temporal structure is invisible to column-wise profiling.*

Many datasets have an implicit or explicit time axis. v0.8 adds the temporal
statistics that matter for forecasting and monitoring workloads. This is the
point where profiling stops being purely cross-sectional.

**Deliverables:**

- [ ] Datetime type inference: recognise ISO-8601 / common date-time formats
      in string columns, promote to a `ColumnType::DateTime`.
- [ ] `TimeSeriesStats`: monotonicity check (is the column sorted?),
      frequency/periodicity estimate, gap detection (missing time steps).
- [ ] Autocorrelation at a configurable set of lags (pure-Rust, reusing
      `correlation_matrix`-style dot products on lagged slices).
- [ ] `QualityKind::NonMonotonicTime` and `QualityKind::TimeGaps` findings.
- [ ] Seasonal-strength hint via a simple seasonal-trend decomposition
      (classical STL-lite, additive) — behind a feature flag given its cost.

### v0.9 — Extension & quality rules

> *Theme: let users encode their own data contracts.*

The built-in checks are opinionated defaults. v0.9 makes the quality layer
extensible so a team can codify its own domain rules (schema constraints,
allowed-value sets, regex patterns) and get them reported alongside the
built-ins.

**Deliverables:**

- [ ] `QualityCheck` trait: `fn check(&self, profile: &DatasetProfile) ->
      Vec<QualityIssue>`, so users (or downstream crates) can plug in custom
      rules consumed by `run_checks`.
- [ ] Built-in rule helpers: `expect_unique(columns)`, `expect_in_set(column,
      allowed)`, `expect_range(column, min, max)`, `expect_not_null(columns)`.
- [ ] A `Schema` / expectations object that bundles a set of rules and can be
      serialised (serde feature) — the foundation for "data contract" files
      checked into a repo or CI.
- [ ] `validate(profile, schema) -> ValidationReport` separating *profile*
      (descriptive) from *validation* (pass/fail against a contract).
- [ ] Exit-code-aware CLI mode: `datarust-profile check data.csv contract.json`
      returns non-zero on validation failure — a CI gate.

### v1.0 — Stability

> *Theme: lock in the API and ship a stability guarantee.*

v1.0 is not primarily about new features — it is the release where the public
API is frozen and the maintenance promise begins.

**Deliverables:**

- [ ] Public-API audit: consistent naming, complete doc comments, no
      undocumented public items.
- [ ] Freeze the `DatasetProfile` / `ColumnProfile` / report schema and version
      the JSON output (`"schema_version"`) so downstream consumers can pin.
- [ ] MSRV review — confirm alignment with `datarust`'s MSRV (currently 1.70)
      and decide whether to hold or bump.
- [ ] Publish a v1.0 stability statement: what counts as a breaking change
      going forward, and the SemVer commitment.
- [ ] Full pandas-`describe` / ydata-profiling parity matrix in the docs, with
      the gaps explicitly called out.
- [ ] Ensure every public entry point has a runnable example and a parity test.

---

## Explicitly out of scope

These will **not** be pursued, to keep the crate focused. Each is served
better by another layer of the ecosystem.

| Area | Why not |
|---|---|
| **Interactive notebook widgets** | A different deployment target (browser kernel, JS runtime). The JSON report is the bridge: a notebook extension or `egui`/`tauri` app can consume it. |
| **GPU-accelerated profiling** | Conflicts with the zero-dependency, CPU-first identity inherited from `datarust`. Profiling is I/O- and allocation-bound, not compute-bound, at the scales this crate targets. |
| **Geo-spatial / image profiling** | Niche and dependency-heavy. The image/file metadata analysis in ydata-profiling belongs in a dedicated crate; `datarust-profile` stays tabular. |
| **Model explainability (SHAP/LIME)** | Different concern (model introspection, not data profiling). Belongs with `datarust`'s planned `inspection` module, not here. |
| **Visual charting library** | Embedding SVG/canvas rendering bloats the binary and the API surface. The HTML report uses pure CSS bars/heat-maps; richer plotting is a rendering concern left to downstream tools consuming the JSON. |

## Under consideration (post-1.0)

Not committed, but worth evaluating once the core is stable:

- **DuckDB / Arrow input** — a feature-gated reader for columnar formats, the
  natural partner to streaming profiling on large analytical datasets.
- **Differential privacy mode** — clamp/histogram-with-noise so a profile can
  be shared without leaking individual rows. Niche but valuable for regulated
  domains.
- **`f32` support** — mirror `datarust`'s post-1.0 `f32` investigation; halves
  memory for wide tables if it lands there.
- **Incremental / rolling profiles** — update a `DatasetProfile` with new
  batches without re-scanning history, the foundation for online data
  monitoring.
- **Anomaly detection beyond IQR** — isolation-forest or robust-zscore
  per-column outliers, if a dep-free implementation is tractable.

---

## How to contribute

The checkboxes above double as a contribution guide. If you want to help:

- Pick an unchecked item that has no architectural prerequisite, or team up
  on one that does.
- Every statistic / check should land with: doc comments, a parity test
  (pandas `describe` or ydata-profiling reference value where one exists),
  and coverage in the HTML/JSON renderers.
- Open an issue before starting non-trivial work, to avoid duplicated effort
  and to align on API shape.

Progress against this roadmap is tracked in `CHANGELOG.md` under each
release; this file is updated as priorities evolve.
