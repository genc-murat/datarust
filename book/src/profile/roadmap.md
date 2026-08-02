# Roadmap

The canonical, detailed roadmap lives in
[`ROADMAP.md`][repo] inside the `datarust-profile` crate. This page gives a
higher-level tour of *why* each phase is ordered the way it is, and what it
unlocks.

[repo]: https://github.com/genc-murat/datarust/blob/main/crates/datarust-profile/ROADMAP.md

## The destination

`datarust-profile` is heading toward v1.0: a complete one-line data-profiling
and data-quality toolkit for the datarust ecosystem — zero external
dependencies by default, reusing `datarust::stats` for the numerical kernels,
owning the interpretation layer (summaries, flags, reports).

## Where we are — v0.3.0

Pairwise relationships & interaction analysis landed: Pearson correlation matrix, Cramér's V categorical association, point-biserial correlation, target-leakage detection hints (`profile_table_with_target`), correlation heatmaps in the HTML report, and two new data-quality checks (`HighCorrelation`, `TargetLeakage`). Eight quality checks now run out of the box.

## The release track

### v0.3 — Relationships & interaction (Shipped)

Columns in isolation miss collinearity, redundant features, and leakage risk.
v0.3 introduced pairwise and dataset-wide relationship analysis, reusing
`datarust::stats`'s correlation matrix, plus a pure-Rust Cramér's V for
categorical pairs and point-biserial correlation for binary categorical ⇄ numeric pairs.
A target-leakage hint flags features highly correlated with a designated target column,
and correlation heatmaps are rendered in the HTML report.

### v0.4 — Data loading & ergonomics

Meet the user at the file, not the in-memory `Matrix`. A `csv` feature adds
one-call profiling from a path, a `cli` binary generates reports from the
shell, and a streaming single-pass mode profiles large files without holding
them in memory.

### v0.5 — Missingness & comparison

Promote missing values from a count to a structured analysis (which columns
go missing *together*), and add dataset comparison: schema diff plus
per-column distributional drift (population stability index), the backbone of
production data monitoring.

### v0.6 — Text & semantics

Richer categorical profiling: string-length distributions, casing/format flags
that catch silent duplicates (`"USA"` vs `"usa "`), and identifier-vs-feature
scoring beyond the current binary `NearUnique` check.

### v0.7 — Performance (In progress)

The hot paths are already competitive with the reference tools: row
deduplication is hash-based instead of an O(n²) scan, categorical columns are
gathered as borrowed `&str` slices (no per-cell `String` clones) with all
infer/profile/relationships paths generic over `T: AsRef<str>`, missing
detection no longer allocates, and Cramér's V runs over pre-encoded integer
codes. A `criterion` benchmark suite now guards throughput
(`profile_str_matrix` 100 000 × 20: 269 ms → 132 ms). Remaining: a `rayon`
feature for parallel per-column statistics.

### v0.8 — Time series

Temporal structure is invisible to column-wise profiling. v0.8 adds datetime
type inference, monotonicity and gap detection, autocorrelation at configurable
lags, and seasonal-strength hints.

### v0.9 — Extension & quality rules

A `QualityCheck` trait lets teams codify their own data contracts (uniqueness,
allowed-value sets, ranges, regex patterns), serialised as a schema file and
checked in CI with a non-zero exit code on failure.

### v1.0 — Stability

Freeze the public API, version the JSON output schema, publish the SemVer
stability statement, and document a full parity matrix against pandas
`describe` / ydata-profiling.

---

For the granular deliverables and the explicit out-of-scope list, see the
[canonical roadmap on GitHub][repo].
