# datarust-profile

**One-call data profiling and data-quality reports** for the [datarust] ecosystem.

`datarust-profile` takes a numeric [`Matrix`], a categorical [`StrMatrix`], a
mixed table of both — or, behind the `csv` feature, a CSV file path — and
produces a [`DatasetProfile`] describing shape, memory footprint, duplicate
rows, per-column descriptive statistics (mean, std, five-number summary,
cardinality, top value) and a list of data-quality findings. Profiles can be
rendered to JSON (via the `serde` feature), to a self-contained HTML report,
or to a terminal table (via the `pretty` feature) with no extra dependencies.

It is the Rust-native, dependency-free counterpart of tools like
[ydata-profiling] / [fg-data-profiling]: instead of a Python package wrapping
pandas, it is a small crate built on [datarust]'s statistics primitives and
designed to be embedded in CLIs, services, notebooks, and CI pipelines.

[datarust]: https://crates.io/crates/datarust
[ydata-profiling]: https://github.com/ydataai/ydata-profiling
[fg-data-profiling]: https://github.com/Data-Centric-AI-Community/fg-data-profiling

## Install

```toml
[dependencies]
datarust-profile = "0.4"
```

For JSON output, enable the `serde` feature (this also pulls in
`datarust/serde`):

```toml
[dependencies]
datarust-profile = { version = "0.4", features = ["serde"] }
```

The default build has **zero external dependencies** beyond `datarust` itself.

## Features

| Feature    | What it enables                                                              |
|------------|------------------------------------------------------------------------------|
| *(default)*| In-memory profiling of `Matrix` / `StrMatrix` tables, HTML reports.           |
| `csv`      | Pure-Rust RFC-4180 reader + one-call CSV profiling (`profile_csv`, `from_csv`).|
| `streaming`| Single-pass bounded-memory CSV profiler (`profile_csv_streaming`).            |
| `pretty`   | Terminal-table renderer (`report::to_terminal`).                              |
| `cli`      | The `datarust-profile` binary (implies `csv`, `streaming`, `pretty`, `serde`).|
| `serde`    | JSON serialization of profiles and quality findings.                          |

## Quick start

```rust
use datarust::Matrix;
use datarust_profile::{profile_matrix, report};

let m = Matrix::from_rows(vec![
    vec![1.0, 10.0],
    vec![2.0, 12.0],
    vec![3.0, f64::NAN],
]).unwrap();

let profile = profile_matrix(&m, Some(&["x".into(), "y".into()])).unwrap();

println!("{}x{}, {} duplicates",
    profile.n_rows, profile.n_columns, profile.duplicate_rows);

for col in &profile.columns {
    println!("{:<6} {:<11} missing={:.0}%",
        col.name, col.column_type, col.missing_fraction * 100.0);
}

// Self-contained HTML report with CSS histograms and correlation heatmaps (no extra deps):
std::fs::write("profile.html", report::to_html(&profile)).unwrap();
```

## Profiling CSV files (`csv` / `streaming` features)

```rust,ignore
use datarust_profile::{profile_csv, profile_csv_streaming, DatasetProfile};

// One call from a path (pure-Rust RFC-4180 reader; quoted fields, CRLF, BOM).
let profile = profile_csv(std::path::Path::new("data.csv"), None)?;

// Forced per-column types: parse failures become missing values.
use datarust_profile::ColumnType;
let forced = DatasetProfile::from_csv_with_types(
    std::path::Path::new("data.csv"),
    &[ColumnType::Numeric, ColumnType::Categorical],
)?;

// Million-row files in one bounded-memory pass (Welford + reservoir
// sampling; exact mean/variance, approximate quantiles above 10 000 rows).
let streamed = profile_csv_streaming(std::path::Path::new("big.csv"))?;

// Terminal summary (pretty feature), HTML, JSON:
println!("{}", datarust_profile::report::to_terminal_auto(&streamed));
```

The CSV reader feeds the exact same `StrMatrix` path as `profile_str_matrix`,
so `profile_csv` and in-memory profiling produce identical profiles on the
same rows (asserted by a bit-identity test).

## The CLI (`cli` feature)

```text
datarust-profile data.csv -o report.html --json report.json --pretty
datarust-profile data.csv --streaming --target churn
datarust-profile data.csv --delimiter ';' --no-header
datarust-profile data.csv --types numeric,categorical,numeric
```

Exit codes: `0` success, `1` runtime error (I/O or parsing), `2` usage error
(from clap). The CLI is ad-hoc inspection; CI-grade contract checks
(`validate` with non-zero exit) are planned for v0.9.

## What it computes

For each column, depending on its inferred type:

| Numeric                                       | Categorical                                    |
|-----------------------------------------------|------------------------------------------------|
| `count`, `missing_count`, `missing_fraction`  | `count`, `missing_count`, `missing_fraction`   |
| `mean`, `std` (sample, ddof = 1)              | `unique` (cardinality)                         |
| five-number summary: min/Q1/median/Q3/max     | `top` (most frequent value)                    |
| `skewness`, `kurtosis` (excess, Fisher)       | `freq` (count of `top`)                        |
| `histogram` (equal-width, Sturges bins)       | `imbalance_ratio` (`freq / present`)           |
| `outlier_count`, `outlier_fraction` (IQR rule)| `top_values` (top-N value/count pairs)         |

**Pairwise Relationships (`Relationships` block):**
- **Pearson correlation matrix** over numeric columns (`CorrelationMatrix`).
- **Cramér's V matrix** over categorical columns (pure-Rust χ² association).
- **Point-biserial correlation** between binary categorical and numeric columns (`PointBiserialEntry`).

Dataset-wide: `n_rows`, `n_columns`, estimated `memory_bytes`, exact
`duplicate_rows` and `duplicate_fraction`, optional `target_column`.

### Data-quality findings

`quality::run_checks` scans the profile against configurable
[`Thresholds`] and emits [`QualityIssue`]s across 8 categories:

- **HighMissing** — missing fraction at/above threshold.
- **ConstantColumn** — numeric column with near-zero variance.
- **NearUnique** — categorical column whose cardinality ≈ row count (likely an identifier).
- **DuplicateRows** — exact-duplicate rows present.
- **Outliers** — values outside Tukey IQR fences.
- **Imbalance** — categorical column dominated by a single value.
- **HighCorrelation** — pair of numeric columns exceeding correlation threshold (`|r| >= 0.95`).
- **TargetLeakage** — feature column strongly correlated (`|r| >= 0.90` or `V >= 0.90`) with designated target column.

Each finding carries a [`Severity`] (`Info` / `Warning` / `Critical`) and an
optional column name. The HTML and JSON renderers include findings by default.

## Output formats

| Format | Feature  | Notes                                              |
|--------|----------|----------------------------------------------------|
| HTML   | —        | Single self-contained `.html`, inline CSS, no JS.  |
| JSON   | `serde`  | Pretty-printed; schema mirrors the in-memory types. |
| Terminal | `pretty` | Fixed-width table, auto ANSI color on TTY, honors `NO_COLOR`. |
| CLI    | `cli`    | `datarust-profile` binary: CSV → HTML/JSON/terminal. |

```rust
#[cfg(feature = "serde")]
{
    use datarust_profile::report::{to_json, JsonReport};
    let json = to_json(&JsonReport::from_profile(&profile)).unwrap();
    std::fs::write("profile.json", json).unwrap();
}
```

## Relationship to `datarust`

`datarust-profile` is a sibling crate in the [datarust workspace][repo]. It
reuses `datarust`'s `Matrix` / `StrMatrix` containers and its `stats` module
(`mean`, `std`, `quantile`, `median_sorted`) rather than reimplementing them, so
profiling stays consistent with the rest of the ecosystem. Column-type
inference, missing-value handling, cardinality counting, and the report
renderers live in this crate.

[repo]: https://github.com/genc-murat/datarust

## License

MIT.
