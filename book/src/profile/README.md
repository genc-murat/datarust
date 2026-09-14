# datarust-profile

**One-call data profiling and data-quality reports** for the [datarust] ecosystem.

[datarust]: https://crates.io/crates/datarust

[![crates.io](https://img.shields.io/crates/v/datarust-profile.svg)](https://crates.io/crates/datarust-profile)
[![docs.rs](https://docs.rs/datarust-profile/badge.svg)](https://docs.rs/datarust-profile)
[![CI](https://github.com/genc-murat/datarust/actions/workflows/ci.yml/badge.svg)](https://github.com/genc-murat/datarust/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/genc-murat/datarust/blob/main/crates/datarust-profile/LICENSE)

```rust
use datarust::Matrix;
use datarust_profile::{profile_matrix, report};

let m = Matrix::from_rows(vec![
    vec![1.0, 10.0],
    vec![2.0, 12.0],
    vec![3.0, f64::NAN], // missing income in row 3
])?;

let profile = profile_matrix(&m, Some(&["age".into(), "income".into()]))?;
println!("{}x{}, {} duplicate rows",
    profile.n_rows, profile.n_columns, profile.duplicate_rows);

// Self-contained HTML report — no dependencies, no JavaScript.
std::fs::write("profile.html", report::to_html(&profile))?;
```

Since v0.4, profiling can also start straight at the file — a pure-Rust
RFC-4180 CSV reader (`csv` feature), a single-pass streaming profiler for
large files (`streaming` feature), a terminal summary (`pretty` feature),
and a `datarust-profile` CLI binary (`cli` feature):

```rust
use datarust_profile::DatasetProfile;

let profile = DatasetProfile::from_csv("customers.csv")?;      // csv feature
let big = datarust_profile::profile_csv_streaming("logs.csv")?; // streaming feature
```

```sh
datarust-profile customers.csv --pretty --target churn
```

> **Default build has zero external dependencies** beyond `datarust` itself.
> Statistics flow through `datarust::stats`; the profiling crate owns the
> *interpretation* (summaries, quality flags, reports). JSON output is opt-in
> via the `serde` feature; CSV, streaming, terminal rendering, and the CLI
> are opt-in via their own features.

---

## What it computes

For each column, depending on its inferred type:

| Numeric                                       | Categorical                                    |
|-----------------------------------------------|------------------------------------------------|
| `count`, `missing_count`, `missing_fraction`  | `count`, `missing_count`, `missing_fraction`   |
| `mean`, `std` (sample, ddof = 1)              | `unique` (cardinality)                         |
| five-number summary: min / Q1 / median / Q3 / max | `top` (most frequent value)                |
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

[`quality::run_checks`](https://docs.rs/datarust-profile/latest/datarust_profile/quality/checks/fn.run_checks.html)
scans the profile against configurable [`Thresholds`](https://docs.rs/datarust-profile/latest/datarust_profile/quality/checks/struct.Thresholds.html)
and emits [`QualityIssue`](https://docs.rs/datarust-profile/latest/datarust_profile/quality/checks/struct.QualityIssue.html)s across 8 categories:

- **HighMissing** — missing fraction at/above threshold.
- **ConstantColumn** — numeric column with near-zero variance.
- **NearUnique** — categorical column whose cardinality ≈ row count (likely an identifier).
- **DuplicateRows** — exact-duplicate rows present.
- **Outliers** — values outside Tukey IQR fences.
- **Imbalance** — categorical column dominated by a single value.
- **HighCorrelation** — pair of numeric columns exceeding correlation threshold (`|r| >= 0.95`).
- **TargetLeakage** — feature column strongly correlated (`|r| >= 0.90` or `V >= 0.90`) with designated target column.

Each finding carries a [`Severity`](https://docs.rs/datarust-profile/latest/datarust_profile/types/enum.Severity.html)
(`Info` / `Warning` / `Critical`) and an optional column name. The HTML and
JSON renderers include findings and correlation heatmaps by default.

## Output formats

| Format | Feature  | Notes                                              |
|--------|----------|----------------------------------------------------|
| HTML   | —        | Single self-contained `.html`, inline CSS, no JS.  |
| Terminal | `pretty` | Fixed-width table + findings for shell inspection; auto TTY/`NO_COLOR` detection. |
| JSON   | `serde`  | Pretty-printed; schema mirrors the in-memory types. |
| CLI    | `cli`    | `datarust-profile <CSV>` binary: HTML/JSON/terminal out, `--streaming`, `--target`, `--delimiter`, `--types`. |

```rust
#[cfg(feature = "serde")]
{
    use datarust_profile::report::{to_json, JsonReport};
    let json = to_json(&JsonReport::from_profile(&profile))?;
    std::fs::write("profile.json", json)?;
}
```

---

## Relationship to `datarust`

`datarust-profile` is a sibling crate in the [datarust workspace][repo]. It
reuses `datarust`'s `Matrix` / `StrMatrix` containers and its `stats` module
(`mean`, `std`, `quantile`, `median_sorted`) rather than reimplementing them.
Column-type inference, missing-value handling, cardinality counting, and the
report renderers live in this crate.

[repo]: https://github.com/genc-murat/datarust

## Next steps

- **New to profiling?** Start with the [Quick Start](./quickstart.md).
- **Want the full API surface?** It's on [docs.rs][docsrs].

[docsrs]: https://docs.rs/datarust-profile
