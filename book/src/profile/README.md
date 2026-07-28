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

> **Default build has zero external dependencies** beyond `datarust` itself.
> Statistics flow through `datarust::stats`; the profiling crate owns the
> *interpretation* (summaries, quality flags, reports). JSON output is opt-in
> via the `serde` feature.

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

Dataset-wide: `n_rows`, `n_columns`, estimated `memory_bytes`, exact
`duplicate_rows` and `duplicate_fraction`.

### Data-quality findings

[`quality::run_checks`](https://docs.rs/datarust-profile/latest/datarust_profile/quality/checks/fn.run_checks.html)
scans the profile against configurable [`Thresholds`](https://docs.rs/datarust-profile/latest/datarust_profile/quality/checks/struct.Thresholds.html)
and emits [`QualityIssue`](https://docs.rs/datarust-profile/latest/datarust_profile/quality/checks/struct.QualityIssue.html)s:

- **HighMissing** — missing fraction at/above threshold.
- **ConstantColumn** — numeric column with near-zero variance.
- **NearUnique** — categorical column whose cardinality ≈ row count (likely an identifier).
- **DuplicateRows** — exact-duplicate rows present.
- **Outliers** — values outside the Tukey IQR fences.
- **Imbalance** — categorical column dominated by a single value.

Each finding carries a [`Severity`](https://docs.rs/datarust-profile/latest/datarust_profile/types/enum.Severity.html)
(`Info` / `Warning` / `Critical`) and an optional column name. The HTML and
JSON renderers include findings by default.

## Output formats

| Format | Feature  | Notes                                              |
|--------|----------|----------------------------------------------------|
| HTML   | —        | Single self-contained `.html`, inline CSS, no JS.  |
| JSON   | `serde`  | Pretty-printed; schema mirrors the in-memory types. |

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
