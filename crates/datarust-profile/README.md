# datarust-profile

**One-call data profiling and data-quality reports** for the [datarust] ecosystem.

`datarust-profile` takes a numeric [`Matrix`], a categorical [`StrMatrix`], or a
mixed table of both, and produces a [`DatasetProfile`] describing shape, memory
footprint, duplicate rows, per-column descriptive statistics (mean, std,
five-number summary, cardinality, top value) and a list of data-quality
findings. Profiles can be rendered to JSON (via the `serde` feature) or to a
self-contained HTML report with no extra dependencies.

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
datarust-profile = "0.1"
```

For JSON output, enable the `serde` feature (this also pulls in
`datarust/serde`):

```toml
[dependencies]
datarust-profile = { version = "0.1", features = ["serde"] }
```

The default build has **zero external dependencies** beyond `datarust` itself.

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

// Self-contained HTML report (no extra deps):
std::fs::write("profile.html", report::to_html(&profile)).unwrap();
```

## What it computes

For each column, depending on its inferred type:

| Numeric                           | Categorical                          |
|-----------------------------------|--------------------------------------|
| `count`, `missing_count`, `missing_fraction` | `count`, `missing_count`, `missing_fraction` |
| `mean`, `std` (sample, ddof = 1)  | `unique` (cardinality)               |
| five-number summary: min/Q1/median/Q3/max | `top` (most frequent value)   |
|                                   | `freq` (count of `top`)              |

Dataset-wide: `n_rows`, `n_columns`, estimated `memory_bytes`, exact
`duplicate_rows` and `duplicate_fraction`.

### Data-quality findings

`quality::run_checks` scans the profile against configurable
[`Thresholds`] and emits [`QualityIssue`]s:

- **HighMissing** — missing fraction at/above threshold.
- **ConstantColumn** — numeric column with near-zero variance.
- **NearUnique** — categorical column whose cardinality ≈ row count (likely an
  identifier, not a feature).
- **DuplicateRows** — exact-duplicate rows present.

Each finding carries a [`Severity`] (`Info` / `Warning` / `Critical`) and an
optional column name. The HTML and JSON renderers include findings by default.

## Output formats

| Format | Feature  | Notes                                              |
|--------|----------|----------------------------------------------------|
| HTML   | —        | Single self-contained `.html`, inline CSS, no JS.  |
| JSON   | `serde`  | Pretty-printed; schema mirrors the in-memory types. |

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
