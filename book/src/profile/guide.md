# Profiling Guide

How to profile numeric, categorical, and mixed data, and how to act on the
data-quality findings. Types and functions live in
[`datarust_profile`](https://docs.rs/datarust-profile/latest/datarust_profile/index.html);
the statistics they report flow from [`datarust::stats`](https://docs.rs/datarust/latest/datarust/stats/index.html).

All public APIs return [`Result`](https://docs.rs/datarust-profile/latest/datarust_profile/type.Result.html);
invalid input is a recoverable [`ProfileError`](https://docs.rs/datarust-profile/latest/datarust_profile/enum.ProfileError.html),
never a panic.

## Profiling numeric data

Pass a [`Matrix`](https://docs.rs/datarust/latest/datarust/struct.Matrix.html)
to [`profile_matrix`](https://docs.rs/datarust-profile/latest/datarust_profile/fn.profile_matrix.html).
Non-finite values (`NaN`, `±inf`) are treated as missing and counted
separately — they never poison the mean or standard deviation.

```rust
use datarust::Matrix;
use datarust_profile::profile_matrix;

let m = Matrix::from_rows(vec![
    vec![1.0, 100.0],
    vec![2.0, 110.0],
    vec![3.0, 105.0],
    vec![4.0, 95.0],
    vec![5.0, f64::NAN], // missing reading in column 1
])?;

let p = profile_matrix(&m, Some(&["index".into(), "reading".into()]))?;
```

Each numeric column yields a [`NumericStats`](https://docs.rs/datarust-profile/latest/datarust_profile/struct.NumericStats.html):

| Field | Meaning |
|---|---|
| `mean`, `std` | Central tendency and spread (sample std, ddof = 1). |
| `five` | [`FiveNumber`](https://docs.rs/datarust-profile/latest/datarust_profile/struct.FiveNumber.html): min / Q1 / median / Q3 / max. |
| `skewness` | Fisher–Pearson third moment. `0` ≈ symmetric; positive ⇒ right tail. |
| `kurtosis` | Excess (Fisher) fourth moment. `0` ≈ normal; positive ⇒ heavy tails. |
| `histogram` | [`Histogram`](https://docs.rs/datarust-profile/latest/datarust_profile/struct.Histogram.html) with Sturges-bin edges and counts. |
| `outlier_count`, `outlier_fraction` | Values beyond the Tukey IQR fences (`Q1 − 1.5·IQR`, `Q3 + 1.5·IQR`). |

The histogram bin count follows Sturges' rule (`ceil(log2(n) + 1)`, floored at
1). Read `histogram.counts` directly, or use `histogram.nbins()` /
`histogram.max_count()` for rendering.

### Fast path

`profile_matrix` routes the mean, variance, and five-number summary through
`datarust`'s fused flat-buffer helpers (`column_mean_var_flat`,
`column_quantiles_many_flat`) over `Matrix::as_slice()`, avoiding per-column
`Vec` allocation on wide tables. Columns containing `NaN` fall back to the
per-column path, since the flat helpers are not NaN-aware.

## Profiling categorical data

[`profile_str_matrix`](https://docs.rs/datarust-profile/latest/datarust_profile/fn.profile_str_matrix.html)
infers each column's type. A column is `Numeric` when every non-empty cell
parses as `f64`; otherwise it is `Categorical`. Empty markers recognised as
missing: `""`, `NA`, `N/A`, `null`, `NaN`, `None`, `-`, `?`.

```rust
use datarust::StrMatrix;
use datarust_profile::profile_str_matrix;

let s = StrMatrix::from_strings(vec![
    vec!["Istanbul", "basic"],
    vec!["Ankara", "premium"],
    vec!["Izmir", "basic"],
    vec!["Istanbul", "basic"],
])?;

let p = profile_str_matrix(&s, Some(&["city".into(), "tier".into()]))?;
```

A [`CategoricalStats`](https://docs.rs/datarust-profile/latest/datarust_profile/struct.CategoricalStats.html)
reports `unique` (cardinality), `top` / `freq` (most frequent value and its
count), `imbalance_ratio` (`freq` ÷ non-missing cells), and `top_values` — the
top-N `(value, count)` pairs, sorted descending, for frequency charts.

## Mixed tables

Real datasets mix numeric and categorical columns.
[`profile_table`](https://docs.rs/datarust-profile/latest/datarust_profile/fn.profile_table.html)
takes an optional numeric [`Matrix`] and an optional categorical
[`StrMatrix`] side by side, with one shared row count. `names` must list every
column across both blocks (numerics first, then categoricals).

```rust
use datarust_profile::profile_table;

let p = profile_table(
    Some(&numeric),
    Some(&categorical),
    &["age".into(), "income".into(), "city".into(), "tier".into()],
)?;
```

Duplicate-row detection works across both blocks: a row is a duplicate only if
its numeric *and* categorical cells match an earlier row exactly.

## Data quality checks

[`run_checks`](https://docs.rs/datarust-profile/latest/datarust_profile/quality/checks/fn.run_checks.html)
turns a profile into a list of [`QualityIssue`](https://docs.rs/datarust-profile/latest/datarust_profile/quality/checks/struct.QualityIssue.html)s.
The thresholds are conservative defaults; tune them via
[`Thresholds`](https://docs.rs/datarust-profile/latest/datarust_profile/quality/checks/struct.Thresholds.html).

```rust
use datarust_profile::quality::{Thresholds, QualityKind};
use datarust_profile::quality::checks::run_checks;

let mut t = Thresholds::default();
t.outlier_fraction = 0.02;  // flag columns with ≥2% outliers
t.imbalance_ratio = 0.90;   // flag categoricals dominated ≥90% by one value

for issue in run_checks(&p, &t) {
    println!("{:?} [{}] {}: {}",
        issue.kind, issue.severity,
        issue.column.as_deref().unwrap_or("(dataset)"),
        issue.message);
}
```

### The six checks

| Kind | Scope | Fires when |
|---|---|---|
| [`HighMissing`](https://docs.rs/datarust-profile/latest/datarust_profile/quality/checks/enum.QualityKind.html) | column | `missing_fraction ≥ threshold.missing_fraction` (default `0.5`). Severity escalates to `Critical` at `0.9`. |
| [`ConstantColumn`](https://docs.rs/datarust-profile/latest/datarust_profile/quality/checks/enum.QualityKind.html) | column (numeric) | variance ≤ `threshold.near_zero_variance` (default `1e-12`). |
| [`NearUnique`](https://docs.rs/datarust-profile/latest/datarust_profile/quality/checks/enum.QualityKind.html) | column (categorical) | `unique / count ≥ threshold.near_unique_ratio` (default `0.98`) — likely an identifier, not a feature. |
| [`Outliers`](https://docs.rs/datarust-profile/latest/datarust_profile/quality/checks/enum.QualityKind.html) | column (numeric) | `outlier_fraction ≥ threshold.outlier_fraction` (default `0.05`). Severity escalates to `Warning` at `0.2`. |
| [`Imbalance`](https://docs.rs/datarust-profile/latest/datarust_profile/quality/checks/enum.QualityKind.html) | column (categorical) | `imbalance_ratio ≥ threshold.imbalance_ratio` (default `0.95`). Always `Critical`. |
| [`DuplicateRows`](https://docs.rs/datarust-profile/latest/datarust_profile/quality/checks/enum.QualityKind.html) | dataset | `duplicate_rows > 0`. Severity escalates to `Warning` at `0.1`. |

Each `QualityIssue` carries a [`Severity`](https://docs.rs/datarust-profile/latest/datarust_profile/types/enum.Severity.html)
(`Info`, `Warning`, `Critical`) and an optional column name (dataset-wide
findings like `DuplicateRows` set `column` to `None`).

## Rendering reports

The [`report`](https://docs.rs/datarust-profile/latest/datarust_profile/report/index.html)
module renders a profile plus its findings. HTML is always available; JSON
needs the `serde` feature.

```rust
use datarust_profile::report;

// HTML: self-contained, no dependencies.
let html = report::to_html(&p);

// Bring your own findings (e.g. with custom thresholds):
let findings = run_checks(&p, &t);
let html = report::to_html_with(&p, &findings);

// JSON (serde feature):
#[cfg(feature = "serde")]
let json = report::to_json(&report::JsonReport::from_profile(&p))?;
```

The HTML report uses a responsive card grid: numeric cards carry the summary
statistics, a CSS mini-histogram, and the outlier count; categorical cards
carry the top value, the imbalance ratio, and a frequency bar chart. Findings
appear in a severity-coloured list at the top.
