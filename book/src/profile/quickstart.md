# Quick Start

A 60-second tour: profile a numeric matrix, read the results, and render a
report. Everything here works with the default feature set — no `serde`
required for the HTML report.

## Install

```toml
[dependencies]
datarust = "0.6"
datarust-profile = "0.2"
```

For JSON output, enable the `serde` feature (this also pulls in `datarust/serde`):

```toml
[dependencies]
datarust-profile = { version = "0.2", features = ["serde"] }
```

## Profile a numeric matrix

[`profile_matrix`](https://docs.rs/datarust-profile/latest/datarust_profile/fn.profile_matrix.html)
takes a [`datarust::Matrix`](https://docs.rs/datarust/latest/datarust/struct.Matrix.html)
and returns a [`DatasetProfile`](https://docs.rs/datarust-profile/latest/datarust_profile/struct.DatasetProfile.html).
Missing values are encoded as `NaN` and excluded from every statistic.

```rust
use datarust::Matrix;
use datarust_profile::profile_matrix;

let m = Matrix::from_rows(vec![
    vec![1.0, 10.0],
    vec![2.0, 20.0],
    vec![3.0, 30.0],
    vec![4.0, 40.0],
    vec![5.0, f64::NAN], // missing in column 1
])?;

let p = profile_matrix(&m, Some(&["x".into(), "y".into()]))?;
```

Column names are optional — pass `None` and they default to `x0..x{n-1}`.

## Read the numeric profile

Each column carries a [`NumericStats`](https://docs.rs/datarust-profile/latest/datarust_profile/struct.NumericStats.html)
block: central tendency, spread, shape, and outliers.

```rust
let col = &p.columns[0];
let n = col.numeric.as_ref().unwrap();

println!("mean={:.2}  std={:.2}", n.mean, n.std);
println!("min={:.1}  Q1={:.1}  median={:.1}  Q3={:.1}  max={:.1}",
    n.five.min, n.five.q1, n.five.median, n.five.q3, n.five.max);

// v0.2: distributional shape.
println!("skewness={:.3}  kurtosis={:.3}", n.skewness, n.kurtosis);
println!("histogram bins: {} (counts: {:?})", n.histogram.nbins(), n.histogram.counts);
println!("outliers: {} ({:.1}%)", n.outlier_count, n.outlier_fraction * 100.0);
```

Skewness near zero means a symmetric column; large positive values indicate a
right tail. Excess kurtosis near zero matches a normal distribution; positive
values indicate heavier tails, negative values a flatter peak.

## Profile categorical data

[`profile_str_matrix`](https://docs.rs/datarust-profile/latest/datarust_profile/fn.profile_str_matrix.html)
infers each column's type: if every non-empty cell parses as `f64` it is treated
as numeric, otherwise categorical.

```rust
use datarust::StrMatrix;
use datarust_profile::profile_str_matrix;

let s = StrMatrix::from_strings(vec![
    vec!["25", "Istanbul", "basic"],
    vec!["40", "Ankara", "premium"],
    vec!["31", "Izmir", "basic"],
])?;

let p = profile_str_matrix(&s, Some(&["age".into(), "city".into(), "tier".into()]))?;

let city = &p.columns[1];
let c = city.categorical.as_ref().unwrap();
println!("{} unique values; top is '{}' ({} rows, {:.0}% of data)",
    c.unique, c.top, c.freq, c.imbalance_ratio * 100.0);
```

`imbalance_ratio` is the share of the most frequent value: `1.0` means a single
value dominates the whole column — a data-quality smell.

## Mixed tables

[`profile_table`](https://docs.rs/datarust-profile/latest/datarust_profile/fn.profile_table.html)
takes a numeric block and a categorical block side by side, sharing the same
row count. This is the natural fit for real-world CSV data.

```rust
use datarust_profile::profile_table;

let p = profile_table(
    Some(&numeric_matrix),
    Some(&categorical_str_matrix),
    &["age".into(), "income".into(), "city".into(), "tier".into()],
)?;
```

## Render a report

The HTML renderer produces a single self-contained document — inline CSS, no
JavaScript, no external assets. The JSON renderer needs the `serde` feature.

```rust
use datarust_profile::report;

// Always available:
let html = report::to_html(&p);
std::fs::write("profile.html", html)?;

#[cfg(feature = "serde")]
{
    let json = report::to_json(&report::JsonReport::from_profile(&p))?;
    std::fs::write("profile.json", json)?;
}
```

The HTML report lays columns out as a responsive card grid: numeric cards show
the summary statistics, a CSS mini-histogram, and the outlier count; categorical
cards show the top value, the imbalance ratio, and a frequency bar chart.

## Next steps

- The [Profiling Guide](./guide.md) covers numeric, categorical, and mixed
  profiling in depth, plus the full set of data-quality checks.
- The [API reference][docsrs] documents every type and function.

[docsrs]: https://docs.rs/datarust-profile
