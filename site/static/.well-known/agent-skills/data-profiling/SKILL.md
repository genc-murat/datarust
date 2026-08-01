# Data Profiling & Quality Analysis with datarust-profile

Use `datarust-profile` for automated dataset profiling, statistical summaries, data quality checks, target-leakage detection, and HTML/JSON report generation in Rust.

## Features
- Column statistics: mean, std, min, max, median, IQR, null count, histograms.
- Data quality checks: duplicate rows, constant columns, high missingness, target leakage warnings.
- Pairwise relationships: Pearson correlation, Cramér's V, point-biserial correlation.
- HTML & JSON report exports.

## Quick Start
```rust
use datarust_profile::Profiler;

let report = Profiler::builder().build().profile(&dataset)?;
println!("{}", report.to_json());
```
