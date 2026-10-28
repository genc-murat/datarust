---
name: data-profiling
description: Automated dataset profiling, data quality checks, and report generation in Rust with datarust-profile. Use this skill when inspecting tabular datasets, evaluating data quality, detecting target leakage, or generating EDA summaries.
compatibility: Claude, Cursor, OpenAI, Antigravity, LLM agents
when_to_use: When an agent or developer needs to analyze tabular datasets, compute descriptive statistics, check for duplicate rows or missing values, detect target leakage, or export HTML and JSON profiling reports in Rust.
---

# Data Profiling & Quality Analysis with datarust-profile

## When to use this skill
Use this skill when:
- Analyzing or profiling tabular datasets (CSV files, numeric and categorical matrices) in Rust.
- Checking data quality: identifying missing values, duplicate rows, constant columns, or outlier distributions.
- Detecting target leakage between feature columns and the prediction target.
- Calculating pairwise correlation metrics (Pearson correlation, Cramér's V, point-biserial correlation).
- Generating automated exploratory data analysis (EDA) reports in HTML or JSON formats.

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
