# I Wanted to Profile My Dataset in Rust. So I Built the Thing That Does It.

*How datarust-profile turns one function call into a full data-quality report — distributions, outliers, missing patterns, and a self-contained HTML page, with zero dependencies.*

---

Every data project starts the same way. You open a CSV. You don't know what's in it. Maybe the header row has a typo. Maybe half the "income" column is empty. Maybe there's a stray `999` where a zero should be — sentinel value from some legacy ETL job nobody remembers. You need to look before you model.

In Python, you reach for `df.describe()` or, if you're being thorough, `ydata-profiling` (formerly `pandas-profiling`). One line of code, and you get distributions, missing-value heatmaps, correlation matrices, and a nice HTML report you can send to your team. It's genuinely one of the best tools in the Python data stack.

Then you try to do the same thing in Rust.

You write a loop. You count the missing values manually. You compute the mean with a hand-rolled accumulator. You think *surely someone has packaged this* — and you find either a thin wrapper around a Python library (back to the runtime tax), or a half-finished crate that gives you a `Vec<Stats>` and nothing else. No HTML report. No outlier detection. No sense of what the data actually *looks like*.

So I built it. [**datarust-profile**](https://crates.io/crates/datarust-profile) just shipped v0.2, and in this article I want to walk through what it does, how it's built on the `datarust` statistics layer, and where it's going.

## The pitch in one paragraph

`datarust-profile` takes a numeric `Matrix`, a categorical `StrMatrix`, or a mixed table of both, and returns a `DatasetProfile` — a structured object describing the shape of your data: per-column statistics (mean, std, five-number summary, skewness, kurtosis, histograms, outliers), type inference, duplicate-row detection, and a list of data-quality findings. You can render it to a self-contained HTML report or serialize it to JSON. Zero external dependencies by default, built on `datarust::stats`.

Here's the whole workflow, start to finish:

```rust
use datarust::StrMatrix;
use datarust_profile::{profile_str_matrix, report};

// Rows as a CSV reader would yield them: strings, mixed types, missing markers.
let table = StrMatrix::from_strings(vec![
    vec!["25", "Istanbul", "basic"],
    vec!["40", "Ankara", "premium"],
    vec!["31", "Izmir", "NA"],        // missing plan
    vec!["NA", "Istanbul", "basic"],  // missing age
    vec!["52", "Istanbul", "basic"],
    vec!["37", "Ankara", "basic"],
])?;

let profile = profile_str_matrix(&table, Some(&["age".into(), "city".into(), "plan".into()]))?;

// Inspect it programmatically...
println!("{} rows × {} columns, {} duplicates",
    profile.n_rows, profile.n_columns, profile.duplicate_rows);

for col in &profile.columns {
    println!("  {:<6} {:<11} missing {:.0}%",
        col.name, col.column_type, col.missing_fraction * 100.0);
}

// ...or render a report.
std::fs::write("report.html", report::to_html(&profile))?;
```

That `report.html` is a single file — inline CSS, no JavaScript, no external assets — that lays out every column as a card with its summary statistics, a mini histogram, and the outlier count. You can email it, stick it in a PR, or serve it as a static file.

Let me break down what's actually happening under the hood, because the design choices are the interesting part.

## The data model: describe, don't transform

The first decision was philosophical: is this a *profiler* (it reads your data and tells you what's there) or a *transformer* (it changes your data)? In `datarust`, transformers implement `fit` / `transform` / `fit_transform`. That's the sklearn contract. But profiling doesn't transform anything — it produces a *description*.

So `datarust-profile` doesn't implement any `datarust` trait. It has its own entry points — `profile_matrix`, `profile_str_matrix`, `profile_table` — that return a `DatasetProfile` value. That value is a plain data structure. You read it. You serialize it. You render it. You don't fit it onto new data, because that's not what profiling is.

This keeps the boundary clean: `datarust` owns the numerical kernels and the transformer pipeline; `datarust-profile` owns the *interpretation* of numbers into summaries and quality flags. The statistics flow through `datarust::stats` — `mean`, `std`, `quantile`, `median_sorted` — so the two crates never disagree about what a quartile is.

## Numeric profiling: beyond the five-number summary

A `describe()` gives you count, mean, std, min, Q1, median, Q3, max. That's the baseline — every profiling tool does it. `datarust-profile` does too, but v0.2 adds the pieces that tell you about the *shape* of the distribution:

```rust
let n = col.numeric.as_ref().unwrap();

// Central tendency and spread (the describe() basics).
println!("mean {:.2}  std {:.2}", n.mean, n.std);

// The five-number summary, from a single sort.
println!("min {}  Q1 {}  median {}  Q3 {}  max {}",
    n.five.min, n.five.q1, n.five.median, n.five.q3, n.five.max);

// v0.2: distributional shape.
println!("skewness {:+.3}", n.skewness);   // +0.0 = symmetric, large + = right tail
println!("kurtosis {:+.3}", n.kurtosis);   // 0 = normal, + = heavy-tailed, - = flat

// A histogram — equal-width bins, count by Sturges' rule.
println!("histogram ({} bins): {:?}", n.histogram.nbins(), n.histogram.counts);

// Outliers beyond the Tukey IQR fences.
println!("outliers {} ({:.1}%)", n.outlier_count, n.outlier_fraction * 100.0);
```

The skewness and kurtosis matter more than you'd think. A column with skewness of +2.5 is telling you "most values are small, but there's a long upper tail" — which is exactly the shape of reaction times, income, and session durations. If you feed that to a linear model without a log transform, the model will be dominated by the tail. The histogram makes it visible; the skewness number quantifies it.

The outlier detection uses the Tukey IQR rule — values below `Q1 - 1.5·IQR` or above `Q3 + 1.5·IQR`. It's not the only way to find outliers (z-scores, isolation forests, robust z-scores all exist), but it's the one that needs no distributional assumptions and no hyperparameters. It's what `ydata-profiling` uses. It's a good default.

## Categorical profiling: cardinality and the silent duplicate

Categorical columns are where the sneaky bugs live. `"USA"`, `"usa "`, `"U.S.A"` — three strings that a human reads as the same value but a model treats as three distinct categories. `datarust-profile` doesn't do fuzzy matching (yet), but it does tell you the cardinality, the most frequent value, and how much that value dominates:

```rust
let c = col.categorical.as_ref().unwrap();

println!("{} unique values", c.unique);
println!("top: {:?} ({} rows, {:.1}% of data)",
    c.top, c.freq, c.imbalance_ratio * 100.0);

// The full frequency table, capped for display.
for (value, count) in &c.top_values {
    println!("  {:<12} ×{}", value, count);
}
```

The `imbalance_ratio` is `freq / present_count`. If it's `0.97`, one value covers 97% of the column — which usually means the column carries almost no signal, or the other values are rare-error categories. The `Imbalance` quality check flags anything above 95% by default. That's the categorical analogue of `ConstantColumn` (a numeric column with zero variance): both are telling you "this column probably won't help your model."

Type inference is automatic: a column is `Numeric` if every non-missing cell parses as `f64`, otherwise `Categorical`. So `"25"`, `"40"`, `"NA"` becomes a numeric column with one missing value. `"Istanbul"`, `"Ankara"` stays categorical. You don't have to tell it which is which.

## Data quality: the six checks

A profile is descriptive; a quality check is *opinionated*. `datarust-profile` ships six, each with configurable thresholds:

```rust
use datarust_profile::quality::checks::run_checks;
use datarust_profile::quality::Thresholds;

let findings = run_checks(&profile, &Thresholds::default());
for issue in &findings {
    let scope = issue.column.as_deref().unwrap_or("dataset");
    println!("[{:?}] {}: {}", issue.severity, scope, issue.message);
}
```

| Check | Fires when |
|---|---|
| `HighMissing` | A column's missing fraction crosses the threshold (default 0.5). |
| `ConstantColumn` | A numeric column has near-zero variance (default 1e-12). |
| `NearUnique` | A categorical column's cardinality ≈ row count — likely an identifier. |
| `Outliers` | A numeric column has values beyond the IQR fences (default 5%). |
| `Imbalance` | A categorical column is dominated by one value (default 95%). |
| `DuplicateRows` | The dataset contains exact-duplicate rows. |

The thresholds are all `pub` fields on a single `Thresholds` struct. You can tune them before running:

```rust
let strict = Thresholds {
    missing_fraction: 0.3,   // flag columns with >30% missing
    outlier_fraction: 0.02,  // flag columns with even a few outliers
    imbalance_ratio: 0.85,   // flag 85%+ domination
    ..Thresholds::default()
};
```

Each finding carries a `Severity` — `Info`, `Warning`, or `Critical`. A column with 60% missing is a `Warning`; one with 95% missing is `Critical`. The HTML and JSON renderers color-code accordingly.

This is where profiling stops being "nice to have" and starts being a CI gate. You can serialize a `Thresholds` config, run `datarust-profile` against your data in a pipeline, and fail the build if a finding crosses severity. That's the v0.9 destination — data contracts as code.

## The HTML report: a card grid, not a table

The first version of the HTML renderer used a wide table — one row per column, 14 columns of statistics. It worked, but it was fragile (the cell-counting broke more than once) and it was hard to read on a laptop screen.

v0.2 replaces it with a responsive card grid. Each column becomes a card:

- **Numeric cards** show the summary statistics, a CSS mini-histogram (pure divs, no JavaScript), the five-number summary on one line, and the outlier count.
- **Categorical cards** show the cardinality, the top value with its imbalance ratio, and a frequency bar chart of the top values.

```
┌─────────────────────────────────────┐
│ age  [numeric]            0% missing│
├─────────────────────────────────────┤
│ mean 34.0  std 7.35  skew 0.12      │
│ kurt -1.30  outliers 0 (0%)         │
│ ▁▂▄▆▅▃▁  (CSS div histogram)        │
│ min 25  Q1 29.5  med 35.5  Q3 40    │
└─────────────────────────────────────┘
```

Findings appear in a severity-colored list at the top — red for `Critical`, orange for `Warning`, blue for `Info`. The whole report is one HTML file with inline CSS and a dark-mode media query. No build step, no framework, no `node_modules`.

## What's not there yet (honest limitations)

This is v0.2 of a young crate. Here's what's missing:

- **No correlation analysis.** You can't see which columns are redundant yet. `datarust::stats` already has `correlation_matrix` and `covariance_matrix` — wiring them into a `Relationships` block with a heatmap is the v0.3 plan.
- **No CSV reader.** You construct the `StrMatrix` yourself from your data source, same as `datarust`. A `csv` feature flag is coming in v0.4, along with a CLI binary (`datarust-profile data.csv -o report.html`).
- **No dataset comparison.** If you want to compare this week's data against last week's to detect drift, you can't yet. v0.5 adds `compare_profiles` with a population stability index.
- **No streaming mode.** The whole dataset lives in memory. For files that don't fit, a single-pass streaming profiler (Welford for moments, t-digest for quantiles) is planned for v0.4.
- **No text-shape analysis.** String-length distributions, casing flags, and the silent-duplicate detection (`"USA"` vs `"usa "`) arrive in v0.6.

If you need any of those today, `ydata-profiling` is still the tool to reach for — it's mature, feature-complete, and runs in Python. `datarust-profile` is the tool you reach for when you need profiling *inside a Rust binary* — a CLI, a service, a WASM module, a CI step — without a Python runtime in sight.

## Why it exists

The honest answer is: because `datarust` already had the statistics. The `mean`, `std`, `quantile`, and `median_sorted` functions were sitting right there, used by the scalers and selectors, but never exposed in a way that answered the question *what does this column look like?* Once you have those primitives, a profiler is mostly interpretation — deciding what numbers matter, what thresholds are suspicious, and how to render the result.

That's the `datarust` ecosystem bet in miniature. The numerical kernels live in the core crate; the workflows that consume them — preprocessing, profiling, eventually model inspection — live in sibling crates that share the same `Matrix` type and the same statistics layer. `cargo add datarust-profile` pulls in `datarust` as a dependency. The two crates never disagree about what a standard deviation is, because there's only one implementation.

The v0.2 release adds the distributional depth that turns a `describe()` into a real picture: skewness, kurtosis, histograms, outliers, and a rebuilt card-grid HTML report. The foundation is solid. The gaps are known. The [roadmap](https://github.com/genc-murat/datarust/blob/main/crates/datarust-profile/ROADMAP.md) is public.

If you've ever wanted a one-call data-quality report in Rust, give it a try:

```sh
cargo add datarust-profile
```

---

*datarust-profile is MIT-licensed and available on [crates.io](https://crates.io/crates/datarust-profile), with documentation at [docs.rs/datarust-profile](https://docs.rs/datarust-profile). It's the second crate in the [datarust workspace](https://github.com/genc-murat/datarust) — a scikit-learn-style ML toolkit and a one-call data-profiling companion, both pure Rust with zero dependencies by default. The profile roadmap lives at [github.com/genc-murat/datarust](https://github.com/genc-murat/datarust/blob/main/crates/datarust-profile/ROADMAP.md) — if you want to contribute a correlation heatmap or a CSV reader, now's a good time.*
