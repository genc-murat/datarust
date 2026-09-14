# Your Data Profiler Shouldn't Make You Parse the CSV First

*How datarust-profile v0.4 meets you at the file — a pure-Rust CSV reader, a single-pass streaming profiler for files bigger than memory, and a CLI that turns any CSV into a quality report in one command.*

---

Every data-profiling tool has the same awkward first five minutes. You have a CSV. The tool wants a dataframe, or a matrix, or a JSON config, or a Python process. So you write the loading code — and the loading code is where the lies begin.

I've seen it go like this: someone profiles a dataset in a notebook, finds nothing alarming, ships the model. Six months later the same file is profiled in production and the missing-rate has doubled. Nothing changed about the data. What changed was the loader: the notebook's CSV reader silently coerced empty strings to `NaN` and treated `"1,5"` (a European decimal) as two columns; the production parser did neither. The profiler was fine both times. The *ingestion* was the bug.

[datarust-profile](https://crates.io/crates/datarust-profile) v0.3 was a profiler you fed matrices to — accurate, fast, zero dependencies, but it made you do the loading. v0.4 removes that step, and guards it:

```rust
use datarust_profile::DatasetProfile;

let profile = DatasetProfile::from_csv("customers.csv")?;
```

That line is the release. But the interesting engineering is in what it took to make it trustworthy.

## Writing a CSV reader you can't fool

The obvious move was to pull in the `csv` crate — battle-tested, fast, one line in `Cargo.toml`. But datarust-profile's whole identity is that its default build has **zero external dependencies**, and a profiler's job is to tell you the truth about your file. When the parser is someone else's black box, "truth about your file" quietly becomes "truth about your file as interpreted by a library you didn't read."

So v0.4 ships its own RFC-4180 reader (the `csv` feature, still dependency-free): quoted fields with `""` escapes, delimiters and newlines inside quotes, LF and CRLF endings, UTF-8 BOM stripping, configurable delimiter/quote bytes, header detection. And where many readers are forgiving, this one is **strict** — a ragged row is an error that names the physical line number, not a silent `null`-padded row that poisons your missing-rate statistics.

Type inference follows the crate's existing rules, and non-parseable cells flip a column from numeric to categorical — visibly, in the profile, where you can argue with it.

## The streaming profiler: one pass, bounded memory

The in-memory path profiles an `n × p` table exactly. For a 500 MB CSV that's fine. For the 40 GB log export someone dropped in your bucket, it isn't.

`profile_csv_streaming` (the `streaming` feature) reads the file **once, forward only**, and keeps per-column accumulators instead of the data:

```rust
let profile = datarust_profile::profile_csv_streaming("events.csv")?;
```

The design question was: which statistics survive a single pass, honestly? The answer became the crate's documented fidelity contract:

- **Exact**: row and cell counts, missing counts, mean, variance, min/max, skewness and kurtosis (single-pass Pébay recurrences for the third and fourth central moments), categorical frequencies, and — opt-in, since it costs memory — duplicate-row detection.
- **Approximate, with the approximation stated**: quantiles, histograms, and outlier counts come from a reservoir sample (default 10 000 values per column; exact whenever the table fits inside it). Pearson correlation is computed pairwise-complete, over exactly the rows where both columns hold values.
- **Decided online**: column types. A numeric column converts to categorical at its first non-parseable cell — a one-pass trade-off the docs say out loud rather than hide.

That last category is where streaming implementations usually cheat. The contract is on the module's front page precisely so nobody has to discover the difference in production.

## `--pretty`: the report your terminal deserved

Profiling is a loop: look, wonder, check. HTML reports end that loop in a browser tab. v0.4's `pretty` feature ends it in the shell — a fixed-width table (type, missing %, μ/σ or top values) plus the quality findings with CRIT/WARN/INFO tags, ANSI color only when stdout is a TTY, `NO_COLOR` respected:

```
Dataset profile
4 rows × 3 columns, 0 duplicate rows, 96 B
------------------------------------------
column  type        missing  summary
------------------------------------------
age     numeric     0 ( 0.0%)  μ=40.00 σ=9.83 ...
city    categorical 0 ( 0.0%)  3 unique, top "Istanbul" 50.0%
...
Findings
  CRIT income  |r| = 0.99 with age — leakage risk
```

## One command, no Rust required

All of it wraps up in a CLI binary (the `cli` feature):

```sh
datarust-profile customers.csv --pretty --target churn
datarust-profile events.csv --streaming -o report.html
datarust-profile survey.csv --delimiter ';' --types numeric,categorical,categorical
```

`--target` validates against the header *before* profiling the file — a typo fails in milliseconds, not after streaming 40 GB. Exit codes are boring on purpose: `0` success, `1` runtime error, `2` usage error.

## The point

None of this is glamorous. A CSV reader, some running sums, a table in a terminal. But the pattern keeps repeating across this project: the value isn't in any single statistic, it's in removing the seams where truth gets lost. The parser you didn't write, the loader that reshapes your data, the "approximately" you didn't notice — v0.4 is about closing those doors in the profiling path.

`cargo add datarust-profile` (or `cargo install datarust-profile --features cli`) — and the [guide](https://docs.rs/datarust-profile) has the full fidelity contract if you're the type who reads it. You should be.
