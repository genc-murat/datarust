# How to Tell a Number from a Zip Code

*A column that parses as f64 isn't always numeric. datarust-profile's type inference handles 14 edge cases — and gets two wrong by design.*

---

I fed `datarust-profile` 14 columns of CSV-like data. The columns included ages, cities, incomes, ratings, zip codes, dates, boolean flags, alphanumeric codes, European-formatted decimals, and customer IDs. The profiler had to decide: is this column *numeric* (fit for mean, std, quantiles, histograms) or *categorical* (fit for cardinality, top values, imbalance ratio)?

The answer depends on a single rule — and that rule is right 12 times out of 14. The two "wrong" answers are deliberate design decisions.

## The Rule

Open `infer.rs` in `datarust-profile`. It's 44 lines:

```rust
pub fn infer_column(cells: &[String]) -> ColumnType {
    let any_non_missing = cells.iter().any(|c| !is_missing(c));
    if !any_non_missing {
        return ColumnType::Categorical;
    }
    let all_numeric = cells
        .iter()
        .all(|c| is_missing(c) || c.trim().parse::<f64>().is_ok());
    if all_numeric {
        ColumnType::Numeric
    } else {
        ColumnType::Categorical
    }
}
```

Every non-missing cell parses as `f64`? → Numeric. Otherwise → Categorical. Missing values (`""`, `"NA"`, `"null"`, `"NaN"`, `"None"`, `"-"`, `"?"`) are skipped.

That's the entire heuristic. Let's see how it performs.

## The Test

14 columns, 14 rows each, covering the full spectrum of real-world CSV data:

```
column          inferred      count missing   unique
-----------------------------------------------------
age             Numeric          14       7%        0
city            Categorical      14       0%        4
income          Numeric          14       7%        0
rating          Numeric          14      21%        0
zip_code        Numeric          14       0%        0
joined          Categorical      14      21%       11
employed        Categorical      14       0%        2
children        Numeric          14       0%        0
tier            Categorical      14      14%        3
code            Categorical      14       0%       14
score           Numeric          14      21%        0
rate            Categorical      14      21%       11
signup          Categorical      14      21%       11
customer_id     Categorical      14       0%       14
```

14 out of 14 columns are inferred the way you'd expect. But two of them are ambiguous.

## The Easy Cases (✓ Correct)

**`age` → Numeric.** `["34", "45", "29", ..., "NA"]` — every non-missing cell parses as f64. The `"NA"` is recognized as missing. Correct.

**`city` → Categorical.** `["Istanbul", "Ankara", "Izmir", ...]` — none of these parse as f64. Straightforward.

**`income` → Numeric.** Like age: numbers with NA markers. Missing values don't block inference.

**`rating` → Numeric.** Decimal ratings like `"4.5"`, `"3.8"`. f64 handles decimals natively.

**`joined` → Categorical.** ISO dates like `"2024-01-15"` — not valid f64. They stay categorical. This is correct for profiling (you want counts, not means), but it means dates can't be sorted chronologically from the profile. A future version may add date detection.

**`employed` → Categorical.** `"yes"`, `"no"` — boolean fields are correctly categorical.

**`children` → Numeric.** `0`, `1`, `2`, `3` — small integers parse cleanly.

**`tier` → Categorical.** Mixed `"basic"`, `"pro"`, `"enterprise"`, and empty strings. The empty cells are missing; the text values prevent numeric inference.

**`code` → Categorical.** `"A01"`, `"B02"` — letters in every value. No numeric parse possible.

**`score` → Numeric.** Integers with NA markers. Parses cleanly.

**`signup` → Categorical.** Same as `joined` — ISO dates aren't f64.

**`customer_id` → Categorical.** `"C001"` through `"C014"` — alphanumeric, so categorical.

## The Ambiguous Cases (✓ by the rule, ✗ by semantics)

**`zip_code` → Numeric.** `["34000", "06000", "35000", ...]` — every value parses as f64. The profiler says Numeric.

But a zip code isn't a number. You don't compute the mean zip code or the standard deviation of postal codes. A mean of `29285` is meaningless. The cardinality (4 unique values out of 14) would be more informative: zip codes are categorical, possibly ordinal (if they map to geographic proximity).

The profiler can't know this. The inference rule sees valid f64 values and treats them as numbers. This is the same problem every profiling tool faces — `pandas.read_csv` converts `"34000"` to `int` unless you pass `dtype=str`.

**`rate` → Categorical.** `["3,14", "2,71", "1,59", ...]` — European decimal notation with a comma separator. Rust's f64 parser doesn't recognize commas, so these columns fall through to Categorical.

This is the opposite problem: semantically numeric, technically not f64. A user expecting European decimal formatting will see `rate` as categorical with 11 unique values, not as a numeric column with mean and std.

## The Design Tradeoff

The inference is conservative by design: if *any* single value fails to parse as f64, the column is Categorical. This avoids silently corrupting data that looks numeric but isn't:

```
"555-0100"  → Categorical (contains dash — correct)
"2024Q1"    → Categorical (contains letter — correct)
"0x1A3F"    → Categorical (hex — correct)
"1,000,000" → Categorical (commas — debatable)
```

The alternative — trying to be smart about different formats — introduces dependencies (date parsers, locale-aware number parsers, hex detectors) and creates ambiguity (is `"5-10"` a range, a date, or a hyphenated code?). The 44-line rule is simple, predictable, and wrong only in ways the caller can fix with explicit column-typing.

## What the Profiler Does After Inference

Once the type is decided, the profiler takes different paths:

**Numeric columns** get the full treatment:
- Mean, std, five-number summary
- Skewness, kurtosis
- Equal-width histogram with Sturges' rule
- IQR outlier detection

**Categorical columns** get a different set:
- Cardinality (unique value count)
- Top value and its frequency
- Imbalance ratio (top frequency / count)
- Top-N values list

```rust
match col.column_type {
    ColumnType::Numeric => {
        let n = col.numeric.as_ref().unwrap();
        println!("  mean {:.1}, outliers {}", n.mean, n.outlier_count);
    }
    ColumnType::Categorical => {
        let c = col.categorical.as_ref().unwrap();
        println!("  top {:?} ({:.0}%)", c.top, c.imbalance_ratio * 100.0);
    }
}
```

A zip code column profiled as Numeric gets a histogram and a five-number summary (which are useless). The same column, if you override the type to Categorical, gets a frequency table and an imbalance ratio (which are useful). The profiler does the right thing 12 out of 14 times — but zip codes and comma-separated decimals need human judgment.

## Explicit Override

If your data contains columns that the inference miscategorizes, you can override before profiling:

```rust
// Parse zip codes manually as strings, keep them in the categorical block
let cat_block = StrMatrix::from_strings(vec![
    zip_code_col,
    city_col,
    // ...
])?;

// Parse actual numbers into the numeric block
let num_block = Matrix::new(vec![
    age_col,
    income_col,
    rating_col,
    // ...
])?;

let profile = datarust_profile::profile_table(
    Some(&num_block),
    Some(&cat_block),
    &names,
)?;
```

The `profile_table` entry point lets you split your data explicitly: numeric columns go in the `Matrix`, categorical columns go in the `StrMatrix`. No inference, no surprises.

## Try It

```bash
cargo add datarust-profile
```

```rust
use datarust::StrMatrix;
use datarust_profile::profile_str_matrix;

let data = StrMatrix::from_strings(rows)?;
let profile = profile_str_matrix(&data, Some(&["age", "city", "zip"]))?;

for col in &profile.columns {
    println!("  {} → {:?}", col.name, col.column_type);
}
```

Type inference is a heuristic — not a guarantee. 44 lines of code handle 12 out of 14 real-world columns correctly. Zip codes pretend to be numbers. European decimals pretend not to be. The profiler documents what it decided, and `profile_table` lets you override when the heuristic gets it wrong.
