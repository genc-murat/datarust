# Skewness +36, Kurtosis +2753: What Your Distribution Looks Like to a Model

*A normal, an exponential, a uniform, a bimodal, a Cauchy, and a constant walk into a profiler — and the numbers tell you which ones will break your model*

---

I profiled seven synthetic columns — 5000 samples each — through `datarust-profile`. The results range from "textbook normal" (skew ≈ 0, kurt ≈ 0) to "this column will break your model" (skew ≈ +36, kurt ≈ +2753).

Distributional shape isn't a curiosity. It determines whether your model can learn from a column, whether outliers dominate the loss, and whether a log transform is needed before you fit a single parameter.

Here's what the profile numbers reveal.

## The Summary Table

```
column              mean      std     skew     kurt        min        max outliers
--------------------------------------------------------------------------------
normal              0.00     1.01    -0.01    -0.03    -3.80       3.64       32
right_skew          1.00     0.98    +1.95    +5.37     0.00       8.26      236
left_skew           1.98     1.01    -1.89    +4.83    -4.98       3.00      227
uniform             5.06     2.88    -0.01    -1.21     0.01      10.00        0
bimodal             0.04     2.07    -0.04    -1.77    -3.87       4.14        0
cauchy              0.36   100.64   +36.37 +2753.39 -3134.89    6021.61      750
constant            5.00     0.00    +0.00    +0.00     5.00       5.00        0
```

Each row is a distribution. Each number is from `datarust-profile`'s `NumericStats`. The story is in the skewness, kurtosis, and outlier count.

## Normal — The Baseline

```
    mean    0.00    std  1.01
    skew   -0.01    kurt -0.03
    IQR:  [-0.68, 0.69]
    range: [-3.80, 3.64]
    outliers: 32 (0.6%)

    [   -2.73,    -2.20)    52 ██
    [   -2.20,    -1.67)   173 ███████
    [   -1.67,    -1.14)   395 ███████████████
    [   -1.14,    -0.61)   730 █████████████████████████████
    [   -0.61,    -0.08)   971 ██████████████████████████████████████
    [   -0.08,     0.45)  1021 ████████████████████████████████████████
    [    0.45,     0.98)   808 ████████████████████████████████
    [    0.98,     1.51)   501 ████████████████████
    [    1.51,     2.04)   223 █████████
    [    2.04,     2.57)    84 ███
    [    2.57,     3.10)    20 █
```

Skewness near 0, kurtosis near 0. The bell curve is symmetric and the tails decay at the normal rate. 32 outliers (0.6%) — about what the IQR rule catches at 1.5× the interquartile range for a normal distribution.

This is what you want a numeric column to look like. Linear models, distance-based methods, and neural networks all assume this shape implicitly.

## Exponential — The Right Tail That Never Ends

```
    mean    1.00    std  0.98
    skew   +1.95    kurt +5.37
    IQR:  [0.29, 1.39]
    range: [0.00, 8.26]
    outliers: 236 (4.7%)

    [    0.00,     0.59)  2189 ████████████████████████████████████████
    [    0.59,     1.18)  1264 ███████████████████████
    [    1.18,     1.77)   716 █████████████
    [    1.77,     2.36)   370 ███████
    [    2.36,     2.95)   208 ████
    [    2.95,     3.54)   108 ██
    [    3.54,     4.13)    65 █
    [    4.13,     4.72)    35 █
```

Skewness +1.95: most values cluster at 0, but the tail stretches to 8.26. Kurtosis +5.37: the tail is heavy — values beyond 3σ are far more common than a normal distribution would predict.

236 outliers (4.7%) — eight times the normal rate. The IQR fence catches these because the right tail keeps generating extreme values. Every one of those 236 points will exert disproportionate influence on a linear model.

**What this means for modeling:** Log-transform this column before fitting. A `log(income)` or `log(session_duration)` collapses the tail, brings skewness toward 0, and makes the column usable for a linear model.

## Bimodal — The Antimode That Splits Your Data

```
    mean    0.04    std  2.07
    skew   -0.04    kurt -1.77
    IQR:  [-1.99, 2.02]
    range: [-3.87, 4.14]
    outliers: 0

    [   -3.30,    -2.73)   169 ██████
    [   -2.73,    -2.15)   763 █████████████████████████████
    [   -2.15,    -1.58)  1050 ████████████████████████████████████████
    [   -1.58,    -1.01)   389 ███████████████
    [   -1.01,    -0.44)    55 ██
    [    0.13,     0.71)    14 █
    [    0.71,     1.28)   180 ███████
    [    1.28,     1.85)   806 ███████████████████████████████
    [    1.85,     2.42)  1041 ████████████████████████████████████████
    [    2.42,     2.99)   470 ██████████████████
    [    2.99,     3.57)    48 ██
```

Skewness −0.04 (symmetric), kurtosis −1.77 (platykurtic — flat). The profile numbers say "roughly symmetric, light-tailed" — which they are! But the histogram tells the real story: two peaks at −2 and +2 with a deep valley at 0.

The mean of 0.04 is between the peaks — it represents nobody. A model that predicts the mean will be wrong for every sample. The standard deviation of 2.07 is inflated by the gap between clusters. Zero outliers — the IQR rule doesn't fire because the data is tightly clustered within two peaks, but the distribution is not what any modeling assumption expects.

**What this means for modeling:** The summary statistics (mean, std, skew, kurt) are misleading for a bimodal column. The histogram reveals the truth. A bimodal column often indicates a latent categorical variable — encode it and split it.

## Cauchy — The Distribution That Breaks Summary Statistics

```
    mean    0.36    std 100.64
    skew  +36.37    kurt +2753.39
    IQR:  [-0.98, 1.00]
    range: [-3134.89, 6021.61]
    outliers: 750 (15.0%)

    [ -518.75,   135.29)  4983 ████████████████████████████████████████
```

Skewness +36.37, kurtosis +2753.39. The mean of 0.36 is meaningless — the next sample will change it. The standard deviation of 100.64 is meaningless — it's dominated by extreme draws. The range spans from −3134 to +6021. 750 outliers (15%) — the IQR fence catches these because the Cauchy distribution generates extreme values continuously.

The histogram shows 4983 of 5000 values in one bin, but that bin stretches from −519 to 135. The remaining 17 values are scattered across the extreme tails. A single histogram bin covering 99.7% of the data with a width of 654 units is not a useful visualization — it's a sign that the column is dominated by extreme outliers.

**What this means for modeling:** This column cannot be used in a linear model without preprocessing. The extreme values will dominate the loss function. Winsorization (capping at the 1st and 99th percentiles) or robust scaling (using median and IQR instead of mean and variance) is required. Even a log transform won't help — the Cauchy distribution has no finite moments beyond the first, and the skewness/kurtosis numbers themselves are unreliable because they're driven by single extreme draws.

## Constant — The Silent Zero

```
    mean    5.00    std  0.00
    skew   +0.00    kurt +0.00
    outliers: 0
```

Standard deviation zero. Variance zero. The column contributes nothing. `datarust-profile`'s `ConstantColumn` quality check flags it immediately.

**What this means for modeling:** Drop it. A constant column adds zero information and can cause singular matrices in linear models.

## What Each Statistic Tells You

**Skewness** measures symmetry. A skewness of +1.95 means "most values are small, but the tail stretches far to the right." This is income, reaction times, session durations. Log-transform before modeling.

**Kurtosis** measures tail weight. A kurtosis of +5.37 means "extreme values are far more common than a normal distribution." This is financial returns, network latencies, insurance claims. Robust methods help.

A kurtosis of −1.21 means "the tails are lighter than normal — values cluster in a bounded range." This is uniform data, exam scores (bounded by 0–100), and controlled measurements.

**Outlier count** (via IQR rule) catches values beyond 1.5× the interquartile range. For a normal distribution, this is about 0.7% of samples. The exponential distribution produces 4.7% outliers — seven times the normal rate. The Cauchy distribution produces 15% outliers — the column is mostly outliers.

## How to Check

```rust
use datarust::Matrix;
use datarust_profile::profile_matrix;

let matrix = Matrix::new(rows)?;
let profile = profile_matrix(&matrix, Some(&column_names))?;

for col in &profile.columns {
    if let Some(s) = &col.numeric {
        println!("{:>15}  skew {:+.3}  kurt {:+.3}  outliers {} ({:.1}%)",
            col.name, s.skewness, s.kurtosis,
            s.outlier_count, s.outlier_fraction * 100.0);
    }
}
```

The profile numbers tell you what the model sees. A normal distribution and a constant look the same in a `.describe()` table (both have a mean and a std), but their skewness, kurtosis, and outlier count are worlds apart.

A column with skewness > +1.0 needs a log transform before it can contribute to a linear model. A column with kurtosis > +10.0 needs robust preprocessing. A column with `std = 0` needs to be dropped. And a column with vanishingly few bins in the histogram (like the Cauchy) needs a fundamentally different scaling approach.

The profile tells you this before you write a single line of model code.

```bash
cargo add datarust-profile
```

```rust
use datarust::Matrix;
use datarust_profile::profile_matrix;

let m = Matrix::new(rows)?;
let p = profile_matrix(&m, None)?;
// Skewness, kurtosis, outliers — before you fit a model
```
