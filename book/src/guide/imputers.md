# Imputers

Missing-value completion. Live in [`datarust::imputer`](https://docs.rs/datarust/latest/datarust/imputer/index.html). Both implement [`Transformer`](https://docs.rs/datarust/latest/datarust/trait.Transformer.html).

Missing values are represented as `f64::NAN` in the `Matrix`.

## SimpleImputer

Fill missing values with a per-column statistic or a constant.

```rust
use datarust::imputer::{SimpleImputer, ImputeStrategy};

// Mean (default), median, most_frequent, or constant
let mut imp = SimpleImputer::new()
    .strategy(ImputeStrategy::Median); // or Mean, MostFrequent, Constant(0.0)
let out = imp.fit_transform(&x_with_nans)?;
```

Strategies:
- `Mean` — column mean (default)
- `Median` — column median, robust to outliers
- `MostFrequent` — most common value
- `Constant(v)` — a fixed fill value

## KnnImputer

Impute using k-Nearest Neighbors. Distance is computed over **co-observed features only** (features where neither sample is missing).

```rust
use datarust::imputer::{KnnImputer, KnnWeights};

let mut imp = KnnImputer::new()
    .n_neighbors(5)
    .weights(KnnWeights::Distance); // or Uniform
let out = imp.fit_transform(&x_with_nans)?;
```

KnnImputer is more accurate than SimpleImputer when features are correlated, but slower — it computes pairwise distances over all samples.

## When to use which?

| Scenario | Imputer |
|---|---|
| Quick baseline | `SimpleImputer` (mean or median) |
| Categorical data | `SimpleImputer` (most_frequent or constant) |
| Correlated features, accuracy matters | `KnnImputer` |
