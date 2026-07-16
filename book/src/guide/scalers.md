# Scalers

Feature scaling and distribution-shaping transformers. All implement [`Transformer`](https://docs.rs/datarust/latest/datarust/trait.Transformer.html) (`Matrix → Matrix`) and live in [`datarust::scaler`](https://docs.rs/datarust/latest/datarust/scaler/index.html).

## StandardScaler

Standardize by removing the mean and scaling to unit variance. **Population** standard deviation (`ddof = 0`), matching sklearn.

```rust
use datarust::scaler::StandardScaler;
use datarust::traits::Transformer;

let mut s = StandardScaler::new()
    .with_mean(true)   // default: center
    .with_std(true);   // default: scale to unit variance
let out = s.fit_transform(&x)?;
```

## MinMaxScaler

Scale each feature to a given range (default `[0, 1]`).

```rust
use datarust::scaler::MinMaxScaler;

let mut s = MinMaxScaler::new().feature_range(-1.0, 1.0);
let out = s.fit_transform(&x)?;
```

## RobustScaler

Outlier-robust scaling using the median and interquartile range.

```rust
use datarust::scaler::RobustScaler;

let mut s = RobustScaler::new()
    .with_centering(true)
    .with_scaling(true);
let out = s.fit_transform(&x)?;
```

## MaxAbsScaler

Scale by dividing by the maximum absolute value per feature. Preserves sparsity.

## Normalizer

Row-wise normalization (not column-wise). Each **sample** is scaled to unit norm.

```rust
use datarust::scaler::{Normalizer, Norm};

let mut n = Normalizer::new().norm(Norm::L2); // or Norm::L1, Norm::Max
```

## Binarizer

Threshold features to 0/1.

```rust
use datarust::scaler::Binarizer;

let mut b = Binarizer::new().threshold(0.5);
```

## KBinsDiscretizer

Continuous-to-discrete bin discretization.

```rust
use datarust::scaler::{KBinsDiscretizer, BinStrategy, KBinsEncode};

let mut k = KBinsDiscretizer::new()
    .strategy(BinStrategy::Quantile) // or Uniform, KMeans
    .encode(KBinsEncode::Ordinal)    // or OneHotDense
    .n_bins(5);
```

## QuantileTransformer

Transform features to follow a uniform or normal distribution. Robust to outliers.

```rust
use datarust::scaler::{QuantileTransformer, OutputDistribution};

let mut q = QuantileTransformer::new()
    .output(OutputDistribution::Normal); // or Uniform
```

## PowerTransformer

Gaussianize features via Yeo-Johnson or Box-Cox, with automatic lambda estimation.

```rust
use datarust::scaler::{PowerTransformer, PowerMethod};

let mut p = PowerTransformer::new().method(PowerMethod::YeoJohnson); // or BoxCox
```

## When to use which?

| Scenario | Recommended scaler |
|---|---|
| Normal-ish data, no outliers | `StandardScaler` |
| Bounded range needed (e.g. neural nets) | `MinMaxScaler` |
| Data with outliers | `RobustScaler` |
| Sparse data | `MaxAbsScaler` (preserves zeros) |
| Non-Gaussian → Gaussian needed | `PowerTransformer` or `QuantileTransformer` |
| Sample-level normalization | `Normalizer` |
