# Encoders

Categorical encoding transformers. Live in [`datarust::encoder`](https://docs.rs/datarust/latest/datarust/encoder/index.html).

## LabelEncoder

1-D string ↔ int mapping for target labels. Implements [`LabelTransformer`](https://docs.rs/datarust/latest/datarust/trait.LabelTransformer.html).

```rust
use datarust::encoder::LabelEncoder;
use datarust::traits::LabelTransformer;

let labels = vec!["cat".to_string(), "dog".to_string(), "cat".to_string()];
let mut le = LabelEncoder::new();
let codes = le.fit_transform(&labels)?;   // [0, 1, 0]
let back = le.inverse_transform(&codes)?; // ["cat", "dog", "cat"]
```

## OneHotEncoder

Encode categorical features as a one-hot numeric matrix. Supports dense and CSR sparse output.

```rust
use datarust::encoder::{OneHotEncoder, HandleUnknown};

let mut ohe = OneHotEncoder::new()
    .handle_unknown(HandleUnknown::Ignore) // or Error
    .sparse_output(true);                  // CSR SparseMatrix output
let out = ohe.fit_transform(&str_matrix)?;
```

The CSR [`SparseMatrix`](https://docs.rs/datarust/latest/datarust/struct.SparseMatrix.html) output stores only the `1.0` positions, saving significant memory for high-cardinality columns.

## OrdinalEncoder

Encode categorical features as integer codes with optional user-defined ordering.

```rust
use datarust::encoder::{OrdinalEncoder, OrdinalCategories};

// Auto-detect categories from data:
let mut oe = OrdinalEncoder::new();

// Or specify explicit ordering (e.g. "low" < "medium" < "high"):
let mut oe = OrdinalEncoder::new()
    .categories(OrdinalCategories::Manual(vec![vec!["low".into(), "medium".into(), "high".into()]]));
```

## TargetEncoder

Supervised encoder: replaces categories with the smoothed mean of the target. Useful for high-cardinality features. Implements [`TargetTransformer`](https://docs.rs/datarust/latest/datarust/trait.TargetTransformer.html) — requires `y` during fit.

```rust
use datarust::encoder::{TargetEncoder, UnknownTarget};

let mut te = TargetEncoder::new()
    .smoothing(1.0)
    .handle_unknown(UnknownTarget::GlobalMean); // or NaN, Error
let out = te.fit_transform(&str_matrix, &y)?;
```

## FrequencyEncoder

Replace categories with their count or proportion in the training data.

```rust
use datarust::encoder::{FrequencyEncoder, UnknownFrequency};

let mut fe = FrequencyEncoder::new()
    .proportion(true)                          // proportion vs raw count
    .handle_unknown(UnknownFrequency::Zero);   // or Error
```

## Choosing an encoder

| Situation | Encoder |
|---|---|
| Target labels (1-D) | `LabelEncoder` |
| Low-cardinality features | `OneHotEncoder` |
| High-cardinality features | `TargetEncoder` |
| Ordinal relationships | `OrdinalEncoder` (with manual ordering) |
| Count-based signal | `FrequencyEncoder` |
