# API Reference (docs.rs)

The complete API reference for datarust is hosted on **[docs.rs](https://docs.rs/datarust)** — the standard home for Rust crate documentation.

> [**→ Open the full API reference on docs.rs**](https://docs.rs/datarust)

## What's there

docs.rs builds the rustdoc documentation with **all features enabled** (`serde`, `rayon`, `matrixmultiply`, `datasets`), so you see the complete API surface including:

- Every `pub struct`, `pub enum`, `pub trait`, and `pub fn` with doc comments.
- Type signatures, trait implementations, and method docs.
- Inline code examples from doc comments (runnable via `cargo test --doc`).
- Cross-links between types and modules.

## Direct links to key modules

| Module | docs.rs link |
|---|---|
| `scaler` | [datarust::scaler](https://docs.rs/datarust/latest/datarust/scaler/index.html) |
| `encoder` | [datarust::encoder](https://docs.rs/datarust/latest/datarust/encoder/index.html) |
| `imputer` | [datarust::imputer](https://docs.rs/datarust/latest/datarust/imputer/index.html) |
| `decomposition` | [datarust::decomposition](https://docs.rs/datarust/latest/datarust/decomposition/index.html) |
| `linear_model` | [datarust::linear_model](https://docs.rs/datarust/latest/datarust/linear_model/index.html) |
| `metrics` | [datarust::metrics](https://docs.rs/datarust/latest/datarust/metrics/index.html) |
| `model_selection` | [datarust::model_selection](https://docs.rs/datarust/latest/datarust/model_selection/index.html) |
| `compose` | [datarust::compose](https://docs.rs/datarust/latest/datarust/compose/index.html) |
| `cluster` | [datarust::cluster](https://docs.rs/datarust/latest/datarust/cluster/index.html) |
| `datasets` | [datarust::datasets](https://docs.rs/datarust/latest/datarust/datasets/index.html) |
| `traits` | [datarust::traits](https://docs.rs/datarust/latest/datarust/traits/index.html) |

## Direct links to key traits

| Trait | docs.rs link |
|---|---|
| `Transformer` | [datarust::Transformer](https://docs.rs/datarust/latest/datarust/trait.Transformer.html) |
| `Regressor` | [datarust::Regressor](https://docs.rs/datarust/latest/datarust/trait.Regressor.html) |
| `Predictor` | [datarust::Predictor](https://docs.rs/datarust/latest/datarust/trait.Predictor.html) |
| `Classifier` | [datarust::Classifier](https://docs.rs/datarust/latest/datarust/trait.Classifier.html) |
| `PredictProba` | [datarust::PredictProba](https://docs.rs/datarust/latest/datarust/trait.PredictProba.html) |
| `Clusterer` | [datarust::Clusterer](https://docs.rs/datarust/latest/datarust/trait.Clusterer.html) |
| `Params` | [datarust::Params](https://docs.rs/datarust/latest/datarust/trait.Params.html) |
| `CategoricalTransformer` | [datarust::CategoricalTransformer](https://docs.rs/datarust/latest/datarust/trait.CategoricalTransformer.html) |
| `FeatureNames` | [datarust::FeatureNames](https://docs.rs/datarust/latest/datarust/trait.FeatureNames.html) |

## Building locally

You can generate the same docs offline:

```sh
cargo doc --all-features --no-deps --open
```

The `--open` flag opens the generated HTML in your browser.
