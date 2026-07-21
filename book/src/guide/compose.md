# Compose: Pipeline & ColumnTransformer

Chain transformers and dispatch different columns to different transformers. Live in [`datarust::pipeline`](https://docs.rs/datarust/latest/datarust/pipeline/index.html) and [`datarust::compose`](https://docs.rs/datarust/latest/datarust/compose/index.html).

## Pipeline

Chain multiple transformers sequentially. Each step fits on the output of the previous one.

```rust
use datarust::pipeline::Pipeline;
use datarust::scaler::{StandardScaler, MinMaxScaler, RobustScaler};
use datarust::transformer_kind::TransformerKind;

let mut pipe = Pipeline::new()
    .push("standard", TransformerKind::StandardScaler(StandardScaler::new()))
    .push("minmax",   TransformerKind::MinMaxScaler(MinMaxScaler::new()))
    .push("robust",   TransformerKind::RobustScaler(RobustScaler::new()));

let out = pipe.fit_transform(&x)?;
```

Pipelines are **serializable** under the `serde` feature — fit, save to JSON, load and `transform` in a different process.

## Supervised pipeline

Attach a final estimator with `with_estimator` to fit preprocessing and a model
together. During `fit(X, y)`, supervised selectors such as `SelectKBest` receive
the training targets before the final estimator is fitted.

```rust
use datarust::linear_model::LogisticRegression;
use datarust::pipeline::Pipeline;
use datarust::selection::{ScoreFunc, SelectKBest};
use datarust::traits::Predictor;
use datarust::transformer_kind::TransformerKind;

let mut model = Pipeline::new()
    .push("select", TransformerKind::SelectKBest(SelectKBest::new(ScoreFunc::FClassif, 5)?))
    .with_estimator(LogisticRegression::new());
model.fit(&x, &y)?;
let labels = model.predict(&x)?;
```

### Runtime step inspection

```rust
pipe.names();           // ["standard", "minmax", "robust"]
pipe.get_step("minmax");  // Option<&TransformerKind>
pipe.set_step("robust", TransformerKind::StandardScaler(StandardScaler::new()));
pipe.insert_step(0, "impute", /* ... */);
pipe.remove_step("standard");
```

## ColumnTransformer

Dispatch different columns to different transformers — the workhorse for **mixed numeric + categorical** data.

```rust
use datarust::compose::{ColumnTransformer, Remainder, Table};
use datarust::encoder::OneHotEncoder;
use datarust::scaler::StandardScaler;
use datarust::categorical_kind::CategoricalTransformerKind;
use datarust::transformer_kind::TransformerKind;

let mut ct = ColumnTransformer::new()
    .add_numeric(
        "nums",
        vec![0, 1, 2],           // column indices
        TransformerKind::StandardScaler(StandardScaler::new()),
    )
    .add_categorical(
        "cats",
        vec![3, 4],
        CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new()),
    )
    .remainder(Remainder::Passthrough); // keep un-specified columns

// Table bundles numeric (Matrix) + categorical (StrMatrix) of equal row count
let out = ct.fit_transform_to_table(&table)?;
// out.numeric    — the scaled + one-hot-encoded numeric block
// out.categorical — the transformed categorical block
```

### Supervised encoders

For `TargetEncoder`, use `fit_with_target` / `fit_transform_with_target`:

```rust
ct.fit_transform_with_target(&table, &y)?;
```

## When to use what

| Need | Use |
|---|---|
| Sequential transforms on one matrix | `Pipeline` |
| Preprocessing + final supervised estimator | `Pipeline::with_estimator` / `SupervisedPipeline` |
| Different columns → different transformers | `ColumnTransformer` |
| Mixed numeric + categorical in one call | `ColumnTransformer` + `Table` |
| Serializable fitted pipeline | `Pipeline` or `ColumnTransformer` + `serde` |
