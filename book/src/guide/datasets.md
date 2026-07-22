# Datasets

Classic toy datasets for examples, tests, and onboarding. Live in
[`datarust::datasets`](https://docs.rs/datarust/latest/datarust/datasets/index.html).
Enable with the `datasets` feature:

```toml
[dependencies]
datarust = { version = "*", features = ["datasets"] }
```

The data is compiled into the binary as `const` arrays — no file I/O, no
network access, no external dependencies. Each loader returns a [`Dataset`]
struct with `features()`, `targets()`, `feature_names()`, and
`target_names()`.

[`Dataset`]: https://docs.rs/datarust/latest/datarust/datasets/struct.Dataset.html

## Available datasets

| Dataset | Samples | Features | Classes | Task |
|---|---|---|---|---|
| **Iris** | 150 | 4 | 3 (50 each) | Classification |
| **Breast Cancer** | 569 | 30 | 2 (357 / 212) | Binary classification |
| **Wine** | 178 | 13 | 3 | Multiclass classification |
| **Diabetes** | 442 | 10 | — | Regression |

## Usage

```rust
use datarust::datasets;

let iris = datasets::iris::load();

let x = iris.features();           // Matrix 150×4
let y = iris.targets();            // &[f64], values {0, 1, 2}
let names = iris.feature_names();  // &["sepal_length", "sepal_width", ...]
let classes = iris.target_names(); // &["setosa", "versicolor", "virginica"]

assert_eq!(iris.n_samples(), 150);
assert_eq!(iris.n_features(), 4);
assert_eq!(iris.n_classes(), 3);
```

Feed directly into a model:

```rust
use datarust::datasets::iris;
use datarust::linear_model::LogisticRegression;
use datarust::traits::Predictor;

let data = iris::load();
let x = data.features();
let y = data.targets().to_vec();

let mut model = LogisticRegression::new().with_max_iter(200);
model.fit(&x, &y)?;
let accuracy = model.score(&x, &y)?;
println!("Iris accuracy: {:.1}%", accuracy * 100.0);
```

## Choosing a dataset

| Goal | Dataset |
|---|---|
| Quick multiclass classification demo | **Iris** — small, fast, well-separated |
| Binary classification benchmark | **Breast Cancer** — 30 features, imbalanced |
| Multiclass with more features | **Wine** — 13 chemical features |
| Regression baseline | **Diabetes** — continuous target, 10 features |
