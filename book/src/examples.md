# Examples

Runnable examples from the `examples/` directory. Each can be run with `cargo run --example <name>`.

## Basic preprocessing

`cargo run --example basic_preprocessing`

Standardize, scale, and encode a small mixed-type dataset. Demonstrates `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, and the `Transformer` trait.

## Pipeline workflow

`cargo run --example pipeline_workflow`

Build a 3-step pipeline (StandardScaler → MinMaxScaler → RobustScaler), fit and transform, then inspect individual steps via the runtime ergonomics API.

## Target encoding

`cargo run --example target_encoding`

Use `TargetEncoder` with smoothed mean encoding on a high-cardinality categorical feature, including the `TargetTransformer` trait and `fit_with_target`.

## Regression workflow

`cargo run --example regression_workflow`

End-to-end regression on synthetic house-price data: generate data, `train_test_split`, fit a `StandardScaler` **on train only** (data-leakage prevention), train a `Ridge`, score test R², then run 5-fold `cross_val_score`. Shows the full preprocess → train → evaluate → cross-validate loop on a single realistic dataset.

## Classification workflow

`cargo run --example classification_workflow`

Customer-churn binary classification with mixed numeric + categorical features. A `ColumnTransformer` applies `StandardScaler` to numeric columns and `OneHotEncoder` to the categorical column, feeding a `LogisticRegression`. Reports the full metric suite (`accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`, `log_loss`), compares decision thresholds (0.3 / 0.5 / 0.7) to show the precision–recall trade-off, and finishes with **stratified** 5-fold cross-validation driven manually (`StratifiedKFold::split`, since `cross_val_score` only accepts `KFold`).

## Regularization comparison

`cargo run --example regularization_comparison`

Compares `Ridge` (L2) and `Lasso` (L1) across multiple `alpha` values on an 8-feature dataset where only 3 features carry signal, one feature is collinear, and the rest are pure noise. Ridge shrinks all coefficients but zeros none; Lasso's soft-thresholding drives the noise features to **exactly zero** (automatic feature selection). The collinear feature would make `LinearRegression` singular — both regularized solvers handle it.

## Model persistence

`cargo run --example model_persistence --features serde`

Production-style model persistence. Trains a `SupervisedPipeline<Ridge>` (StandardScaler → PCA → Ridge), writes it to a JSON file with `save_json`, reloads it with `load_json`, and confirms the restored model is still `is_fitted()` and produces bit-identical predictions **without refitting**. Requires the `serde` feature.

## KMeans clustering

`cargo run --example kmeans_clustering`

Unsupervised clustering on synthetic 2-D point data. Generates three well-separated blobs, fits a `KMeans` with k-means++ initialization, inspects the learned centroids against the true blob centers, predicts cluster assignments for new points, compares k-means++ vs random initialization by inertia, and (with `--features serde`) round-trips the fitted model through JSON. Demonstrates the `Clusterer` trait.

## Benchmark comparison

`cargo run --release --example bench_compare_rust`

The deterministic benchmark harness that produces the numbers on the [Performance](./performance.md) page. Uses a xorshift64 PRNG (seed 42) so Rust and Python generate identical data. Pass a repetition count as an argument:

```sh
cargo run --release --features matrixmultiply --example bench_compare_rust 15
```
