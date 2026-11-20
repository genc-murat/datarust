# Nearest Neighbors

k-Nearest-Neighbor classification and regression. Mirrors `sklearn.neighbors`.
Live in [`datarust::neighbors`](https://docs.rs/datarust/latest/datarust/neighbors/index.html).

The first release is brute-force Euclidean — the same distance kernel behind
`KnnImputer` — with tree acceleration (ball/KD) explicitly planned post-0.8.

## KNeighborsClassifier

Majority vote (or distance-weighted vote) over the `k` nearest training rows.

```rust
use datarust::neighbors::{KNeighborsClassifier, KnnWeights};
use datarust::traits::{Classifier, PredictProba, Predictor};

let mut knn = KNeighborsClassifier::new()
    .with_n_neighbors(5)          // sklearn default
    .with_weights(KnnWeights::Uniform);  // or KnnWeights::Distance

knn.fit(&x, &y)?;
let pred = knn.predict(&x_new)?;
let proba = knn.predict_proba(&x_new)?;  // (n_samples, n_classes), classes() order
let acc = knn.score(&x_te, &y_te)?;
```

Aggregation rules (all sklearn-parity-tested):

- **Uniform**: majority vote; exact ties break to the smallest winning class
  (sklearn's `argmax` over the sorted `classes_`).
- **Distance**: each neighbor votes with inverse Euclidean weight `1/d` —
  **except** when any neighbor coincides with the query (zero distance): then
  the vote is winner-takes-all, only those neighbors vote. This reproduces
  sklearn's `_get_weights` exactly.
- `predict_proba` returns vote fractions (uniform) or normalized vote weights
  (distance), in sorted-class order. Gapped labels (`{2, 5, 9}`) are preserved.

## KNeighborsRegressor

Mean (or distance-weighted mean) of the neighbors' targets, with the same
weighting rules as the classifier.

```rust
use datarust::neighbors::{KNeighborsRegressor, KnnWeights};
use datarust::traits::{Predictor, Regressor};

let mut knn = KNeighborsRegressor::new()
    .with_n_neighbors(5)
    .with_weights(KnnWeights::Distance);

knn.fit(&x, &y)?;
let pred = knn.predict(&x_new)?;
let r2 = knn.score(&x_te, &y_te)?;
```

## Notes

- `n_neighbors` and `weights` are **prediction-only** hyperparameters: they
  can be changed after `fit` (and tuned via `GridSearchCV` /
  `RandomizedSearchCV` — both estimators implement `Params`).
- `n_neighbors` must not exceed the training sample count; `fit` validates
  up front and `predict` re-checks after post-fit parameter changes (sklearn
  errors in `kneighbors` the same way).
- Distance ties are resolved deterministically by training order, so every
  fit/predict is reproducible.
- Features must be finite (`NaN`/inf rejected like the other estimators).
