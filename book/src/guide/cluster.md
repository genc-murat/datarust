# Clustering

Unsupervised clustering estimators. Live in [`datarust::cluster`](https://docs.rs/datarust/latest/datarust/cluster/index.html). All implement the [`Clusterer`](https://docs.rs/datarust/latest/datarust/trait.Clusterer.html) trait — the unsupervised counterpart to `Predictor`. `fit` takes only `X` (no target `y`); `predict` returns cluster indices as `Vec<usize>`.

## KMeans

k-means clustering via Lloyd's algorithm with k-means++ initialization. Partitions `n` observations into `k` clusters by minimizing within-cluster sum of squares (inertia). Each iteration assigns every point to its nearest centroid, then recomputes centroids as the mean of their members.

```rust
use datarust::cluster::{KMeans, KMeansInit};
use datarust::traits::Clusterer;

let mut km = KMeans::new()
    .with_n_clusters(3)                         // default 8
    .with_init(KMeansInit::KMeansPlusPlus)      // default; or KMeansInit::Random
    .with_n_init(10)                            // restarts, keep best (default 10)
    .with_max_iter(300)                         // default 300
    .with_tol(1e-4)                             // default 1e-4
    .with_random_state(42);                     // deterministic seed

let labels = km.fit_predict(&x)?;               // Vec<usize>, one cluster per row
let centers = km.cluster_centers();             // &[Vec<f64>], one centroid per cluster
let inertia = km.inertia();                     // f64, sum of squared distances
let iters = km.n_iter();                        // Lloyd's iterations of best run

let new_labels = km.predict(&new_x)?;           // assign new points to nearest centroid
```

**Initialization strategies** (`KMeansInit`):

- `KMeansPlusPlus` (default) — first centroid is uniform random; each subsequent
  centroid is sampled with probability proportional to its squared distance
  from the nearest already-chosen centroid. Produces well-spread initial
  centroids and is the scikit-learn default.
- `Random` — choose `n_clusters` distinct points uniformly at random from the
  data.

**How it works:**

1. Initialize `k` centroids using the chosen strategy.
2. **Assignment step** — each point is assigned to its nearest centroid (lowest
   squared Euclidean distance).
3. **Update step** — each centroid is recomputed as the mean of its members.
   Empty clusters keep their previous centroid.
4. Repeat until centroid movement drops below `tol` or `max_iter` is reached.
5. Steps 1–4 run `n_init` times with different seeds; the lowest-inertia result
   is kept.

**Reproducibility:** `with_random_state(seed)` makes results fully
deterministic — the same seed yields identical labels, centroids, and inertia.
KMeans uses the crate's internal xorshift64 PRNG, preserving the
zero-dependency ethos (no `rand` crate).

**Serialization:** KMeans derives `Serialize`/`Deserialize` under the `serde`
feature — fitted centroids, labels, and inertia round-trip through JSON, and
the restored model serves predictions without refitting.

## Choosing a clustering algorithm

| Goal | Estimator |
|---|---|
| Spherical, equally-sized blobs | `KMeans` |
| Arbitrary-shape clusters | _(DBSCAN — planned, see [Roadmap](../roadmap.md))_ |
| Hierarchical structure | _(AgglomerativeClustering — planned)_ |

## Evaluating clusters

Without ground-truth labels, the **silhouette score** assesses clustering
quality from the data alone:

```rust
use datarust::cluster::metrics::silhouette_score;

let s = silhouette_score(&x, &labels)?;  // f64 in [-1, 1]
```

For each sample, the silhouette coefficient is `(b − a) / max(a, b)` where `a`
is the mean intra-cluster distance and `b` is the mean distance to the nearest
other cluster. Values near `1` indicate well-separated clusters; near `0`
indicate overlapping clusters. The `inertia()` (within-cluster sum of squares,
lower is better) is also useful for comparing runs with different
`n_clusters` or `random_state` values.
