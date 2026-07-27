---
title: "KMeans Picked k=4 Because Inertia Kept Dropping. Silhouette Said k=3."
subtitle: "A clustering metric that actually measures cluster quality"
author: "Murat Genc"
date: "2026-07-26"
tags: ["machine-learning", "rust", "datarust", "clustering", "unsupervised"]
series: "datarust-v06"
---

Every KMeans tutorial ends with "plot the elbow and pick where the curve bends." The problem: the elbow is almost never sharp. Inertia drops monotonically with k, so you're squinting at a curve trying to find a拐点 that might not exist. Silhouette score replaces that ambiguity with a number that actually measures cluster quality.

## The Metric

For each sample $i$, silhouette computes:

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$

where $a_i$ is the mean distance to other points in the same cluster, and $b_i$ is the mean distance to the nearest other cluster. The score ranges from -1 to +1: high values mean well-separated clusters, near-zero means overlapping, negative means the sample is in the wrong cluster.

The overall silhouette score is the mean of all $s_i$.

## Experiment 1: Three Well-Separated Blobs

Three clusters at `(0,0)`, `(10,10)`, `(20,20)` with standard deviation 0.5:

```
=== Three well-separated blobs ===
True labels silhouette: 0.9327

KMeans with different k:
k    inertia    silhouette
2        3025.6      0.7172
3          49.8      0.9327
4          41.8      0.7372
5          33.5      0.5497
```

Inertia keeps dropping (3025 → 49 → 41 → 33), but silhouette peaks sharply at k=3 with 0.9327. This is what a clear signal looks like: the silhouette curve has a genuine maximum, not a vague elbow.

Notice k=2 gets 0.7172 — not bad, because two of the three clusters are still well-separated from each other. But k=3 is unambiguously better.

## Experiment 2: Overlapping Blobs

Two clusters at `(0,0)` and `(3,0)` with standard deviation 2.0:

```
=== Two overlapping blobs ===
True labels silhouette: 0.1594

KMeans with different k:
k    inertia    silhouette
2         503.3      0.4185
3         346.3      0.4032
4         258.2      0.4240
```

True labels get only 0.1594 — the clusters genuinely overlap. KMeans with k=2 gets 0.4185, which is higher than the true labels because KMeans forces a hard boundary that creates the *illusion* of separation. This is the first trap: silhouette measures geometric separation, not ground-truth accuracy.

All k values produce similar silhouette scores (0.41-0.42), which is the real signal: there's no clear cluster structure here.

## Experiment 3: Choosing k

Three clusters at `(0,0)`, `(8,0)`, `(4,7)` with standard deviation 0.8:

```
k    silhouette  inertia
2        0.5475      1813.8
3        0.8194       182.7
4        0.6603       156.6
5        0.5106       130.7
6        0.3455      113.0

Best k by silhouette: 3 (score: 0.8194)
```

Inertia drops continuously: 1813 → 182 → 156 → 130 → 113. No elbow. But silhouette has a clear peak at k=3 with 0.8194, then drops sharply. This is the practical value: you don't need to guess where the elbow is.

The drop from k=3 to k=4 (0.8194 → 0.6603) is substantial — KMeans is splitting a natural cluster, and silhouette detects it.

## Experiment 4: When There Are No Clusters

Single blob centered at `(5,5)` with standard deviation 1.0:

```
=== Single blob (no natural clusters) ===
Artificial split silhouette: -0.0164 (low = no real separation)

KMeans on single blob:
k    silhouette  inertia
2        0.3850        52.8
3        0.3629        38.1
4        0.3600        28.3
```

All k values produce similar, low silhouette scores (~0.36-0.38). No peak. This is the "I forced KMeans on data without clusters" scenario, and silhouette correctly tells you there's nothing meaningful here.

The artificial split (first half vs second half) gets -0.0164 — essentially random, which is what you'd expect from splitting a single distribution.

## When Silhouette Misleads

**Small datasets.** Silhouette is unstable with fewer than ~50 samples per cluster. The metric averages over all samples, and a few outliers can swing the score.

**Non-spherical clusters.** Silhouette uses Euclidean distance, so it assumes roughly spherical clusters. Elongated or crescent-shaped clusters will get low scores even if KMeans finds them correctly.

**High-dimensional data.** Distance concentration in high dimensions makes all inter-point distances similar, pushing silhouette toward zero regardless of cluster quality.

**Unequal cluster sizes.** A tiny cluster surrounded by a large one can get high silhouette even if the separation isn't meaningful.

## The Inertia Trap

The standard advice is "look for the elbow in the inertia plot." The problem:

```text
k=2:  inertia=3025.6
k=3:  inertia=  49.8   ← 98% drop!
k=4:  inertia=  41.8   ← 16% drop
k=5:  inertia=  33.5   ← 20% drop
```

The 98% drop from k=2 to k=3 looks like an elbow, but the subsequent drops are also significant. Without silhouette, you'd have to subjectively decide which drop is "the elbow." With silhouette, k=3 scores 0.9327 and k=4 scores 0.7372 — the choice is quantitative, not visual.

## Practical Pattern

```rust
use datarust::cluster::metrics::silhouette_score;
use datarust::cluster::KMeans;
use datarust::traits::Clusterer;

let mut best_k = 2;
let mut best_score = f64::NEG_INFINITY;

for k in 2..=10 {
    let mut km = KMeans::new()
        .with_n_clusters(k)
        .with_n_init(10)
        .with_random_state(42);
    let labels = km.fit_predict(&x)?;
    let score = silhouette_score(&x, &labels)?;
    if score > best_score {
        best_score = score;
        best_k = k;
    }
}
```

This is the "silhouette method" for choosing k: try a range, pick the k with the highest score. It's not foolproof, but it's more reliable than squinting at inertia curves.

## Tradeoffs

Silhouette is $O(n^2)$ in the worst case (it computes pairwise distances), which makes it expensive for large datasets. The datarust implementation uses a single-pass algorithm that's faster in practice but still quadratic. For datasets with more than ~10,000 samples, consider subsampling before computing silhouette.

The alternative is the "silhouette samples" variant, which returns per-sample scores instead of the mean. This is useful for detecting which specific samples are misclassified, but costs the same $O(n^2)$ computation.

For large datasets, the practical approach is: use inertia for the initial k selection, then compute silhouette on a subsample to validate.

## Try It

```bash
cargo add datarust
```

```rust
use datarust::cluster::metrics::silhouette_score;
use datarust::cluster::KMeans;
use datarust::traits::Clusterer;
use datarust::Matrix;

let x = Matrix::new(vec![
    vec![0.0, 0.0], vec![0.1, 0.1], vec![0.0, 0.1],
    vec![10.0, 10.0], vec![10.1, 10.1], vec![10.0, 10.1],
]).unwrap();

let mut km = KMeans::new().with_n_clusters(2).with_n_init(10);
let labels = km.fit_predict(&x).unwrap();
let score = silhouette_score(&x, &labels).unwrap();
println!("silhouette = {score:.4}");
```

If you're still plotting inertia curves to choose k, you're doing it wrong. Silhouette gives you a number. Use it.
