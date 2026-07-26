# One KMeans Run Split the Big Cluster and Merged the Two Small Ones.

*A practical datarust experiment with local minima, random seeds, k-means++, multiple restarts, and the segmentation that looked valid until I counted who was inside it.*

---

I asked KMeans for three clusters.

The dataset contained three groups: one large group of 200 observations and two smaller groups of 40 each.

One run found exactly that:

```text
[200] [40] [40]
```

Another run, using the same rows and the same `k = 3`, produced this:

```text
[115] [85] [40 + 40]
```

It split the large group in half and merged the two small groups together.

Nothing crashed. Lloyd's algorithm converged. Every row received a cluster label, all three clusters were non-empty, and the model could predict labels for new observations.

It had simply reached a worse local minimum.

Across 100 random seeds with one initialization each, only 23 random-initialized runs recovered the known groups. Single-run k-means++ did not rescue this deliberately awkward geometry: it succeeded for 19 of 100 seeds.

With 20 restarts, both initialization methods found the best solution for all 100 base seeds in this experiment.

That is why I treat `random_state` as a reproducibility control and `n_init` as optimization insurance. A fixed seed can reproduce a bad answer perfectly.

Let's measure the entire failure with [datarust](https://crates.io/crates/datarust).

## Three groups with unequal sizes

The two-dimensional dataset contains:

```text
group 0: 200 points around (0, 0), spread 0.80
group 1:  40 points around (8,-2), spread 0.45
group 2:  40 points around (8, 2), spread 0.45
```

The groups are intentionally unbalanced. Most randomly selected observations come from the large left-hand blob. An unlucky initialization can place two centers there and only one on the right.

Once that happens, Lloyd's iterations may settle into a stable compromise:

- Two centroids divide the large blob.
- One centroid sits between the two smaller blobs.
- No update step moves a centroid across the empty space to separate the small groups.

I run four seed sweeps:

1. Random initialization, one run per seed
2. k-means++ initialization, one run per seed
3. Random initialization, 20 restarts per base seed
4. k-means++ initialization, 20 restarts per base seed

Because this data is synthetic, I retain the true group labels and calculate the best label-permutation agreement. In production clustering there is usually no such answer key; it exists here so we can distinguish “found a low objective” from “recovered the structure we planted.”

Here is the complete Rust program:

```rust
use datarust::cluster::{
    metrics::silhouette_score, KMeans, KMeansInit,
};
use datarust::traits::Clusterer;
use datarust::Matrix;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 as f64 + 1.0)
            / (u64::MAX as f64 + 2.0)
    }

    fn normal(&mut self) -> f64 {
        let u1 =
            self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt()
            * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn add_blob(
    rows: &mut Vec<Vec<f64>>,
    truth: &mut Vec<usize>,
    rng: &mut Rng,
    n: usize,
    center: [f64; 2],
    spread: f64,
    label: usize,
) {
    for _ in 0..n {
        rows.push(vec![
            center[0] + spread * rng.normal(),
            center[1] + spread * rng.normal(),
        ]);
        truth.push(label);
    }
}

fn best_accuracy(
    predicted: &[usize],
    truth: &[usize],
) -> f64 {
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    permutations
        .iter()
        .map(|mapping| {
            predicted
                .iter()
                .zip(truth)
                .filter(|(p, t)| mapping[**p] == **t)
                .count()
        })
        .max()
        .unwrap_or(0) as f64
        / predicted.len() as f64
}

#[derive(Clone)]
struct Run {
    seed: u64,
    inertia: f64,
    silhouette: f64,
    accuracy: f64,
}

fn run(
    x: &Matrix,
    truth: &[usize],
    init: KMeansInit,
    seed: u64,
    n_init: usize,
) -> Run {
    let mut model = KMeans::new()
        .with_n_clusters(3)
        .with_init(init)
        .with_n_init(n_init)
        .with_random_state(seed);
    let labels = model.fit_predict(x).unwrap();

    Run {
        seed,
        inertia: model.inertia(),
        silhouette: silhouette_score(x, &labels).unwrap(),
        accuracy: best_accuracy(&labels, truth),
    }
}

fn summarize(name: &str, runs: &[Run]) {
    let mut sorted = runs.to_vec();
    sorted.sort_by(|a, b| {
        a.inertia.total_cmp(&b.inertia)
    });
    let good = runs
        .iter()
        .filter(|run| run.accuracy > 0.95)
        .count();

    println!(
        "{name:<15} good seeds {:>3}/{} | \
         inertia min/median/max {:>8.2} / {:>8.2} / {:>8.2}",
        good,
        runs.len(),
        sorted[0].inertia,
        sorted[sorted.len() / 2].inertia,
        sorted[sorted.len() - 1].inertia,
    );
}

fn print_run(name: &str, run: &Run) {
    println!(
        "{name:<20} seed {:>2} | inertia {:>8.2} | \
         silhouette {:.3} | truth agreement {:.1}%",
        run.seed,
        run.inertia,
        run.silhouette,
        run.accuracy * 100.0,
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Rng::new(2026);
    let mut rows = Vec::new();
    let mut truth = Vec::new();

    add_blob(
        &mut rows,
        &mut truth,
        &mut rng,
        200,
        [0.0, 0.0],
        0.8,
        0,
    );
    add_blob(
        &mut rows,
        &mut truth,
        &mut rng,
        40,
        [8.0, -2.0],
        0.45,
        1,
    );
    add_blob(
        &mut rows,
        &mut truth,
        &mut rng,
        40,
        [8.0, 2.0],
        0.45,
        2,
    );
    let x = Matrix::new(rows)?;

    let random_runs: Vec<Run> = (0..100)
        .map(|seed| {
            run(
                &x,
                &truth,
                KMeansInit::Random,
                seed,
                1,
            )
        })
        .collect();
    let plus_runs: Vec<Run> = (0..100)
        .map(|seed| {
            run(
                &x,
                &truth,
                KMeansInit::KMeansPlusPlus,
                seed,
                1,
            )
        })
        .collect();
    let random_restarted: Vec<Run> = (0..100)
        .map(|seed| {
            run(
                &x,
                &truth,
                KMeansInit::Random,
                seed,
                20,
            )
        })
        .collect();
    let plus_restarted: Vec<Run> = (0..100)
        .map(|seed| {
            run(
                &x,
                &truth,
                KMeansInit::KMeansPlusPlus,
                seed,
                20,
            )
        })
        .collect();

    summarize("random x1", &random_runs);
    summarize("k-means++ x1", &plus_runs);
    summarize("random x20", &random_restarted);
    summarize("k-means++ x20", &plus_restarted);

    let best_random = random_runs
        .iter()
        .min_by(|a, b| a.inertia.total_cmp(&b.inertia))
        .unwrap();
    let worst_random = random_runs
        .iter()
        .max_by(|a, b| a.inertia.total_cmp(&b.inertia))
        .unwrap();

    println!();
    print_run("best random run", best_random);
    print_run("worst random run", worst_random);

    let mut restarted = KMeans::new()
        .with_n_clusters(3)
        .with_init(KMeansInit::Random)
        .with_n_init(20)
        .with_random_state(worst_random.seed);
    let restarted_labels =
        restarted.fit_predict(&x)?;
    let restarted_run = Run {
        seed: worst_random.seed,
        inertia: restarted.inertia(),
        silhouette: silhouette_score(
            &x,
            &restarted_labels,
        )?,
        accuracy: best_accuracy(
            &restarted_labels,
            &truth,
        ),
    };
    print_run("same seed, 20 runs", &restarted_run);

    println!(
        "\ncluster compositions \
         (rows are predicted clusters; columns are true groups)"
    );

    for (name, init, seed, n_init) in [
        (
            "worst single",
            KMeansInit::Random,
            worst_random.seed,
            1,
        ),
        (
            "20 restarts",
            KMeansInit::Random,
            worst_random.seed,
            20,
        ),
    ] {
        let mut model = KMeans::new()
            .with_n_clusters(3)
            .with_init(init)
            .with_n_init(n_init)
            .with_random_state(seed);
        let labels = model.fit_predict(&x)?;
        let mut counts = [[0usize; 3]; 3];

        for (&pred, &actual) in labels.iter().zip(&truth) {
            counts[pred][actual] += 1;
        }
        println!("{name:<12}: {:?}", counts);
    }

    Ok(())
}
```

This is the output I measured:

```text
random x1       good seeds  23/100 | inertia min/median/max   320.53 /   542.61 /   549.15
k-means++ x1    good seeds  19/100 | inertia min/median/max   320.53 /   543.60 /   549.17
random x20      good seeds 100/100 | inertia min/median/max   320.53 /   320.53 /   320.53
k-means++ x20   good seeds 100/100 | inertia min/median/max   320.53 /   320.53 /   320.53

best random run      seed  2 | inertia   320.53 | silhouette 0.807 | truth agreement 100.0%
worst random run     seed 47 | inertia   549.15 | silhouette 0.405 | truth agreement 55.4%
same seed, 20 runs   seed 47 | inertia   320.53 | silhouette 0.807 | truth agreement 100.0%

cluster compositions (rows are predicted clusters; columns are true groups)
worst single: [[115, 0, 0], [85, 0, 0], [0, 40, 40]]
20 restarts : [[0, 40, 0], [0, 0, 40], [200, 0, 0]]
```

The worst solution was not slightly different. It answered a different segmentation question.

## Lloyd's algorithm cannot reconsider everything at once

KMeans alternates between two simple operations:

1. Assign every observation to its nearest centroid.
2. Replace each centroid with the mean of its assigned observations.

Each update does not increase the within-cluster sum of squares. The process eventually stops when centroid movement becomes small.

That guarantees convergence to a stable solution. It does not guarantee convergence to the globally smallest inertia.

In the bad run, two initial centers found different regions of the 200-point blob. The third covered both right-hand groups. After assignment, the resulting means still occupied those same broad regions. No single local update had a reason to abandon half of the large group, cross the gap, and split the smaller groups.

The arrangement was locally self-consistent:

```text
predicted cluster 0: 115 from true group 0
predicted cluster 1:  85 from true group 0
predicted cluster 2:  40 from group 1 + 40 from group 2
```

Converged does not mean globally best. It means the current assignment and centroid updates have stopped changing enough.

## The seed reproduced luck; it did not improve it

`with_random_state(47)` makes the initialization sequence deterministic. It is valuable for debugging, tests, repeatable experiments, and artifact recreation.

With a single random initialization, seed 47 deterministically returns the worst run in our 0-to-99 sweep:

```text
inertia:         549.15
silhouette:        0.405
truth agreement:  55.4%
```

Fixing the seed did not make that solution statistically privileged. It only made it repeatable.

This is why I do not search seeds until a clustering plot looks attractive and then present the chosen seed as a modeling parameter. That is manual selection on the answer. If I compare initializations, I use the KMeans objective and independent diagnostics through a declared restart policy.

The seed controls which random sequence is explored. `n_init` controls how many candidates from that sequence get a chance.

## n_init keeps the lowest-inertia run

In datarust, each restart receives a distinct seed derived from the base seed. The implementation runs Lloyd's algorithm independently and retains the result with the lowest inertia.

For the previously bad base seed:

```rust
let mut model = KMeans::new()
    .with_n_clusters(3)
    .with_init(KMeansInit::Random)
    .with_n_init(20)
    .with_random_state(47);
```

One of those 20 candidates enters the basin of the better solution. Its inertia of `320.53` beats the bad candidate's `549.15`, so it becomes the fitted result.

Across the full base-seed sweep, the effect was decisive:

```text
one random init:  23/100 recovered the groups
20 random inits: 100/100 recovered the groups
```

Twenty is not a proof and not a universal optimum. It was enough for this dataset and these 100 base seeds. A harder geometry can require more restarts, a different algorithm, or an admission that compact spherical clusters are the wrong model.

The useful practice is to measure stability as `n_init` increases rather than inherit a number as folklore.

## k-means++ improves a draw, not the objective landscape

k-means++ chooses initial centers sequentially, favoring observations far from the nearest center already selected. It usually spreads the starting centroids more effectively than uniform random selection.

It remains probabilistic. It does not enumerate every initialization or change the local nature of Lloyd's updates.

In this deliberately unbalanced dataset, one k-means++ initialization did not outperform one random initialization across our particular 100 seeds:

```text
random x1:     23 good seeds
k-means++ x1:  19 good seeds
```

I would not generalize those counts into “random is better than k-means++.” Change the sample, cluster sizes, spreads, or seed range and the proportions can change. k-means++ has strong motivation as an initialization heuristic and is datarust's default.

The narrower conclusion is supported directly by the experiment:

> A better initialization strategy is not a guarantee that one initialization is enough.

With 20 restarts, both methods reached the same best inertia for all 100 base seeds.

## Inertia selected the optimization winner

For a fixed dataset, scale, and `k`, lower inertia means observations sit closer—in squared Euclidean distance—to their assigned centroids.

The two representative solutions had:

```text
good solution inertia: 320.53
bad solution inertia:  549.15
```

That makes inertia the correct quantity for choosing among KMeans restarts. It is the objective the algorithm is trying to minimize.

The comparison has a boundary: I do not use raw inertia to declare that a larger `k` is automatically better. Adding centers almost always reduces inertia. Restart selection holds `k` constant; model selection over different cluster counts needs additional evidence such as silhouette, stability, interpretability, and downstream usefulness.

Low inertia also does not prove that the clusters are meaningful to the business. It proves that the fitted centers compress this Euclidean geometry better.

In our synthetic data, the planted labels happen to agree with the lowest-inertia result. Real data does not promise that kindness.

## Silhouette exposed the bad geometry

The silhouette score compares how close each observation is to its own cluster with how close it is to the nearest alternative cluster.

The good solution scored:

```text
0.807
```

The bad solution scored:

```text
0.405
```

Merging the separated right-hand blobs increases their within-cluster distances. Splitting the large left blob creates two neighboring clusters with a fuzzy artificial border. Both effects reduce silhouette.

That makes silhouette a useful independent diagnostic here. It is not part of the KMeans fitting objective, and I would not replace multi-start optimization with “run once and hope silhouette catches it.” I use restarts to fit KMeans well, then use silhouette and domain inspection to evaluate what was fitted.

On non-convex data, unequal-density populations, or business-defined segments, silhouette can prefer a geometrically tidy answer that is still operationally irrelevant.

## Truth agreement required label matching

Cluster IDs are arbitrary. One correct run may label the large group `0`; another may call it `2`.

Directly comparing label vectors would mark those runs as different even if every partition were identical.

The example tries all six permutations of three labels and reports the best agreement. That is why the correct solution reaches 100% regardless of its numeric cluster IDs.

For more clusters, I would use an assignment algorithm to match centers or labels rather than enumerate every permutation. For deployment, I attach business names only after profiling fitted centroids, and I remap new model clusters to the previous version before measuring segment migration.

A stable partition does not imply stable cluster numbers.

## Defaults are a starting point, not an experiment report

`KMeans::new()` currently uses k-means++ and ten initializations by default. The example explicitly sets one or 20 so their effect is visible.

In real work, I record:

- initialization strategy,
- `n_init`,
- random state,
- `k`,
- feature order and fitted scaler,
- inertia and silhouette,
- cluster counts and centroids,
- stability across seeds and resampled data.

If results keep changing after increasing `n_init`, the issue may not be initialization. The dataset may support several nearly equivalent partitions. A few boundary observations may be unstable. Feature weighting may be unclear. KMeans' spherical-cluster assumption may simply be inappropriate.

More restarts optimize the chosen objective more reliably. They cannot turn the wrong objective into the right product model.

## Restarts cost time and buy evidence

Twenty initializations perform up to twenty independent fits. On a large dataset, that cost is real.

I usually begin with k-means++, a fixed base seed for reproducibility, and a moderate restart count. Then I inspect the distribution of inertia across single runs and increase `n_init` until the best result and cluster profiles are stable enough for the decision.

Useful compromises include:

- developing on a representative sample,
- comparing restart counts before the final full-data fit,
- running independent seeds in parallel at the orchestration level,
- keeping the best fitted artifact rather than rerunning it during every request,
- retraining offline and using only `predict` in production.

I do not reduce `n_init` to one merely because one chosen seed was good yesterday. That turns an observed optimization property into an undocumented dependency on luck.

## The worst run was telling a coherent wrong story

The bad clustering could have survived a superficial review.

It contained three non-empty clusters. The largest population genuinely had internal variation, so splitting it produced two plausible-looking centroids. The merged smaller populations both lived on the right, so their combined profile also looked coherent at a distance.

Only comparison made the failure obvious:

```text
single seed 47:
inertia 549.15, silhouette 0.405, composition 115 / 85 / (40+40)

same base seed, 20 restarts:
inertia 320.53, silhouette 0.807, composition 200 / 40 / 40
```

The seed did not need to change. The model needed more opportunities to start.

So before I name a cluster, build a dashboard around it, or send it to a campaign system, I ask a question that one converged run cannot answer:

> If the initial centroids had landed somewhere else, would I still be telling the same story?

On this dataset, one run usually said no.

Twenty restarts made the answer stable across every base seed we tested.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
