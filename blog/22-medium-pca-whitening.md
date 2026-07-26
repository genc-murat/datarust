# PCA Reconstructed Everything. Whitening Still Found the Wrong Clusters.

*A practical datarust guide to PCA whitening, unit-variance components, and the moment a harmless-looking option rewrites distance.*

---

Both PCA models reconstructed the original data almost perfectly.

```text
plain PCA RMSE:      5.491e-16
whitened PCA RMSE:   6.326e-16
```

They kept both components. They used the same principal axes. They retained the same information.

Then I ran the same two-cluster KMeans model on their outputs.

The ordinary PCA representation recovered the real customer segments with `99.5%` accuracy. The whitened representation ignored those segments and recovered an unrelated measurement band with `100%` accuracy.

Nothing had been dropped. The geometry had changed.

That is the part of PCA whitening I find easiest to underestimate. Setting `whiten(true)` sounds like a tidy normalization step after dimensionality reduction. Mathematically, it gives every retained principal component unit variance. Operationally, it decides that a naturally quiet direction deserves the same distance budget as a dominant one.

Sometimes that is exactly what a model needs. Sometimes it promotes low-variance nuisance into a first-class signal.

Let's build both outcomes into one small experiment with [datarust](https://crates.io/crates/datarust).

## Two real segments and two irrelevant bands

The synthetic dataset has 200 observations and two features.

The first feature contains the structure I care about:

```text
segment 0: centered near -5
segment 1: centered near +5
noise:     standard deviation 2
```

The second feature is an unrelated acquisition artifact. Half of every segment was measured near `-0.5`; the other half was measured near `+0.5`. The bands are extremely clean, but they say nothing about segment membership.

```text
                   nuisance band
                 -0.5         +0.5
segment 0          50           50
segment 1          50           50
```

This distinction matters. “Low variance” does not mean “random,” and “easy to cluster” does not mean “useful.” The nuisance feature has less numeric spread than the segment feature, but its two bands are sharper.

I fit ordinary PCA and whitened PCA while keeping all components. Then I compare projected variance, reconstruction, a pair of controlled distances, and KMeans assignments.

Here is the complete Rust program:

```rust
use datarust::cluster::{metrics::silhouette_score, KMeans};
use datarust::decomposition::{PCAComponents, PCA};
use datarust::traits::{Clusterer, Transformer};
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
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt()
            * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn sample_variances(x: &Matrix) -> Vec<f64> {
    let means = x.column_mean();
    (0..x.ncols())
        .map(|j| {
            (0..x.nrows())
                .map(|i| {
                    let d = x.get(i, j) - means[j];
                    d * d
                })
                .sum::<f64>()
                / (x.nrows() - 1) as f64
        })
        .collect()
}

fn rmse(a: &Matrix, b: &Matrix) -> f64 {
    (a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        / a.as_slice().len() as f64)
        .sqrt()
}

fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn row(x: &Matrix, i: usize) -> Vec<f64> {
    (0..x.ncols()).map(|j| x.get(i, j)).collect()
}

fn best_binary_accuracy(
    predicted: &[usize],
    expected: &[usize],
) -> f64 {
    let direct = predicted
        .iter()
        .zip(expected)
        .filter(|(a, b)| a == b)
        .count();
    let flipped = predicted.len() - direct;
    direct.max(flipped) as f64 / predicted.len() as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Rng::new(0xC0FFEE);
    let mut rows = Vec::new();
    let mut segment = Vec::new();
    let mut nuisance_band = Vec::new();

    for class in 0..2 {
        for i in 0..100 {
            let x = if class == 0 { -5.0 } else { 5.0 }
                + 2.0 * rng.normal();
            let band = i % 2;
            let y = if band == 0 { -0.5 } else { 0.5 }
                + 0.02 * rng.normal();

            rows.push(vec![x, y]);
            segment.push(class);
            nuisance_band.push(band);
        }
    }
    let data = Matrix::new(rows)?;

    let mut plain = PCA::new(PCAComponents::All);
    let plain_scores = plain.fit_transform(&data)?;
    let plain_rebuilt = plain.inverse_transform(&plain_scores)?;

    let mut white =
        PCA::new(PCAComponents::All).whiten(true);
    let white_scores = white.fit_transform(&data)?;
    let white_rebuilt = white.inverse_transform(&white_scores)?;

    println!(
        "explained variance: {:?}",
        plain.explained_variance(),
    );
    println!(
        "explained ratio:    {:?}",
        plain.explained_variance_ratio(),
    );
    println!(
        "plain score var:    {:?}",
        sample_variances(&plain_scores),
    );
    println!(
        "white score var:    {:?}",
        sample_variances(&white_scores),
    );
    println!(
        "plain RMSE:         {:.3e}",
        rmse(&data, &plain_rebuilt),
    );
    println!(
        "white RMSE:         {:.3e}",
        rmse(&data, &white_rebuilt),
    );

    let mean = plain.mean();
    let checkpoints = Matrix::new(vec![
        vec![mean[0] - 5.0, mean[1]],
        vec![mean[0] + 5.0, mean[1]],
        vec![mean[0], mean[1] - 0.5],
        vec![mean[0], mean[1] + 0.5],
    ])?;
    let plain_checkpoints = plain.transform(&checkpoints)?;
    let white_checkpoints = white.transform(&checkpoints)?;

    println!(
        "plain distance (segment / nuisance): {:.3} / {:.3}",
        distance(
            &row(&plain_checkpoints, 0),
            &row(&plain_checkpoints, 1),
        ),
        distance(
            &row(&plain_checkpoints, 2),
            &row(&plain_checkpoints, 3),
        ),
    );
    println!(
        "white distance (segment / nuisance): {:.3} / {:.3}",
        distance(
            &row(&white_checkpoints, 0),
            &row(&white_checkpoints, 1),
        ),
        distance(
            &row(&white_checkpoints, 2),
            &row(&white_checkpoints, 3),
        ),
    );

    let mut plain_kmeans = KMeans::new()
        .with_n_clusters(2)
        .with_n_init(20)
        .with_random_state(42);
    let plain_labels =
        plain_kmeans.fit_predict(&plain_scores)?;

    let mut white_kmeans = KMeans::new()
        .with_n_clusters(2)
        .with_n_init(20)
        .with_random_state(42);
    let white_labels =
        white_kmeans.fit_predict(&white_scores)?;

    println!(
        "plain KMeans — segment acc: {:.3}, nuisance acc: {:.3}, silhouette: {:.3}",
        best_binary_accuracy(&plain_labels, &segment),
        best_binary_accuracy(&plain_labels, &nuisance_band),
        silhouette_score(&plain_scores, &plain_labels)?,
    );
    println!(
        "white KMeans — segment acc: {:.3}, nuisance acc: {:.3}, silhouette: {:.3}",
        best_binary_accuracy(&white_labels, &segment),
        best_binary_accuracy(&white_labels, &nuisance_band),
        silhouette_score(&white_scores, &white_labels)?,
    );

    Ok(())
}
```

This is the output I measured:

```text
explained variance: [26.513550992294103, 0.2529337496320315]
explained ratio:    [0.9905503560863245, 0.00944964391367554]
plain score var:    [26.513550992294093, 0.25293374963203163]
white score var:    [0.9999999999999999, 1.0000000000000002]
plain RMSE:         5.491e-16
white RMSE:         6.326e-16
plain distance (segment / nuisance): 10.000 / 1.000
white distance (segment / nuisance): 1.942 / 1.988
plain KMeans — segment acc: 0.995, nuisance acc: 0.505, silhouette: 0.727
white KMeans — segment acc: 0.500, nuisance acc: 1.000, silhouette: 0.524
```

The ordinary projection says one direction is roughly one hundred times more variable than the other. Whitening says both directions have variance one.

KMeans listens to the second description.

## Ordinary PCA rotated the data

PCA first subtracts the feature means. It then finds orthonormal directions ordered by the variance of the centered data.

For component `j`, the ordinary projected coordinate is:

```text
z_j = (x - mean) · component_j
```

The two projected variances in this run are:

```text
component 0: 26.5136
component 1:  0.2529
```

Because all components were kept, this ordinary PCA transform is a rotation after centering. Rotations preserve Euclidean distance. Two original observations ten units apart remain ten units apart in PCA space.

That is visible in the controlled checkpoint comparison:

```text
segment-axis difference:  10.000
nuisance-axis difference:  1.000
```

The segment direction dominates KMeans' squared-distance objective. Its clusters match the intended segments for 199 of 200 observations.

Ordinary PCA decorrelated the coordinates. It did not give them equal variance.

## Whitening divided by the natural spread

Whitened PCA adds another operation:

```text
white_z_j = z_j / sqrt(explained_variance_j)
```

The wide first component is divided by about `sqrt(26.51)`. The narrow second component is divided by about `sqrt(0.253)`, a much smaller number. Dividing by that smaller number expands the quiet component far more aggressively.

Afterward, both sample variances are one:

```text
white score var: [1.0, 1.0]
```

That makes the ten-unit segment separation worth `1.942` in whitened space. The one-unit nuisance separation becomes worth `1.988`.

The raw segment difference is ten times larger, yet whitening makes the two differences nearly equal. Because the nuisance bands are much tighter internally, KMeans chooses them.

This is not merely rescaling an output table for presentation. Any downstream method based on Euclidean distance, dot products, or regularized coefficients sees a different optimization problem.

## The explained ratio did not become fifty-fifty

There is a subtle distinction in the fitted PCA state.

The model still reports the variance found in the original centered data:

```text
explained ratio: [0.99055, 0.00945]
```

Whitening does not rewrite history and claim that the original directions each explained half of the variance. It scales the *projected coordinates* after those directions and eigenvalues have been learned.

So these statements can both be true:

- Component zero explains about 99.1% of the original variance.
- Both whitened output columns have variance one.

The first describes the training data. The second describes the representation handed to the next model.

Confusing those two spaces makes a whitened scatter plot surprisingly easy to misread.

## Perfect reconstruction did not protect the clusters

Both reconstruction errors are floating-point noise because both components were retained.

During the whitened inverse transform, datarust multiplies each coordinate by the same standard deviation that `transform` divided out, projects back through the components, and adds the learned mean.

The scaling is reversible. That is why no information was lost.

But KMeans never calls `inverse_transform`. It works directly in the whitened coordinates, where the meaning of a unit has changed.

This is why reconstruction error cannot answer a downstream geometry question. A transform may be perfectly invertible and still change:

- nearest neighbors,
- cluster assignments,
- RBF-kernel similarities,
- anomaly rankings,
- coefficient penalties,
- and any threshold based on projected distance.

Invertibility tells me whether I can recover the input. It does not tell me whether the new metric matches the task.

## Whitening is not StandardScaler before PCA

The two operations are sometimes described as interchangeable normalization. They are not.

`StandardScaler` before PCA scales the original feature columns first. PCA then learns its axes from that newly scaled covariance or correlation structure. Changing the input scale can rotate the principal components themselves.

PCA whitening learns principal axes from the centered input as given, projects onto those axes, and then scales each component by its own explained standard deviation.

In short:

```text
StandardScaler -> PCA:
scale original features, then learn directions

PCA with whitening:
learn directions, then scale component scores
```

Those pipelines answer different questions. I choose between them based on what feature units mean and what geometry the downstream estimator should receive, not because both happen to produce values near unit scale.

## When whitening earns its place

The synthetic example is intentionally hostile to whitening. There are also legitimate reasons to use it.

I consider whitening when:

- Large variance mostly reflects measurement units rather than importance.
- Low-variance principal directions contain validated predictive signal.
- The downstream estimator behaves better with roughly isotropic input.
- I am comparing distances in component space and want variance-normalized directions.
- Cross-validation shows a repeatable improvement on the actual target metric.

For example, a quiet defect signature may matter more than a high-variance operating-load direction. Without whitening, the operating regime could dominate nearest-neighbor distance. Equalizing component variances can expose the defect structure.

That useful scenario and this nuisance-band failure share the same mechanism. Whitening cannot know which low-variance direction is semantically important. It only knows the eigenvalue.

## What I validate before enabling it

I now treat `whiten(true)` as a model choice, not a cosmetic option.

My checklist is:

1. Inspect the explained variances, not only their cumulative ratio.
2. Compare downstream metrics with and without whitening on held-out data.
3. Inspect which samples become nearest neighbors in each representation.
4. Check component loadings for sensor artifacts, batch effects, and near-constant noise.
5. Fit PCA on training data only, then reuse that fitted transform for validation and production.
6. Save the whitening choice with the model artifact; serving must use the same geometry as training.

For unsupervised work, a silhouette score alone is not enough. The whitened nuisance clusters in this example still achieved `0.524`, which could look respectable without domain labels. I also examine stability, cluster profiles, known cohorts, and whether the discovered split supports the decision I am trying to make.

Our controlled labels made the mistake obvious. Production data is rarely that generous.

## The option changed what “close” meant

Ordinary PCA and whitened PCA learned the same directions from the same 200 rows. Both retained all components. Both reconstructed every input value.

Yet their downstream stories were opposite:

```text
ordinary PCA + KMeans: 99.5% real segment agreement
whitened PCA + KMeans: 100% nuisance-band agreement
```

The whitening step removed the variance hierarchy. It promoted a direction explaining less than 1% of original variance until one standardized unit there counted as much as one standardized unit in the dominant direction.

That can rescue quiet signal. It can also amplify a quiet distraction.

So before I enable whitening, I no longer ask only whether unit-variance components look cleaner. I ask the question the downstream model will actually answer:

> After every retained direction receives equal variance, are the right observations still close to one another?

In this experiment, the answer was no — even though the reconstruction was perfect.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
