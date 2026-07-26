# TruncatedSVD Explained 91%. It Mostly Learned the Word Everyone Used.

*A practical datarust comparison of PCA and TruncatedSVD — and why centering can reveal variation while quietly destroying sparsity.*

---

The first component explained 91.18%.

That sounded excellent until I looked at what the component contained.

Almost all of its weight pointed at one term — the term that appeared in every document with exactly the same count.

It did not separate sports from technology or cooking. It mostly described what all three topics had in common.

This is the central difference between PCA and `TruncatedSVD`. PCA asks which directions explain variation *around the mean*. Truncated SVD, without centering, asks which directions carry the most squared magnitude from the origin.

Those questions can produce similar-looking APIs and very different representations.

The usual advice is simple: use PCA for dense centered data and TruncatedSVD for sparse count or TF-IDF matrices. That advice is useful, but the reason behind it matters more than the slogan. Centering changes both the geometry and the storage problem.

Let's watch it happen with [datarust](https://crates.io/crates/datarust).

## Ninety documents, three topics, one universal term

The experiment uses a deliberately clean document-term matrix with 90 rows and 10 columns.

There are three groups of 30 identical documents:

```text
column 0: a common term, count 20 in every document
columns 1–3: sports terms, counts 6, 4, 3
columns 4–6: technology terms, counts 6, 4, 3
columns 7–9: cooking terms, counts 6, 4, 3
```

A sports row looks like this:

```text
[20, 6, 4, 3, 0, 0, 0, 0, 0, 0]
```

A technology row is:

```text
[20, 0, 0, 0, 6, 4, 3, 0, 0, 0]
```

And a cooking row is:

```text
[20, 0, 0, 0, 0, 0, 0, 6, 4, 3]
```

I reduce the matrix to two dimensions with both PCA and `TruncatedSVD`, reconstruct the original rows, and inspect how strongly each component loads on the universal first column.

Here is the complete Rust program:

```rust
use datarust::decomposition::{
    PCAComponents, TruncatedSVD, PCA,
};
use datarust::traits::Transformer;
use datarust::Matrix;

fn density(x: &Matrix) -> f64 {
    let nonzero = x
        .as_slice()
        .iter()
        .filter(|&&value| value.abs() > 1e-12)
        .count();
    nonzero as f64 / x.as_slice().len() as f64
}

fn rmse(original: &Matrix, reconstructed: &Matrix) -> f64 {
    let mse = original
        .as_slice()
        .iter()
        .zip(reconstructed.as_slice())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / original.as_slice().len() as f64;
    mse.sqrt()
}

fn centroid(
    projected: &Matrix,
    start: usize,
    end: usize,
) -> Vec<f64> {
    (0..projected.ncols())
        .map(|col| {
            (start..end)
                .map(|row| projected.get(row, col))
                .sum::<f64>()
                / (end - start) as f64
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rows = Vec::new();

    for _ in 0..30 {
        rows.push(vec![
            20.0, 6.0, 4.0, 3.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
    }
    for _ in 0..30 {
        rows.push(vec![
            20.0, 0.0, 0.0, 0.0,
            6.0, 4.0, 3.0,
            0.0, 0.0, 0.0,
        ]);
    }
    for _ in 0..30 {
        rows.push(vec![
            20.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            6.0, 4.0, 3.0,
        ]);
    }

    let x = Matrix::new(rows)?;

    let mut pca = PCA::new(PCAComponents::Count(2));
    let pca_projected = pca.fit_transform(&x)?;
    let pca_reconstructed =
        pca.inverse_transform(&pca_projected)?;

    let mut svd = TruncatedSVD::new(2)?;
    let svd_projected = svd.fit_transform(&x)?;
    let svd_reconstructed =
        svd.inverse_transform(&svd_projected)?;

    let centered_rows: Vec<Vec<f64>> = (0..x.nrows())
        .map(|i| {
            (0..x.ncols())
                .map(|j| x.get(i, j) - pca.mean()[j])
                .collect()
        })
        .collect();
    let centered = Matrix::new(centered_rows)?;

    println!("Matrix shape: {} x {}", x.nrows(), x.ncols());
    println!("Raw density:      {:.1}%", density(&x) * 100.0);
    println!(
        "Centered density: {:.1}%\n",
        density(&centered) * 100.0,
    );

    println!(
        "PCA explained ratios:          {:?}",
        pca.explained_variance_ratio(),
    );
    println!(
        "TruncatedSVD explained ratios: {:?}",
        svd.explained_variance_ratio(),
    );
    println!(
        "PCA reconstruction RMSE:          {:.6}",
        rmse(&x, &pca_reconstructed),
    );
    println!(
        "TruncatedSVD reconstruction RMSE: {:.6}\n",
        rmse(&x, &svd_reconstructed),
    );

    println!("Absolute loading on always-present term (column 0)");
    println!(
        "PCA component 0:          {:.6}",
        pca.components()[0][0].abs(),
    );
    println!(
        "PCA component 1:          {:.6}",
        pca.components()[1][0].abs(),
    );
    println!(
        "TruncatedSVD component 0: {:.6}",
        svd.components()[0][0].abs(),
    );
    println!(
        "TruncatedSVD component 1: {:.6}\n",
        svd.components()[1][0].abs(),
    );

    for (name, start, end) in [
        ("sports", 0, 30),
        ("tech", 30, 60),
        ("cooking", 60, 90),
    ] {
        let p = centroid(&pca_projected, start, end);
        let s = centroid(&svd_projected, start, end);
        println!(
            "{name:<7} PCA centroid [{:8.4}, {:8.4}]   \
             SVD centroid [{:8.4}, {:8.4}]",
            p[0], p[1], s[0], s[1],
        );
    }

    Ok(())
}
```

This is the output I measured:

```text
Matrix shape: 90 x 10
Raw density:      40.0%
Centered density: 90.0%

PCA explained ratios:          [0.5000000000000004, 0.5000000000000003]
TruncatedSVD explained ratios: [0.9117859725234998, 0.04410701373825016]
PCA reconstruction RMSE:          0.000000
TruncatedSVD reconstruction RMSE: 1.425950

Absolute loading on always-present term (column 0)
PCA component 0:          0.000000
PCA component 1:          0.000000
TruncatedSVD component 0: 0.975513
TruncatedSVD component 1: 0.000000

sports  PCA centroid [  3.4578,  -5.3582]   SVD centroid [ 20.5020,  -2.6810]
tech    PCA centroid [  2.9114,   5.6737]   SVD centroid [ 20.5020,  -3.6704]
cooking PCA centroid [ -6.3692,  -0.3155]   SVD centroid [ 20.5020,   6.3514]
```

The result is not “PCA wins.” The result is that centering changed the question.

## PCA removed what every document shared

PCA begins by subtracting the training mean of each feature.

For column zero, the mean is 20 because every document contains exactly 20 occurrences. After centering:

```text
20 - 20 = 0
```

The common term has no variation, so it has no role in either principal component. Both absolute loadings are exactly `0.000000`.

PCA then works on topic deviations from the average document. A sports row is above the mean in sports columns and below it in technology and cooking columns. The representation describes how a document differs from the corpus baseline.

There are three topic centroids, but after their mean is removed, they lie in a two-dimensional plane: the three centered group vectors sum to zero. Two principal components therefore preserve the entire structure.

That is why the explained variance ratios are approximately `0.5` and `0.5`, and reconstruction RMSE is zero. During `inverse_transform`, PCA adds the learned mean back, restoring the universal count of 20 along with the topic-specific counts.

The constant term did not need a component. It lived in the mean.

## TruncatedSVD kept the origin fixed

Truncated SVD does not subtract a feature mean. Every document remains a positive vector extending from the all-zero origin.

Column zero contains `20`, much larger than the topic counts `6`, `4`, and `3`, and it appears in all 90 rows. It contributes enormous squared magnitude to the matrix.

The first SVD component responds accordingly:

```text
absolute loading on column 0 = 0.975513
```

All three topic centroids receive exactly the same first coordinate, `20.5020`. That coordinate mostly describes the shared document direction. It does not distinguish sports, technology, and cooking.

With two components available, SVD spends one on this global direction and only one on topic contrast. The uncentered matrix has rank three — one shared direction plus two independent contrasts — so a two-component reconstruction cannot be exact. Its RMSE is `1.425950`.

Nothing went wrong. TruncatedSVD preserved the largest directions in the matrix it was given.

## The two “explained ratios” do not share a denominator

It is tempting to compare these lines directly:

```text
PCA first component:          50.00%
TruncatedSVD first component: 91.18%
```

That comparison is misleading.

PCA's denominator is centered variance: squared deviations from feature means. A constant feature contributes zero.

In the current datarust implementation, TruncatedSVD derives its ratios from the eigenvalues of uncentered `XᵀX`. Its denominator is the total squared energy relative to the origin. A large always-positive mean direction can dominate.

So `91.18%` does not mean the first SVD component preserves more topic information than both PCA components. In this example it preserves a direction shared by every topic.

Explained variance is not a semantic score. It tells me how much of a particular geometric quantity was retained, and I need to know which geometry produced it.

## Centering turned zeros into data

If centering reveals the useful topic variation so cleanly, why not always use PCA?

Look at the density:

```text
raw matrix:      40.0% nonzero
centered matrix: 90.0% nonzero
```

A zero in a sports document's technology column means the term did not occur. The corpus mean for that technology term is positive, so centering turns the zero into a negative deviation.

Almost every zero becomes a nonzero number.

Our 90-by-10 example is tiny. A real document-term matrix might contain millions of rows, hundreds of thousands of columns, and well below 1% nonzero entries. Explicit centering can turn a manageable sparse representation into an impossible dense one.

Truncated SVD avoids that operation. Products involving zero entries can remain sparse in an implementation designed for sparse storage, which is why the method is widely used for latent semantic analysis and other high-dimensional count data.

There is one current-library detail worth stating plainly: datarust's `TruncatedSVD` transformer currently accepts a dense `Matrix`, not a `SparseMatrix` directly. The mathematics avoids centering, but the present API does not yet provide an end-to-end CSR input path for this transformer. I would not claim sparse-memory savings that the current call signature cannot deliver.

The conceptual distinction still matters, and a future sparse execution path can preserve it without changing the fitted geometry.

## The common direction is not always useless

The universal column in this synthetic dataset was intentionally uninformative. Real uncentered magnitude can carry signal.

In a user–item matrix, the first SVD direction may describe overall activity or item popularity. In term data, document length and common vocabulary may matter. In purchase counts, a general spending factor can be predictive.

Centering removes the global mean from the components, but it does not prove that the mean is irrelevant to the downstream task.

The right response is not automatically “delete component one.” I inspect component loadings, projected values, and downstream validation metrics. If the first direction is dominated by document length or ubiquitous tokens when I care about topic, I might use TF-IDF weighting, remove stop words, normalize rows, or adjust the vocabulary before decomposition.

Preprocessing defines what SVD sees as energy.

## Component signs are arbitrary

The PCA centroids in this run are:

```text
sports   [ 3.4578, -5.3582]
tech     [ 2.9114,  5.6737]
cooking  [-6.3692, -0.3155]
```

Another mathematically valid solver could multiply either component by `-1`, reflecting every point across an axis. The reconstructed matrix, pairwise distances, and explained ratios would remain unchanged.

I therefore do not attach meaning to “positive component 0” by itself. I interpret the relative loadings and positions together, and I avoid tests that require a decomposition component to have a particular sign.

The same applies to TruncatedSVD.

## Reconstruction exposed where the component budget went

Projection plots can look convincing even when important information was lost. Reconstruction makes the compression cost measurable:

```rust
let projected = svd.fit_transform(&x_train)?;
let approximate = svd.inverse_transform(&projected)?;
```

PCA reconstructed this controlled dataset exactly with two dimensions because centering reduced the topic structure to rank two and the mean was stored separately.

TruncatedSVD needed three dimensions for exact reconstruction of the uncentered rank-three matrix. With only two, some topic-specific counts were approximated.

In real data, exact reconstruction is neither expected nor necessarily desirable. But per-row or per-feature reconstruction error can reveal which documents, products, or rare terms the low-dimensional representation serves poorly.

A high explained ratio and a low reconstruction error still do not guarantee usefulness for classification, search, or recommendation. They are geometric diagnostics, not product metrics.

## What I use each method for

I start with PCA when:

- The data is already dense or can be centered safely.
- Variation around the feature mean is the structure I want.
- A global offset should live in the stored mean rather than consume a component.
- PCA's inverse transform and centered interpretation fit the task.

I start with TruncatedSVD when:

- The input is conceptually sparse and centering would destroy that structure.
- Zeros have a meaningful absence interpretation.
- The origin and global magnitude direction are acceptable parts of the geometry.
- I am working toward latent semantic, user–item, or count-based representations.

In either case, I fit the decomposition on training data only and reuse it for validation and production. The feature vocabulary and exact column order are part of the artifact. A new term cannot be inserted into column 0 without changing every component loading's meaning.

## The first component answered the question I asked

This experiment's most seductive number was `91.18%`.

It would have been easy to read it as “one dimension contains nearly everything important.” What it actually said was:

> One uncentered direction contains nearly all of this matrix's squared energy.

Inspection revealed that the direction mostly represented a term shared by everyone.

PCA removed that common level, found the two independent topic contrasts, and reconstructed the data exactly. TruncatedSVD preserved the zero origin and avoided densifying center subtraction, but used one of its two components on the global direction.

Neither algorithm misunderstood the matrix.

The difference was the baseline:

- PCA asked how documents differ from the average document.
- TruncatedSVD asked how documents extend from zero.

Before choosing between them, that is the question I now answer first.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
