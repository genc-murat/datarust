//! KMeans clustering workflow: synthetic blob generation → fit → inspect
//! centroids, labels, inertia → predict new points → compare k-means++ vs
//! random initialization.
//!
//! Scenario: discover structure in unlabeled 2-D point data. Three visually
//! distinct point clouds are generated with a deterministic PRNG; KMeans must
//! recover the three clusters up to a label permutation.
//!
//! Run: `cargo run --example kmeans_clustering`
//! (optionally with `--features serde` for the serialization demo)

use datarust::cluster::{KMeans, KMeansInit};
use datarust::traits::Clusterer;
use datarust::Matrix;

/// Simple deterministic PRNG (xorshift64) for reproducible synthetic data.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Approximately normal with mean 0, std `sigma` (simplified Box–Muller).
    fn normal(&mut self, sigma: f64) -> f64 {
        let u = self.next_f64();
        let v = self.next_f64();
        sigma * (u.ln() * -2.0).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// Build `k` well-separated blobs of `points_per_blob` 2-D points each.
fn make_blobs(k: usize, points_per_blob: usize, spread: f64) -> (Matrix, Vec<usize>) {
    let mut rng = Rng::new(42);
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(k * points_per_blob);
    let mut truth: Vec<usize> = Vec::with_capacity(k * points_per_blob);
    for blob in 0..k {
        let cx = (blob as f64) * 15.0; // 15 units apart — well separated
        let cy = (blob as f64) * 8.0;
        for _ in 0..points_per_blob {
            rows.push(vec![cx + rng.normal(spread), cy + rng.normal(spread)]);
            truth.push(blob);
        }
    }
    (Matrix::new(rows).unwrap(), truth)
}

/// Fraction of labels that agree with the truth up to a permutation of labels.
fn agreement(predicted: &[usize], truth: &[usize]) -> f64 {
    // Map each predicted label to the most common true label among its members.
    let mut best_fit: std::collections::HashMap<usize, std::collections::HashMap<usize, usize>> =
        std::collections::HashMap::new();
    for (p, t) in predicted.iter().zip(truth.iter()) {
        *best_fit.entry(*p).or_default().entry(*t).or_insert(0) += 1;
    }
    let mut mapping: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (p, counts) in &best_fit {
        let winner = counts
            .iter()
            .max_by_key(|(_, v)| *v)
            .map(|(t, _)| *t)
            .unwrap_or(0);
        mapping.insert(*p, winner);
    }
    let correct = predicted
        .iter()
        .zip(truth.iter())
        .filter(|(p, t)| mapping.get(p) == Some(t))
        .count();
    correct as f64 / predicted.len() as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Generate three well-separated blobs ────────────────────────
    let (x, truth) = make_blobs(3, 40, 0.5);
    println!("=== KMeans Clustering ===");
    println!(
        "Data: {} points, {} features (3 blobs, 15 units apart)\n",
        x.nrows(),
        x.ncols()
    );

    // ── 2. Fit KMeans with k-means++ initialization ──────────────────
    // k-means++ spreads initial centroids far apart, which dramatically
    // improves solution quality over random initialization.
    let mut km = KMeans::new()
        .with_n_clusters(3)
        .with_init(KMeansInit::KMeansPlusPlus)
        .with_n_init(10)
        .with_random_state(42);
    let labels = km.fit_predict(&x)?;

    println!("=== Fit Result (k-means++) ===");
    println!("Converged in {} iterations", km.n_iter());
    println!(
        "Inertia (within-cluster sum of squares): {:.4}",
        km.inertia()
    );
    println!(
        "Cluster agreement with truth: {:.1}%",
        100.0 * agreement(&labels, &truth)
    );
    println!();

    // ── 3. Inspect the learned centroids ──────────────────────────────
    // The true blob centers are at (0,0), (15,8), (30,16).
    println!("=== Cluster Centroids ===");
    let true_centers = [[0.0, 0.0], [15.0, 8.0], [30.0, 16.0]];
    println!(
        "{:<10} {:<20} {:<20}",
        "Cluster", "Learned center", "True center (nearest)"
    );
    for (i, center) in km.cluster_centers().iter().enumerate() {
        // Find the nearest true center for display.
        let nearest = true_centers
            .iter()
            .min_by_key(|tc| ((tc[0] - center[0]).powi(2) + (tc[1] - center[1]).powi(2)) as u64)
            .copied()
            .unwrap();
        println!(
            "{:<10} ({:6.2}, {:6.2})      ({:6.1}, {:6.1})",
            i, center[0], center[1], nearest[0], nearest[1]
        );
    }
    println!();

    // ── 4. Predict new points ─────────────────────────────────────────
    // Points near each true center should be assigned to the corresponding
    // cluster learned during fit.
    let test = Matrix::new(vec![vec![0.5, 0.5], vec![15.5, 8.5], vec![29.5, 15.5]])?;
    let predicted = km.predict(&test)?;
    println!("=== Predicting New Points ===");
    println!("{:<18} {:<10}", "Point", "Cluster");
    for (i, p) in predicted.iter().enumerate() {
        println!("({:5.1}, {:5.1})    {}", test.get(i, 0), test.get(i, 1), p);
    }
    println!();

    // ── 5. k-means++ vs random initialization ────────────────────────
    // On well-separated data both strategies find the three clusters, but
    // k-means++ typically achieves equal or lower inertia because it avoids
    // poor initial centroid placements.
    println!("=== Initialization Comparison ===");
    println!("{:<18} {:<12} {:<10}", "Strategy", "Inertia", "Agreement");

    let mut km_pp = KMeans::new()
        .with_n_clusters(3)
        .with_init(KMeansInit::KMeansPlusPlus)
        .with_random_state(0);
    let labels_pp = km_pp.fit_predict(&x)?;
    println!(
        "{:<18} {:<12.4} {:<9.1}%",
        "k-means++",
        km_pp.inertia(),
        100.0 * agreement(&labels_pp, &truth)
    );

    let mut km_rand = KMeans::new()
        .with_n_clusters(3)
        .with_init(KMeansInit::Random)
        .with_random_state(0);
    let labels_rand = km_rand.fit_predict(&x)?;
    println!(
        "{:<18} {:<12.4} {:<9.1}%",
        "random",
        km_rand.inertia(),
        100.0 * agreement(&labels_rand, &truth)
    );

    // ── 6. Serialization round-trip (serde feature) ──────────────────
    #[cfg(feature = "serde")]
    {
        let json = datarust::serialize::to_json(&km)?;
        let restored: KMeans = datarust::serialize::from_json(&json)?;
        let restored_pred = restored.predict(&test)?;
        assert_eq!(predicted, restored_pred);
        println!("\n=== Serialization (serde) ===");
        println!("JSON size: {} bytes", json.len());
        println!("Restored model is_fitted? {}", restored.is_fitted());
        println!("Predictions identical after round-trip: OK");
    }
    #[cfg(not(feature = "serde"))]
    {
        println!("\n(serde feature not enabled — skipping serialization demo)");
    }

    Ok(())
}
