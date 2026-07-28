//! Integration tests for `cluster::KMeans`.
//!
//! Mirrors the sklearn-parity testing style used elsewhere in the crate:
//! deterministic datasets with known structure, exact label recovery on
//! well-separated blobs, and round-trip serialization under the `serde` feature.

#![cfg(feature = "serde")]

use datarust::cluster::{KMeans, KMeansInit};
use datarust::serialize::{from_json, to_json};
use datarust::traits::Clusterer;
use datarust::Matrix;

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// Build `n_blobs` clusters of `points_per_blob` points each, each blob a tight
/// cloud around a center on the integer grid.
fn make_blobs(n_blobs: usize, points_per_blob: usize, spread: f64) -> (Matrix, Vec<usize>) {
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();
    let mut k = 0;
    for blob in 0..n_blobs {
        let center = (blob as f64) * 20.0; // well-separated: 20 units apart
        for i in 0..points_per_blob {
            let _ = i;
            for (dx, dy) in [(spread, 0.0), (-spread, 0.0), (0.0, spread), (0.0, -spread)] {
                rows.push(vec![center + dx, center + dy]);
                labels.push(blob);
                k += 1;
                if k >= n_blobs * points_per_blob {
                    break;
                }
            }
            if k >= n_blobs * points_per_blob {
                break;
            }
        }
    }
    (Matrix::new(rows).unwrap(), labels)
}

#[test]
fn recovers_three_blobs_up_to_label_permutation() {
    let (x, true_labels) = make_blobs(3, 12, 0.1);
    let mut km = KMeans::new()
        .with_n_clusters(3)
        .with_n_init(10)
        .with_random_state(42);
    let pred = km.fit_predict(&x).unwrap();

    // Map each predicted label to the true label of its first member.
    let mut mapping: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (p, t) in pred.iter().zip(true_labels.iter()) {
        mapping.entry(*p).or_insert(*t);
    }
    // Every predicted label must map consistently to one true label.
    let correct = pred
        .iter()
        .zip(true_labels.iter())
        .filter(|(p, t)| mapping.get(p) == Some(t))
        .count();
    assert_eq!(
        correct,
        pred.len(),
        "cluster labels do not match truth (up to permutation): {:?}",
        pred
    );
}

#[test]
fn centroids_near_true_centers() {
    let (x, _) = make_blobs(3, 20, 0.05);
    let mut km = KMeans::new()
        .with_n_clusters(3)
        .with_n_init(10)
        .with_random_state(0);
    km.fit(&x).unwrap();
    let true_centers = [0.0, 20.0, 40.0];
    let mut matched = 0;
    for center in km.cluster_centers() {
        let close = true_centers
            .iter()
            .any(|&tc| approx(center[0], tc, 0.5) && approx(center[1], tc, 0.5));
        if close {
            matched += 1;
        }
    }
    assert_eq!(matched, 3, "centroids: {:?}", km.cluster_centers());
}

#[test]
fn predict_new_points_matches_nearest_centroid() {
    let (x, _) = make_blobs(2, 8, 0.1);
    let mut km = KMeans::new()
        .with_n_clusters(2)
        .with_n_init(5)
        .with_random_state(1);
    km.fit(&x).unwrap();
    // Points near 0 and near 20 should land in different clusters.
    let test = Matrix::new(vec![vec![0.0, 0.0], vec![20.0, 20.0]]).unwrap();
    let pred = km.predict(&test).unwrap();
    assert_ne!(pred[0], pred[1]);
}

#[test]
fn deterministic_with_same_seed() {
    let (x, _) = make_blobs(3, 10, 0.2);
    let mut a = KMeans::new().with_n_clusters(3).with_random_state(7);
    let mut b = KMeans::new().with_n_clusters(3).with_random_state(7);
    let la = a.fit_predict(&x).unwrap();
    let lb = b.fit_predict(&x).unwrap();
    assert_eq!(la, lb);
    assert_eq!(a.cluster_centers(), b.cluster_centers());
    assert!(approx(a.inertia(), b.inertia(), 1e-12));
}

#[test]
fn inertia_non_negative_and_lower_with_more_clusters() {
    let (x, _) = make_blobs(3, 15, 0.3);
    let mut km3 = KMeans::new().with_n_clusters(3).with_random_state(0);
    km3.fit(&x).unwrap();
    let mut km5 = KMeans::new().with_n_clusters(5).with_random_state(0);
    km5.fit(&x).unwrap();
    assert!(km3.inertia() >= 0.0);
    // More clusters can only reduce (or match) the inertia on the training set.
    assert!(km5.inertia() <= km3.inertia() + 1e-9);
}

#[test]
fn n_clusters_zero_rejected() {
    let x = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
    let mut km = KMeans::new().with_n_clusters(0);
    assert!(km.fit(&x).is_err());
}

#[test]
fn n_clusters_exceeds_samples_rejected() {
    let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0]]).unwrap();
    let mut km = KMeans::new().with_n_clusters(10);
    assert!(km.fit(&x).is_err());
}

#[test]
fn predict_before_fit_errors() {
    let km = KMeans::new().with_n_clusters(2);
    let x = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
    let err = km.predict(&x).unwrap_err();
    assert!(matches!(err, datarust::DatarustError::NotFitted(_)));
}

#[test]
fn predict_feature_mismatch_errors() {
    let x = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let mut km = KMeans::new().with_n_clusters(2).with_random_state(0);
    km.fit(&x).unwrap();
    let wrong = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
    let err = km.predict(&wrong).unwrap_err();
    assert!(matches!(err, datarust::DatarustError::ShapeMismatch { .. }));
}

#[test]
fn kmeans_plus_plus_finds_clusters() {
    let (x, _) = make_blobs(3, 15, 0.1);
    let mut km = KMeans::new()
        .with_n_clusters(3)
        .with_init(KMeansInit::KMeansPlusPlus)
        .with_random_state(0);
    let pred = km.fit_predict(&x).unwrap();
    let unique: std::collections::BTreeSet<usize> = pred.iter().copied().collect();
    assert_eq!(unique.len(), 3);
}

#[test]
fn random_init_finds_clusters() {
    let (x, _) = make_blobs(3, 15, 0.1);
    let mut km = KMeans::new()
        .with_n_clusters(3)
        .with_init(KMeansInit::Random)
        .with_random_state(0);
    let pred = km.fit_predict(&x).unwrap();
    let unique: std::collections::BTreeSet<usize> = pred.iter().copied().collect();
    assert_eq!(unique.len(), 3);
}

#[test]
fn n_iter_at_least_one() {
    let (x, _) = make_blobs(2, 10, 0.1);
    let mut km = KMeans::new().with_n_clusters(2).with_random_state(0);
    km.fit(&x).unwrap();
    assert!(km.n_iter() >= 1);
}

#[test]
fn serde_round_trip_preserves_predictions() {
    let (x, _) = make_blobs(3, 12, 0.1);
    let mut km = KMeans::new().with_n_clusters(3).with_random_state(42);
    km.fit(&x).unwrap();
    let before = km.predict(&x).unwrap();

    let json = to_json(&km).unwrap();
    let restored: KMeans = from_json(&json).unwrap();
    let after = restored.predict(&x).unwrap();

    assert_eq!(before, after);
    assert!(approx(restored.inertia(), km.inertia(), 1e-12));
    assert!(restored.is_fitted());
}

#[test]
fn serde_preserves_cluster_centers() {
    let (x, _) = make_blobs(2, 10, 0.1);
    let mut km = KMeans::new().with_n_clusters(2).with_random_state(0);
    km.fit(&x).unwrap();

    let json = to_json(&km).unwrap();
    let restored: KMeans = from_json(&json).unwrap();

    for (a, b) in km
        .cluster_centers()
        .iter()
        .zip(restored.cluster_centers().iter())
    {
        for (va, vb) in a.iter().zip(b.iter()) {
            assert!(approx(*va, *vb, 1e-12));
        }
    }
}
