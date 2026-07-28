//! Unsupervised clustering estimators.
//!
//! Provides [`KMeans`], the Lloyd's-algorithm k-means clustering estimator
//! with k-means++ initialization. All clustering estimators implement the
//! [`Clusterer`](crate::traits::Clusterer) trait: `fit(X)` learns cluster
//! structure, `predict(X)` assigns new points to their nearest cluster, and
//! `fit_predict(X)` returns the cluster index assigned to each training row.
//!
//! Unlike supervised [`Predictor`](crate::traits::Predictor)s, clustering
//! estimators take no target `y` and return `Vec<usize>` cluster indices rather
//! than regression targets or class labels.

/// k-means clustering (Lloyd's algorithm, k-means++ initialization).
pub mod kmeans;
/// Clustering evaluation metrics (silhouette score).
pub mod metrics;

pub use kmeans::{KMeans, KMeansInit};
