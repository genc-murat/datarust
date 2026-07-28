//! Missing-value imputation.

/// K-nearest-neighbors imputer.
pub mod knn;
/// Simple (per-column statistic) imputer.
pub mod simple;

pub use knn::{KnnImputer, KnnWeights};
pub use simple::{ImputeStrategy, SimpleImputer};
