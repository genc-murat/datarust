//! Missing-value imputation.

pub mod knn;
pub mod simple;

pub use knn::{KnnImputer, KnnWeights};
pub use simple::{ImputeStrategy, SimpleImputer};
