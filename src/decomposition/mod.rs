//! Dimensionality reduction: PCA and Truncated SVD.

pub mod jacobi;
/// Principal Component Analysis.
pub mod pca;
/// Truncated Singular Value Decomposition.
pub mod truncated_svd;

pub use pca::{PCAComponents, PCA};
pub use truncated_svd::{SVDComponents, TruncatedSVD};
