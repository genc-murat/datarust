//! Dimensionality reduction: PCA and Truncated SVD.

pub mod jacobi;
/// Principal Component Analysis.
pub mod pca;
/// Randomized SVD backend (used by PCA for large low-rank inputs).
pub mod randomized_svd;
/// Truncated Singular Value Decomposition.
pub mod truncated_svd;

pub use pca::{PCAComponents, PCA};
pub use truncated_svd::{SVDComponents, TruncatedSVD};
