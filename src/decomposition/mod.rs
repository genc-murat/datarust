//! Dimensionality reduction: PCA and Truncated SVD.

pub mod jacobi;
pub mod pca;
pub mod truncated_svd;

pub use pca::{PCAComponents, PCA};
pub use truncated_svd::TruncatedSVD;
