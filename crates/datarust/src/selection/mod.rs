//! Feature selection.

/// Select the `k` best features according to a scoring function.
pub mod select_k_best;
/// Remove low-variance features.
pub mod variance_threshold;

pub use select_k_best::{ScoreFunc, SelectKBest};
pub use variance_threshold::VarianceThreshold;
