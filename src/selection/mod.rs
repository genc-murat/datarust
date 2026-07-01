//! Feature selection.

pub mod select_k_best;
pub mod variance_threshold;

pub use select_k_best::{ScoreFunc, SelectKBest};
pub use variance_threshold::VarianceThreshold;
