//! Model-selection utilities mirroring `sklearn.model_selection`.
//!
//! Provides train/test splitting, K-fold cross-validation splitters, and a
//! generic [`cross_val_score`] that works with any [`Predictor`](crate::traits::Predictor)
//! + `Clone` estimator. A shared PRNG lives in
//! [`rng`](crate::model_selection::rng).

/// Cross-validation scoring driver.
pub mod cross_val;
/// K-fold and stratified K-fold splitters.
pub mod kfold;
/// Shared deterministic PRNG (xorshift64).
pub mod rng;
/// Train/test split.
pub mod split;

pub use cross_val::cross_val_score;
pub use kfold::{KFold, StratifiedKFold};
pub use split::{train_test_split, TrainTestSplit};
