//! Model-selection utilities mirroring `sklearn.model_selection`.
//!
//! Provides train/test splitting, K-fold cross-validation splitters, a
//! generic [`cross_val_score`] that works with any [`Predictor`]
//! + `Clone` estimator, the [`CrossValidator`] trait, and hyperparameter
//! search ([`GridSearchCV`] / [`RandomizedSearchCV`]) plus CV-driven
//! diagnostics ([`validation_curve`] / [`learning_curve`]). A shared PRNG
//! lives in [`crate::rng`] (re-exported here as
//! [`rng`](crate::model_selection::rng) for backwards compatibility).

/// Cross-validation scoring driver.
pub mod cross_val;
/// Cross-validator trait (KFold / StratifiedKFold implement it).
pub mod cross_validator;
/// CV-driven diagnostics: validation and learning curves.
pub mod diagnostics;
/// Exhaustive grid search over parameter values.
pub mod grid_search;
/// K-fold and stratified K-fold splitters.
pub mod kfold;
/// Randomized search over parameter distributions.
pub mod randomized_search;
/// Shared deterministic PRNG (xorshift64). Re-export of [`crate::rng`].
pub mod rng;
/// Train/test split.
pub mod split;

pub use cross_val::cross_val_score;
pub use cross_validator::CrossValidator;
pub use diagnostics::{learning_curve, validation_curve};
pub use grid_search::{CVResults, GridSearchCV, ParamGrid};
pub use kfold::{KFold, StratifiedKFold};
pub use randomized_search::{ParamDistribution, ParamSpace, RandomizedSearchCV};
pub use split::{train_test_split, TrainTestSplit};
