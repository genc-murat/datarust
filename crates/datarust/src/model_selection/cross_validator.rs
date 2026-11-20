//! Abstraction over cross-validation splitting strategies so that
//! `cross_val_score`, `GridSearchCV`, `learning_curve`, etc. can work with
//! `KFold`, `StratifiedKFold`, or custom splitters through the same interface.

use crate::error::Result;
use crate::matrix::Matrix;

/// A cross-validation splitting strategy.
///
/// Requires `Debug` so that search wrappers (e.g.
/// [`GridSearchCV`](crate::model_selection::GridSearchCV)) can derive `Debug`
/// while holding a boxed splitter.
pub trait CrossValidator: std::fmt::Debug {
    /// Number of folds this strategy produces.
    fn n_splits(&self) -> usize;

    /// Returns `n_splits` pairs of `(train_indices, test_indices)`.
    ///
    /// `x` provides the sample count for KFold; `y` provides the class labels
    /// for stratified splitters. Implementations ignore whichever they don't
    /// need.
    fn split(&self, x: &Matrix, y: &[f64]) -> Result<Vec<(Vec<usize>, Vec<usize>)>>;
}
