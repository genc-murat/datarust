//! Model-evaluation metrics mirroring `sklearn.metrics`.
//!
//! Regression metrics live in [`regression`](crate::metrics::regression),
//! classification metrics in
//! [`classification`](crate::metrics::classification).

/// Classification metrics: accuracy, precision, recall, F1, confusion matrix, log loss.
pub mod classification;
/// Regression metrics: MSE, MAE, R², max error, explained variance.
pub mod regression;
