//! Regression estimators with a `fit`/`predict` API.
//!
//! Provides ordinary least squares ([`LinearRegression`]), L2-regularized
//! ([`Ridge`]), L1-regularized ([`Lasso`]) linear models, and binary
//! [`LogisticRegression`] for classification. All share the
//! [`Regressor`](crate::traits::Regressor) trait and the [`crate::linalg`]
//! solver foundation.

/// L1-regularized regression (coordinate descent).
pub mod lasso;
/// Ordinary least-squares linear regression.
pub mod linear_regression;
/// Binary logistic regression (IRLS solver).
pub mod logistic_regression;
/// L2-regularized ridge regression.
pub mod ridge;

pub use lasso::Lasso;
pub use linear_regression::{LinearRegression, LinearSolver};
pub use logistic_regression::{LogisticRegression, LogisticSolver};
pub use ridge::{Ridge, RidgeSolver};
