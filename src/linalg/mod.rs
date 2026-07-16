//! Linear-algebra primitives shared by estimators.
//!
//! This module hosts internal solver kernels (currently Cholesky decomposition
//! for symmetric positive-definite systems) that are reused across linear
//! models. Keeping them here — rather than inlined in each estimator — gives
//! Ridge, Lasso and Logistic Regression a common, tested foundation.

/// Cholesky decomposition and SPD system solver.
pub mod cholesky;
