//! Classic toy datasets for examples, tests, and onboarding.
//!
//! Mirrors `sklearn.datasets` — small, well-known datasets compiled directly
//! into the binary as `const` arrays. No file I/O, no network access, no
//! external dependencies.

#![allow(clippy::large_const_arrays)]
#![allow(clippy::approx_constant)]
//!
//! Enable with the `datasets` feature:
//!
//! ```toml
//! [dependencies]
//! datarust = { version = "*", features = ["datasets"] }
//! ```
//!
//! ```rust
//! use datarust::datasets::iris;
//!
//! let data = iris::load();
//! let x = data.features();   // Matrix 150×4
//! let y = data.targets();    // Vec<f64>, 3 classes {0, 1, 2}
//! ```

/// A loaded dataset: features, targets, and metadata.
#[derive(Debug, Clone)]
pub struct Dataset {
    data: Vec<Vec<f64>>,
    target: Vec<f64>,
    feature_names: &'static [&'static str],
    target_names: &'static [&'static str],
}

impl Dataset {
    /// Creates a dataset from a const 2-D array and const target slice.
    pub(crate) fn from_const(
        data: Vec<Vec<f64>>,
        target: Vec<f64>,
        feature_names: &'static [&'static str],
        target_names: &'static [&'static str],
    ) -> Self {
        Self {
            data,
            target,
            feature_names,
            target_names,
        }
    }

    /// Returns the feature matrix as a [`Matrix`](crate::Matrix).
    pub fn features(&self) -> crate::Matrix {
        crate::Matrix::new(self.data.clone()).expect("embedded dataset data is well-formed")
    }

    /// Returns the target vector as a flat `&[f64]`.
    pub fn targets(&self) -> &[f64] {
        &self.target
    }

    /// Names of the feature columns (e.g. `["sepal_length", ...]`).
    pub fn feature_names(&self) -> &[&str] {
        self.feature_names
    }

    /// Names of the target classes (e.g. `["setosa", "versicolor", ...]`).
    pub fn target_names(&self) -> &[&str] {
        self.target_names
    }

    /// Number of samples.
    pub fn n_samples(&self) -> usize {
        self.data.len()
    }

    /// Number of features.
    pub fn n_features(&self) -> usize {
        self.feature_names.len()
    }

    /// Number of distinct classes (classification datasets only).
    pub fn n_classes(&self) -> usize {
        self.target_names.len()
    }
}

// ── Dataset modules ──────────────────────────────────────────────────

/// Breast Cancer binary classification dataset (569 samples, 30 features).
pub mod breast_cancer;
/// Diabetes regression dataset (442 samples, 10 features).
pub mod diabetes;
/// Iris flower classification dataset (150 samples, 4 features, 3 classes).
pub mod iris;
/// Wine classification dataset (178 samples, 13 features, 3 classes).
pub mod wine;
