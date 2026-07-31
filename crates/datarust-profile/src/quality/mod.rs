//! Data-quality checks and findings.

pub mod checks;

pub use checks::{run_checks, QualityIssue, QualityKind, Thresholds};
