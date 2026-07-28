//! Composing multiple transformers.

/// Apply different transformers to different columns of a dataset.
pub mod column_transformer;

/// Result container preserving numeric / categorical separation.
pub mod output;

pub use column_transformer::{ColumnSpec, ColumnTransformer, Remainder, Table};
pub use output::Output;
