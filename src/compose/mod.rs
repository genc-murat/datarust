//! Composing multiple transformers.

pub mod column_transformer;

pub use column_transformer::{ColumnSpec, ColumnTransformer, Remainder, Table};
