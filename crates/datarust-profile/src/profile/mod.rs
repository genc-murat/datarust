//! Dataset and column profiling.

pub mod column;
pub mod dataset;
pub(crate) mod distribution;
pub mod relationships;

pub use column::{CategoricalStats, ColumnProfile, FiveNumber, Histogram, NumericStats};
pub use dataset::DatasetProfile;
pub use relationships::{CorrelationMatrix, PointBiserialEntry, Relationships};
