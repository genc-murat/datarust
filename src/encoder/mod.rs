//! Categorical encoders.

/// Frequency/count categorical encoder.
pub mod frequency;
/// Label encoder for target labels.
pub mod label;
/// One-hot categorical encoder.
pub mod onehot;
/// Ordinal categorical encoder.
pub mod ordinal;
/// Target (mean) categorical encoder.
pub mod target;

pub use frequency::{FrequencyEncoder, UnknownFrequency};
pub use label::LabelEncoder;
pub use onehot::{DropStrategy, HandleUnknown, OneHotEncoder};
pub use ordinal::{OrdinalCategories, OrdinalEncoder, OrdinalHandleUnknown};
pub use target::{TargetEncoder, UnknownTarget};
