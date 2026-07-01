//! Categorical encoders.

pub mod frequency;
pub mod label;
pub mod onehot;
pub mod ordinal;
pub mod target;

pub use frequency::FrequencyEncoder;
pub use label::LabelEncoder;
pub use onehot::{DropStrategy, HandleUnknown, OneHotEncoder};
pub use ordinal::{OrdinalCategories, OrdinalEncoder, OrdinalHandleUnknown};
pub use target::{TargetEncoder, UnknownTarget};
