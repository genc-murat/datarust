//! Report renderers for a [`crate::DatasetProfile`].
//!
//! - `json` — structured JSON serialization (requires the `serde` feature).
//! - `html` — a dependency-free, self-contained HTML report.

pub mod html;
#[cfg(feature = "serde")]
pub mod json;

pub use html::to_html;
#[cfg(feature = "serde")]
pub use json::{to_json, to_json_file, JsonReport};
