//! Report renderers for a [`crate::DatasetProfile`].
//!
//! - `json` — structured JSON serialization (requires the `serde` feature).
//! - `html` — a dependency-free, self-contained HTML report.
//! - `terminal` — a fixed-width terminal-table summary (requires the
//!   `pretty` feature).

pub mod html;
#[cfg(feature = "serde")]
pub mod json;
#[cfg(feature = "pretty")]
pub mod terminal;

pub use html::{to_html, to_html_with};
#[cfg(feature = "serde")]
pub use json::{to_json, to_json_file, JsonReport};
#[cfg(feature = "pretty")]
pub use terminal::{to_terminal, to_terminal_auto, to_terminal_with};
