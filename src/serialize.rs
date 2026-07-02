//! Optional JSON serialization for fitted transformers (gated on the
//! `serde` feature).
//!
//! Leaf transformers derive `Serialize`/`Deserialize`; use [`save_json`] and
//! [`load_json`] to persist fitted state. [`Pipeline`] and
//! [`ColumnTransformer`] are also serializable via the [`TransformerKind`]
//! enum wrapper.
//!
//! [`Pipeline`]: crate::pipeline::Pipeline
//! [`ColumnTransformer`]: crate::compose::ColumnTransformer
//! [`TransformerKind`]: crate::transformer_kind::TransformerKind

use std::path::Path;

use crate::error::Result;

/// Serialize a fitted transformer to a pretty-printed JSON string.
pub fn to_json<T: serde::Serialize>(t: &T) -> Result<String> {
    Ok(serde_json::to_string_pretty(t)?)
}

/// Deserialize a transformer from a JSON string.
pub fn from_json<T: serde::de::DeserializeOwned>(s: &str) -> Result<T> {
    Ok(serde_json::from_str(s)?)
}

/// Serialize a fitted transformer and write it to the file at `path`.
///
/// `path` accepts anything implementing [`AsRef<Path>`] (e.g. `&str`, `String`,
/// `&Path`, `PathBuf`).
pub fn save_json<T: serde::Serialize, P: AsRef<Path>>(t: &T, path: P) -> Result<()> {
    let s = to_json(t)?;
    std::fs::write(path, s)?;
    Ok(())
}

/// Read a transformer from the JSON file at `path`.
///
/// `path` accepts anything implementing [`AsRef<Path>`] (e.g. `&str`, `String`,
/// `&Path`, `PathBuf`).
pub fn load_json<T: serde::de::DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<T> {
    let s = std::fs::read_to_string(path)?;
    from_json(&s)
}
