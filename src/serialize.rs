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

use crate::error::{DatarustError, Result};

/// Serialize a fitted transformer to a JSON string.
pub fn to_json<T: serde::Serialize>(t: &T) -> Result<String> {
    serde_json::to_string_pretty(t).map_err(|e| DatarustError::InvalidInput(e.to_string()))
}

/// Deserialize a transformer from a JSON string.
pub fn from_json<T: serde::de::DeserializeOwned>(s: &str) -> Result<T> {
    serde_json::from_str(s).map_err(|e| DatarustError::InvalidInput(e.to_string()))
}

/// Serialize a fitted transformer and write it to the file at `path`.
pub fn save_json<T: serde::Serialize>(t: &T, path: &str) -> Result<()> {
    let s = to_json(t)?;
    std::fs::write(path, s).map_err(|e| DatarustError::InvalidInput(e.to_string()))
}

/// Read a transformer from the JSON file at `path`.
pub fn load_json<T: serde::de::DeserializeOwned>(path: &str) -> Result<T> {
    let s =
        std::fs::read_to_string(path).map_err(|e| DatarustError::InvalidInput(e.to_string()))?;
    from_json(&s)
}
