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

#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::*;
    use crate::matrix::Matrix;
    use crate::scaler::StandardScaler;
    use crate::traits::Transformer;

    #[test]
    fn to_from_json_round_trips_scaler_params() {
        let x = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]).unwrap();
        let mut scaler = StandardScaler::new();
        scaler.fit(&x).unwrap();
        let json = to_json(&scaler).unwrap();
        let restored: StandardScaler = from_json(&json).unwrap();
        // Fitted parameters (mean, std) must survive the round trip.
        assert_eq!(restored.mean(), scaler.mean());
        assert_eq!(restored.std(), scaler.std());
    }

    #[test]
    fn save_json_creates_file_and_load_restores() {
        let x = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]).unwrap();
        let mut scaler = StandardScaler::new();
        scaler.fit(&x).unwrap();
        let path = std::env::temp_dir().join("datarust_serialize_test.json");
        // Clean any leftover from a prior run.
        let _ = std::fs::remove_file(&path);

        save_json(&scaler, &path).unwrap();
        assert!(path.exists(), "save_json did not create the file");

        let restored: StandardScaler = load_json(&path).unwrap();
        // Loaded scaler must produce the same transform output as the original.
        let original_out = scaler.transform(&x).unwrap();
        let restored_out = restored.transform(&x).unwrap();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!(
                    (original_out.get(i, j) - restored_out.get(i, j)).abs() < 1e-12,
                    "i={i} j={j}"
                );
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_json_missing_file_errors() {
        let path = std::env::temp_dir().join("datarust_does_not_exist_12345.json");
        let _ = std::fs::remove_file(&path);
        let result: Result<StandardScaler> = load_json(&path);
        assert!(result.is_err(), "loading a missing file should error");
        // The error must surface as the Io variant.
        assert!(matches!(
            result.unwrap_err(),
            crate::error::DatarustError::Io(_)
        ));
    }

    #[test]
    fn from_json_malformed_errors() {
        let result: Result<StandardScaler> = from_json("not valid json {");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::error::DatarustError::Serde(_)
        ));
    }
}
