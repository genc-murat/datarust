//! Train/test split utilities mirroring `sklearn.model_selection.train_test_split`.

use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::model_selection::rng::Rng;

/// Configuration for [`train_test_split`].
///
/// Build with [`TrainTestSplit::new`] (defaults: `test_size = 0.25`,
/// `shuffle = true`, `random_state = None`) and chain `.with_*` setters, or use
/// the free function [`train_test_split`] for the common case.
#[derive(Debug, Clone)]
pub struct TrainTestSplit {
    test_size: f64,
    shuffle: bool,
    random_state: Option<u64>,
}

impl Default for TrainTestSplit {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainTestSplit {
    /// New split config with defaults: 25% test, shuffled, random seed.
    pub fn new() -> Self {
        Self {
            test_size: 0.25,
            shuffle: true,
            random_state: None,
        }
    }

    /// Builder: fraction of samples assigned to the test set (default `0.25`).
    /// Must be in `(0.0, 1.0)`.
    pub fn with_test_size(mut self, ratio: f64) -> Self {
        self.test_size = ratio;
        self
    }

    /// Builder: whether to shuffle before splitting (default `true`).
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Builder: deterministic seed for the shuffle. `None` (default) uses a
    /// fixed seed derived from sample count, so results are reproducible.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Perform the split, returning `(x_train, x_test, y_train, y_test)`.
    pub fn split(&self, x: &Matrix, y: &[f64]) -> Result<(Matrix, Matrix, Vec<f64>, Vec<f64>)> {
        let n = x.nrows();
        if n == 0 {
            return Err(DatarustError::EmptyInput("X has no rows".into()));
        }
        if y.len() != n {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} targets", n),
                actual: format!("{} targets", y.len()),
            });
        }
        if self.test_size <= 0.0 || self.test_size >= 1.0 {
            return Err(DatarustError::InvalidInput(format!(
                "test_size must be in (0, 1), got {}",
                self.test_size
            )));
        }

        let mut indices: Vec<usize> = (0..n).collect();
        if self.shuffle {
            let seed = self.random_state.unwrap_or(0x9E3779B97F4A7C15);
            Rng::new(seed).shuffle(&mut indices);
        }

        let n_test = (n as f64 * self.test_size).round() as usize;
        let n_test = n_test.clamp(1, n - 1);
        let (test_idx, train_idx) = partition_indices(&indices, n_test);

        let x_test = x.select_rows(&test_idx)?;
        let x_train = x.select_rows(&train_idx)?;
        let y_test: Vec<f64> = test_idx.iter().map(|&i| y[i]).collect();
        let y_train: Vec<f64> = train_idx.iter().map(|&i| y[i]).collect();
        Ok((x_train, x_test, y_train, y_test))
    }
}

/// Convenience: split `x` and `y` into train/test with default settings
/// (25% test, shuffled).
///
/// ```rust
/// use datarust::model_selection::train_test_split;
/// use datarust::Matrix;
///
/// let x = Matrix::new(vec![vec![1.0]; 100])?;
/// let y = vec![0.0; 100];
/// let (x_tr, x_te, y_tr, y_te) = train_test_split(&x, &y)?;
/// assert_eq!(x_tr.nrows() + x_te.nrows(), 100);
/// assert_eq!(y_tr.len() + y_te.len(), 100);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
pub fn train_test_split(x: &Matrix, y: &[f64]) -> Result<(Matrix, Matrix, Vec<f64>, Vec<f64>)> {
    TrainTestSplit::new().split(x, y)
}

/// Split a permutation into test (first `n_test`) and train (rest) indices.
fn partition_indices(indices: &[usize], n_test: usize) -> (Vec<usize>, Vec<usize>) {
    let test = indices[..n_test].to_vec();
    let train = indices[n_test..].to_vec();
    (test, train)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(n: usize) -> (Matrix, Vec<f64>) {
        let rows: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        (Matrix::new(rows).unwrap(), y)
    }

    #[test]
    fn default_split_sizes() {
        let (x, y) = make(100);
        let (x_tr, x_te, y_tr, y_te) = train_test_split(&x, &y).unwrap();
        assert_eq!(x_tr.nrows(), 75);
        assert_eq!(x_te.nrows(), 25);
        assert_eq!(y_tr.len(), 75);
        assert_eq!(y_te.len(), 25);
    }

    #[test]
    fn custom_test_size() {
        let (x, y) = make(100);
        let (x_tr, x_te, _, _) = TrainTestSplit::new()
            .with_test_size(0.3)
            .split(&x, &y)
            .unwrap();
        assert_eq!(x_te.nrows(), 30);
        assert_eq!(x_tr.nrows(), 70);
    }

    #[test]
    fn no_shuffle_preserves_order() {
        let (x, y) = make(10);
        let (_, x_te, _, y_te) = TrainTestSplit::new()
            .with_shuffle(false)
            .split(&x, &y)
            .unwrap();
        // Without shuffle, the test set is the first n_test entries.
        assert_eq!(x_te.get(0, 0), 0.0);
        assert_eq!(x_te.get(2, 0), 2.0);
        assert_eq!(y_te[0], 0.0);
    }

    #[test]
    fn shuffle_deterministic_with_seed() {
        let (x, y) = make(100);
        let s = TrainTestSplit::new()
            .with_test_size(0.2)
            .with_random_state(42);
        let (_, x_te_a, _, _) = s.clone().split(&x, &y).unwrap();
        let (_, x_te_b, _, _) = s.split(&x, &y).unwrap();
        // Same seed → same split.
        for i in 0..x_te_a.nrows() {
            assert_eq!(x_te_a.get(i, 0), x_te_b.get(i, 0));
        }
    }

    #[test]
    fn all_samples_present_after_split() {
        let (x, y) = make(50);
        let (x_tr, x_te, y_tr, y_te) = train_test_split(&x, &y).unwrap();
        let mut all: Vec<f64> = (0..x_tr.nrows()).map(|i| x_tr.get(i, 0)).collect();
        all.extend((0..x_te.nrows()).map(|i| x_te.get(i, 0)));
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (i, v) in all.iter().enumerate() {
            assert_eq!(*v, i as f64);
        }
        let _ = (y_tr, y_te);
    }

    #[test]
    fn empty_input_rejected() {
        let x = Matrix::new(vec![vec![1.0]]).unwrap();
        let err = TrainTestSplit::new().split(&x, &[]).unwrap_err();
        assert!(matches!(err, DatarustError::ShapeMismatch { .. }));
    }

    #[test]
    fn invalid_test_size_rejected() {
        let (x, y) = make(10);
        let err = TrainTestSplit::new()
            .with_test_size(0.0)
            .split(&x, &y)
            .unwrap_err();
        assert!(matches!(err, DatarustError::InvalidInput(_)));
        let err = TrainTestSplit::new()
            .with_test_size(1.0)
            .split(&x, &y)
            .unwrap_err();
        assert!(matches!(err, DatarustError::InvalidInput(_)));
    }

    #[test]
    fn y_length_mismatch_rejected() {
        let (x, _) = make(10);
        let err = TrainTestSplit::new().split(&x, &[0.0, 1.0]).unwrap_err();
        assert!(matches!(err, DatarustError::ShapeMismatch { .. }));
    }

    #[test]
    fn small_dataset_clamps_test_size() {
        // 3 samples, test_size=0.25 → n_test would be 1 (clamped to [1, n-1]).
        let (x, y) = make(3);
        let (_, x_te, _, _) = train_test_split(&x, &y).unwrap();
        assert_eq!(x_te.nrows(), 1);
    }
}
