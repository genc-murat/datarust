//! K-fold and stratified K-fold cross-validation splitters.
//!
//! Mirror `sklearn.model_selection.KFold` and `StratifiedKFold`. Each splitter
//! produces an iterator of `(train_indices, test_indices)` via `split()`.

use crate::error::{DatarustError, Result};
use crate::model_selection::rng::Rng;

/// K-fold cross-validation splitter.
///
/// Divides `n_samples` into `n_splits` consecutive folds (optionally shuffled);
/// each fold serves as the test set once, with the remaining folds as training.
#[derive(Debug, Clone)]
pub struct KFold {
    n_splits: usize,
    shuffle: bool,
    random_state: Option<u64>,
}

impl Default for KFold {
    fn default() -> Self {
        Self::new()
    }
}

impl KFold {
    /// New K-fold with `n_splits = 5`, `shuffle = false`.
    pub fn new() -> Self {
        Self {
            n_splits: 5,
            shuffle: false,
            random_state: None,
        }
    }

    /// Builder: number of folds (default `5`). Must be `>= 2`.
    pub fn with_n_splits(mut self, n: usize) -> Self {
        self.n_splits = n;
        self
    }

    /// Builder: whether to shuffle indices before folding (default `false`).
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Builder: deterministic seed for the shuffle.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Returns an iterator of `(train_indices, test_indices)` pairs, one per
    /// fold. The union of all test sets covers every sample exactly once.
    ///
    /// Errors if `n_splits > n_samples` or `n_splits < 2`.
    pub fn split(
        &self,
        n_samples: usize,
    ) -> Result<impl Iterator<Item = (Vec<usize>, Vec<usize>)> + '_> {
        if n_samples == 0 {
            return Err(DatarustError::EmptyInput("n_samples is 0".into()));
        }
        if self.n_splits < 2 {
            return Err(DatarustError::InvalidInput(format!(
                "n_splits must be >= 2, got {}",
                self.n_splits
            )));
        }
        if self.n_splits > n_samples {
            return Err(DatarustError::InvalidInput(format!(
                "n_splits ({}) cannot be greater than n_samples ({})",
                self.n_splits, n_samples
            )));
        }

        let mut indices: Vec<usize> = (0..n_samples).collect();
        if self.shuffle {
            let seed = self.random_state.unwrap_or(0x9E3779B97F4A7C15);
            Rng::new(seed).shuffle(&mut indices);
        }

        // Fold boundaries: sklearn distributes remainder to the first folds.
        let n_splits = self.n_splits;
        let fold_sizes = fold_sizes(n_samples, n_splits);
        Ok(fold_sizes
            .into_iter()
            .scan(0usize, move |start, fold_size| {
                let test = indices[*start..*start + fold_size].to_vec();
                let train: Vec<usize> = indices[..*start]
                    .iter()
                    .chain(indices[*start + fold_size..].iter())
                    .copied()
                    .collect();
                *start += fold_size;
                Some((train, test))
            }))
    }
}

/// Stratified K-fold cross-validation splitter.
///
/// Each fold preserves the class ratio of the full dataset (for binary
/// classification targets in `{0.0, 1.0}`). Useful when classes are imbalanced.
#[derive(Debug, Clone)]
pub struct StratifiedKFold {
    n_splits: usize,
    shuffle: bool,
    random_state: Option<u64>,
}

impl Default for StratifiedKFold {
    fn default() -> Self {
        Self::new()
    }
}

impl StratifiedKFold {
    /// New stratified splitter with `n_splits = 5`, `shuffle = false`.
    pub fn new() -> Self {
        Self {
            n_splits: 5,
            shuffle: false,
            random_state: None,
        }
    }

    /// Builder: number of folds (default `5`). Must be `>= 2`.
    pub fn with_n_splits(mut self, n: usize) -> Self {
        self.n_splits = n;
        self
    }

    /// Builder: whether to shuffle within each class before folding (default `false`).
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Builder: deterministic seed for the shuffle.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Returns an iterator of `(train_indices, test_indices)` pairs that
    /// preserve the class balance of `y`. `y` holds binary `{0.0, 1.0}` labels.
    pub fn split(&self, y: &[f64]) -> Result<impl Iterator<Item = (Vec<usize>, Vec<usize>)> + '_> {
        let n = y.len();
        if n == 0 {
            return Err(DatarustError::EmptyInput("y is empty".into()));
        }
        if self.n_splits < 2 {
            return Err(DatarustError::InvalidInput(format!(
                "n_splits must be >= 2, got {}",
                self.n_splits
            )));
        }
        if self.n_splits > n {
            return Err(DatarustError::InvalidInput(format!(
                "n_splits ({}) cannot be greater than n_samples ({})",
                self.n_splits, n
            )));
        }

        // Group sample indices by class.
        let mut class0: Vec<usize> = Vec::new();
        let mut class1: Vec<usize> = Vec::new();
        for (i, &label) in y.iter().enumerate() {
            if label >= 0.5 {
                class1.push(i);
            } else {
                class0.push(i);
            }
        }
        if self.shuffle {
            let seed = self.random_state.unwrap_or(0x9E3779B97F4A7C15);
            let mut rng = Rng::new(seed);
            rng.shuffle(&mut class0);
            rng.shuffle(&mut class1);
        }

        // Assign each class's samples round-robin across the folds.
        let n_splits = self.n_splits;
        let mut folds: Vec<Vec<usize>> = vec![Vec::new(); n_splits];
        for (f, idx) in class0.iter().enumerate() {
            folds[f % n_splits].push(*idx);
        }
        for (f, idx) in class1.iter().enumerate() {
            folds[f % n_splits].push(*idx);
        }

        Ok(folds.into_iter().map(move |test_idx| {
            let test_set: std::collections::HashSet<usize> = test_idx.iter().copied().collect();
            let train: Vec<usize> = (0..n).filter(|i| !test_set.contains(i)).collect();
            (train, test_idx)
        }))
    }
}

/// Compute fold sizes, distributing the remainder to the first folds (sklearn style).
fn fold_sizes(n_samples: usize, n_splits: usize) -> Vec<usize> {
    let base = n_samples / n_splits;
    let rem = n_samples % n_splits;
    (0..n_splits)
        .map(|i| base + if i < rem { 1 } else { 0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kfold_default_five_folds() {
        let kf = KFold::new();
        let folds: Vec<_> = kf.split(20).unwrap().collect();
        assert_eq!(folds.len(), 5);
        for (train, test) in &folds {
            assert_eq!(train.len() + test.len(), 20);
        }
    }

    #[test]
    fn kfold_each_sample_tested_once() {
        let kf = KFold::new().with_n_splits(4);
        let folds: Vec<_> = kf.split(12).unwrap().collect();
        let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.iter().copied()).collect();
        all_test.sort();
        assert_eq!(all_test, (0..12).collect::<Vec<_>>());
    }

    #[test]
    fn kfold_train_test_disjoint() {
        let kf = KFold::new().with_n_splits(3);
        for (train, test) in kf.split(9).unwrap() {
            let tr: std::collections::HashSet<usize> = train.iter().copied().collect();
            let te: std::collections::HashSet<usize> = test.iter().copied().collect();
            assert!(tr.is_disjoint(&te));
        }
    }

    #[test]
    fn kfold_remainder_distributed() {
        // 10 samples, 3 folds → sizes 4, 3, 3.
        let sizes = fold_sizes(10, 3);
        assert_eq!(sizes, vec![4, 3, 3]);
    }

    #[test]
    fn kfold_shuffle_deterministic() {
        let kf = KFold::new()
            .with_n_splits(3)
            .with_shuffle(true)
            .with_random_state(7);
        let a: Vec<_> = kf.split(15).unwrap().collect();
        let b: Vec<_> = kf.split(15).unwrap().collect();
        assert_eq!(a, b);
    }

    #[test]
    fn kfold_shuffle_still_covers_all() {
        let kf = KFold::new()
            .with_n_splits(4)
            .with_shuffle(true)
            .with_random_state(1);
        let folds: Vec<_> = kf.split(20).unwrap().collect();
        let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.iter().copied()).collect();
        all_test.sort();
        assert_eq!(all_test, (0..20).collect::<Vec<_>>());
    }

    #[test]
    fn kfold_n_splits_too_large_rejected() {
        let kf = KFold::new().with_n_splits(11);
        // Force the Result to resolve by collecting; the error path never yields.
        let res: Result<Vec<(Vec<usize>, Vec<usize>)>> = kf.split(10).map(|it| it.collect());
        assert!(matches!(res, Err(DatarustError::InvalidInput(_))));
    }

    #[test]
    fn kfold_n_splits_too_small_rejected() {
        let kf = KFold::new().with_n_splits(1);
        let res: Result<Vec<(Vec<usize>, Vec<usize>)>> = kf.split(10).map(|it| it.collect());
        assert!(matches!(res, Err(DatarustError::InvalidInput(_))));
    }

    #[test]
    fn stratified_preserves_class_balance() {
        // 20 samples, 10 of class 0, 10 of class 1.
        let y: Vec<f64> = (0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }).collect();
        let skf = StratifiedKFold::new().with_n_splits(4);
        let folds: Vec<_> = skf.split(&y).unwrap().collect();
        assert_eq!(folds.len(), 4);
        for (_, test) in &folds {
            let n1 = test.iter().filter(|&&i| y[i] >= 0.5).count();
            let n0 = test.len() - n1;
            // Each fold should have roughly balanced classes.
            assert!(n0 > 0 && n1 > 0, "fold has no class diversity: {test:?}");
        }
    }

    #[test]
    fn stratified_covers_all_samples() {
        let y: Vec<f64> = (0..16)
            .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        let skf = StratifiedKFold::new().with_n_splits(4);
        let folds: Vec<_> = skf.split(&y).unwrap().collect();
        let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.iter().copied()).collect();
        all_test.sort();
        assert_eq!(all_test, (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn stratified_train_test_disjoint() {
        let y: Vec<f64> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let skf = StratifiedKFold::new().with_n_splits(2);
        for (train, test) in skf.split(&y).unwrap() {
            let tr: std::collections::HashSet<usize> = train.iter().copied().collect();
            let te: std::collections::HashSet<usize> = test.iter().copied().collect();
            assert!(tr.is_disjoint(&te));
        }
    }
}
