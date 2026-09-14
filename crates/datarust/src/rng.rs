//! Deterministic seedable PRNG (xorshift64), shared crate-wide.
//!
//! One PRNG implementation for the whole crate — `model_selection` splitters,
//! `cluster::KMeans`, `decomposition::randomized_svd`, and `tree` estimators —
//! keeps runs fully reproducible with zero external dependencies: no `rand`
//! crate, identical sequences for identical seeds on every platform.
//!
//! Moved here (from `model_selection::rng`) in v0.7; the old path re-exports
//! this type so existing code keeps compiling.

/// xorshift64 uniform PRNG. Identical seeds produce identical sequences.
///
/// Seed `0` is mapped to a non-zero splinter constant to avoid the degenerate
/// all-zero state.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Create a new generator. `seed == 0` is remapped to a fixed non-zero
    /// constant (the golden-ratio splinter) so the stream is never stuck at 0.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    /// Advance the state and return the next raw `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Uniform `f64` in `[0, 1)`.
    pub fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform `f64` in `[lo, hi)`. Panics if `lo >= hi` or either bound is
    /// non-finite.
    pub fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        assert!(
            lo.is_finite() && hi.is_finite() && lo < hi,
            "next_range requires finite lo < hi"
        );
        // Avoid forming `hi - lo`, which can overflow for finite endpoints
        // such as `(-f64::MAX, f64::MAX)`. This convex combination remains
        // finite for every finite `lo < hi`.
        let unit = self.next_unit();
        lo * (1.0 - unit) + hi * unit
    }

    /// Uniform `usize` in `[0, hi)`. `hi == 0` returns `0` (no-op).
    ///
    /// The float-based mapping is retained for compatibility with the
    /// pre-v0.7 model-selection RNG stream; the final clamp protects the
    /// documented bound in the extreme `usize` case. Prefer `next_u64` /
    /// `next_unit` with integer rejection sampling when unbiased
    /// subsampling matters.
    pub fn next_usize(&mut self, hi: usize) -> usize {
        self.next_index(hi)
    }

    /// Uniform `usize` in `[0, hi)`. `hi == 0` returns `0`.
    pub fn next_index(&mut self, hi: usize) -> usize {
        if hi == 0 {
            return 0;
        }
        let index = (self.next_unit() * hi as f64) as usize;
        index.min(hi - 1)
    }

    /// Standard normal sample via the Box–Muller transform.
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit().max(1e-300);
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        r * theta.cos()
    }

    /// Shuffle a slice in place using the Fisher–Yates algorithm.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        if n < 2 {
            return;
        }
        for i in (1..n).rev() {
            let j = self.next_index(i + 1);
            slice.swap(i, j);
        }
    }
}

impl Default for Rng {
    fn default() -> Self {
        Rng::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinism_same_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let mut diffs = 0;
        for _ in 0..10 {
            if a.next_u64() != b.next_u64() {
                diffs += 1;
            }
        }
        assert!(diffs > 5);
    }

    #[test]
    fn next_unit_in_unit_interval() {
        let mut rng = Rng::new(7);
        for _ in 0..1000 {
            let u = rng.next_unit();
            assert!((0.0..1.0).contains(&u), "next_unit out of range: {u}");
        }
    }

    #[test]
    fn next_usize_bounded() {
        let mut rng = Rng::new(99);
        for _ in 0..1000 {
            let k = rng.next_usize(10);
            assert!(k < 10, "next_usize out of range: {k}");
        }
        assert_eq!(rng.next_usize(0), 0);
    }

    #[test]
    fn next_index_bounded() {
        let mut rng = Rng::new(99);
        for _ in 0..1000 {
            let k = rng.next_index(10);
            assert!(k < 10, "next_index out of range: {k}");
        }
        assert_eq!(rng.next_index(0), 0);
    }

    #[test]
    fn next_range_bounded() {
        let mut rng = Rng::new(3);
        for _ in 0..1000 {
            let v = rng.next_range(-2.0, 5.0);
            assert!((-2.0..5.0).contains(&v), "next_range out of range: {v}");
        }
    }

    #[test]
    fn next_range_handles_extreme_finite_bounds() {
        let mut rng = Rng::new(123);
        for _ in 0..1000 {
            let value = rng.next_range(-f64::MAX, f64::MAX);
            assert!(value.is_finite());
            assert!((-f64::MAX..f64::MAX).contains(&value));
        }
    }

    #[test]
    fn next_usize_preserves_legacy_sequence() {
        let mut rng = Rng::new(42);
        let mut expected = Rng::new(42);
        for _ in 0..100 {
            let old_style = (expected.next_unit() * 37.0) as usize;
            assert_eq!(rng.next_usize(37), old_style.min(36));
        }
    }

    #[test]
    fn next_normal_finite() {
        let mut rng = Rng::new(11);
        for _ in 0..1000 {
            assert!(rng.next_normal().is_finite());
        }
    }

    #[test]
    fn shuffle_is_permutation() {
        let mut rng = Rng::new(5);
        let mut data: Vec<usize> = (0..64).collect();
        let sorted = data.clone();
        rng.shuffle(&mut data);
        data.sort_unstable();
        assert_eq!(data, sorted);
    }

    #[test]
    fn zero_seed_is_remapped_not_stuck() {
        let mut rng = Rng::new(0);
        let a = rng.next_u64();
        let b = rng.next_u64();
        assert_ne!(a, b, "all-zero state would be stuck");
        assert_ne!(a, 0);
    }
}
