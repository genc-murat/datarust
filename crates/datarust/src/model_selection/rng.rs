//! Deterministic seedable PRNG (xorshift64), shared by `model_selection` and
//! `decomposition::randomized_svd`.
//!
//! Keeping one PRNG implementation here (rather than duplicating it) preserves
//! the crate's zero-dependency ethos: no `rand` crate, fully reproducible runs.

/// xorshift64 uniform PRNG. Identical seeds produce identical sequences.
///
/// Seed `0` is mapped to a non-zero splinter constant to avoid the degenerate
/// all-zero state.
pub(crate) struct Rng {
    state: u64,
}

impl Rng {
    /// Create a new generator. `seed == 0` is remapped to a fixed non-zero
    /// constant (the golden-ratio splinter) so the stream is never stuck at 0.
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    /// Advance the state and return the next raw `u64`.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Uniform `f64` in `[0, 1)`.
    pub(crate) fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform `f64` in `[lo, hi)`. Panics if `lo >= hi`.
    #[allow(dead_code)]
    pub(crate) fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_unit()
    }

    /// Uniform `usize` in `[0, hi)`. `hi == 0` returns `0` (no-op).
    pub(crate) fn next_usize(&mut self, hi: usize) -> usize {
        if hi == 0 {
            return 0;
        }
        (self.next_unit() * hi as f64) as usize
    }

    /// Standard normal sample via the Box–Muller transform.
    pub(crate) fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit().max(1e-300);
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        r * theta.cos()
    }

    /// Shuffle a slice in place using the Fisher–Yates algorithm.
    pub(crate) fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        if n < 2 {
            return;
        }
        for i in (1..n).rev() {
            let j = self.next_usize(i + 1);
            slice.swap(i, j);
        }
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
    fn next_range_bounded() {
        let mut rng = Rng::new(5);
        for _ in 0..1000 {
            let v = rng.next_range(-3.0, 7.0);
            assert!((-3.0..7.0).contains(&v), "next_range out of range: {v}");
        }
    }

    #[test]
    fn shuffle_preserves_elements() {
        let mut rng = Rng::new(11);
        let mut v: Vec<u32> = (0..50).collect();
        let original = v.clone();
        rng.shuffle(&mut v);
        // Same multiset (sorted equal).
        let mut sorted = v.clone();
        sorted.sort();
        assert_eq!(sorted, original);
    }

    #[test]
    fn shuffle_changes_order_for_large_input() {
        let mut rng = Rng::new(123);
        let mut v: Vec<u32> = (0..100).collect();
        let before = v.clone();
        rng.shuffle(&mut v);
        let diffs = before.iter().zip(v.iter()).filter(|(a, b)| a != b).count();
        assert!(diffs > 50, "shuffle barely changed order: {diffs} diffs");
    }

    #[test]
    fn shuffle_deterministic_same_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        let mut va: Vec<u32> = (0..30).collect();
        let mut vb: Vec<u32> = (0..30).collect();
        a.shuffle(&mut va);
        b.shuffle(&mut vb);
        assert_eq!(va, vb);
    }

    #[test]
    fn seed_zero_remapped() {
        // seed 0 must not be degenerate.
        let mut rng = Rng::new(0);
        let v = rng.next_u64();
        assert_ne!(v, 0);
    }
}
