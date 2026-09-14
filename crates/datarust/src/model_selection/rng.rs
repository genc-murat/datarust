//! Shared deterministic PRNG (xorshift64).
//!
//! Moved to [`crate::rng`] in v0.7; this path re-exports the same type so
//! existing `use crate::model_selection::rng::Rng` imports keep compiling.
//! New code should prefer [`crate::rng::Rng`].

pub use crate::rng::Rng;

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn re_export_is_the_same_type() {
        let mut a = Rng::new(42);
        let mut b = crate::rng::Rng::new(42);
        assert_eq!(a.next_u64(), b.next_u64());
    }
}
