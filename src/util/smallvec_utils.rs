//! Helpers for working with `smallvec` without scattering type aliases.

use smallvec::SmallVec;

/// Fixed inline capacity for short coordinate-style arrays.
pub const INLINE_CAPACITY_6: usize = 6;

/// Convenience alias for a small vector of `f64`.
pub type SmallF64Vec = SmallVec<[f64; INLINE_CAPACITY_6]>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_vec_grows_inline() {
        let mut v: SmallF64Vec = SmallVec::new();
        v.extend_from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
        assert!(v.spilled() == false);
    }
}
