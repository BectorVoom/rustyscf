//! Lightweight wrappers around `bumpalo` to centralise short-lived allocations.

use bumpalo::Bump;

/// Type alias for arena-backed vectors.
pub type BumpVec<'a, T> = bumpalo::collections::Vec<'a, T>;

/// Construct a new bump allocator. Keeping this in one place makes it easier
/// to swap strategies later (e.g., thread-local arenas).
pub fn new_arena() -> Bump {
    Bump::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bump_vec_allocates_from_arena() {
        let arena = new_arena();
        let mut v: BumpVec<'_, u8> = BumpVec::new_in(&arena);
        v.extend_from_slice(&[1, 2, 3]);
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }
}
