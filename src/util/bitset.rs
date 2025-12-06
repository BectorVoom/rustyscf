//! Thin helpers around `bitvec` for occupancy/flag handling.

pub type BitBox = bitvec::boxed::BitBox;
pub type BitVec = bitvec::vec::BitVec;
pub type BitSlice<T = usize, O = bitvec::order::Lsb0> = bitvec::slice::BitSlice<T, O>;

/// Allocate a zeroed bitvec of length `len`.
pub fn zeroed(len: usize) -> BitVec {
    BitVec::repeat(false, len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_creates_false_bits() {
        let bits = zeroed(4);
        assert_eq!(bits.len(), 4);
        assert!(bits.not_any());
    }
}
