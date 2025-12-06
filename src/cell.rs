use crate::error::{Error, InputError, Result};

/// Unit for lattice vectors and atomic coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Unit {
    Angstrom,
    Bohr,
}

/// Periodic simulation cell. Internal fields are placeholder until the
/// numerical backend is implemented.
#[derive(Debug, Clone)]
pub struct Cell {
    pub(crate) atom: String,
    pub(crate) a: [[f64; 3]; 3],
    pub(crate) unit: Unit,
    pub(crate) basis: String,
    pub(crate) pseudo: String,
    pub(crate) spin: i32,
    pub(crate) dimension: i32,
}

impl Cell {
    /// Atomic specification string (PySCF-style).
    pub fn atom(&self) -> &str {
        &self.atom
    }

    /// Lattice vectors (rows) in the configured unit.
    pub fn lattice(&self) -> [[f64; 3]; 3] {
        self.a
    }

    pub fn unit(&self) -> Unit {
        self.unit
    }

    pub fn basis(&self) -> &str {
        &self.basis
    }

    pub fn pseudo(&self) -> &str {
        &self.pseudo
    }

    pub fn spin(&self) -> i32 {
        self.spin
    }

    pub fn dimension(&self) -> i32 {
        self.dimension
    }
}

/// Builder for [`Cell`]. Derives mesh/precision internally in later tasks.
pub struct CellBuilder {
    atom: Option<String>,
    a: Option<[[f64; 3]; 3]>,
    unit: Unit,
    basis: Option<String>,
    pseudo: Option<String>,
    spin: i32,
    dimension: i32,
}

impl CellBuilder {
    /// Create a new builder with sensible defaults (Angstrom, spin=0, dim=3).
    pub fn new() -> Self {
        Self {
            atom: None,
            a: None,
            unit: Unit::Angstrom,
            basis: None,
            pseudo: None,
            spin: 0,
            dimension: 3,
        }
    }

    pub fn with_atom(mut self, atom: &str) -> Self {
        self.atom = Some(atom.to_string());
        self
    }

    pub fn with_a(mut self, a: [[f64; 3]; 3]) -> Self {
        self.a = Some(a);
        self
    }

    pub fn with_unit(mut self, unit: Unit) -> Self {
        self.unit = unit;
        self
    }

    pub fn with_basis(mut self, basis: &str) -> Self {
        self.basis = Some(basis.to_string());
        self
    }

    pub fn with_pseudo(mut self, pseudo: &str) -> Self {
        self.pseudo = Some(pseudo.to_string());
        self
    }

    pub fn with_spin(mut self, spin: i32) -> Self {
        self.spin = spin;
        self
    }

    pub fn with_dimension(mut self, dimension: i32) -> Self {
        self.dimension = dimension;
        self
    }

    /// Build a validated [`Cell`]. Placeholder validation for now.
    pub fn build(self) -> Result<Cell> {
        let atom = self.atom.ok_or_else(|| {
            Error::InputError(InputError::InvalidAtomSpec {
                message: "atom specification missing".into(),
            })
        })?;

        let a = self.a.ok_or_else(|| {
            Error::InputError(InputError::InvalidParameter {
                parameter: "a".into(),
                message: "lattice vectors missing".into(),
            })
        })?;

        let basis = self.basis.ok_or_else(|| {
            Error::InputError(InputError::InvalidParameter {
                parameter: "basis".into(),
                message: "basis set missing".into(),
            })
        })?;

        let pseudo = self.pseudo.ok_or_else(|| {
            Error::InputError(InputError::InvalidParameter {
                parameter: "pseudo".into(),
                message: "pseudopotential missing".into(),
            })
        })?;

        if self.dimension != 3 {
            return Err(Error::InputError(InputError::UnsupportedDimension {
                dimension: self.dimension,
            }));
        }

        Ok(Cell {
            atom,
            a,
            unit: self.unit,
            basis,
            pseudo,
            spin: self.spin,
            dimension: self.dimension,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_atom_is_invalid_atom_spec() {
        let err = CellBuilder::new()
            .with_a([[0.0; 3]; 3])
            .build()
            .unwrap_err();
        match err {
            Error::InputError(InputError::InvalidAtomSpec { message }) => {
                assert!(message.contains("atom"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn missing_lattice_vectors_is_invalid_parameter() {
        let err = CellBuilder::new()
            .with_atom("H 0 0 0")
            .with_basis("gth-dzvp")
            .with_pseudo("gth-pbe")
            .build()
            .unwrap_err();
        match err {
            Error::InputError(InputError::InvalidParameter { parameter, .. }) => {
                assert_eq!(parameter, "a");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn missing_basis_is_invalid_parameter() {
        let err = CellBuilder::new()
            .with_atom("H 0 0 0")
            .with_a([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            .with_pseudo("gth-pbe")
            .build()
            .unwrap_err();
        match err {
            Error::InputError(InputError::InvalidParameter { parameter, .. }) => {
                assert_eq!(parameter, "basis");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn missing_pseudo_is_invalid_parameter() {
        let err = CellBuilder::new()
            .with_atom("H 0 0 0")
            .with_a([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            .with_basis("gth-dzvp")
            .build()
            .unwrap_err();
        match err {
            Error::InputError(InputError::InvalidParameter { parameter, .. }) => {
                assert_eq!(parameter, "pseudo");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn unsupported_dimension_is_rejected() {
        let err = CellBuilder::new()
            .with_atom("H 0 0 0")
            .with_a([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            .with_basis("gth-dzvp")
            .with_pseudo("gth-pbe")
            .with_dimension(2)
            .build()
            .unwrap_err();
        match err {
            Error::InputError(InputError::UnsupportedDimension { dimension }) => {
                assert_eq!(dimension, 2);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn cell_getters_expose_fields() {
        let cell = CellBuilder::new()
            .with_atom("H 0 0 0")
            .with_a([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            .with_basis("gth-dzvp")
            .with_pseudo("gth-pbe")
            .build()
            .unwrap();

        assert_eq!(cell.atom(), "H 0 0 0");
        assert_eq!(
            cell.lattice(),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );
        assert_eq!(cell.unit(), Unit::Angstrom);
        assert_eq!(cell.basis(), "gth-dzvp");
        assert_eq!(cell.pseudo(), "gth-pbe");
        assert_eq!(cell.spin(), 0);
        assert_eq!(cell.dimension(), 3);
    }
}
