//! rustyscf: library-first skeleton for cubecl-based band-structure workflows.
//!
//! This crate currently provides public type definitions and builders; numerical
//! kernels land in later milestones. The API is kept stable and documented so
//! downstream crates can prototype against it.
//!
//! # Quickstart
//!
//! ```no_run
//! use rustyscf::{CellBuilder, Unit, KMesh, ScfBuilder, Method, BandStructureBuilder};
//!
//! let cell = CellBuilder::new()
//!     .with_atom("H 0 0 0; H 1 1 1")
//!     .with_a([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
//!     .with_unit(Unit::Angstrom)
//!     .with_basis("gth-dzvp")
//!     .with_pseudo("gth-pbe")
//!     .build()
//!     .unwrap();
//!
//! let scf = ScfBuilder::new(&cell)
//!     .with_method(Method::KRHF)
//!     .with_kmesh(KMesh::new([2, 2, 2]))
//!     .run();
//!
//! // BandStructureBuilder produces band energies along a path; kernels are
//! // currently placeholders that return zero-valued bands with correct shapes.
//! let _bands = BandStructureBuilder::new(&cell, &scf.unwrap()).run();
//! ```

pub mod backend;
pub mod band;
pub mod cell;
pub mod error;
pub mod integrals;
pub mod kpoints;
pub mod linalg;
pub mod scf;
pub mod util;

// Re-exports for ergonomic `use rustyscf::*;` patterns
pub use crate::backend::{BackendConfig, BackendKind};
pub use crate::band::{BandLinalgBackend, BandStructure, BandStructureBuilder, BandStructureResult};
pub use crate::cell::{Cell, CellBuilder, Unit};
pub use crate::error::{Error, Result};
pub use crate::kpoints::{KMesh, KPath, KPathSegment, KPoint};
pub use crate::linalg::{CpuEigenSolver, CubeclEigenSolver, Matrix};
pub use crate::scf::{Method, ScfBuilder, ScfResult};
