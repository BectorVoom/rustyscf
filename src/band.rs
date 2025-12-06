use crate::cell::Cell;
use crate::error::{Error, Result};
use crate::kpoints::{KPath, KPoint};
use crate::scf::ScfResult;

/// Builder for band-structure evaluation using an SCF result.
pub struct BandStructureBuilder<'a> {
    cell: &'a Cell,
    scf: &'a ScfResult,
    kpath: Option<KPath>,
}

/// Band-structure data container.
#[derive(Debug, Default)]
pub struct BandStructureResult {
    pub kpoints: Vec<KPoint>,
    pub energies: Vec<Vec<f64>>, // kpoint-major
    pub occupations: Option<Vec<Vec<f64>>>,
}

impl<'a> BandStructureBuilder<'a> {
    pub fn new(cell: &'a Cell, scf: &'a ScfResult) -> Self {
        Self {
            cell,
            scf,
            kpath: None,
        }
    }

    pub fn with_kpath(mut self, kpath: KPath) -> Self {
        self.kpath = Some(kpath);
        self
    }

    /// Compute band energies along the configured path. Placeholder for now.
    pub fn run(self) -> Result<BandStructureResult> {
        let _ = (self.cell, self.scf, self.kpath);
        Err(Error::InternalError(
            "Band-structure kernel not implemented in Task 1".into(),
        ))
    }
}

impl BandStructureResult {
    pub fn energies(&self) -> &[Vec<f64>] {
        &self.energies
    }

    pub fn energy(&self, k_idx: usize, band_idx: usize) -> Option<f64> {
        self.energies
            .get(k_idx)
            .and_then(|b| b.get(band_idx))
            .copied()
    }

    pub fn kpoints(&self) -> &[KPoint] {
        &self.kpoints
    }
}
