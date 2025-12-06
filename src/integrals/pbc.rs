//! Placeholder PBC integral builders. Numerical kernels will be replaced with
//! cubecl-backed implementations; these functions currently return zeros/identity
//! with correct shapes to allow end-to-end flow during scaffolding.

use crate::backend::{make_backend, BackendConfig, MemoryUsage, MatrixHandle};
use crate::backend::traits::Backend;
use crate::backend::fock::FockWorkspace;
use crate::cell::Cell;
use crate::error::Result;
use crate::kpoints::KPoint;
use crate::linalg::Matrix;
use crate::scf::DensityKpts;

/// Trait for supplying PBC AO integrals; enables swapping CPU/GPU backends.
pub trait PbcIntegralProvider: Send + Sync {
    fn overlap_at_k(&self, cell: &Cell, k: &KPoint, nao: usize) -> Result<Matrix>;
    fn kinetic_at_k(&self, cell: &Cell, k: &KPoint, nao: usize) -> Result<Matrix>;
    fn v_nuc_at_k(&self, cell: &Cell, k: &KPoint, nao: usize) -> Result<Matrix>;
    fn j_k_at_k(
        &self,
        cell: &Cell,
        density: &DensityKpts,
        k_band: &KPoint,
        nao: usize,
    ) -> Result<(Matrix, Matrix)>;

    /// Hint to downstream code to pick GPU-accelerated paths when available.
    fn prefers_gpu(&self) -> bool {
        false
    }
}

/// CPU provider placeholder; currently returns simple zero/identity matrices.
pub struct CpuIntegralProvider;

impl PbcIntegralProvider for CpuIntegralProvider {
    fn overlap_at_k(&self, _cell: &Cell, _k: &KPoint, nao: usize) -> Result<Matrix> {
        Ok(Matrix::identity(nao))
    }

    fn kinetic_at_k(&self, _cell: &Cell, _k: &KPoint, nao: usize) -> Result<Matrix> {
        Ok(Matrix::zeros(nao, nao))
    }

    fn v_nuc_at_k(&self, _cell: &Cell, _k: &KPoint, nao: usize) -> Result<Matrix> {
        Ok(Matrix::zeros(nao, nao))
    }

    fn j_k_at_k(
        &self,
        _cell: &Cell,
        _density: &DensityKpts,
        _k_band: &KPoint,
        nao: usize,
    ) -> Result<(Matrix, Matrix)> {
        Ok((Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)))
    }
}

/// Cubecl-backed provider backed by a persistent backend instance to avoid
/// runtime re-registration panics. Currently reuses CPU kernels for AO pieces
/// but exercises cubecl memory / fock skeleton for J to validate plumbing.
pub struct CubeclIntegralProvider {
    backend: Box<dyn Backend>,
}

impl CubeclIntegralProvider {
    pub fn new(config: BackendConfig) -> Result<Self> {
        let backend = make_backend(&config)?;
        Ok(Self { backend })
    }
}

impl PbcIntegralProvider for CubeclIntegralProvider {
    fn overlap_at_k(&self, _cell: &Cell, _k: &KPoint, nao: usize) -> Result<Matrix> {
        // TODO: replace with cubecl overlap kernel. For now, stage identity on
        // device then read back to exercise the cubecl memory path.
        let mut host = vec![0.0; nao * nao];
        for i in 0..nao {
            host[i * nao + i] = 1.0;
        }
        let buf = self.backend.upload_f64(&host, MemoryUsage::Transient)?;
        let data = self.backend.read_f64(&buf, host.len())?;
        self.backend.free_f64(buf)?;
        Ok(Matrix {
            nrow: nao,
            ncol: nao,
            data,
        })
    }

    fn kinetic_at_k(&self, _cell: &Cell, _k: &KPoint, nao: usize) -> Result<Matrix> {
        let buf = self
            .backend
            .alloc_f64(nao * nao, MemoryUsage::Transient)?;
        let data = self.backend.read_f64(&buf, nao * nao)?;
        self.backend.free_f64(buf)?;
        Ok(Matrix {
            nrow: nao,
            ncol: nao,
            data,
        })
    }

    fn v_nuc_at_k(&self, _cell: &Cell, _k: &KPoint, nao: usize) -> Result<Matrix> {
        let buf = self
            .backend
            .alloc_f64(nao * nao, MemoryUsage::Transient)?;
        let data = self.backend.read_f64(&buf, nao * nao)?;
        self.backend.free_f64(buf)?;
        Ok(Matrix {
            nrow: nao,
            ncol: nao,
            data,
        })
    }

    fn j_k_at_k(
        &self,
        _cell: &Cell,
        _density: &DensityKpts,
        _k_band: &KPoint,
        nao: usize,
    ) -> Result<(Matrix, Matrix)> {
        // Exercise cubecl pipeline: run minimal J skeleton on a tiny grid to
        // validate runtime and memory without producing meaningful values.
        let backend = self.backend.as_ref();
        let grid = [2, 2, 2];
        let ngrid = grid.iter().product();
        let mut workspace = FockWorkspace::new(backend, ngrid)?;
        let coulomb = backend.alloc_f64(ngrid, MemoryUsage::Transient)?;

        let mut j_buf = backend.alloc_f64(nao * nao, MemoryUsage::Transient)?;
        backend.write_f64(&mut j_buf, &vec![0.0; nao * nao])?;
        let j_handle = MatrixHandle::new(j_buf, nao, nao);

        let _ = crate::backend::fock::build_j_skeleton(
            backend,
            grid,
            &coulomb,
            &mut workspace,
            &mut [j_handle],
            None,
            None,
        );

        // Return zero matrices as placeholders until true J/K kernels are wired.
        Ok((Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)))
    }

    fn prefers_gpu(&self) -> bool {
        true
    }
}
