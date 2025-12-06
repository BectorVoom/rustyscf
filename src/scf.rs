use crate::backend::BackendConfig;
use crate::backend::{BackendError, make_backend};
use crate::backend::FockWorkspace;
use crate::cell::Cell;
use crate::error::{Error, Result};
use crate::kpoints::{KMesh, KPoint};
use crate::integrals::pbc::{CpuIntegralProvider, PbcIntegralProvider};
use crate::linalg::Matrix;

/// SCF methods supported in v0.1.0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    KRHF,
}

/// Builder for SCF runs. Numerical routines are added later; this type fixes
/// the public surface and defaults.
pub struct ScfBuilder<'a> {
    cell: &'a Cell,
    method: Method,
    kmesh: Option<KMesh>,
    conv_tol: f64,
    max_cycle: usize,
    backend: BackendConfig,
    integral_provider: Option<Box<dyn PbcIntegralProvider + Send + Sync + 'static>>,
}

/// Placeholder SCF result container.
pub struct ScfResult {
    pub converged: bool,
    pub cell: Cell,
    pub kmesh: KMesh,
    pub kpoints: Vec<KPoint>,
    pub density: DensityKpts,
    pub nao: usize,
    pub fermi_level: Option<f64>,
    pub integral_provider: Box<dyn PbcIntegralProvider + Send + Sync + 'static>,
}

/// Minimal density matrix container on the SCF sampling k-mesh.
#[derive(Debug, Default, Clone)]
pub struct DensityKpts {
    /// Density matrices flattened (k-major) for placeholder use.
    pub dms: Vec<Vec<f64>>,
    pub nao: usize,
}

impl DensityKpts {
    pub fn zero(nao: usize, nkpts: usize) -> Self {
        let dm_size = nao * nao;
        let mut dms = Vec::with_capacity(nkpts);
        for _ in 0..nkpts {
            dms.push(vec![0.0; dm_size]);
        }
        Self { dms, nao }
    }
}

/// Trait describing the data required by band-structure workflows.
pub trait PeriodicScf {
    fn cell(&self) -> &Cell;
    fn kmesh(&self) -> &[KPoint];
    fn density(&self) -> &DensityKpts;
    fn fermi_level(&self) -> Option<f64>;
    fn ao_dimension(&self) -> usize;
    fn converged(&self) -> bool;
    fn integral_provider(&self) -> &dyn PbcIntegralProvider;

    /// Build Fock and overlap matrices at an arbitrary band k-point.
    fn build_fock_at_k(&self, k_band: &KPoint) -> Result<(Matrix, Matrix)>;
}

impl<'a> ScfBuilder<'a> {
    pub fn new(cell: &'a Cell) -> Self {
        Self {
            cell,
            method: Method::KRHF,
            kmesh: None,
            conv_tol: 1e-8,
            max_cycle: 10,
            backend: BackendConfig::cpu(),
            integral_provider: None,
        }
    }

    pub fn with_method(mut self, method: Method) -> Self {
        self.method = method;
        self
    }

    pub fn with_kmesh(mut self, kmesh: KMesh) -> Self {
        self.kmesh = Some(kmesh);
        self
    }

    pub fn with_conv_tol(mut self, tol: f64) -> Self {
        self.conv_tol = tol;
        self
    }

    pub fn with_max_cycle(mut self, max_cycle: usize) -> Self {
        self.max_cycle = max_cycle;
        self
    }

    pub fn with_backend(mut self, backend: BackendConfig) -> Self {
        self.backend = backend;
        self
    }

    pub fn with_integral_provider(
        mut self,
        provider: Box<dyn PbcIntegralProvider + Send + Sync + 'static>,
    ) -> Self {
        self.integral_provider = Some(provider);
        self
    }

    /// Execute the SCF procedure. Currently returns a stub result.
    pub fn run(self) -> Result<ScfResult> {
        // Enforce dimension scope: only 3D periodic systems supported in v0.1.0.
        if self.cell.dimension != 3 {
            return Err(Error::InputError(
                crate::error::InputError::UnsupportedDimension {
                    dimension: self.cell.dimension,
                },
            ));
        }

        // Validate required inputs
        let kmesh = self.kmesh.ok_or_else(|| {
            Error::InputError(crate::error::InputError::InvalidParameter {
                parameter: "kmesh".into(),
                message: "k-point mesh missing".into(),
            })
        })?;

        // Enforce MVP scope: only 4x4x4 meshes allowed for now.
        if kmesh.dims != [4, 4, 4] {
            return Err(Error::InputError(
                crate::error::InputError::UnsupportedKMesh {
                    kmesh: [
                        kmesh.dims[0] as usize,
                        kmesh.dims[1] as usize,
                        kmesh.dims[2] as usize,
                    ],
                },
            ));
        }

        // Simple resource gate: if a max memory is set and too low, fail fast.
        const ESTIMATED_MB: usize = 512;
        if self.backend.max_memory_mb < ESTIMATED_MB {
            return Err(BackendError::OutOfMemory {
                requested_mb: Some(ESTIMATED_MB),
                limit_mb: Some(self.backend.max_memory_mb),
            }
            .into());
        }

        // Construct backend; execute a minimal backend op to validate wiring.
        let backend = make_backend(&self.backend).map_err(Error::from)?;
        // Basic backend smoke test: sum over zero buffer and run placeholder J skeleton.
        let buf = backend
            .alloc_f64(1, crate::backend::MemoryUsage::Transient)
            .map_err(Error::from)?;
        let _ = backend.sum_f64(&buf, 1).map_err(Error::from)?;
        backend.free_f64(buf).map_err(Error::from)?;

        // Placeholder J-build skeleton on a tiny grid to validate FFT plumbing.
        let grid = [2, 2, 2];
        let ngrid = grid.iter().product();
        let mut workspace = FockWorkspace::new(backend.as_ref(), ngrid).map_err(Error::from)?;
        let coulomb = backend
            .alloc_f64(ngrid, crate::backend::MemoryUsage::Transient)
            .map_err(Error::from)?;
        // fill v(G)=0 to keep it cheap
        backend
            .write_f64(&mut coulomb.clone(), &vec![0.0; ngrid])
            .map_err(Error::from)?;
        let mut j_mat = backend
            .alloc_f64(1, crate::backend::MemoryUsage::Transient)
            .map_err(Error::from)?;
        backend
            .write_f64(&mut j_mat, &[0.0])
            .map_err(Error::from)?;
        let j_handle = crate::backend::MatrixHandle::new(j_mat, 1, 1);
        let _ = crate::backend::fock::build_j_skeleton(
            backend.as_ref(),
            grid,
            &coulomb,
            &mut workspace,
            &mut [j_handle],
            None,
            None,
        )
        .map_err(|e| Error::ResourceError {
            message: format!("J-build skeleton failed: {e:?}"),
            requested_memory_mb: None,
            limit_memory_mb: None,
        })?;

        // Placeholder successful result for CPU path until kernels exist.
        let _ = (
            self.cell,
            self.method,
            kmesh,
            self.conv_tol,
            self.max_cycle,
            self.backend,
        );

        let nao = 1; // placeholder: single AO per cell for stub behavior
        let kpoints = kmesh.generate_points();
        let density = DensityKpts::zero(nao, kpoints.len());
        let integral_provider: Box<dyn PbcIntegralProvider + Send + Sync> =
            match self.integral_provider {
                Some(p) => p,
                None => Box::new(CpuIntegralProvider),
            };

        Ok(ScfResult {
            converged: true,
            cell: self.cell.clone(),
            kmesh,
            kpoints,
            density,
            nao,
            fermi_level: Some(0.0),
            integral_provider,
        })
    }
}

impl PeriodicScf for ScfResult {
    fn cell(&self) -> &Cell {
        &self.cell
    }

    fn kmesh(&self) -> &[KPoint] {
        &self.kpoints
    }

    fn density(&self) -> &DensityKpts {
        &self.density
    }

    fn fermi_level(&self) -> Option<f64> {
        self.fermi_level
    }

    fn ao_dimension(&self) -> usize {
        self.nao
    }

    fn converged(&self) -> bool {
        self.converged
    }

    fn integral_provider(&self) -> &dyn PbcIntegralProvider {
        self.integral_provider.as_ref()
    }

    fn build_fock_at_k(&self, _k_band: &KPoint) -> Result<(Matrix, Matrix)> {
        let nao = self.nao;
        let overlap = self.integral_provider.overlap_at_k(&self.cell, _k_band, nao)?;
        let kinetic = self.integral_provider.kinetic_at_k(&self.cell, _k_band, nao)?;
        let v_nuc = self.integral_provider.v_nuc_at_k(&self.cell, _k_band, nao)?;
        let (j, k) = self
            .integral_provider
            .j_k_at_k(&self.cell, &self.density, _k_band, nao)?;

        // F = T + V_nuc + J - K  (DFT XC will be added later)
        let mut fock = Matrix::zeros(nao, nao);
        for i in 0..nao {
            for j_idx in 0..nao {
                let idx = i * nao + j_idx;
                fock.data[idx] = kinetic.data[idx]
                    + v_nuc.data[idx]
                    + j.data[idx]
                    - k.data[idx];
            }
        }

        Ok((fock, overlap))
    }
}

impl std::fmt::Debug for ScfResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScfResult")
            .field("converged", &self.converged)
            .field("nao", &self.nao)
            .field("kmesh", &self.kmesh)
            .field("kpoints_len", &self.kpoints.len())
            .field("fermi_level", &self.fermi_level)
            .finish()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::CellBuilder;
    use crate::kpoints::KMesh;

    fn minimal_cell() -> Cell {
        CellBuilder::new()
            .with_atom("H 0 0 0")
            .with_a([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            .with_basis("gth-dzvp")
            .with_pseudo("gth-pbe")
            .build()
            .unwrap()
    }

    #[test]
    fn missing_kmesh_is_invalid_parameter() {
        let cell = minimal_cell();
        let err = ScfBuilder::new(&cell).run().unwrap_err();
        match err {
            Error::InputError(crate::error::InputError::InvalidParameter { parameter, .. }) => {
                assert_eq!(parameter, "kmesh");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn unsupported_kmesh_is_rejected() {
        let cell = minimal_cell();
        let err = ScfBuilder::new(&cell)
            .with_kmesh(KMesh::new([2, 2, 2]))
            .run()
            .unwrap_err();
        match err {
            Error::InputError(crate::error::InputError::UnsupportedKMesh { kmesh }) => {
                assert_eq!(kmesh, [2, 2, 2]);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn wgpu_backend_maps_to_resource_error() {
        let cell = minimal_cell();
        let err = ScfBuilder::new(&cell)
            .with_kmesh(KMesh::new([4, 4, 4]))
            .with_backend(BackendConfig::wgpu())
            .run()
            .unwrap_err();
        match err {
            Error::ResourceError { message, .. } => {
                assert!(message.contains("wgpu"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn low_memory_limit_maps_to_resource_error() {
        let cell = minimal_cell();
        let err = ScfBuilder::new(&cell)
            .with_kmesh(KMesh::new([4, 4, 4]))
            .with_backend(BackendConfig::cpu().with_max_memory_mb(128))
            .run()
            .unwrap_err();
        match err {
            Error::ResourceError {
                requested_memory_mb,
                limit_memory_mb,
                ..
            } => {
                assert_eq!(requested_memory_mb, Some(512));
                assert_eq!(limit_memory_mb, Some(128));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn happy_path_cpu_returns_converged_stub() {
        let cell = minimal_cell();
        let res = ScfBuilder::new(&cell)
            .with_kmesh(KMesh::new([4, 4, 4]))
            .run()
            .unwrap();
        assert!(res.converged);
    }

    #[test]
    fn cell_dimension_not_3_is_rejected() {
        let mut cell = minimal_cell();
        cell.dimension = 2;
        let err = ScfBuilder::new(&cell)
            .with_kmesh(KMesh::new([4, 4, 4]))
            .run()
            .unwrap_err();
        match err {
            Error::InputError(crate::error::InputError::UnsupportedDimension { dimension }) => {
                assert_eq!(dimension, 2);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
