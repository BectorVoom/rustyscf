use crate::backend::BackendConfig;
use crate::backend::{BackendError, make_backend};
use crate::cell::Cell;
use crate::error::{Error, Result};
use crate::kpoints::KMesh;

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
}

/// Placeholder SCF result container.
#[derive(Debug, Default)]
pub struct ScfResult {
    pub converged: bool,
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
        let buf = backend
            .alloc_f64(1, crate::backend::MemoryUsage::Transient)
            .map_err(Error::from)?;
        // Sum should return 0.0 for zero-initialised buffer on CPU; on WGPU this
        // will currently return a kernel failure until kernels are wired.
        let _ = backend.sum_f64(&buf, 1).map_err(Error::from)?;
        backend.free_f64(buf).map_err(Error::from)?;

        // Placeholder successful result for CPU path until kernels exist.
        let _ = (
            self.cell,
            self.method,
            kmesh,
            self.conv_tol,
            self.max_cycle,
            self.backend,
        );

        Ok(ScfResult { converged: true })
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
