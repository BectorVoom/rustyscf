mod error;
mod kernels;
mod memory;
mod runtime;
mod traits;
mod types;

use std::sync::Arc;

use crate::backend::error::Result;
use crate::backend::kernels::{fft, linalg, reduce};
use crate::backend::memory::{BackendMemory, MemoryManager};
use crate::backend::runtime::build_runtime;
use crate::backend::traits::{Backend, BackendFft, BackendLinalg, BackendReduce};
pub use crate::backend::types::*;
#[allow(unused_imports)]
pub use crate::backend::traits::*;

/// Concrete backend wrapping a Cube runtime; composes linalg/fft/reduce/memory facades.
pub struct CubeBackend {
    runtime: Arc<runtime::CubeRuntimeHandle>,
    memory: MemoryManager,
    linalg: LinalgImpl,
    fft: FftImpl,
    reduce: ReduceImpl,
}

impl CubeBackend {
    pub fn new(config: BackendConfig) -> Result<Self> {
        let runtime = Arc::new(build_runtime(&config)?);
        let memory = MemoryManager::new(runtime.clone(), config.max_memory_mb);
        let linalg = LinalgImpl {
            runtime: runtime.clone(),
        };
        let fft = FftImpl {
            runtime: runtime.clone(),
        };
        let reduce = ReduceImpl {
            runtime: runtime.clone(),
        };
        Ok(Self {
            runtime,
            memory,
            linalg,
            fft,
            reduce,
        })
    }
}

impl Backend for CubeBackend {
    fn kind(&self) -> BackendKind {
        self.runtime.kind()
    }

    fn info(&self) -> BackendInfo {
        BackendInfo {
            kind: self.kind(),
            device_name: self.runtime.device_name(),
            max_memory_mb: self.memory.max_memory_mb(),
        }
    }

    fn linalg(&self) -> &dyn BackendLinalg {
        &self.linalg
    }

    fn fft(&self) -> &dyn BackendFft {
        &self.fft
    }

    fn reduce(&self) -> &dyn BackendReduce {
        &self.reduce
    }

    fn memory(&self) -> &dyn BackendMemory {
        &self.memory
    }
}

/// Factory used by higher layers.
pub fn make_backend(config: &BackendConfig) -> Result<Box<dyn Backend>> {
    let backend = CubeBackend::new(config.clone())?;
    Ok(Box::new(backend))
}

/// Linalg facade delegating to kernel wrappers.
struct LinalgImpl {
    runtime: Arc<runtime::CubeRuntimeHandle>,
}

impl BackendLinalg for LinalgImpl {
    fn matmul(
        &self,
        a: &MatrixHandle,
        b: &MatrixHandle,
        c: &mut MatrixHandle,
        alpha: f64,
        beta: f64,
    ) -> Result<()> {
        linalg::gemm(self.runtime.as_ref(), a, b, c, alpha, beta)
    }

    fn eig_hermitian(
        &self,
        a: &MatrixHandle,
        evals: &mut VectorHandle,
        evecs: &mut MatrixHandle,
    ) -> Result<()> {
        // Reuse existing placeholder; evecs currently unused.
        let mut a_owned = MatrixHandle::new(a.buffer.clone(), a.rows, a.cols);
        linalg::eigh(self.runtime.as_ref(), &mut a_owned, evals)?;
        // evecs not computed yet; keep placeholder semantics.
        let _ = evecs;
        Ok(())
    }
}

/// FFT facade delegating to kernel wrappers.
struct FftImpl {
    runtime: Arc<runtime::CubeRuntimeHandle>,
}

impl BackendFft for FftImpl {
    fn fft3d_forward(&self, grid: &mut ComplexGridHandle) -> Result<()> {
        fft::fft3d_forward(self.runtime.as_ref(), grid)
    }

    fn fft3d_inverse(&self, grid: &mut ComplexGridHandle) -> Result<()> {
        fft::fft3d_inverse(self.runtime.as_ref(), grid)
    }
}

/// Reduce facade delegating to kernel wrappers.
struct ReduceImpl {
    runtime: Arc<runtime::CubeRuntimeHandle>,
}

impl BackendReduce for ReduceImpl {
    fn sum_f64(&self, buf: &DeviceBuffer<f64>, len: usize) -> Result<f64> {
        reduce::sum_f64(self.runtime.as_ref(), buf, len)
    }
}

pub use error::BackendError;
pub use runtime::CubeRuntimeHandle;
#[cfg(feature = "wgpu-backend")]
pub use runtime::WgpuRuntime;
