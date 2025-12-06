use crate::backend::memory::{BackendMemory, ScfMemoryParams};
use crate::backend::types::{
    BackendKind, BackendInfo, ComplexGridHandle, DeviceBuffer, MatrixHandle, MemoryUsage,
    VectorHandle,
};
use crate::backend::error::Result;

/// Linear algebra operations (GEMM + Hermitian eigendecomposition).
pub trait BackendLinalg: Send + Sync {
    fn matmul(
        &self,
        a: &MatrixHandle,
        b: &MatrixHandle,
        c: &mut MatrixHandle,
        alpha: f64,
        beta: f64,
    ) -> Result<()>;

    fn eig_hermitian(
        &self,
        a: &MatrixHandle,
        evals: &mut VectorHandle,
        evecs: &mut MatrixHandle,
    ) -> Result<()>;
}

/// 3D FFT (split-complex) operations.
pub trait BackendFft: Send + Sync {
    fn fft3d_forward(&self, grid: &mut ComplexGridHandle) -> Result<()>;
    fn fft3d_inverse(&self, grid: &mut ComplexGridHandle) -> Result<()>;
}

/// Reductions.
pub trait BackendReduce: Send + Sync {
    fn sum_f64(&self, buf: &DeviceBuffer<f64>, len: usize) -> Result<f64>;
}

/// Unified backend surface exposed to higher-level modules.
pub trait Backend: Send + Sync {
    /// Backend selection (CPU/WGPU).
    fn kind(&self) -> BackendKind;
    /// Runtime/memory metadata.
    fn info(&self) -> BackendInfo;

    /// Subsystems
    fn linalg(&self) -> &dyn BackendLinalg;
    fn fft(&self) -> &dyn BackendFft;
    fn reduce(&self) -> &dyn BackendReduce;
    fn memory(&self) -> &dyn BackendMemory;

    /// Convenience wrappers delegating to subsystem traits.
    fn matmul(
        &self,
        a: &MatrixHandle,
        b: &MatrixHandle,
        c: &mut MatrixHandle,
        alpha: f64,
        beta: f64,
    ) -> Result<()> {
        self.linalg().matmul(a, b, c, alpha, beta)
    }

    fn eig_hermitian(
        &self,
        a: &MatrixHandle,
        evals: &mut VectorHandle,
        evecs: &mut MatrixHandle,
    ) -> Result<()> {
        self.linalg().eig_hermitian(a, evals, evecs)
    }

    fn fft3d_forward(&self, grid: &mut ComplexGridHandle) -> Result<()> {
        self.fft().fft3d_forward(grid)
    }

    fn fft3d_inverse(&self, grid: &mut ComplexGridHandle) -> Result<()> {
        self.fft().fft3d_inverse(grid)
    }

    fn sum_f64(&self, buf: &DeviceBuffer<f64>, len: usize) -> Result<f64> {
        self.reduce().sum_f64(buf, len)
    }

    /// Convenience helpers for common memory ops.
    fn alloc_f64(&self, len: usize, usage: MemoryUsage) -> Result<DeviceBuffer<f64>> {
        self.memory().alloc_f64(len, usage)
    }
    fn upload_f64(&self, data: &[f64], usage: MemoryUsage) -> Result<DeviceBuffer<f64>> {
        self.memory().upload_f64(data, usage)
    }
    fn free_f64(&self, buf: DeviceBuffer<f64>) -> Result<()> {
        self.memory().free_f64(buf)
    }
    fn write_f64(&self, buf: &mut DeviceBuffer<f64>, data: &[f64]) -> Result<()> {
        self.memory().write_f64(buf, data)
    }
    fn read_f64(&self, buf: &DeviceBuffer<f64>, len: usize) -> Result<Vec<f64>> {
        self.memory().read_f64(buf, len)
    }
    fn max_memory_mb(&self) -> usize {
        self.memory().max_memory_mb()
    }
    fn used_memory_mb(&self) -> usize {
        self.memory().used_memory_mb()
    }
    fn estimate_scf_memory_mb(&self, params: &ScfMemoryParams) -> usize {
        self.memory().estimate_scf_memory_mb(params)
    }
}
