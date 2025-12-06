use crate::backend::error::{BackendError, Result};
use crate::backend::runtime::CubeRuntimeHandle;
use crate::backend::types::{BackendKind, DeviceBuffer, MemoryUsage};
use std::sync::Arc;

/// Parameters used to estimate memory footprint for an SCF/Band job.
#[derive(Debug, Clone)]
pub struct ScfMemoryParams {
    /// Number of k-points.
    pub kpoints: usize,
    /// Number of atomic orbitals (AOs) / basis functions.
    pub aos: usize,
    /// Number of bands (occupied + virtual) considered.
    pub bands: usize,
    /// FFT mesh dimensions (nx, ny, nz).
    pub fft_mesh: [usize; 3],
    /// Count of temporary work buffers needed simultaneously.
    pub scratch_buffers: usize,
}

impl ScfMemoryParams {
    pub fn new(kpoints: usize, aos: usize, bands: usize, fft_mesh: [usize; 3]) -> Self {
        Self {
            kpoints,
            aos,
            bands,
            fft_mesh,
            scratch_buffers: 2, // density + fock scratch by default
        }
    }

    pub fn with_scratch_buffers(mut self, scratch_buffers: usize) -> Self {
        self.scratch_buffers = scratch_buffers.max(1);
        self
    }
}

/// Memory abstraction that hides CPU/WGPU allocation details.
pub trait BackendMemory: Send + Sync {
    fn backend_kind(&self) -> BackendKind;
    fn alloc_f64(&self, len: usize, usage: MemoryUsage) -> Result<DeviceBuffer<f64>>;
    fn upload_f64(&self, data: &[f64], usage: MemoryUsage) -> Result<DeviceBuffer<f64>>;
    fn free_f64(&self, buf: DeviceBuffer<f64>) -> Result<()>;
    fn write_f64(&self, buf: &mut DeviceBuffer<f64>, data: &[f64]) -> Result<()>;
    fn read_f64(&self, buf: &DeviceBuffer<f64>, len: usize) -> Result<Vec<f64>>;
    fn max_memory_mb(&self) -> usize;
    fn used_memory_mb(&self) -> usize;

    /// Rough upper bound for SCF workload in MB. Conservative by design to
    /// avoid over-commit on GPU memory.
    fn estimate_scf_memory_mb(&self, params: &ScfMemoryParams) -> usize {
        // matrices: kpoints * (aos x aos) for density/fock intermediates
        let ao_mat = params
            .aos
            .saturating_mul(params.aos)
            .saturating_mul(params.kpoints.max(1));
        // wavefunction/band amplitudes per k-point
        let band_mat = params
            .bands
            .saturating_mul(params.aos)
            .saturating_mul(params.kpoints.max(1));
        // FFT grid (split complex, so 2x)
        let fft_cells = params
            .fft_mesh
            .iter()
            .copied()
            .fold(1usize, |acc, v| acc.saturating_mul(v))
            .saturating_mul(2); // re + im
        // scratch buffers
        let scratch = ao_mat.saturating_mul(params.scratch_buffers.max(1));

        let total_f64 = ao_mat
            .saturating_add(band_mat)
            .saturating_add(fft_cells)
            .saturating_add(scratch);
        let total_bytes = total_f64.saturating_mul(std::mem::size_of::<f64>());
        (total_bytes + (1024 * 1024 - 1)) / (1024 * 1024)
    }
}

/// Internal memory manager that dispatches to CPU or WGPU runtime helpers.
pub(crate) struct MemoryManager {
    runtime: Arc<CubeRuntimeHandle>,
    max_memory_mb: usize,
}

impl MemoryManager {
    pub fn new(runtime: Arc<CubeRuntimeHandle>, max_memory_mb: usize) -> Self {
        Self {
            runtime,
            max_memory_mb,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::runtime::build_runtime;
    use crate::backend::types::{BackendConfig, BackendKind, MemoryUsage};
    #[cfg(feature = "wgpu-backend")]
    use crate::backend::make_backend;
    #[cfg(feature = "wgpu-backend")]
    use crate::backend::error::BackendError;

    #[test]
    fn estimate_is_conservative_and_positive() {
        let dummy = MemoryManager::new(Arc::new(CubeRuntimeHandle::Cpu(crate::backend::runtime::CpuRuntime::new(&BackendConfig::cpu()))), usize::MAX);
        let params = ScfMemoryParams::new(4, 64, 32, [20, 20, 20]).with_scratch_buffers(3);
        let mb = dummy.estimate_scf_memory_mb(&params);
        assert!(mb > 0);
    }

    #[test]
    fn cpu_alloc_tracks_usage() {
        let cfg = BackendConfig::cpu().with_max_memory_mb(4);
        let runtime = Arc::new(build_runtime(&cfg).expect("cpu runtime"));
        assert!(matches!(runtime.kind(), BackendKind::Cpu));
        let memory = MemoryManager::new(runtime.clone(), cfg.max_memory_mb);

        // allocate just over 1 MiB to ensure accounting is visible in MB units
        let over_one_mb = (1024 * 1024 / std::mem::size_of::<f64>()) + 8;
        let buf = memory
            .alloc_f64(over_one_mb, MemoryUsage::Persistent)
            .expect("alloc" );
        assert!(memory.used_memory_mb() >= 1);
        memory.free_f64(buf).unwrap();
    }

    #[cfg(feature = "wgpu-backend")]
    #[test]
    fn wgpu_alloc_upload_read_roundtrip() {
        let cfg = BackendConfig::wgpu().with_max_memory_mb(256);
        let backend = match std::panic::catch_unwind(|| make_backend(&cfg)) {
            Ok(Ok(b)) => b,
            Ok(Err(BackendError::DeviceUnavailable { message })) => {
                eprintln!("skipping wgpu test: {message}");
                return;
            }
            Err(_) => {
                eprintln!("skipping wgpu test: cubecl wgpu init panicked (no adapter?)");
                return;
            }
            Ok(Err(e)) => panic!("unexpected backend error: {e:?}"),
        };

        let data = vec![1.5f64, 2.5, -4.0, 8.0];
        let buf = backend.upload_f64(&data, MemoryUsage::Transient).expect("upload");
        let roundtrip = backend.read_f64(&buf, data.len()).expect("read");
        assert_eq!(roundtrip, data);

        let sum = backend.sum_f64(&buf, data.len()).expect("sum");
        assert!((sum - data.iter().sum::<f64>()).abs() < 1e-9);

        backend.free_f64(buf).unwrap();
    }
}

impl BackendMemory for MemoryManager {
    fn backend_kind(&self) -> BackendKind {
        self.runtime.kind()
    }

    fn alloc_f64(&self, len: usize, usage: MemoryUsage) -> Result<DeviceBuffer<f64>> {
        match self.runtime.as_ref() {
            CubeRuntimeHandle::Cpu(cpu) => cpu.alloc_f64(len, usage),
            #[cfg(feature = "wgpu-backend")]
            CubeRuntimeHandle::Wgpu(wgpu) => wgpu.alloc_f64(len, usage),
            #[cfg(not(feature = "wgpu-backend"))]
            CubeRuntimeHandle::Wgpu(wgpu) => Err(BackendError::DeviceUnavailable {
                message: format!("wgpu backend unavailable: {}", wgpu.reason()),
            }),
        }
    }

    fn upload_f64(&self, data: &[f64], usage: MemoryUsage) -> Result<DeviceBuffer<f64>> {
        match self.runtime.as_ref() {
            CubeRuntimeHandle::Cpu(cpu) => cpu.upload_f64(data, usage),
            #[cfg(feature = "wgpu-backend")]
            CubeRuntimeHandle::Wgpu(wgpu) => wgpu.upload_f64(data, usage),
            #[cfg(not(feature = "wgpu-backend"))]
            CubeRuntimeHandle::Wgpu(wgpu) => Err(BackendError::DeviceUnavailable {
                message: format!("wgpu backend unavailable: {}", wgpu.reason()),
            }),
        }
    }

    fn free_f64(&self, buf: DeviceBuffer<f64>) -> Result<()> {
        match self.runtime.as_ref() {
            CubeRuntimeHandle::Cpu(cpu) => {
                cpu.free(buf);
                Ok(())
            }
            #[cfg(feature = "wgpu-backend")]
            CubeRuntimeHandle::Wgpu(wgpu) => {
                wgpu.free(buf);
                Ok(())
            }
            #[cfg(not(feature = "wgpu-backend"))]
            CubeRuntimeHandle::Wgpu(_) => {
                drop(buf);
                Ok(())
            }
        }
    }

    fn write_f64(&self, buf: &mut DeviceBuffer<f64>, data: &[f64]) -> Result<()> {
        match self.runtime.as_ref() {
            CubeRuntimeHandle::Cpu(cpu) => cpu.write_f64(buf, data),
            #[cfg(feature = "wgpu-backend")]
            CubeRuntimeHandle::Wgpu(wgpu) => wgpu.write_f64(buf, data),
            #[cfg(not(feature = "wgpu-backend"))]
            CubeRuntimeHandle::Wgpu(wgpu) => Err(BackendError::DeviceUnavailable {
                message: format!("wgpu backend unavailable: {}", wgpu.reason()),
            }),
        }
    }

    fn read_f64(&self, buf: &DeviceBuffer<f64>, len: usize) -> Result<Vec<f64>> {
        match self.runtime.as_ref() {
            CubeRuntimeHandle::Cpu(cpu) => cpu.read_f64(buf, len),
            #[cfg(feature = "wgpu-backend")]
            CubeRuntimeHandle::Wgpu(wgpu) => wgpu.read_f64(buf, len),
            #[cfg(not(feature = "wgpu-backend"))]
            CubeRuntimeHandle::Wgpu(wgpu) => Err(BackendError::DeviceUnavailable {
                message: format!("wgpu backend unavailable: {}", wgpu.reason()),
            }),
        }
    }

    fn max_memory_mb(&self) -> usize {
        self.max_memory_mb
    }

    fn used_memory_mb(&self) -> usize {
        match self.runtime.as_ref() {
            CubeRuntimeHandle::Cpu(cpu) => cpu.used_memory_mb(),
            #[cfg(feature = "wgpu-backend")]
            CubeRuntimeHandle::Wgpu(wgpu) => wgpu.used_memory_mb(),
            #[cfg(not(feature = "wgpu-backend"))]
            CubeRuntimeHandle::Wgpu(_) => 0,
        }
    }
}
