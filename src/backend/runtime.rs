use crate::backend::error::{BackendError, Result};
use crate::backend::types::{
    BackendConfig, BackendKind, DeviceBuffer, DeviceBufferRaw, MemoryUsage,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use cubecl_runtime::client::ComputeClient;
#[cfg(feature = "wgpu-backend")]
use bytemuck::cast_slice;
use cubecl_common::device::{Device, DeviceState};

#[cfg(feature = "wgpu-backend")]
use cubecl_wgpu::{init_device, init_setup, AutoGraphicsApi, RuntimeOptions as WgpuRuntimeOptions, WgpuDevice};

/// Unified runtime selector used by the backend.
#[derive(Debug)]
pub enum CubeRuntimeHandle {
    Cpu(CpuRuntime),
    Wgpu(WgpuRuntime),
}

impl CubeRuntimeHandle {
    pub fn kind(&self) -> BackendKind {
        match self {
            CubeRuntimeHandle::Cpu(_) => BackendKind::Cpu,
            CubeRuntimeHandle::Wgpu(_) => BackendKind::Wgpu,
        }
    }

    pub fn device_name(&self) -> String {
        match self {
            CubeRuntimeHandle::Cpu(_) => "cpu".to_string(),
            #[cfg(feature = "wgpu-backend")]
            CubeRuntimeHandle::Wgpu(wgpu) => wgpu
                .adapter_info
                .name
                .clone(),
            #[cfg(not(feature = "wgpu-backend"))]
            CubeRuntimeHandle::Wgpu(wgpu) => format!("wgpu-unavailable ({})", wgpu.reason()),
        }
    }

    /// Convenience: expose CPU client for cube kernels (for cubecl-kernels feature).
    #[cfg(feature = "cubecl-kernels")]
    pub fn cpu_client(&self) -> Option<&ComputeClient<cubecl_cpu::compute::server::CpuServer>> {
        match self {
            CubeRuntimeHandle::Cpu(cpu) => Some(cpu.client()),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub(crate) struct MemoryAccountant {
    used_bytes: AtomicUsize,
    pub(crate) max_bytes: usize,
}

impl MemoryAccountant {
    fn new(max_memory_mb: usize) -> Self {
        let max_bytes = max_memory_mb.saturating_mul(1024 * 1024);
        Self {
            used_bytes: AtomicUsize::new(0),
            max_bytes,
        }
    }

    fn try_reserve(&self, bytes: usize) -> Result<()> {
        let mut current = self.used_bytes.load(Ordering::SeqCst);
        loop {
            let next = current.saturating_add(bytes);
            if next > self.max_bytes {
                return Err(BackendError::OutOfMemory {
                    requested_mb: Some(bytes / (1024 * 1024).max(1)),
                    limit_mb: Some(self.max_bytes / (1024 * 1024).max(1)),
                });
            }
            match self
                .used_bytes
                .compare_exchange(current, next, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => return Ok(()),
                Err(updated) => current = updated,
            }
        }
    }

    fn release(&self, bytes: usize) {
        self.used_bytes.fetch_sub(bytes, Ordering::SeqCst);
    }

    fn used_memory_mb(&self) -> usize {
        self.used_bytes.load(Ordering::SeqCst) / (1024 * 1024).max(1)
    }
}

/// CPU runtime using host allocations; keeps logic minimal while aligning with
/// the single-source kernel concept.
pub struct CpuRuntime {
    memory: MemoryAccountant,
    client: ComputeClient<cubecl_cpu::compute::server::CpuServer>,
}

impl core::fmt::Debug for CpuRuntime {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CpuRuntime").finish()
    }
}

impl CpuRuntime {
    pub(crate) fn client(&self) -> &ComputeClient<cubecl_cpu::compute::server::CpuServer> {
        &self.client
    }
    pub fn new(cfg: &BackendConfig) -> Self {
        let device = cubecl_cpu::CpuDevice::default();
        let client = match std::panic::catch_unwind(|| {
            let server = cubecl_cpu::compute::server::CpuServer::init(device.to_id());
            ComputeClient::init(&device, server)
        }) {
            Ok(c) => c,
            Err(_) => ComputeClient::load(&device),
        };
        Self {
            memory: MemoryAccountant::new(cfg.max_memory_mb),
            client,
        }
    }

    pub fn alloc_f64(&self, len: usize, usage: MemoryUsage) -> Result<DeviceBuffer<f64>> {
        let bytes = len
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(BackendError::OutOfMemory {
                requested_mb: None,
                limit_mb: Some(self.memory.max_bytes / (1024 * 1024).max(1)),
            })?;

        self.memory.try_reserve(bytes)?;
        let zeroes = vec![0u8; bytes];
        let handle = self.client.create(&zeroes);
        Ok(DeviceBuffer::new(DeviceBufferRaw::CubeHandle(handle), len, bytes, usage))
    }

    pub fn upload_f64(&self, data: &[f64], usage: MemoryUsage) -> Result<DeviceBuffer<f64>> {
        let len = data.len();
        let bytes = len
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(BackendError::OutOfMemory {
                requested_mb: None,
                limit_mb: Some(self.memory.max_bytes / (1024 * 1024).max(1)),
            })?;
        self.memory.try_reserve(bytes)?;
        let bytes_slice = bytemuck::cast_slice(data);
        let handle = self.client.create(bytes_slice);
        Ok(DeviceBuffer::new(DeviceBufferRaw::CubeHandle(handle), len, bytes, usage))
    }

    pub fn free<T>(&self, buf: DeviceBuffer<T>) {
        self.memory.release(buf.bytes());
        drop(buf);
    }

    pub fn write_f64(&self, buf: &mut DeviceBuffer<f64>, data: &[f64]) -> Result<()> {
        if data.len() > buf.len {
            return Err(BackendError::KernelFailure {
                message: "write_f64 length exceeds buffer".into(),
            });
        }
        let bytes_slice = bytemuck::cast_slice(data);
        let handle = self.client.create(bytes_slice);
        if bytes_slice.len() != buf.bytes {
            self.memory.release(buf.bytes());
            self.memory.try_reserve(bytes_slice.len())?;
            buf.bytes = bytes_slice.len();
            buf.len = data.len();
        }
        buf.raw = DeviceBufferRaw::CubeHandle(handle);
        Ok(())
    }

    pub fn used_memory_mb(&self) -> usize {
        self.memory.used_memory_mb()
    }

    pub fn read_f64(&self, buf: &DeviceBuffer<f64>, len: usize) -> Result<Vec<f64>> {
        let handle = match &buf.raw {
            DeviceBufferRaw::CubeHandle(h) => h,
            _ => {
                return Err(BackendError::KernelFailure {
                    message: "read_f64 expects CubeHandle".into(),
                })
            }
        };
        let shape = [len];
        let strides = [1usize];
        let desc = handle.copy_descriptor(&shape, &strides, std::mem::size_of::<f64>());
        let bytes_vec = self.client.read_one_tensor(desc);
        let slice_f64 = bytemuck::try_cast_slice::<u8, f64>(&bytes_vec)
            .map_err(|_| BackendError::KernelFailure {
                message: "cpu read_f64: bytemuck cast failed".into(),
            })?;
        Ok(slice_f64.iter().take(len).copied().collect())
    }
}

/// WGPU runtime placeholder; constructed only when the `wgpu-backend` feature is enabled.
pub struct WgpuRuntime {
    reason: String,
    #[cfg(feature = "wgpu-backend")]
    client: ComputeClient<cubecl_wgpu::WgpuServer>,
    #[cfg(feature = "wgpu-backend")]
    pub adapter_info: wgpu::AdapterInfo,
    #[cfg(feature = "wgpu-backend")]
    memory: MemoryAccountant,
}

impl core::fmt::Debug for WgpuRuntime {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WgpuRuntime")
            .field("reason", &self.reason)
            .finish()
    }
}

impl WgpuRuntime {
    #[cfg(feature = "wgpu-backend")]
    pub fn new(_cfg: &BackendConfig) -> Result<Self> {
        let device = WgpuDevice::DefaultDevice;
        let mut options_setup = WgpuRuntimeOptions::default();
        let mut options_run = WgpuRuntimeOptions::default();
        if let Some(max_tasks) = _cfg.options.wgpu.as_ref().and_then(|o| o.max_tasks) {
            options_setup.tasks_max = max_tasks as usize;
            options_run.tasks_max = max_tasks as usize;
        }
        let setup = init_setup::<AutoGraphicsApi>(&device, options_setup);
        let adapter_info = setup.adapter.get_info();
        let device = init_device(setup, options_run);
        let client = ComputeClient::load(&device);

        Ok(Self {
            reason: format!(
                "wgpu runtime constructed ({} {:?})",
                adapter_info.name, adapter_info.backend
            ),
            client,
            adapter_info,
            memory: MemoryAccountant::new(_cfg.max_memory_mb),
        })
    }

    pub fn unavailable(reason: String) -> Self {
        Self {
            reason,
            #[cfg(feature = "wgpu-backend")]
            client: ComputeClient::load(&cubecl_wgpu::WgpuDevice::DefaultDevice),
            #[cfg(feature = "wgpu-backend")]
            adapter_info: wgpu::AdapterInfo {
                name: "unavailable".into(),
                vendor: 0,
                device: 0,
                device_type: wgpu::DeviceType::Other,
                backend: wgpu::Backend::Vulkan,
                driver: "".into(),
                driver_info: "".into(),
            },
            #[cfg(feature = "wgpu-backend")]
            memory: MemoryAccountant::new(0),
        }
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }

    #[cfg(feature = "wgpu-backend")]
    pub(crate) fn client(&self) -> &ComputeClient<cubecl_wgpu::WgpuServer> {
        &self.client
    }

    #[cfg(feature = "wgpu-backend")]
    pub fn alloc_f64(&self, len: usize, usage: MemoryUsage) -> Result<DeviceBuffer<f64>> {
        let bytes = len
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(BackendError::OutOfMemory {
                requested_mb: None,
                limit_mb: Some(self.memory.max_bytes / (1024 * 1024).max(1)),
            })?;
        self.memory.try_reserve(bytes)?;
        let zeroes = vec![0u8; bytes];
        let handle = self.client.create(&zeroes);
        Ok(DeviceBuffer::new(DeviceBufferRaw::CubeHandle(handle), len, bytes, usage))
    }

    #[cfg(feature = "wgpu-backend")]
    pub fn upload_f64(&self, data: &[f64], usage: MemoryUsage) -> Result<DeviceBuffer<f64>> {
        let len = data.len();
        let bytes = len
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(BackendError::OutOfMemory {
                requested_mb: None,
                limit_mb: Some(self.memory.max_bytes / (1024 * 1024).max(1)),
            })?;
        self.memory.try_reserve(bytes)?;
        let bytes_slice = cast_slice(data);
        let handle = self.client.create(bytes_slice);
        Ok(DeviceBuffer::new(DeviceBufferRaw::CubeHandle(handle), len, bytes, usage))
    }

    #[cfg(feature = "wgpu-backend")]
    pub fn free<T>(&self, buf: DeviceBuffer<T>) {
        self.memory.release(buf.bytes());
        drop(buf);
    }

    #[cfg(feature = "wgpu-backend")]
    pub fn write_f64(&self, buf: &mut DeviceBuffer<f64>, data: &[f64]) -> Result<()> {
        if data.len() > buf.len {
            return Err(BackendError::KernelFailure {
                message: "write_f64 length exceeds buffer".into(),
            });
        }
        let bytes_slice = cast_slice(data);
        // Allocate a fresh handle and swap; release previous accounting if size differs.
        let new_handle = self.client.create(bytes_slice);

        // update accounting if size changed (unlikely)
        if bytes_slice.len() != buf.bytes {
            self.memory.release(buf.bytes());
            self.memory.try_reserve(bytes_slice.len())?;
            buf.bytes = bytes_slice.len();
            buf.len = data.len();
        }
        buf.raw = DeviceBufferRaw::CubeHandle(new_handle);
        Ok(())
    }

    #[cfg(feature = "wgpu-backend")]
    pub fn used_memory_mb(&self) -> usize {
        self.memory.used_memory_mb()
    }

    #[cfg(feature = "wgpu-backend")]
    pub fn read_f64(&self, buf: &DeviceBuffer<f64>, len: usize) -> Result<Vec<f64>> {
        let handle = match &buf.raw {
            DeviceBufferRaw::CubeHandle(h) => h,
            _ => {
                return Err(BackendError::KernelFailure {
                    message: "read_f64 expects CubeHandle".into(),
                })
            }
        };
        let shape = [len];
        let strides = [1usize];
        let desc = handle.copy_descriptor(&shape, &strides, std::mem::size_of::<f64>());
        let bytes_vec = self.client.read_one_tensor(desc);
        let slice_f64 = bytemuck::try_cast_slice::<u8, f64>(&bytes_vec)
            .map_err(|_| BackendError::KernelFailure {
                message: "wgpu read_f64: bytemuck cast failed".into(),
            })?;
        Ok(slice_f64.iter().take(len).copied().collect())
    }
}

/// Construct the appropriate runtime from configuration.
pub fn build_runtime(cfg: &BackendConfig) -> Result<CubeRuntimeHandle> {
        match cfg.kind {
            BackendKind::Cpu => Ok(CubeRuntimeHandle::Cpu(CpuRuntime::new(cfg))),
            BackendKind::Wgpu => {
                #[cfg(feature = "wgpu-backend")]
                {
                    let rt = WgpuRuntime::new(cfg)?;
                    Ok(CubeRuntimeHandle::Wgpu(rt))
                }
                #[cfg(not(feature = "wgpu-backend"))]
                {
                Ok(CubeRuntimeHandle::Wgpu(WgpuRuntime::unavailable(
                    "wgpu feature disabled; build with `--features wgpu-backend`".into(),
                )))
            }
        }
    }
}
