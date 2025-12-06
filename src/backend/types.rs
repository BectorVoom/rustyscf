use std::marker::PhantomData;
use num_complex::Complex;

/// High-level backend selection exposed to builders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// CubeCL CPU runtime.
    Cpu,

    /// CubeCL WGPU runtime.
    Wgpu,
}

/// Runtime metadata surfaced to higher layers.
#[derive(Clone, Debug)]
pub struct BackendInfo {
    pub kind: BackendKind,
    pub device_name: String,
    pub max_memory_mb: usize,
}

/// Backend configuration passed from high-level builders.
#[derive(Clone, Debug)]
pub struct BackendConfig {
    pub kind: BackendKind,
    /// Maximum memory in megabytes the backend may allocate. `usize::MAX`
    /// means “no explicit cap”.
    pub max_memory_mb: usize,
    /// Optional concurrency hint (threads for CPU, batching for GPU).
    pub concurrency: Option<usize>,
    /// Backend-specific knobs (currently only WGPU options).
    pub options: BackendOptions,
}

impl BackendConfig {
    pub fn cpu() -> Self {
        Self {
            kind: BackendKind::Cpu,
            max_memory_mb: usize::MAX,
            concurrency: None,
            options: BackendOptions::default(),
        }
    }

    pub fn wgpu() -> Self {
        Self {
            kind: BackendKind::Wgpu,
            max_memory_mb: usize::MAX,
            concurrency: None,
            options: BackendOptions::default(),
        }
    }

    pub fn with_max_memory_mb(mut self, mb: usize) -> Self {
        self.max_memory_mb = mb;
        self
    }

    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.concurrency = Some(n);
        self
    }

    pub fn with_options(mut self, options: BackendOptions) -> Self {
        self.options = options;
        self
    }
}

/// Aggregated backend options; extensible without breaking callers.
#[derive(Clone, Debug, Default)]
pub struct BackendOptions {
    pub wgpu: Option<WgpuOptions>,
}

/// Optional WGPU-specific tuning knobs.
#[derive(Clone, Debug)]
pub struct WgpuOptions {
    /// Override for task batching (`CUBECL_WGPU_MAX_TASKS`).
    pub max_tasks: Option<u32>,
}

/// Memory usage classification for device buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryUsage {
    Persistent,
    Transient,
}

/// Logical buffer on a device (CPU or GPU), parameterised by element type.
#[derive(Debug, Clone)]
pub struct DeviceBuffer<T> {
    pub(crate) raw: DeviceBufferRaw,
    pub(crate) bytes: usize,
    pub(crate) len: usize,
    #[allow(dead_code)]
    pub(crate) usage: MemoryUsage,
    _marker: PhantomData<T>,
}

impl<T> DeviceBuffer<T> {
    pub(crate) fn new(raw: DeviceBufferRaw, len: usize, bytes: usize, usage: MemoryUsage) -> Self {
        Self {
            raw,
            bytes,
            len,
            usage,
            _marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Internal accessor for CubeCL allocation handle.
    pub(crate) fn cube_handle(&self) -> Option<&CubeHandle> {
        self.raw.handle()
    }
}

use cubecl_runtime::server::Handle as CubeHandle;

/// Internal buffer handle variants; concrete types are intentionally hidden
/// from callers to keep the public surface stable.
#[derive(Debug, Clone)]
pub(crate) enum DeviceBufferRaw {
    /// CubeCL device handle (CPU or GPU) backed by runtime storage.
    CubeHandle(CubeHandle),
    #[allow(dead_code)]
    Mock,
}

impl DeviceBufferRaw {
    pub(crate) fn handle(&self) -> Option<&CubeHandle> {
        match self {
            DeviceBufferRaw::CubeHandle(h) => Some(h),
            _ => None,
        }
    }
}

/// Matrix view used by kernel wrappers.
#[derive(Debug)]
pub struct MatrixHandle {
    pub buffer: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl MatrixHandle {
    pub fn new(buffer: DeviceBuffer<f64>, rows: usize, cols: usize) -> Self {
        Self { buffer, rows, cols }
    }
}

/// Vector view used by kernel wrappers.
#[derive(Debug)]
pub struct VectorHandle {
    pub buffer: DeviceBuffer<f64>,
    pub len: usize,
}

impl VectorHandle {
    pub fn new(buffer: DeviceBuffer<f64>, len: usize) -> Self {
        Self { buffer, len }
    }
}

/// Split-complex grid (real + imaginary buffers) for FFT routines.
#[derive(Debug)]
pub struct ComplexGridHandle {
    pub re: DeviceBuffer<f64>,
    pub im: DeviceBuffer<f64>,
    pub dims: [usize; 3],
}

impl ComplexGridHandle {
    pub fn new(re: DeviceBuffer<f64>, im: DeviceBuffer<f64>, dims: [usize; 3]) -> Self {
        Self { re, im, dims }
    }
}

/// Convert host-side complex slice into split real/imag vectors.
pub fn host_complex_to_split(src: &[Complex<f64>]) -> (Vec<f64>, Vec<f64>) {
    let mut re = Vec::with_capacity(src.len());
    let mut im = Vec::with_capacity(src.len());
    for c in src {
        re.push(c.re);
        im.push(c.im);
    }
    (re, im)
}

/// Convert split real/imag slices back into host-side complex values.
pub fn split_to_host_complex(re: &[f64], im: &[f64]) -> Vec<Complex<f64>> {
    assert_eq!(re.len(), im.len());
    re.iter()
        .zip(im.iter())
        .map(|(r, i)| Complex::new(*r, *i))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        BackendConfig, BackendKind, BackendOptions, DeviceBuffer, DeviceBufferRaw, MemoryUsage,
        WgpuOptions, host_complex_to_split, split_to_host_complex,
    };
    use num_complex::Complex;

    #[test]
    fn backend_config_builders_roundtrip() {
        let cfg = BackendConfig::cpu()
            .with_max_memory_mb(1024)
            .with_concurrency(4)
            .with_options(BackendOptions {
                wgpu: Some(WgpuOptions { max_tasks: Some(8) }),
            });
        assert_eq!(cfg.kind, BackendKind::Cpu);
        assert_eq!(cfg.max_memory_mb, 1024);
        assert_eq!(cfg.concurrency, Some(4));
        assert_eq!(cfg.options.wgpu.unwrap().max_tasks, Some(8));
    }

    #[test]
    fn device_buffer_tracks_len_and_bytes() {
        let buf: DeviceBuffer<f64> =
            DeviceBuffer::new(DeviceBufferRaw::Mock, 10, 10 * std::mem::size_of::<f64>(), MemoryUsage::Persistent);
        assert_eq!(buf.len(), 10);
        assert_eq!(buf.bytes(), 10 * std::mem::size_of::<f64>());
    }

    #[test]
    fn complex_split_roundtrip() {
        let src: Vec<Complex<f64>> = vec![
            Complex::new(1.0, -2.0),
            Complex::new(3.5, 0.25),
            Complex::new(-7.0, 4.0),
        ];
        let (re, im) = host_complex_to_split(&src);
        let back = split_to_host_complex(&re, &im);
        assert_eq!(src, back);
    }
}
