use crate::backend::error::{BackendError, Result};
use crate::backend::runtime::CubeRuntimeHandle;
use crate::backend::types::DeviceBuffer;
#[cfg(feature = "wgpu-backend")]
use cubecl_core::frontend::TensorHandleRef;
#[cfg(feature = "cpu-backend")]
use cubecl_core::frontend::TensorHandleRef as CpuTensorHandleRef;
#[cfg(feature = "wgpu-backend")]
use cubecl_reduce::{instructions::Sum, reduce};
#[cfg(feature = "cpu-backend")]
use cubecl_reduce::{instructions::Sum as CpuSum, reduce as cpu_reduce};
#[cfg(feature = "wgpu-backend")]
use cubecl_wgpu::WgpuRuntime;
#[cfg(feature = "wgpu-backend")]
use bytemuck::cast_slice;
#[cfg(feature = "cpu-backend")]
use bytemuck::cast_slice as cpu_cast_slice;

/// Sum a buffer of f64 values using the active runtime.
pub fn sum_f64(runtime: &CubeRuntimeHandle, buffer: &DeviceBuffer<f64>, len: usize) -> Result<f64> {
    #[cfg(feature = "cubecl-kernels")]
    {
        if let Err(e) = crate::backend::kernels::cube::sum_f64(buffer, len) {
            tracing::debug!("cubecl sum_f64 fallback: {e:?}");
        } else {
            return crate::backend::kernels::cube::sum_f64(buffer, len);
        }
    }

    match runtime {
        CubeRuntimeHandle::Cpu(cpu) => {
            let handle = buffer.cube_handle().ok_or_else(|| BackendError::KernelFailure {
                message: "cpu sum_f64 expects CubeHandle".into(),
            })?;
            let shape = [len];
            let strides = [1usize];
            let elem_size = std::mem::size_of::<f64>();
            let input = unsafe {
                CpuTensorHandleRef::<cubecl_cpu::CpuRuntime>::from_raw_parts(
                    handle,
                    &strides,
                    &shape,
                    elem_size,
                )
            };
            let output_alloc = cpu.client().empty_tensor(&[1usize], elem_size);
            let output = unsafe {
                CpuTensorHandleRef::<cubecl_cpu::CpuRuntime>::from_raw_parts(
                    &output_alloc.handle,
                    &output_alloc.strides,
                    &[1usize],
                    elem_size,
                )
            };

            cpu_reduce::<cubecl_cpu::CpuRuntime, f64, f64, CpuSum>(
                cpu.client(),
                input,
                output,
                0,
                None,
                (),
            )
            .map_err(|e| BackendError::KernelFailure {
                message: format!("cpu reduce sum failed: {e:?}"),
            })?;

            let bytes = cpu
                .client()
                .read_tensor(vec![output_alloc.handle.copy_descriptor(
                    &[1usize],
                    &output_alloc.strides,
                    elem_size,
                )]);
            let scalar = cpu_cast_slice::<u8, f64>(&bytes[0])
                .first()
                .copied()
                .ok_or_else(|| BackendError::KernelFailure {
                    message: "cpu reduce sum readback empty".into(),
                })?;
            Ok(scalar)
        }
        #[cfg(feature = "wgpu-backend")]
        CubeRuntimeHandle::Wgpu(wgpu_rt) => {
            let handle = buffer.cube_handle().ok_or_else(|| BackendError::KernelFailure {
                message: "wgpu sum_f64 expects CubeHandle".into(),
            })?;
            let shape = [len];
            let strides = [1usize];
            let elem_size = std::mem::size_of::<f64>();
            // Build TensorHandleRef for input and a single-element output.
            let input = unsafe {
                TensorHandleRef::<cubecl_wgpu::WgpuRuntime>::from_raw_parts(
                    handle,
                    &strides,
                    &shape,
                    elem_size,
                )
            };
            let output_alloc = wgpu_rt
                .client()
                .empty_tensor(&[1usize], elem_size);
            let output = unsafe {
                TensorHandleRef::<cubecl_wgpu::WgpuRuntime>::from_raw_parts(
                    &output_alloc.handle,
                    &output_alloc.strides,
                    &[1usize],
                    elem_size,
                )
            };

            reduce::<cubecl_wgpu::WgpuRuntime, f64, f64, Sum>(
                wgpu_rt.client(),
                input,
                output,
                0,
                None,
                (),
            )
            .map_err(|e| BackendError::KernelFailure {
                message: format!("wgpu reduce sum failed: {e:?}"),
            })?;

            let bytes = wgpu_rt
                .client()
                .read_tensor(vec![output_alloc.handle.copy_descriptor(
                    &[1usize],
                    &output_alloc.strides,
                    elem_size,
                )]);
            let scalar = bytemuck::cast_slice::<u8, f64>(&bytes[0])
                .first()
                .copied()
                .ok_or_else(|| BackendError::KernelFailure {
                    message: "wgpu reduce sum readback empty".into(),
                })?;
            Ok(scalar)
        }
        #[cfg(not(feature = "wgpu-backend"))]
        CubeRuntimeHandle::Wgpu(_wgpu) => Err(BackendError::KernelFailure {
            message: "wgpu sum_f64 not implemented yet".into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::runtime::{build_runtime, CubeRuntimeHandle};
    use crate::backend::types::{BackendConfig, BackendKind, MemoryUsage};
    #[test]
    fn cpu_sum_matches_reference() {
        let cfg = BackendConfig::cpu();
        let runtime = build_runtime(&cfg).unwrap();
        assert!(matches!(runtime.kind(), BackendKind::Cpu));
        let cpu = match &runtime {
            CubeRuntimeHandle::Cpu(cpu) => cpu,
            _ => panic!("expected cpu runtime"),
        };

        let data: Vec<f64> = (1..=8).map(|v| v as f64).collect();
        let len = data.len();
        let mut buf = cpu.alloc_f64(len, MemoryUsage::Transient).unwrap();
        cpu.write_f64(&mut buf, &data).unwrap();

        let sum = sum_f64(&runtime, &buf, len).unwrap();
        assert_eq!(sum, data.iter().sum::<f64>());
    }

    #[cfg(feature = "wgpu-backend")]
    #[test]
    fn wgpu_sum_matches_reference() {
        let cfg = BackendConfig::wgpu().with_max_memory_mb(64);
        let backend = match std::panic::catch_unwind(|| make_backend(&cfg)) {
            Ok(Ok(b)) => b,
            Ok(Err(_)) | Err(_) => {
                eprintln!("skipping wgpu_sum_matches_reference (no adapter)");
                return;
            }
        };

        let data: Vec<f64> = (1..=6).map(|v| v as f64).collect();
        let len = data.len();
        let buf = backend.upload_f64(&data, MemoryUsage::Transient).unwrap();
        let sum = backend.sum_f64(&buf, len).unwrap();
        assert!((sum - data.iter().sum::<f64>()).abs() < 1e-9);
    }
}
