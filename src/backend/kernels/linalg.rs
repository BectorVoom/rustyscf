use crate::backend::error::{BackendError, Result};
use crate::backend::runtime::CubeRuntimeHandle;
use crate::backend::types::{MatrixHandle, VectorHandle};

#[cfg(feature = "wgpu-backend")]
use cubecl_matmul::{launch as matmul_launch, MatmulInputHandle, Strategy};
#[cfg(feature = "wgpu-backend")]
use cubecl_std::tensor::TensorHandle;
#[cfg(feature = "wgpu-backend")]
use cubecl_wgpu::WgpuRuntime;

/// Placeholder GEMM implementation. Real CubeCL kernels will land later.
pub fn gemm(
    runtime: &CubeRuntimeHandle,
    a: &MatrixHandle,
    b: &MatrixHandle,
    c: &mut MatrixHandle,
    alpha: f64,
    beta: f64,
) -> Result<()> {
    #[cfg(feature = "cubecl-kernels")]
    {
        if let Err(e) = crate::backend::kernels::cube::gemm(a, b, c, alpha, beta) {
            // fall through to CPU/host fallback on failure
            tracing::debug!("cubecl gemm fallback: {e:?}");
        } else {
            return Ok(());
        }
    }

    match runtime {
        CubeRuntimeHandle::Cpu(cpu) => host_gemm(cpu, a, b, c, alpha, beta),
        #[cfg(feature = "wgpu-backend")]
        CubeRuntimeHandle::Wgpu(wgpu) => {
            // WGPU path: use cubecl-matmul based on Obsidian GEMM example.
            let client = wgpu.client();

            let a_handle = a
                .buffer
                .cube_handle()
                .ok_or_else(|| BackendError::KernelFailure {
                    message: "wgpu gemm: missing cube handle for A".into(),
                })?;
            let b_handle = b
                .buffer
                .cube_handle()
                .ok_or_else(|| BackendError::KernelFailure {
                    message: "wgpu gemm: missing cube handle for B".into(),
                })?;
            let c_handle = c
                .buffer
                .cube_handle()
                .ok_or_else(|| BackendError::KernelFailure {
                    message: "wgpu gemm: missing cube handle for C".into(),
                })?;

            let a_shape = vec![a.rows, a.cols];
            let b_shape = vec![b.rows, b.cols];
            let c_shape = vec![c.rows, c.cols];
            let a_strides = vec![a.cols, 1];
            let b_strides = vec![b.cols, 1];
            let c_strides = vec![c.cols, 1];

            let lhs = MatmulInputHandle::Normal(TensorHandle::<cubecl_wgpu::WgpuRuntime, f64>::new(
                a_handle.clone(),
                a_shape,
                a_strides,
            ));
            let rhs = MatmulInputHandle::Normal(TensorHandle::<cubecl_wgpu::WgpuRuntime, f64>::new(
                b_handle.clone(),
                b_shape,
                b_strides,
            ));
            let out = TensorHandle::<cubecl_wgpu::WgpuRuntime, f64>::new(
                c_handle.clone(),
                c_shape,
                c_strides,
            );

            // cubecl-matmul currently does not apply alpha/beta; emulate via C init.
            if beta != 0.0 {
                // read -> scale -> write
                let mut c_host = wgpu.read_f64(&c.buffer, c.rows * c.cols)?;
                for val in &mut c_host {
                    *val *= beta;
                }
                wgpu.write_f64(&mut c.buffer, &c_host)?;
            }

            matmul_launch::<cubecl_wgpu::WgpuRuntime, f64>(&Strategy::Auto, client, lhs, rhs, out)
                .map_err(|e| BackendError::KernelFailure {
                    message: format!("wgpu matmul failed: {e:?}"),
                })?;

            if alpha != 1.0 {
                // scale result by alpha
                let mut c_host = wgpu.read_f64(&c.buffer, c.rows * c.cols)?;
                for val in &mut c_host {
                    *val *= alpha;
                }
                wgpu.write_f64(&mut c.buffer, &c_host)?;
            }
            Ok(())
        }
        #[cfg(not(feature = "wgpu-backend"))]
        CubeRuntimeHandle::Wgpu(wgpu) => Err(BackendError::KernelFailure {
            message: format!("wgpu gemm not available: {}", wgpu.reason()),
        }),
    }
}

fn host_gemm(
    runtime: &crate::backend::runtime::CpuRuntime,
    a: &MatrixHandle,
    b: &MatrixHandle,
    c: &mut MatrixHandle,
    alpha: f64,
    beta: f64,
) -> Result<()> {
    let m = a.rows;
    let k = a.cols;
    let n = b.cols;
    if b.rows != k || c.rows != m || c.cols != n {
        return Err(BackendError::KernelFailure {
            message: format!(
                "gemm shape mismatch: a {}x{}, b {}x{}, c {}x{}",
                a.rows, a.cols, b.rows, b.cols, c.rows, c.cols
            ),
        });
    }

    let a_data = runtime.read_f64(&a.buffer, a.rows * a.cols)?;
    let b_data = runtime.read_f64(&b.buffer, b.rows * b.cols)?;
    let mut c_data = runtime.read_f64(&c.buffer, c.rows * c.cols)?;

    host_gemm_impl(a_data.as_slice(), b_data.as_slice(), c_data.as_mut_slice(), m, k, n, alpha, beta)?;
    runtime.write_f64(&mut c.buffer, &c_data)?;
    Ok(())
}

fn host_gemm_impl(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    k: usize,
    n: usize,
    alpha: f64,
    beta: f64,
) -> Result<()> {
    if a.len() != m * k || b.len() != k * n || c.len() != m * n {
        return Err(BackendError::KernelFailure {
            message: "gemm buffer length mismatch".into(),
        });
    }

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            let idx = i * n + j;
            c[idx] = alpha * acc + beta * c[idx];
        }
    }
    Ok(())
}

/// Placeholder eigen decomposition.
pub fn eigh(
    runtime: &CubeRuntimeHandle,
    _a: &mut MatrixHandle,
    _evals: &mut VectorHandle,
) -> Result<()> {
    match runtime {
        CubeRuntimeHandle::Cpu(_) => Err(BackendError::KernelFailure {
            message: "eigh kernel not implemented yet".into(),
        }),
        CubeRuntimeHandle::Wgpu(wgpu) => Err(BackendError::KernelFailure {
            message: format!("wgpu eigh not available yet ({})", wgpu.reason()),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::runtime::build_runtime;
    use crate::backend::types::{BackendConfig, BackendKind, MemoryUsage, MatrixHandle};
    use crate::backend::runtime::CubeRuntimeHandle;

    #[test]
    fn cpu_gemm_naive_matches_hand_calc() {
        let cfg = BackendConfig::cpu();
        let runtime = build_runtime(&cfg).unwrap();
        assert!(matches!(runtime.kind(), BackendKind::Cpu));

        let cpu = match &runtime {
            CubeRuntimeHandle::Cpu(cpu) => cpu,
            _ => panic!("expected cpu runtime"),
        };

        let mut a_buf = cpu.alloc_f64(4, MemoryUsage::Persistent).unwrap();
        let mut b_buf = cpu.alloc_f64(4, MemoryUsage::Persistent).unwrap();
        let c_buf = cpu.alloc_f64(4, MemoryUsage::Persistent).unwrap();
        cpu.write_f64(&mut a_buf, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        cpu.write_f64(&mut b_buf, &[5.0, 6.0, 7.0, 8.0]).unwrap();

        let a = MatrixHandle::new(a_buf, 2, 2);
        let b = MatrixHandle::new(b_buf, 2, 2);
        let mut c = MatrixHandle::new(c_buf, 2, 2);

        gemm(&runtime, &a, &b, &mut c, 1.0, 0.0).unwrap();

        let vals = cpu.read_f64(&c.buffer, 4).unwrap();
        assert_eq!(vals, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[cfg(feature = "wgpu-backend")]
    #[test]
    fn wgpu_gemm_roundtrip_small_matrix() {
        let cfg = BackendConfig::wgpu().with_max_memory_mb(256);
        // Backend construction may fail on machines without adapters; skip quietly.
        let backend = match std::panic::catch_unwind(|| make_backend(&cfg)) {
            Ok(Ok(b)) => b,
            Ok(Err(_)) | Err(_) => {
                eprintln!("skipping wgpu_gemm_roundtrip_small_matrix (no adapter)");
                return;
            }
        };

        let data_a = [1.0f64, 2.0, 3.0, 4.0]; // 2x2
        let data_b = [5.0f64, 6.0, 7.0, 8.0]; // 2x2
        let mut buf_a = backend.upload_f64(&data_a, MemoryUsage::Transient).unwrap();
        let mut buf_b = backend.upload_f64(&data_b, MemoryUsage::Transient).unwrap();
        let buf_c = backend.alloc_f64(4, MemoryUsage::Transient).unwrap();

        let a = MatrixHandle::new(buf_a, 2, 2);
        let b = MatrixHandle::new(buf_b, 2, 2);
        let mut c = MatrixHandle::new(buf_c, 2, 2);

        backend.matmul(&a, &b, &mut c, 1.0, 0.0).unwrap();
        let vals = backend.read_f64(&c.buffer, 4).unwrap();
        assert_eq!(vals, vec![19.0, 22.0, 43.0, 50.0]);
    }
}
