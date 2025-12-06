use crate::backend::error::{BackendError, Result};
use crate::backend::runtime::CubeRuntimeHandle;
use crate::backend::types::ComplexGridHandle;
use std::f64::consts::TAU;

#[cfg(feature = "cubecl-kernels")]
use crate::backend::kernels::cube;

/// Naive CPU 3D FFT (forward). Intended for small grids only; falls back to an error for large sizes.
pub fn fft3d_forward(runtime: &CubeRuntimeHandle, grid: &mut ComplexGridHandle) -> Result<()> {
    #[cfg(feature = "cubecl-kernels")]
    {
        // Avoid CubeCL path for large meshes to skip O(N^2) fallback kernels.
        let total = grid.dims.iter().product::<usize>();
        let limit = cubecl_fft_threshold(runtime);
        if total <= limit {
            if let Err(e) = cube::fft3d_forward(runtime, grid) {
                tracing::debug!("cubecl fft3d_forward fallback: {e:?}");
            } else {
                return Ok(());
            }
        } else {
            tracing::debug!("cubecl fft3d_forward skipped: total {} > {}", total, limit);
        }
        if let Err(e) = cube::fft3d_forward(runtime, grid) {
            tracing::debug!("cubecl fft3d_forward fallback: {e:?}");
        } else {
            return Ok(());
        }
    }

    match runtime {
        CubeRuntimeHandle::Cpu(cpu) => cpu_fft3d(cpu, grid, false),
        #[cfg(feature = "wgpu-backend")]
        CubeRuntimeHandle::Wgpu(wgpu) => {
            let re = wgpu.read_f64(&grid.re, grid.re.len)?;
            let im = wgpu.read_f64(&grid.im, grid.im.len)?;
            let (re_out, im_out) = cpu_fft3d_host(&re, &im, grid.dims, false)?;
            let mut re_buf = grid.re.clone();
            let mut im_buf = grid.im.clone();
            wgpu.write_f64(&mut re_buf, &re_out)?;
            wgpu.write_f64(&mut im_buf, &im_out)?;
            grid.re = re_buf;
            grid.im = im_buf;
            Ok(())
        }
        #[cfg(not(feature = "wgpu-backend"))]
        CubeRuntimeHandle::Wgpu(wgpu) => Err(BackendError::KernelFailure {
            message: format!("wgpu fft3d_forward not available: {}", wgpu.reason()),
        }),
    }

    #[cfg(feature = "cubecl-kernels")]
    #[test]
    fn cpu_fft_cubecl_round_trip_small_grid() {
        let cfg = BackendConfig::cpu();
        let runtime = build_runtime(&cfg).unwrap();
        let cpu = match &runtime {
            CubeRuntimeHandle::Cpu(cpu) => cpu,
            _ => panic!("expected cpu runtime"),
        };
        assert!(matches!(runtime.kind(), BackendKind::Cpu));

        let dims = [2, 2, 2];
        let total = dims.iter().product();
        let mut re = cpu.alloc_f64(total, MemoryUsage::Persistent).unwrap();
        let mut im = cpu.alloc_f64(total, MemoryUsage::Persistent).unwrap();
        cpu.write_f64(&mut re, &(0..total).map(|v| v as f64).collect::<Vec<_>>())
            .unwrap();
        cpu.write_f64(&mut im, &vec![0.0; total]).unwrap();
        let mut grid = ComplexGridHandle::new(re, im, dims);

        // With cubecl-kernels enabled, this should exercise the CubeCL path.
        fft3d_forward(&runtime, &mut grid).unwrap();
        fft3d_inverse(&runtime, &mut grid).unwrap();

        let re_back = cpu.read_f64(&grid.re, total).unwrap();
        let expected: Vec<f64> = (0..total).map(|v| v as f64).collect();
        for (a, b) in re_back.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }
}

/// Naive CPU 3D FFT (inverse). Intended for small grids only; falls back to an error for large sizes.
pub fn fft3d_inverse(runtime: &CubeRuntimeHandle, grid: &mut ComplexGridHandle) -> Result<()> {
    #[cfg(feature = "cubecl-kernels")]
    {
        let total = grid.dims.iter().product::<usize>();
        let limit = cubecl_fft_threshold(runtime);
        if total <= limit {
            if let Err(e) = cube::fft3d_inverse(runtime, grid) {
                tracing::debug!("cubecl fft3d_inverse fallback: {e:?}");
            } else {
                return Ok(());
            }
        } else {
            tracing::debug!("cubecl fft3d_inverse skipped: total {} > {}", total, limit);
        }
        if let Err(e) = cube::fft3d_inverse(runtime, grid) {
            tracing::debug!("cubecl fft3d_inverse fallback: {e:?}");
        } else {
            return Ok(());
        }
    }

    match runtime {
        CubeRuntimeHandle::Cpu(cpu) => cpu_fft3d(cpu, grid, true),
        #[cfg(feature = "wgpu-backend")]
        CubeRuntimeHandle::Wgpu(wgpu) => {
            let re = wgpu.read_f64(&grid.re, grid.re.len)?;
            let im = wgpu.read_f64(&grid.im, grid.im.len)?;
            let (re_out, im_out) = cpu_fft3d_host(&re, &im, grid.dims, true)?;
            let mut re_buf = grid.re.clone();
            let mut im_buf = grid.im.clone();
            wgpu.write_f64(&mut re_buf, &re_out)?;
            wgpu.write_f64(&mut im_buf, &im_out)?;
            grid.re = re_buf;
            grid.im = im_buf;
            Ok(())
        }
        #[cfg(not(feature = "wgpu-backend"))]
        CubeRuntimeHandle::Wgpu(wgpu) => Err(BackendError::KernelFailure {
            message: format!("wgpu fft3d_inverse not available: {}", wgpu.reason()),
        }),
    }
}

fn cpu_fft3d(
    runtime: &crate::backend::runtime::CpuRuntime,
    grid: &mut ComplexGridHandle,
    inverse: bool,
) -> Result<()> {
    let total = grid.dims.iter().product();
    let re_in = runtime.read_f64(&grid.re, total)?;
    let im_in = runtime.read_f64(&grid.im, total)?;
    let (re_out, im_out) = cpu_fft3d_host(&re_in, &im_in, grid.dims, inverse)?;
    runtime.write_f64(&mut grid.re, &re_out)?;
    runtime.write_f64(&mut grid.im, &im_out)?;
    Ok(())
}

fn cpu_fft3d_host(re_in: &[f64], im_in: &[f64], dims: [usize; 3], inverse: bool) -> Result<(Vec<f64>, Vec<f64>)> {
    let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
    let total = nx * ny * nz;
    const MAX_ELEMENTS: usize = 2048;
    if total > MAX_ELEMENTS {
        return Err(BackendError::KernelFailure {
            message: format!(
                "cpu_fft3d grid too large for naive path ({} elements, limit {})",
                total, MAX_ELEMENTS
            ),
        });
    }
    if re_in.len() < total || im_in.len() < total {
        return Err(BackendError::KernelFailure {
            message: "cpu_fft3d input length mismatch".into(),
        });
    }

    let mut re_out = vec![0.0f64; total];
    let mut im_out = vec![0.0f64; total];
    let sign = if inverse { 1.0 } else { -1.0 };
    let norm = if inverse { 1.0 / total as f64 } else { 1.0 };

    for kx in 0..nx {
        for ky in 0..ny {
            for kz in 0..nz {
                let mut acc_re = 0.0;
                let mut acc_im = 0.0;
                for x in 0..nx {
                    for y in 0..ny {
                        for z in 0..nz {
                            let idx_in = index3(x, y, z, ny, nz);
                            let phase = TAU
                                * (kx as f64 * x as f64 / nx as f64
                                    + ky as f64 * y as f64 / ny as f64
                                    + kz as f64 * z as f64 / nz as f64);
                            let (s, c) = phase.sin_cos();
                            let cos = c;
                            let sin = sign * s;
                            let ar = re_in[idx_in];
                            let ai = im_in[idx_in];
                            acc_re += ar * cos - ai * sin;
                            acc_im += ar * sin + ai * cos;
                        }
                    }
                }
                let idx_out = index3(kx, ky, kz, ny, nz);
                re_out[idx_out] = acc_re * norm;
                im_out[idx_out] = acc_im * norm;
            }
        }
    }

    Ok((re_out, im_out))
}

/// Pick a CubeCL FFT element threshold based on runtime and optional env override.
#[cfg(feature = "cubecl-kernels")]
fn cubecl_fft_threshold(runtime: &CubeRuntimeHandle) -> usize {
    if let Ok(env) = std::env::var("RUSTYSCF_FFT_CUBECL_MAX") {
        if let Ok(v) = env.parse::<usize>() {
            return v;
        }
    }
    match runtime {
        CubeRuntimeHandle::Cpu(_) => 4_096,
        #[cfg(feature = "wgpu-backend")]
        CubeRuntimeHandle::Wgpu(_) => 32_768,
        #[cfg(not(feature = "wgpu-backend"))]
        CubeRuntimeHandle::Wgpu(_) => 4_096,
    }
}

#[inline]
fn index3(x: usize, y: usize, z: usize, ny: usize, nz: usize) -> usize {
    (x * ny + y) * nz + z
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::types::{BackendConfig, BackendKind, MemoryUsage};
    use crate::backend::runtime::CubeRuntimeHandle;
    use crate::backend::ComplexGridHandle;
    use crate::backend::runtime::build_runtime;
    #[test]
    fn cpu_fft3d_round_trip_small_grid() {
        let cfg = BackendConfig::cpu();
        let runtime = build_runtime(&cfg).unwrap();
        let cpu = match &runtime {
            CubeRuntimeHandle::Cpu(cpu) => cpu,
            _ => panic!("expected cpu runtime"),
        };
        assert!(matches!(runtime.kind(), BackendKind::Cpu));

        let dims = [2, 2, 2];
        let total = dims.iter().product();
        let mut re = cpu.alloc_f64(total, MemoryUsage::Persistent).unwrap();
        let mut im = cpu.alloc_f64(total, MemoryUsage::Persistent).unwrap();
        cpu.write_f64(&mut re, &(0..total).map(|v| v as f64).collect::<Vec<_>>())
            .unwrap();
        cpu.write_f64(&mut im, &vec![0.0; total]).unwrap();
        let mut grid = ComplexGridHandle::new(re, im, dims);

        fft3d_forward(&runtime, &mut grid).unwrap();
        fft3d_inverse(&runtime, &mut grid).unwrap();

        let re_back = cpu.read_f64(&grid.re, total).unwrap();
        let expected: Vec<f64> = (0..total).map(|v| v as f64).collect();
        for (a, b) in re_back.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[cfg(all(feature = "wgpu-backend", feature = "cubecl-kernels"))]
    #[test]
    fn wgpu_fft_round_trip_small_grid() {
        let cfg = BackendConfig::wgpu().with_max_memory_mb(128);
        let backend = match std::panic::catch_unwind(|| make_backend(&cfg)) {
            Ok(Ok(b)) => b,
            Ok(Err(_)) | Err(_) => {
                eprintln!("skipping wgpu fft test (no adapter)");
                return;
            }
        };

        let dims = [2, 2, 2];
        let total = dims.iter().product();
        let mut re = backend.alloc_f64(total, MemoryUsage::Transient).unwrap();
        let mut im = backend.alloc_f64(total, MemoryUsage::Transient).unwrap();
        backend
            .write_f64(&mut re, &(0..total).map(|v| v as f64).collect::<Vec<_>>())
            .unwrap();
        backend.write_f64(&mut im, &vec![0.0; total]).unwrap();
        let mut grid = ComplexGridHandle::new(re, im, dims);

        backend.fft3d_forward(&mut grid).unwrap();
        backend.fft3d_inverse(&mut grid).unwrap();

        let re_back = backend.read_f64(&grid.re, total).unwrap();
        let expected: Vec<f64> = (0..total).map(|v| v as f64).collect();
        for (a, b) in re_back.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
