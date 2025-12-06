//! CubeCL single-source kernel scaffolding.
//! This module is gated behind the `cubecl-kernels` feature and is currently a placeholder.
//! Replace the `todo!()` bodies with real CubeCL kernels once available.

use crate::backend::error::{BackendError, Result};
use crate::backend::types::{MatrixHandle, VectorHandle, ComplexGridHandle, DeviceBuffer};
#[cfg(feature = "cubecl-kernels")]
use cubecl::prelude::*;
#[cfg(feature = "cubecl-kernels")]
use cubecl_core::CubeCount;
#[cfg(feature = "cubecl-kernels")]
use cubecl_core::prelude::{Array, CubeDim};
#[cfg(feature = "cubecl-kernels")]
use std::f64::consts::PI;
#[cfg(feature = "cubecl-kernels")]
use cubecl_cpu::{CpuDevice, CpuRuntime};
#[cfg(feature = "cubecl-kernels")]
use cubecl_runtime::client::ComputeClient;
#[cfg(all(feature = "cubecl-kernels", feature = "wgpu-backend"))]
use cubecl_wgpu::{init_device, init_setup, AutoGraphicsApi, RuntimeOptions, WgpuDevice, WgpuRuntime};
#[cfg(all(feature = "cubecl-kernels", feature = "wgpu-backend"))]
use cubecl_std::tensor::TensorHandle as WgpuTensorHandle;
#[cfg(feature = "cubecl-kernels")]
use cubecl_wgpu::WgpuServer;
#[cfg(feature = "cubecl-kernels")]
use cubecl_core::frontend::TensorHandleRef;

/// GEMM via CubeCL (placeholder).
pub fn gemm(
    _a: &MatrixHandle,
    _b: &MatrixHandle,
    _c: &mut MatrixHandle,
    _alpha: f64,
    _beta: f64,
) -> Result<()> {
    // TODO: implement CubeCL gemm kernel and dispatch via runtime.
    Err(BackendError::KernelFailure {
        message: "cubecl gemm not implemented yet".into(),
    })
}

/// Eigh via CubeCL (placeholder).
pub fn eigh(_a: &mut MatrixHandle, _evals: &mut VectorHandle) -> Result<()> {
    Err(BackendError::KernelFailure {
        message: "cubecl eigh not implemented yet".into(),
    })
}

/// FFT forward via CubeCL (placeholder).
pub fn fft3d_forward(runtime: &CubeRuntimeHandle, _grid: &mut ComplexGridHandle) -> Result<()> {
    #[cfg(not(feature = "cubecl-kernels"))]
    {
        return Err(BackendError::KernelFailure {
            message: "cubecl fft3d_forward not implemented yet".into(),
        });
    }

    #[cfg(feature = "cubecl-kernels")]
    {
        launch_fft3d::<false>(runtime, _grid)
    }
}

/// FFT inverse via CubeCL (placeholder).
pub fn fft3d_inverse(runtime: &CubeRuntimeHandle, _grid: &mut ComplexGridHandle) -> Result<()> {
    #[cfg(not(feature = "cubecl-kernels"))]
    {
        return Err(BackendError::KernelFailure {
            message: "cubecl fft3d_inverse not implemented yet".into(),
        });
    }

    #[cfg(feature = "cubecl-kernels")]
    {
        launch_fft3d::<true>(runtime, _grid)
    }
}

/// Sum reduction via CubeCL (placeholder).
pub fn sum_f64(_buffer: &DeviceBuffer<f64>, _len: usize) -> Result<f64> {
    Err(BackendError::KernelFailure {
        message: "cubecl sum_f64 not implemented yet".into(),
    })
}

/// Elementwise multiply in G-space: out = v_g * rho.
#[cfg(feature = "cubecl-kernels")]
pub fn coulomb_mul(
    rho_re: &DeviceBuffer<f64>,
    rho_im: &DeviceBuffer<f64>,
    v_g: &DeviceBuffer<f64>,
    out_re: &mut DeviceBuffer<f64>,
    out_im: &mut DeviceBuffer<f64>,
    ngrid: usize,
) -> Result<()> {
    // Prefer WGPU when available and explicitly requested; otherwise CPU.
    #[cfg(feature = "wgpu-backend")]
    if std::env::var("RUSTYSCF_CUBECL_WGPU")
        .ok()
        .as_deref()
        == Some("1")
    {
        let device = WgpuDevice::DefaultDevice;
        let setup = init_setup::<AutoGraphicsApi>(&device, RuntimeOptions::default());
        let device = init_device(setup, RuntimeOptions::default());
        let client: ComputeClient<_> = ComputeClient::load(&device);
        return launch_coulomb::<WgpuRuntime, _>(&client, rho_re, rho_im, v_g, out_re, out_im, ngrid);
    }

    let client = cubecl_cpu::ComputeClient::load(&cubecl_cpu::CpuDevice::default());
    launch_coulomb::<CpuRuntime, _>(&client, rho_re, rho_im, v_g, out_re, out_im, ngrid)
}

#[cfg(feature = "cubecl-kernels")]
fn launch_coulomb<R: cubecl_core::Runtime, C: cubecl_core::client::ComputeClientLike<R::Server>>(
    client: &C,
    rho_re: &DeviceBuffer<f64>,
    rho_im: &DeviceBuffer<f64>,
    v_g: &DeviceBuffer<f64>,
    out_re: &mut DeviceBuffer<f64>,
    out_im: &mut DeviceBuffer<f64>,
    ngrid: usize,
) -> Result<()> {
    let rho_re_handle = rho_re
        .cube_handle()
        .ok_or_else(|| BackendError::KernelFailure {
            message: "coulomb_mul missing rho_re handle".into(),
        })?;
    let rho_im_handle = rho_im
        .cube_handle()
        .ok_or_else(|| BackendError::KernelFailure {
            message: "coulomb_mul missing rho_im handle".into(),
        })?;
    let v_g_handle = v_g
        .cube_handle()
        .ok_or_else(|| BackendError::KernelFailure {
            message: "coulomb_mul missing v_g handle".into(),
        })?;
    let out_re_handle = out_re
        .cube_handle()
        .ok_or_else(|| BackendError::KernelFailure {
            message: "coulomb_mul missing out_re handle".into(),
        })?;
    let out_im_handle = out_im
        .cube_handle()
        .ok_or_else(|| BackendError::KernelFailure {
            message: "coulomb_mul missing out_im handle".into(),
        })?;

    unsafe {
        let rho_re_ref = TensorHandleRef::<R>::from_raw_parts(
            rho_re_handle,
            &[1],
            &[ngrid],
            std::mem::size_of::<f64>(),
        );
        let rho_im_ref = TensorHandleRef::<R>::from_raw_parts(
            rho_im_handle,
            &[1],
            &[ngrid],
            std::mem::size_of::<f64>(),
        );
        let v_g_ref = TensorHandleRef::<R>::from_raw_parts(
            v_g_handle,
            &[1],
            &[ngrid],
            std::mem::size_of::<f64>(),
        );
        let out_re_ref = TensorHandleRef::<R>::from_raw_parts(
            out_re_handle,
            &[1],
            &[ngrid],
            std::mem::size_of::<f64>(),
        );
        let out_im_ref = TensorHandleRef::<R>::from_raw_parts(
            out_im_handle,
            &[1],
            &[ngrid],
            std::mem::size_of::<f64>(),
        );

        apply_coulomb_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(((ngrid as u32) + 63) / 64),
            CubeDim::new_1d(64),
            rho_re_ref,
            rho_im_ref,
            v_g_ref,
            out_re_ref,
            out_im_ref,
            ngrid,
        );
    }
    Ok(())
}

#[cfg(feature = "cubecl-kernels")]
#[cube(launch_unchecked)]
fn apply_coulomb_kernel(
    rho_re: &Array<f64>,
    rho_im: &Array<f64>,
    v_g: &Array<f64>,
    out_re: &mut Array<f64>,
    out_im: &mut Array<f64>,
    #[comptime] ngrid: usize,
) {
    let idx = ABSOLUTE_POS as usize;
    if idx >= ngrid {
        terminate!();
    }
    let v = v_g[idx];
    out_re[idx] = rho_re[idx] * v;
    out_im[idx] = rho_im[idx] * v;
}

/// Zero-out complex grid buffers (utility for density init).
#[cfg(feature = "cubecl-kernels")]
pub fn zero_complex(
    target_re: &mut DeviceBuffer<f64>,
    target_im: &mut DeviceBuffer<f64>,
    ngrid: usize,
) -> Result<()> {
    let client = cubecl_cpu::ComputeClient::load(&cubecl_cpu::CpuDevice::default());
    let re_handle = target_re
        .cube_handle()
        .ok_or_else(|| BackendError::KernelFailure {
            message: "zero_complex missing re handle".into(),
        })?;
    let im_handle = target_im
        .cube_handle()
        .ok_or_else(|| BackendError::KernelFailure {
            message: "zero_complex missing im handle".into(),
        })?;
    unsafe {
        zero_complex_kernel::launch_unchecked::<CpuRuntime>(
            &client,
            CubeCount::new_1d(((ngrid as u32) + 63) / 64),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts::<f64>(re_handle, ngrid, 1),
            ArrayArg::from_raw_parts::<f64>(im_handle, ngrid, 1),
            ngrid,
        );
    }
    Ok(())
}

#[cfg(feature = "cubecl-kernels")]
#[cube(launch_unchecked)]
fn zero_complex_kernel(
    re: &mut Array<f64>,
    im: &mut Array<f64>,
    #[comptime] ngrid: usize,
) {
    let idx = ABSOLUTE_POS as usize;
    if idx >= ngrid {
        terminate!();
    }
    re[idx] = 0.0;
    im[idx] = 0.0;
}

/// Zero-out matrix buffer (utility for J init).
#[cfg(feature = "cubecl-kernels")]
pub fn zero_matrix(
    target: &mut DeviceBuffer<f64>,
    len: usize,
) -> Result<()> {
    let client = cubecl_cpu::ComputeClient::load(&cubecl_cpu::CpuDevice::default());
    let handle = target
        .cube_handle()
        .ok_or_else(|| BackendError::KernelFailure {
            message: "zero_matrix missing handle".into(),
        })?;
    unsafe {
        zero_matrix_kernel::launch_unchecked::<CpuRuntime>(
            &client,
            CubeCount::new_1d(((len as u32) + 63) / 64),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts::<f64>(handle, len, 1),
            len,
        );
    }
    Ok(())
}

#[cfg(feature = "cubecl-kernels")]
#[cube(launch_unchecked)]
fn zero_matrix_kernel(
    buf: &mut Array<f64>,
    #[comptime] len: usize,
) {
    let idx = ABSOLUTE_POS as usize;
    if idx >= len {
        terminate!();
    }
    buf[idx] = 0.0;
}

#[cfg(feature = "cubecl-kernels")]
fn launch_fft3d<const INVERSE: bool>(
    runtime: &CubeRuntimeHandle,
    grid: &mut ComplexGridHandle,
) -> Result<()> {
    match runtime {
        CubeRuntimeHandle::Cpu(cpu_rt) => launch_fft3d_cpu::<INVERSE>(cpu_rt.client(), grid),
        #[cfg(feature = "wgpu-backend")]
        CubeRuntimeHandle::Wgpu(wgpu_rt) => launch_fft3d_wgpu::<INVERSE>(wgpu_rt.client(), grid)
            .or_else(|e| {
                tracing::debug!("wgpu fft fell back to cpu: {e:?}");
                launch_fft3d_cpu::<INVERSE>(&ComputeClient::load(&CpuDevice::default()), grid)
            }),
        #[cfg(not(feature = "wgpu-backend"))]
        CubeRuntimeHandle::Wgpu(_) => launch_fft3d_cpu::<INVERSE>(&ComputeClient::load(&CpuDevice::default()), grid),
    }
}

#[cfg(feature = "cubecl-kernels")]
fn launch_fft3d_cpu<const INVERSE: bool>(
    client: &ComputeClient<CpuRuntime::Server>,
    grid: &mut ComplexGridHandle,
) -> Result<()> {
    let [nx, ny, nz] = grid.dims;
    let total = nx * ny * nz;
    let cube_dim = CubeDim::new_1d((total as u32).min(64));
    let cubes = CubeCount::new_1d(((total as u32) + cube_dim.x - 1) / cube_dim.x);

    let re_handle = grid
        .re
        .cube_handle()
        .ok_or_else(|| BackendError::KernelFailure {
            message: "missing re handle".into(),
        })?;
    let im_handle = grid
        .im
        .cube_handle()
        .ok_or_else(|| BackendError::KernelFailure {
            message: "missing im handle".into(),
        })?;

    unsafe {
        fft_pass::<INVERSE>::launch_unchecked::<CpuRuntime>(
            client,
            cubes.clone(),
            cube_dim,
            ArrayArg::from_raw_parts::<f64>(re_handle, total, 1),
            ArrayArg::from_raw_parts::<f64>(im_handle, total, 1),
            nx,
            ny,
            nz,
        )
    }

    Ok(())
}

#[cfg(all(feature = "cubecl-kernels", feature = "wgpu-backend"))]
fn launch_fft3d_wgpu<const INVERSE: bool>(
    client: &ComputeClient<WgpuServer>,
    grid: &mut ComplexGridHandle,
) -> Result<()> {
    let [nx, ny, nz] = grid.dims;
    let total = nx * ny * nz;
    let cube_dim = CubeDim::new_1d((total as u32).min(64));
    let cubes = CubeCount::new_1d(((total as u32) + cube_dim.x - 1) / cube_dim.x);

    let re_handle = grid
        .re
        .cube_handle()
        .ok_or_else(|| BackendError::KernelFailure {
            message: "missing re handle".into(),
        })?
        .clone();
    let im_handle = grid
        .im
        .cube_handle()
        .ok_or_else(|| BackendError::KernelFailure {
            message: "missing im handle".into(),
        })?
        .clone();

    // Wrap as TensorHandle for WgpuRuntime
    let re_tensor = WgpuTensorHandle::<WgpuRuntime, f64>::new(re_handle, vec![total], vec![1]);
    let im_tensor = WgpuTensorHandle::<WgpuRuntime, f64>::new(im_handle, vec![total], vec![1]);

    unsafe {
        fft_pass::<INVERSE>::launch_unchecked::<WgpuRuntime>(
            &client,
            cubes.clone(),
            cube_dim,
            ArrayArg::from_raw_parts::<f64>(&re_tensor.handle, total, 1),
            ArrayArg::from_raw_parts::<f64>(&im_tensor.handle, total, 1),
            nx,
            ny,
            nz,
        )
    }

    Ok(())
}

#[cfg(feature = "cubecl-kernels")]
#[cube(launch_unchecked)]
fn fft_pass<const INVERSE: bool>(
    re: &mut Array<f64>,
    im: &mut Array<f64>,
    #[comptime] nx: usize,
    #[comptime] ny: usize,
    #[comptime] nz: usize,
) {
    let idx = ABSOLUTE_POS as usize;
    let total = nx * ny * nz;
    if idx >= total {
        terminate!();
    }
    let mut acc_re = 0.0f64;
    let mut acc_im = 0.0f64;
    let kx = idx % nx;
    let tmp = idx / nx;
    let ky = tmp % ny;
    let kz = tmp / ny;
    let sign = if INVERSE { 1.0 } else { -1.0 };
    let norm = if INVERSE { 1.0 / total as f64 } else { 1.0 };
    let two_pi = 2.0f64 * PI;

    let mut x = 0;
    while x < nx {
        let mut y = 0;
        while y < ny {
            let mut z = 0;
            while z < nz {
                let src = (z * ny + y) * nx + x;
                let angle = two_pi
                    * (kx as f64 * x as f64 / nx as f64
                        + ky as f64 * y as f64 / ny as f64
                        + kz as f64 * z as f64 / nz as f64);
                let (s, c) = angle.sin_cos();
                let sin = sign * s;
                let cos = c;
                let ar = re[src];
                let ai = im[src];
                acc_re += ar * cos - ai * sin;
                acc_im += ar * sin + ai * cos;
                z += 1;
            }
            y += 1;
        }
        x += 1;
    }

    re[idx] = acc_re * norm;
    im[idx] = acc_im * norm;
}
/// Density build placeholder: zeroes rho for now (host fallback exists in fock.rs).
#[cfg(feature = "cubecl-kernels")]
pub fn build_density_zero(
    rho_re: &mut DeviceBuffer<f64>,
    rho_im: &mut DeviceBuffer<f64>,
    ngrid: usize,
) -> Result<()> {
    zero_complex(rho_re, rho_im, ngrid)
}

/// Host fallback density build: rho = 0.0 (for now).
pub fn build_density_host(
    rho_re: &mut DeviceBuffer<f64>,
    rho_im: &mut DeviceBuffer<f64>,
    backend: &dyn crate::backend::traits::Backend,
    ngrid: usize,
) -> Result<()> {
    backend.write_f64(rho_re, &vec![0.0; ngrid])?;
    backend.write_f64(rho_im, &vec![0.0; ngrid])?;
    Ok(())
}

/// Project V_H grid back to J matrices (placeholder zero init via kernel).
#[cfg(feature = "cubecl-kernels")]
pub fn project_j_zero(j: &mut MatrixHandle) -> Result<()> {
    zero_matrix(&mut j.buffer, j.rows * j.cols)
}

/// Density build placeholder: zeroes rho; later replace with AOxAO accumulation.
#[cfg(feature = "cubecl-kernels")]
pub fn build_density_zero(
    rho_re: &mut DeviceBuffer<f64>,
    rho_im: &mut DeviceBuffer<f64>,
    ngrid: usize,
) -> Result<()> {
    zero_complex(rho_re, rho_im, ngrid)
}

/// CubeCL density build for nk=1, layout ao_values[mu][grid], d[mu][nu].
#[cfg(feature = "cubecl-kernels")]
pub fn build_density_cube(
    ao_values: &DeviceBuffer<f64>,
    nao: usize,
    ngrid: usize,
    d_mats: &DeviceBuffer<f64>,
    rho_re: &mut DeviceBuffer<f64>,
    rho_im: &mut DeviceBuffer<f64>,
) -> Result<()> {
    let client = cubecl_cpu::ComputeClient::load(&cubecl_cpu::CpuDevice::default());
    let ao_handle = ao_values.cube_handle().ok_or_else(|| BackendError::KernelFailure { message: "build_density_cube missing ao handle".into() })?;
    let d_handle = d_mats.cube_handle().ok_or_else(|| BackendError::KernelFailure { message: "build_density_cube missing d handle".into() })?;
    let rho_re_handle = rho_re.cube_handle().ok_or_else(|| BackendError::KernelFailure { message: "build_density_cube missing rho_re handle".into() })?;
    let rho_im_handle = rho_im.cube_handle().ok_or_else(|| BackendError::KernelFailure { message: "build_density_cube missing rho_im handle".into() })?;

    unsafe {
        build_density_kernel::launch_unchecked::<CpuRuntime>(
            &client,
            CubeCount::new_1d(((ngrid as u32) + 63) / 64),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts::<f64>(ao_handle, nao * ngrid, 1),
            ArrayArg::from_raw_parts::<f64>(d_handle, nao * nao, 1),
            ArrayArg::from_raw_parts::<f64>(rho_re_handle, ngrid, 1),
            ArrayArg::from_raw_parts::<f64>(rho_im_handle, ngrid, 1),
            ngrid,
            nao,
        );
    }
    Ok(())
}

#[cfg(feature = "cubecl-kernels")]
#[cube(launch_unchecked)]
fn build_density_kernel(
    ao_vals: &Array<f64>, // [mu][grid] flattened mu-major
    d_mats: &Array<f64>,  // [mu][nu] flattened row-major
    rho_re: &mut Array<f64>,
    rho_im: &mut Array<f64>,
    #[comptime] ngrid: usize,
    #[comptime] nao: usize,
) {
    let g = ABSOLUTE_POS as usize;
    if g >= ngrid {
        terminate!();
    }
    let mut acc = 0.0f64;
    let mut mu = 0;
    while mu < nao {
        let mut nu = 0;
        while nu < nao {
            let d_idx = mu * nao + nu;
            let d_val = d_mats[d_idx];
            if d_val != 0.0 {
                let a = ao_vals[mu * ngrid + g];
                let b = ao_vals[nu * ngrid + g];
                acc += d_val * a * b;
            }
            nu += 1;
        }
        mu += 1;
    }
    rho_re[g] = acc;
    rho_im[g] = 0.0;
}

/// CubeCL J projection for nk=1, layout ao_values[mu][grid].
#[cfg(feature = "cubecl-kernels")]
pub fn project_j_cube(
    v_re: &DeviceBuffer<f64>,
    ao_values: &DeviceBuffer<f64>,
    nao: usize,
    ngrid: usize,
    j: &mut MatrixHandle,
) -> Result<()> {
    let client = cubecl_cpu::ComputeClient::load(&cubecl_cpu::CpuDevice::default());
    let v_handle = v_re.cube_handle().ok_or_else(|| BackendError::KernelFailure { message: "project_j_cube missing v handle".into() })?;
    let ao_handle = ao_values.cube_handle().ok_or_else(|| BackendError::KernelFailure { message: "project_j_cube missing ao handle".into() })?;
    let j_handle = j.buffer.cube_handle().ok_or_else(|| BackendError::KernelFailure { message: "project_j_cube missing j handle".into() })?;

    unsafe {
        project_j_kernel::launch_unchecked::<CpuRuntime>(
            &client,
            CubeCount::new_1d(((nao * nao) as u32 + 63) / 64),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts::<f64>(v_handle, ngrid, 1),
            ArrayArg::from_raw_parts::<f64>(ao_handle, nao * ngrid, 1),
            ArrayArg::from_raw_parts::<f64>(j_handle, nao * nao, 1),
            ngrid,
            nao,
        );
    }
    Ok(())
}

#[cfg(feature = "cubecl-kernels")]
#[cube(launch_unchecked)]
fn project_j_kernel(
    v_re: &Array<f64>,
    ao_vals: &Array<f64>,
    j_out: &mut Array<f64>,
    #[comptime] ngrid: usize,
    #[comptime] nao: usize,
) {
    let idx = ABSOLUTE_POS as usize;
    let total = nao * nao;
    if idx >= total {
        terminate!();
    }
    let mu = idx / nao;
    let nu = idx % nao;
    let mut acc = 0.0f64;
    let mut g = 0;
    while g < ngrid {
        let a = ao_vals[mu * ngrid + g];
        let b = ao_vals[nu * ngrid + g];
        acc += v_re[g] * a * b;
        g += 1;
    }
    j_out[idx] = acc;
}
