//! CubeCL single-source kernel scaffolding.
//! This module is gated behind the `cubecl-kernels` feature and is currently a placeholder.
//! Replace the `todo!()` bodies with real CubeCL kernels once available.

use crate::backend::error::{BackendError, Result};
use crate::backend::types::{MatrixHandle, VectorHandle, ComplexGridHandle, DeviceBuffer};
use crate::backend::runtime::CubeRuntimeHandle;

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
    client: &ComputeClient<cubecl_wgpu::WgpuServer>,
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
