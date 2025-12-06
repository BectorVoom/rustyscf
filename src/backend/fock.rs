use crate::backend::error::Result;
use crate::backend::traits::Backend;
use crate::backend::types::{
    ComplexGridHandle, DeviceBuffer, MatrixHandle, MemoryUsage,
};

/// Optional inputs for density/J builds.
pub struct DensityInput {
    pub ao_values: DeviceBuffer<f64>, // layout [k][mu][grid] (flattened; current MVP supports nk=1)
    pub nao: usize,
    pub ngrid: usize,
    pub nk: usize,
}

pub struct DensityMatrices {
    pub d_mats: DeviceBuffer<f64>, // layout [k][mu][nu] flattened
    pub nao: usize,
    pub nk: usize,
}

/// Workspace buffers reused across J-build steps to avoid repeated allocations.
pub struct FockWorkspace {
    pub rho_re: DeviceBuffer<f64>,
    pub rho_im: DeviceBuffer<f64>,
    pub v_g_re: DeviceBuffer<f64>,
    pub v_g_im: DeviceBuffer<f64>,
    pub v_h_re: DeviceBuffer<f64>,
    pub v_h_im: DeviceBuffer<f64>,
}

impl FockWorkspace {
    pub fn new(backend: &dyn Backend, grid_elems: usize) -> Result<Self> {
        let rho_re = backend.alloc_f64(grid_elems, MemoryUsage::Transient)?;
        let rho_im = backend.alloc_f64(grid_elems, MemoryUsage::Transient)?;
        let v_g_re = backend.alloc_f64(grid_elems, MemoryUsage::Transient)?;
        let v_g_im = backend.alloc_f64(grid_elems, MemoryUsage::Transient)?;
        let v_h_re = backend.alloc_f64(grid_elems, MemoryUsage::Transient)?;
        let v_h_im = backend.alloc_f64(grid_elems, MemoryUsage::Transient)?;
        Ok(Self {
            rho_re,
            rho_im,
            v_g_re,
            v_g_im,
            v_h_re,
            v_h_im,
        })
    }
}

/// Skeleton J-build orchestration:
/// 1) Build density grid (placeholder).
/// 2) Forward FFT.
/// 3) Apply Coulomb kernel (placeholder multiply).
/// 4) Inverse FFT.
/// 5) Project to J matrices (placeholder).
pub fn build_j_skeleton(
    backend: &dyn Backend,
    grid_dims: [usize; 3],
    coulomb_kernel: &DeviceBuffer<f64>,
    workspace: &mut FockWorkspace,
    j_mats: &mut [MatrixHandle], // per k
    ao: Option<&DensityInput>,
    d: Option<&DensityMatrices>,
) -> Result<()> {
    let ngrid = grid_dims.iter().product();

    // Step 1: density build.
    build_density(
        backend,
        &mut workspace.rho_re,
        &mut workspace.rho_im,
        ngrid,
        ao,
        d,
    )?;

    let mut rho_grid = ComplexGridHandle::new(workspace.rho_re.clone(), workspace.rho_im.clone(), grid_dims);

    // Step 2: forward FFT rho -> rho(G)
    backend.fft3d_forward(&mut rho_grid)?;

    // Step 3: apply Coulomb kernel (rho(G) * v(G)) -> v_g_re/im
    apply_coulomb_kernel(
        backend,
        &rho_grid.re,
        &rho_grid.im,
        coulomb_kernel,
        &mut workspace.v_g_re,
        &mut workspace.v_g_im,
        ngrid,
    )?;

    // Step 4: inverse FFT to V_H(r)
    let mut v_grid = ComplexGridHandle::new(workspace.v_g_re.clone(), workspace.v_g_im.clone(), grid_dims);
    backend.fft3d_inverse(&mut v_grid)?;

    // Step 5: project V_H back to J matrices
    for j in j_mats.iter_mut() {
        let len = j.rows * j.cols;
        project_j(
            backend,
            &v_grid,
        j,
        len,
        ao,
    )?;
    }

    Ok(())
}

fn build_density(
    backend: &dyn Backend,
    rho_re: &mut DeviceBuffer<f64>,
    rho_im: &mut DeviceBuffer<f64>,
    ngrid: usize,
    ao: Option<&DensityInput>,
    d: Option<&DensityMatrices>,
) -> Result<()> {
    // MVP: support nk=1; otherwise zero.
    if let (Some(ao), Some(d)) = (ao, d) {
        if ao.nk != 1 || d.nk != 1 {
            backend.write_f64(rho_re, &vec![0.0; ngrid])?;
            backend.write_f64(rho_im, &vec![0.0; ngrid])?;
            return Ok(());
        }
        #[cfg(feature = "cubecl-kernels")]
        {
            crate::backend::kernels::cube::build_density_cube(
                &ao.ao_values,
                ao.nao,
                ngrid,
                &d.d_mats,
                rho_re,
                rho_im,
            )?;
            return Ok(());
        }
        #[cfg(not(feature = "cubecl-kernels"))]
        {
            let nao = ao.nao;
            let ao_vals = backend.read_f64(&ao.ao_values, ao.ao_values.len())?;
            let d_vals = backend.read_f64(&d.d_mats, d.d_mats.len())?;
            let mut rho_host = vec![0.0f64; ngrid];
            for mu in 0..nao {
                for nu in 0..nao {
                    let d_idx = mu * nao + nu;
                    let d_val = d_vals.get(d_idx).copied().unwrap_or(0.0);
                    if d_val == 0.0 {
                        continue;
                    }
                    let ao_mu_off = mu * ngrid;
                    let ao_nu_off = nu * ngrid;
                    for g in 0..ngrid {
                        let a = ao_vals[ao_mu_off + g];
                        let b = ao_vals[ao_nu_off + g];
                        rho_host[g] += d_val * a * b;
                    }
                }
            }
            backend.write_f64(rho_re, &rho_host)?;
            backend.write_f64(rho_im, &vec![0.0; ngrid])?;
            return Ok(());
        }
    }

    // No inputs: zero
    backend.write_f64(rho_re, &vec![0.0; ngrid])?;
    backend.write_f64(rho_im, &vec![0.0; ngrid])?;
    Ok(())
}

fn project_j(
    backend: &dyn Backend,
    v_grid: &ComplexGridHandle,
    j: &mut MatrixHandle,
    len: usize,
    ao: Option<&DensityInput>,
) -> Result<()> {
    if let Some(ao) = ao {
        if ao.nk == 1 {
            let nao = ao.nao;
            let ngrid = ao.ngrid;
            #[cfg(feature = "cubecl-kernels")]
            {
                crate::backend::kernels::cube::project_j_cube(
                    &v_grid.re,
                    &ao.ao_values,
                    nao,
                    ngrid,
                    j,
                )?;
                return Ok(());
            }
            #[cfg(not(feature = "cubecl-kernels"))]
            {
                let v_re = backend.read_f64(&v_grid.re, ngrid)?;
                let ao_vals = backend.read_f64(&ao.ao_values, ao.ao_values.len())?;
                let mut j_host = vec![0.0f64; nao * nao];
                for mu in 0..nao {
                    for nu in 0..nao {
                        let ao_mu_off = mu * ngrid;
                        let ao_nu_off = nu * ngrid;
                        let mut acc = 0.0;
                        for g in 0..ngrid {
                            acc += v_re[g] * ao_vals[ao_mu_off + g] * ao_vals[ao_nu_off + g];
                        }
                        j_host[mu * nao + nu] = acc;
                    }
                }
                backend.write_f64(&mut j.buffer, &j_host)?;
                return Ok(());
            }
        }
    }

    // default zero
    backend.write_f64(&mut j.buffer, &vec![0.0; len])?;
    Ok(())
}

fn apply_coulomb_kernel(
    backend: &dyn Backend,
    rho_re: &DeviceBuffer<f64>,
    rho_im: &DeviceBuffer<f64>,
    v_g: &DeviceBuffer<f64>,
    out_re: &mut DeviceBuffer<f64>,
    out_im: &mut DeviceBuffer<f64>,
    ngrid: usize,
) -> Result<()> {
    #[cfg(feature = "cubecl-kernels")]
    {
        crate::backend::kernels::cube::coulomb_mul(rho_re, rho_im, v_g, out_re, out_im, ngrid)
    }

    #[cfg(not(feature = "cubecl-kernels"))]
    {
        // Temporary host-side multiply; replace with CubeCL kernel.
        let rr = backend.read_f64(rho_re, ngrid)?;
        let ri = backend.read_f64(rho_im, ngrid)?;
        let vg = backend.read_f64(v_g, ngrid)?;
        let mut or = Vec::with_capacity(ngrid);
        let mut oi = Vec::with_capacity(ngrid);
        for i in 0..ngrid {
            or.push(rr[i] * vg[i]);
            oi.push(ri[i] * vg[i]);
        }
        backend.write_f64(out_re, &or)?;
        backend.write_f64(out_im, &oi)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{make_backend, BackendConfig, MemoryUsage, MatrixHandle};

    #[test]
    fn j_skeleton_runs_on_cpu() {
        let backend = make_backend(&BackendConfig::cpu()).expect("backend");
        let grid = [2usize, 2, 2];
        let ngrid = grid.iter().product();
        let mut workspace = FockWorkspace::new(backend.as_ref(), ngrid).expect("workspace");
        let coulomb = backend
            .alloc_f64(ngrid, MemoryUsage::Transient)
            .expect("coulomb");
        backend
            .write_f64(&mut coulomb.clone(), &vec![1.0; ngrid])
            .expect("write kernel");
        let j_buf = backend
            .alloc_f64(1, MemoryUsage::Transient)
            .expect("j buf");
        let j_handle = MatrixHandle::new(j_buf, 1, 1);

        build_j_skeleton(
            backend.as_ref(),
            grid,
            &coulomb,
            &mut workspace,
            &mut [j_handle],
            None,
            None,
        )
        .expect("skeleton run");
    }
}
