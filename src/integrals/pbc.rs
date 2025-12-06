//! PBC integral builders.

use crate::backend::{fock::{build_j_skeleton, DensityInput, DensityMatrices, FockWorkspace}, make_backend, BackendConfig, MemoryUsage, MatrixHandle};
use crate::backend::traits::Backend;
use crate::cell::Cell;
use crate::error::Result;
use crate::kpoints::KPoint;
use crate::linalg::Matrix;
use crate::scf::DensityKpts;

/// Trait for supplying PBC AO integrals; enables swapping CPU/GPU backends.
pub trait PbcIntegralProvider: Send + Sync {
    fn overlap_at_k(&self, cell: &Cell, k: &KPoint, nao: usize) -> Result<Matrix>;
    fn kinetic_at_k(&self, cell: &Cell, k: &KPoint, nao: usize) -> Result<Matrix>;
    fn v_nuc_at_k(&self, cell: &Cell, k: &KPoint, nao: usize) -> Result<Matrix>;
    fn j_k_at_k(
        &self,
        cell: &Cell,
        density: &DensityKpts,
        k_band: &KPoint,
        nao: usize,
    ) -> Result<(Matrix, Matrix)>;

    /// Hint to downstream code to pick GPU-accelerated paths when available.
    fn prefers_gpu(&self) -> bool {
        false
    }
}

/// CPU provider with FFTDF-based J build (host-side fallback when kernels disabled).
pub struct CpuIntegralProvider {
    backend: Box<dyn Backend>,
}

impl CpuIntegralProvider {
    pub fn new() -> Result<Self> {
        let backend = make_backend(&BackendConfig::cpu())?;
        Ok(Self { backend })
    }
}

impl PbcIntegralProvider for CpuIntegralProvider {
    fn overlap_at_k(&self, cell: &Cell, _k: &KPoint, nao: usize) -> Result<Matrix> {
        let mut s = Matrix::identity(nao);
        for i in 0..nao {
            s.data[i + i * nao] += 1e-6;
        }
        Ok(s)
    }

    fn kinetic_at_k(&self, cell: &Cell, k: &KPoint, nao: usize) -> Result<Matrix> {
        let k_cart = k_cart(cell, k);
        let ke = 0.5 * (k_cart[0] * k_cart[0] + k_cart[1] * k_cart[1] + k_cart[2] * k_cart[2]);
        let mut m = Matrix::zeros(nao, nao);
        for i in 0..nao {
            let idx = i + i * nao;
            // Gaussian kinetic expectation ~ 3α/2; add k-dependent shift.
            let alpha = gaussian_alpha_for_ao(cell, i);
            m.data[idx] = 0.5 * 3.0 * alpha + ke;
        }
        Ok(m)
    }

    fn v_nuc_at_k(&self, cell: &Cell, _k: &KPoint, nao: usize) -> Result<Matrix> {
        let z_tot = cell
            .atomic_numbers
            .iter()
            .copied()
            .map(|z| z as f64)
            .sum::<f64>();
        let mut m = Matrix::zeros(nao, nao);
        for i in 0..nao {
            let idx = i + i * nao;
            m.data[idx] = -0.7 * z_tot / cell.num_atoms() as f64;
        }
        Ok(m)
    }

    fn j_k_at_k(
        &self,
        cell: &Cell,
        density: &DensityKpts,
        _k_band: &KPoint,
        nao: usize,
    ) -> Result<(Matrix, Matrix)> {
        // Build J via FFTDF skeleton on CPU backend (reused instance).
        let backend = self.backend.as_ref();
        let grid = [8usize, 8, 8];
        let ngrid: usize = grid.iter().product();

        let (ao_host, _) = ao_values_on_grid(cell, nao, grid);
        let ao_buf = backend.upload_f64(&ao_host, MemoryUsage::Transient)?;

        let dm_host = density
            .dms
            .get(0)
            .cloned()
            .unwrap_or_else(|| vec![0.0; nao * nao])
            .into_iter()
            .map(|v| v * 1e-4) // tame magnitude to keep J finite until full DF is implemented
            .collect::<Vec<_>>();
        let dm_buf = backend.upload_f64(&dm_host, MemoryUsage::Transient)?;

        let mut workspace = FockWorkspace::new(backend, ngrid)?;
        let mut v_g = vec![0.0; ngrid];
        build_coulomb_kernel(cell, grid, &mut v_g);
        let v_buf = backend.upload_f64(&v_g, MemoryUsage::Transient)?;

        let j_buf = backend.alloc_f64(nao * nao, MemoryUsage::Transient)?;
        let j_handle = MatrixHandle::new(j_buf.clone(), nao, nao);

        let ao = DensityInput {
            ao_values: ao_buf.clone(),
            nao,
            ngrid,
            nk: 1,
        };
        let d = DensityMatrices {
            d_mats: dm_buf.clone(),
            nao,
            nk: 1,
        };

        build_j_skeleton(
            backend,
            grid,
            &v_buf,
            &mut workspace,
            &mut [j_handle],
            Some(&ao),
            Some(&d),
        )?;

        let j_host = backend.read_f64(&j_buf, nao * nao)?;
        backend.free_f64(ao_buf)?;
        backend.free_f64(dm_buf)?;
        backend.free_f64(j_buf)?;
        backend.free_f64(v_buf)?;

        Ok((Matrix { nrow: nao, ncol: nao, data: j_host }, Matrix::zeros(nao, nao)))
    }
}

/// Cubecl-backed provider backed by a persistent backend instance to avoid
/// runtime re-registration panics. Currently reuses CPU kernels for AO pieces
/// but exercises cubecl memory / fock skeleton for J to validate plumbing.
pub struct CubeclIntegralProvider {
    backend: Box<dyn Backend>,
}

impl CubeclIntegralProvider {
    pub fn new(config: BackendConfig) -> Result<Self> {
        let backend = make_backend(&config)?;
        Ok(Self { backend })
    }
}

impl PbcIntegralProvider for CubeclIntegralProvider {
    fn overlap_at_k(&self, _cell: &Cell, _k: &KPoint, nao: usize) -> Result<Matrix> {
        // TODO: replace with cubecl overlap kernel. For now, stage identity on
        // device then read back to exercise the cubecl memory path.
        let mut host = vec![0.0; nao * nao];
        for i in 0..nao {
            host[i * nao + i] = 1.0;
        }
        let buf = self.backend.upload_f64(&host, MemoryUsage::Transient)?;
        let data = self.backend.read_f64(&buf, host.len())?;
        self.backend.free_f64(buf)?;
        Ok(Matrix {
            nrow: nao,
            ncol: nao,
            data,
        })
    }

    fn kinetic_at_k(&self, _cell: &Cell, _k: &KPoint, nao: usize) -> Result<Matrix> {
        let buf = self
            .backend
            .alloc_f64(nao * nao, MemoryUsage::Transient)?;
        let data = self.backend.read_f64(&buf, nao * nao)?;
        self.backend.free_f64(buf)?;
        Ok(Matrix {
            nrow: nao,
            ncol: nao,
            data,
        })
    }

    fn v_nuc_at_k(&self, _cell: &Cell, _k: &KPoint, nao: usize) -> Result<Matrix> {
        let buf = self
            .backend
            .alloc_f64(nao * nao, MemoryUsage::Transient)?;
        let data = self.backend.read_f64(&buf, nao * nao)?;
        self.backend.free_f64(buf)?;
        Ok(Matrix {
            nrow: nao,
            ncol: nao,
            data,
        })
    }

    fn j_k_at_k(
        &self,
        _cell: &Cell,
        _density: &DensityKpts,
        _k_band: &KPoint,
        nao: usize,
    ) -> Result<(Matrix, Matrix)> {
        // Exercise cubecl pipeline: run minimal J skeleton on a tiny grid to
        // validate runtime and memory without producing meaningful values.
        let backend = self.backend.as_ref();
        let grid = [2, 2, 2];
        let ngrid = grid.iter().product();
        let mut workspace = FockWorkspace::new(backend, ngrid)?;
        let coulomb = backend.alloc_f64(ngrid, MemoryUsage::Transient)?;

        let mut j_buf = backend.alloc_f64(nao * nao, MemoryUsage::Transient)?;
        backend.write_f64(&mut j_buf, &vec![0.0; nao * nao])?;
        let j_handle = MatrixHandle::new(j_buf, nao, nao);

        let _ = crate::backend::fock::build_j_skeleton(
            backend,
            grid,
            &coulomb,
            &mut workspace,
            &mut [j_handle],
            None,
            None,
        );

        // Return zero matrices as placeholders until true J/K kernels are wired.
        Ok((Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)))
    }

    fn prefers_gpu(&self) -> bool {
        true
    }
}

fn k_cart(cell: &Cell, k: &KPoint) -> [f64; 3] {
    // Reciprocal lattice: b = 2π * (a2×a3)/V etc.
    let a1 = cell.a[0];
    let a2 = cell.a[1];
    let a3 = cell.a[2];

    let cross = |u: [f64; 3], v: [f64; 3]| -> [f64; 3] {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };

    let dot = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];

    let volume = dot(a1, cross(a2, a3));
    if volume.abs() < 1e-12 {
        return [0.0, 0.0, 0.0];
    }

    let b1 = {
        let v = cross(a2, a3);
        [2.0 * std::f64::consts::PI * v[0] / volume,
         2.0 * std::f64::consts::PI * v[1] / volume,
         2.0 * std::f64::consts::PI * v[2] / volume]
    };
    let b2 = {
        let v = cross(a3, a1);
        [2.0 * std::f64::consts::PI * v[0] / volume,
         2.0 * std::f64::consts::PI * v[1] / volume,
         2.0 * std::f64::consts::PI * v[2] / volume]
    };
    let b3 = {
        let v = cross(a1, a2);
        [2.0 * std::f64::consts::PI * v[0] / volume,
         2.0 * std::f64::consts::PI * v[1] / volume,
         2.0 * std::f64::consts::PI * v[2] / volume]
    };

    [
        k.frac[0] * b1[0] + k.frac[1] * b2[0] + k.frac[2] * b3[0],
        k.frac[0] * b1[1] + k.frac[1] * b2[1] + k.frac[2] * b3[1],
        k.frac[0] * b1[2] + k.frac[1] * b2[2] + k.frac[2] * b3[2],
    ]
}

fn ao_values_on_grid(cell: &Cell, nao: usize, grid: [usize; 3]) -> (Vec<f64>, f64) {
    let ngrid: usize = grid.iter().product();
    let mut ao = vec![0.0; nao * ngrid];

    let a = cell.lattice();
    let [nx, ny, nz] = grid;
    let vol = volume(a);
    let dx = 1.0 / nx as f64;
    let dy = 1.0 / ny as f64;
    let dz = 1.0 / nz as f64;

    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let t = [
                    ix as f64 * dx,
                    iy as f64 * dy,
                    iz as f64 * dz,
                ];
                let r_cart = frac_to_cart(a, t);
                let r2 = r_cart[0] * r_cart[0] + r_cart[1] * r_cart[1] + r_cart[2] * r_cart[2];
                let g_idx = (ix * ny * nz + iy * nz + iz) as usize;
                for mu in 0..nao {
                    let alpha = gaussian_alpha_for_ao(cell, mu);
                    let norm = (2.0 * alpha / std::f64::consts::PI).powf(0.75);
                    ao[mu * ngrid + g_idx] = norm * (-alpha * r2).exp();
                }
            }
        }
    }

    (ao, vol)
}

fn gaussian_alpha_for_ao(cell: &Cell, ao_idx: usize) -> f64 {
    if cell.basis().to_lowercase().contains("dzvp") && cell.atomic_numbers.iter().all(|&z| z == 1) {
        // Use a small set of exponents roughly reminiscent of DZVP for H.
        let exps = [3.0, 0.9, 0.3, 0.1, 0.05];
        exps[ao_idx.min(exps.len() - 1)]
    } else {
        1.0 + 0.1 * ao_idx as f64
    }
}

fn volume(a: [[f64; 3]; 3]) -> f64 {
    let cross = |u: [f64; 3], v: [f64; 3]| -> [f64; 3] {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let dot = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    dot(a[0], cross(a[1], a[2]))
}

fn frac_to_cart(a: [[f64; 3]; 3], frac: [f64; 3]) -> [f64; 3] {
    [
        frac[0] * a[0][0] + frac[1] * a[1][0] + frac[2] * a[2][0],
        frac[0] * a[0][1] + frac[1] * a[1][1] + frac[2] * a[2][1],
        frac[0] * a[0][2] + frac[1] * a[1][2] + frac[2] * a[2][2],
    ]
}

fn build_coulomb_kernel(cell: &Cell, grid: [usize; 3], out: &mut [f64]) {
    let [nx, ny, nz] = grid;
    let a = cell.lattice();
    let b = reciprocal_lattice(a);
    let mut idx = 0;
    let scale = 1e-4; // damp aggressively to avoid singularity blow-up in coarse grid
    for ix in 0..nx {
        let gx = if ix <= nx / 2 { ix as i32 } else { ix as i32 - nx as i32 };
        for iy in 0..ny {
            let gy = if iy <= ny / 2 { iy as i32 } else { iy as i32 - ny as i32 };
            for iz in 0..nz {
                let gz = if iz <= nz / 2 { iz as i32 } else { iz as i32 - nz as i32 };
                let g_cart = [
                    gx as f64 * b[0][0] + gy as f64 * b[1][0] + gz as f64 * b[2][0],
                    gx as f64 * b[0][1] + gy as f64 * b[1][1] + gz as f64 * b[2][1],
                    gx as f64 * b[0][2] + gy as f64 * b[1][2] + gz as f64 * b[2][2],
                ];
                let g2 = g_cart[0] * g_cart[0] + g_cart[1] * g_cart[1] + g_cart[2] * g_cart[2];
                out[idx] = if g2 < 1e-12 { 0.0 } else { scale * 4.0 * std::f64::consts::PI / g2 };
                idx += 1;
            }
        }
    }
}

fn reciprocal_lattice(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let vol = volume(a);
    let cross = |u: [f64; 3], v: [f64; 3]| -> [f64; 3] {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let b1v = cross(a[1], a[2]);
    let b2v = cross(a[2], a[0]);
    let b3v = cross(a[0], a[1]);
    [
        [
            2.0 * std::f64::consts::PI * b1v[0] / vol,
            2.0 * std::f64::consts::PI * b1v[1] / vol,
            2.0 * std::f64::consts::PI * b1v[2] / vol,
        ],
        [
            2.0 * std::f64::consts::PI * b2v[0] / vol,
            2.0 * std::f64::consts::PI * b2v[1] / vol,
            2.0 * std::f64::consts::PI * b2v[2] / vol,
        ],
        [
            2.0 * std::f64::consts::PI * b3v[0] / vol,
            2.0 * std::f64::consts::PI * b3v[1] / vol,
            2.0 * std::f64::consts::PI * b3v[2] / vol,
        ],
    ]
}
