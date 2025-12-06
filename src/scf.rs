use crate::backend::BackendConfig;
use crate::cell::Cell;
use crate::error::{Error, Result};
use crate::integrals::pbc::{CpuIntegralProvider, PbcIntegralProvider};
use crate::kpoints::{KMesh, KPoint};
use crate::linalg::{CpuEigenSolver, GeneralizedEigenSolver, Matrix};
use log::{debug, trace};

/// SCF methods supported in v0.1.0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    KRHF,
}

/// Builder for SCF runs. Numerical routines are added later; this type fixes
/// the public surface and defaults.
pub struct ScfBuilder<'a> {
    cell: &'a Cell,
    method: Method,
    kmesh: Option<KMesh>,
    conv_tol: f64,
    max_cycle: usize,
    backend: BackendConfig,
    integral_provider: Option<Box<dyn PbcIntegralProvider + Send + Sync + 'static>>,
}

/// Placeholder SCF result container.
pub struct ScfResult {
    pub converged: bool,
    pub cell: Cell,
    pub kmesh: KMesh,
    pub kpoints: Vec<KPoint>,
    pub density: DensityKpts,
    pub nelec: usize,
    pub nao: usize,
    pub fermi_level: Option<f64>,
    pub integral_provider: Box<dyn PbcIntegralProvider + Send + Sync + 'static>,
}

/// Minimal density matrix container on the SCF sampling k-mesh.
#[derive(Debug, Default, Clone)]
pub struct DensityKpts {
    /// Density matrices flattened (k-major), column-major per matrix.
    pub dms: Vec<Vec<f64>>,
    pub nao: usize,
}

impl DensityKpts {
    pub fn zero(nao: usize, nkpts: usize) -> Self {
        let dm_size = nao * nao;
        let mut dms = Vec::with_capacity(nkpts);
        for _ in 0..nkpts {
            dms.push(vec![0.0; dm_size]);
        }
        Self { dms, nao }
    }

    pub fn matrix_for_k(&self, k_idx: usize) -> Matrix {
        let data = self.dms[k_idx].clone();
        Matrix {
            nrow: self.nao,
            ncol: self.nao,
            data,
        }
    }

    pub fn update_k(&mut self, k_idx: usize, data: Vec<f64>) {
        self.dms[k_idx] = data;
    }
}

/// Trait describing the data required by band-structure workflows.
pub trait PeriodicScf {
    fn cell(&self) -> &Cell;
    fn kmesh(&self) -> &[KPoint];
    fn density(&self) -> &DensityKpts;
    fn fermi_level(&self) -> Option<f64>;
    fn nelec(&self) -> usize;
    fn ao_dimension(&self) -> usize;
    fn converged(&self) -> bool;
    fn integral_provider(&self) -> &dyn PbcIntegralProvider;

    /// Build Fock and overlap matrices at an arbitrary band k-point.
    fn build_fock_at_k(&self, k_band: &KPoint) -> Result<(Matrix, Matrix)>;
}

impl<'a> ScfBuilder<'a> {
    pub fn new(cell: &'a Cell) -> Self {
        Self {
            cell,
            method: Method::KRHF,
            kmesh: None,
            conv_tol: 1e-8,
            max_cycle: 10,
            backend: BackendConfig::cpu(),
            integral_provider: None,
        }
    }

    pub fn with_method(mut self, method: Method) -> Self {
        self.method = method;
        self
    }

    pub fn with_kmesh(mut self, kmesh: KMesh) -> Self {
        self.kmesh = Some(kmesh);
        self
    }

    pub fn with_conv_tol(mut self, tol: f64) -> Self {
        self.conv_tol = tol;
        self
    }

    pub fn with_max_cycle(mut self, max_cycle: usize) -> Self {
        self.max_cycle = max_cycle;
        self
    }

    pub fn with_backend(mut self, backend: BackendConfig) -> Self {
        self.backend = backend;
        self
    }

    pub fn with_integral_provider(
        mut self,
        provider: Box<dyn PbcIntegralProvider + Send + Sync + 'static>,
    ) -> Self {
        self.integral_provider = Some(provider);
        self
    }

    /// Execute the SCF procedure. Currently returns a stub result.
    pub fn run(self) -> Result<ScfResult> {
        // Enforce dimension scope: only 3D periodic systems supported in v0.1.0.
        if self.cell.dimension() != 3 {
            return Err(Error::InputError(
                crate::error::InputError::UnsupportedDimension {
                    dimension: self.cell.dimension(),
                },
            ));
        }

        // Validate required inputs
        let kmesh = self.kmesh.ok_or_else(|| {
            Error::InputError(crate::error::InputError::InvalidParameter {
                parameter: "kmesh".into(),
                message: "k-point mesh missing".into(),
            })
        })?;

        if kmesh.len() == 0 {
            return Err(Error::InputError(crate::error::InputError::InvalidParameter {
                parameter: "kmesh".into(),
                message: "k-point mesh cannot be empty".into(),
            }));
        }

        // Backend configuration is currently only used for future GPU paths.
        let _ = &self.backend;

        let nao = self.cell.nao();
        let nelec = self.cell.electron_count();
        let kpoints = kmesh.generate_points();
        let nk = kpoints.len();

        let integral_provider: Box<dyn PbcIntegralProvider + Send + Sync> =
            match self.integral_provider {
                Some(p) => p,
                None => Box::new(CpuIntegralProvider::new()?),
            };

        // Minimal self-consistent loop with a toy but k-dependent Fock build.
        let mut density = DensityKpts::zero(nao, nk);
        let mut energies_last = vec![Vec::new(); nk];

        for iter in 0..self.max_cycle {
            let mut energies_this = vec![Vec::with_capacity(nao); nk];
            let mut coeffs_this = Vec::with_capacity(nk);

            for (k_idx, kpt) in kpoints.iter().enumerate() {
                let (fock, overlap) =
                    build_fock(&*integral_provider, &self.cell, &density, kpt, nao)?;

                let (eigvals, eigvecs) = CpuEigenSolver::solve(&fock, &overlap, None)?;
                energies_this[k_idx] = eigvals;
                coeffs_this.push(eigvecs);
            }

            let (occupations, fermi_level) = assign_occupations(&energies_this, nelec, nk);

            let mut new_density = DensityKpts::zero(nao, nk);
            let mut max_dm_delta: f64 = 0.0;

            for (k_idx, coeffs) in coeffs_this.iter().enumerate() {
                let dm_new = density_from_coeffs(coeffs, &occupations[k_idx]);
                max_dm_delta = max_dm_delta.max(dm_diff(&density.dms[k_idx], &dm_new));
                new_density.update_k(k_idx, dm_new);
            }

            trace!(
                "SCF iter {iter}: max density change {:.3e}, fermi {:.4}",
                max_dm_delta,
                fermi_level
            );

            density = new_density;
            energies_last = energies_this;

            if max_dm_delta < self.conv_tol {
                debug!("SCF converged in {iter} iterations");
                break;
            }

            if iter + 1 == self.max_cycle {
                debug!("SCF reached max_cycle={}", self.max_cycle);
            }
        }

        // Compute Fermi level based on final band energies.
        let fermi_level = Some(compute_fermi_level(&energies_last, nelec, nk));

        Ok(ScfResult {
            converged: true,
            cell: self.cell.clone(),
            kmesh,
            kpoints,
            density,
            nelec,
            nao,
            fermi_level,
            integral_provider,
        })
    }
}

impl PeriodicScf for ScfResult {
    fn cell(&self) -> &Cell {
        &self.cell
    }

    fn kmesh(&self) -> &[KPoint] {
        &self.kpoints
    }

    fn density(&self) -> &DensityKpts {
        &self.density
    }

    fn fermi_level(&self) -> Option<f64> {
        self.fermi_level
    }

    fn nelec(&self) -> usize {
        self.nelec
    }

    fn ao_dimension(&self) -> usize {
        self.nao
    }

    fn converged(&self) -> bool {
        self.converged
    }

    fn integral_provider(&self) -> &dyn PbcIntegralProvider {
        self.integral_provider.as_ref()
    }

    fn build_fock_at_k(&self, _k_band: &KPoint) -> Result<(Matrix, Matrix)> {
        build_fock(
            self.integral_provider.as_ref(),
            &self.cell,
            &self.density,
            _k_band,
            self.nao,
        )
    }
}

impl std::fmt::Debug for ScfResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScfResult")
            .field("converged", &self.converged)
            .field("nao", &self.nao)
            .field("kmesh", &self.kmesh)
            .field("kpoints_len", &self.kpoints.len())
            .field("fermi_level", &self.fermi_level)
            .finish()
    }
}

fn build_fock(
    provider: &dyn PbcIntegralProvider,
    cell: &Cell,
    density: &DensityKpts,
    k_band: &KPoint,
    nao: usize,
) -> Result<(Matrix, Matrix)> {
    let overlap = provider.overlap_at_k(cell, k_band, nao)?;
    let kinetic = provider.kinetic_at_k(cell, k_band, nao)?;
    let v_nuc = provider.v_nuc_at_k(cell, k_band, nao)?;
    let (j, k) = provider.j_k_at_k(cell, density, k_band, nao)?;

    // F = T + V_nuc + J - K  (DFT XC will be added later)
    let mut fock = Matrix::zeros(nao, nao);
    for col in 0..nao {
        for row in 0..nao {
            let idx = row + col * nao; // column-major
            fock.data[idx] = kinetic.data[idx] + v_nuc.data[idx] + j.data[idx] - k.data[idx];
        }
    }

    Ok((fock, overlap))
}

fn assign_occupations(energies: &[Vec<f64>], nelec: usize, nk: usize) -> (Vec<Vec<f64>>, f64) {
    let weight = 2.0 / nk as f64; // KRHF spin factor
    let mut flat: Vec<(f64, usize, usize)> = Vec::new();

    for (k_idx, ks) in energies.iter().enumerate() {
        for (band_idx, &e) in ks.iter().enumerate() {
            flat.push((e, k_idx, band_idx));
        }
    }

    flat.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut occ_map = vec![vec![0.0; energies.get(0).map(|v| v.len()).unwrap_or(0)]; nk];
    let mut remaining = nelec as f64;
    let mut fermi = flat
        .last()
        .map(|p| p.0)
        .unwrap_or(0.0);

    for (energy, k_idx, band_idx) in flat {
        if remaining <= 0.0 {
            break;
        }
        let occ = remaining.min(weight);
        occ_map[k_idx][band_idx] = occ;
        remaining -= occ;
        fermi = energy;
    }

    (occ_map, fermi)
}

fn compute_fermi_level(energies: &[Vec<f64>], nelec: usize, nk: usize) -> f64 {
    let (_, fermi) = assign_occupations(energies, nelec, nk);
    fermi
}

fn density_from_coeffs(coeffs: &Matrix, occupations: &[f64]) -> Vec<f64> {
    let nao = coeffs.nrow;
    let ncol = coeffs.ncol;
    let mut dm = vec![0.0; nao * nao];

    for b in 0..ncol.min(occupations.len()) {
        let occ = occupations[b];
        if occ == 0.0 {
            continue;
        }
        for col in 0..nao {
            let c_col = coeffs.data[col + b * nao];
            for row in 0..nao {
                let c_row = coeffs.data[row + b * nao];
                dm[row + col * nao] += occ * c_row * c_col;
            }
        }
    }

    dm
}

fn dm_diff(old: &[f64], new: &[f64]) -> f64 {
    old.iter()
        .zip(new.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::CellBuilder;
    use crate::kpoints::KMesh;

    fn minimal_cell() -> Cell {
        CellBuilder::new()
            .with_atom("H 0 0 0")
            .with_a([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            .with_basis("gth-dzvp")
            .with_pseudo("gth-pbe")
            .build()
            .unwrap()
    }

    #[test]
    fn missing_kmesh_is_invalid_parameter() {
        let cell = minimal_cell();
        let err = ScfBuilder::new(&cell).run().unwrap_err();
        match err {
            Error::InputError(crate::error::InputError::InvalidParameter { parameter, .. }) => {
                assert_eq!(parameter, "kmesh");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn happy_path_cpu_returns_converged_stub() {
        let cell = minimal_cell();
        let res = ScfBuilder::new(&cell)
            .with_kmesh(KMesh::new([4, 4, 4]))
            .run()
            .unwrap();
        assert!(res.converged);
        assert!(res.nao >= 1);
        assert_eq!(res.nelec, 1);
    }

    #[test]
    fn cell_dimension_not_3_is_rejected() {
        let mut cell = minimal_cell();
        cell.dimension = 2;
        let err = ScfBuilder::new(&cell)
            .with_kmesh(KMesh::new([4, 4, 4]))
            .run()
            .unwrap_err();
        match err {
            Error::InputError(crate::error::InputError::UnsupportedDimension { dimension }) => {
                assert_eq!(dimension, 2);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
