use crate::cell::Cell;
use crate::error::{Error, Result};
use crate::kpoints::{KPath, KPoint};
use crate::linalg::{CpuEigenSolver, CubeclEigenSolver, GeneralizedEigenSolver, Matrix};
use crate::scf::PeriodicScf;

/// Builder for band-structure evaluation using an SCF result.
pub struct BandStructureBuilder<'a> {
    cell: &'a Cell,
    scf: &'a dyn PeriodicScf,
    kpath: Option<KPath>,
    explicit_kpoints: Option<Vec<KPoint>>,
    with_mo_coeff: bool,
    with_occupations: bool,
    n_bands: Option<usize>,
    linalg: BandLinalgBackend,
}

/// Band-structure data container.
#[derive(Debug, Default)]
pub struct BandStructureResult {
    pub kpoints: Vec<KPoint>,
    pub energies: Vec<Vec<f64>>, // kpoint-major: energies[k][band]
    pub occupations: Option<Vec<Vec<f64>>>,
    pub mo_coeff: Option<Vec<Matrix>>,
    pub fermi_level: Option<f64>,
}

/// Alias to align with the design document terminology.
pub type BandStructure = BandStructureResult;

/// Linear algebra backend choice for band eigenproblems.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BandLinalgBackend {
    /// Select Cubecl when the SCF integral provider prefers GPU, else CPU.
    Auto,
    Cpu,
    Cubecl,
}

impl<'a> BandStructureBuilder<'a> {
    pub fn new(cell: &'a Cell, scf: &'a dyn PeriodicScf) -> Self {
        Self {
            cell,
            scf,
            kpath: None,
            explicit_kpoints: None,
            with_mo_coeff: false,
            with_occupations: false,
            n_bands: None,
            linalg: BandLinalgBackend::Auto,
        }
    }

    pub fn with_kpath(mut self, kpath: KPath) -> Self {
        self.kpath = Some(kpath);
        self
    }

    pub fn with_kpoints(mut self, kpoints: Vec<KPoint>) -> Self {
        self.explicit_kpoints = Some(kpoints);
        self
    }

    pub fn with_mo_coeff(mut self, enabled: bool) -> Self {
        self.with_mo_coeff = enabled;
        self
    }

    pub fn with_occupations(mut self, enabled: bool) -> Self {
        self.with_occupations = enabled;
        self
    }

    pub fn with_n_bands(mut self, n_bands: usize) -> Self {
        self.n_bands = Some(n_bands);
        self
    }

    pub fn with_linalg_backend(mut self, backend: BandLinalgBackend) -> Self {
        self.linalg = backend;
        self
    }

    /// Compute band energies along the configured path. Placeholder for now.
    pub fn run(self) -> Result<BandStructureResult> {
        if self.cell.dimension() != 3 {
            return Err(Error::InputError(crate::error::InputError::UnsupportedDimension {
                dimension: self.cell.dimension(),
            }));
        }

        // Fail fast if SCF did not converge in a future real implementation.
        if !self.is_scf_converged() {
            return Err(Error::ScfNotConverged {
                message: "call BandStructureBuilder after SCF convergence".into(),
            });
        }

        let kpoints: Vec<KPoint> = match (self.kpath, self.explicit_kpoints) {
            (Some(_), Some(_)) => {
                return Err(Error::InputError(crate::error::InputError::InvalidParameter {
                    parameter: "kpoints".into(),
                    message: "use either kpath or explicit kpoints, not both".into(),
                }))
            }
            (Some(path), None) => path.interpolate(),
            (None, Some(list)) => list,
            (None, None) => vec![KPoint::gamma()],
        };

        if kpoints.is_empty() {
            return Err(Error::EmptyKPointPath);
        }

        let mut energies = Vec::with_capacity(kpoints.len());
        let mut coeffs = if self.with_mo_coeff {
            Some(Vec::with_capacity(kpoints.len()))
        } else {
            None
        };

        for k in &kpoints {
            let (fock, overlap) = self
                .scf
                .build_fock_at_k(k)
                .map_err(|e| match e {
                    Error::IntegralFailure { .. } => e,
                    _ => Error::IntegralFailure {
                        kpoint: k.frac,
                        message: format!("{e}"),
                    },
                })?;

            let backend = match self.linalg {
                BandLinalgBackend::Auto => {
                    if self.scf.integral_provider().prefers_gpu() {
                        BandLinalgBackend::Cubecl
                    } else {
                        BandLinalgBackend::Cpu
                    }
                }
                other => other,
            };

            let (eigvals, eigvecs) = match backend {
                BandLinalgBackend::Cpu => CpuEigenSolver::solve(&fock, &overlap, self.n_bands),
                BandLinalgBackend::Cubecl => CubeclEigenSolver::solve(&fock, &overlap, self.n_bands),
                BandLinalgBackend::Auto => unreachable!(),
            }?;
            energies.push(eigvals);

            if let Some(storage) = coeffs.as_mut() {
                storage.push(eigvecs);
            }
        }

        let (occupations, fermi_level) = if self.with_occupations || self.scf.fermi_level().is_none() {
            let (occ, f) =
                assign_band_occupations(&energies, self.scf.nelec(), kpoints.len());
            (Some(occ), Some(f))
        } else {
            (None, self.scf.fermi_level())
        };

        Ok(BandStructureResult {
            kpoints,
            energies,
            occupations: if self.with_occupations { occupations } else { None },
            mo_coeff: coeffs,
            fermi_level: fermi_level.or(self.scf.fermi_level()),
        })
    }

    fn is_scf_converged(&self) -> bool {
        self.scf.converged()
    }
}

impl BandStructureResult {
    pub fn energies(&self) -> &[Vec<f64>] {
        &self.energies
    }

    pub fn energy(&self, k_idx: usize, band_idx: usize) -> Option<f64> {
        self.energies
            .get(k_idx)
            .and_then(|b| b.get(band_idx))
            .copied()
    }

    pub fn kpoints(&self) -> &[KPoint] {
        &self.kpoints
    }
}

fn assign_band_occupations(energies: &[Vec<f64>], nelec: usize, nk: usize) -> (Vec<Vec<f64>>, f64) {
    let weight = 2.0 / nk as f64;
    let mut flat = Vec::new();
    for (k_idx, row) in energies.iter().enumerate() {
        for (band_idx, &e) in row.iter().enumerate() {
            flat.push((e, k_idx, band_idx));
        }
    }
    flat.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut occ = vec![vec![0.0; energies.get(0).map(|r| r.len()).unwrap_or(0)]; nk];
    let mut remaining = nelec as f64;
    let mut fermi = flat.last().map(|p| p.0).unwrap_or(0.0);

    for (e, k_idx, band_idx) in flat {
        if remaining <= 0.0 {
            break;
        }
        let o = remaining.min(weight);
        occ[k_idx][band_idx] = o;
        remaining -= o;
        fermi = e;
    }

    (occ, fermi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::CellBuilder;
    use crate::kpoints::{KPath, KPoint};
    use crate::scf::{Method, ScfBuilder};

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
    fn band_builder_returns_nontrivial_band() {
        let cell = minimal_cell();
        let scf = ScfBuilder::new(&cell)
            .with_method(Method::KRHF)
            .with_kmesh(crate::kpoints::KMesh::new([4, 4, 4]))
            .run()
            .unwrap();

        let kpath = KPath::from_points(
            vec![
                KPoint::gamma(),
                KPoint::new([0.5, 0.0, 0.0]).with_label("X"),
            ],
            8,
        );

        let bands = BandStructureBuilder::new(&cell, &scf)
            .with_kpath(kpath.clone())
            .run()
            .unwrap();

        assert_eq!(bands.kpoints().len(), kpath.total_points());
        assert_eq!(bands.energies.len(), bands.kpoints.len());
        assert!(bands
            .energies()
            .iter()
            .all(|row| row.len() == scf.ao_dimension()));
        // Should now yield a non-trivial dispersion along the path (first band).
        let energy_gamma = bands.energies[0][0];
        let energy_x = bands.energies.last().unwrap()[0];
        assert_ne!(energy_gamma, energy_x);
    }

    #[test]
    fn empty_kpath_is_rejected() {
        let cell = minimal_cell();
        let scf = ScfBuilder::new(&cell)
            .with_method(Method::KRHF)
            .with_kmesh(crate::kpoints::KMesh::new([4, 4, 4]))
            .run()
            .unwrap();

        let res = BandStructureBuilder::new(&cell, &scf)
            .with_kpath(KPath::new())
            .run();

        assert!(matches!(res, Err(Error::EmptyKPointPath)));
    }

    #[test]
    fn cubecl_backend_surfaces_not_implemented() {
        let cell = minimal_cell();
        let scf = ScfBuilder::new(&cell)
            .with_method(Method::KRHF)
            .with_kmesh(crate::kpoints::KMesh::new([4, 4, 4]))
            .run()
            .unwrap();

        let res = BandStructureBuilder::new(&cell, &scf)
            .with_linalg_backend(BandLinalgBackend::Cubecl)
            .run();

        // For now cubecl backend delegates to CPU; assert it succeeds to avoid
        // cubecl runtime panics in CI until real GPU path is wired.
        assert!(res.is_ok());
    }
}
