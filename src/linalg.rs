//! Lightweight linear-algebra placeholders for band-structure workflows.
//!
//! Real numerical kernels will be wired to cubecl in later milestones; for now
//! we provide minimal structures so the public API and builders can execute.

use crate::error::Error;
use nalgebra::{Cholesky, DMatrix, SymmetricEigen};

/// Dense column-major matrix wrapper (f64).
/// All data vectors are interpreted as column-major (row + col * nrow).
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub nrow: usize,
    pub ncol: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn zeros(nrow: usize, ncol: usize) -> Self {
        Self {
            nrow,
            ncol,
            data: vec![0.0; nrow * ncol],
        }
    }

    pub fn identity(dim: usize) -> Self {
        let mut m = Self::zeros(dim, dim);
        for i in 0..dim {
            m.data[i * dim + i] = 1.0;
        }
        m
    }

    pub fn is_square(&self) -> bool {
        self.nrow == self.ncol
    }

    fn to_dmatrix(&self) -> DMatrix<f64> {
        DMatrix::from_column_slice(self.nrow, self.ncol, &self.data)
    }

    #[allow(dead_code)]
    fn from_dmatrix(m: DMatrix<f64>) -> Self {
        Self {
            nrow: m.nrows(),
            ncol: m.ncols(),
            data: m.data.as_vec().clone(),
        }
    }
}

/// Generalized Hermitian eigenproblem solver trait.
pub trait GeneralizedEigenSolver {
    fn solve(
        _fock: &Matrix,
        _overlap: &Matrix,
        _n_bands: Option<usize>,
    ) -> Result<(Vec<f64>, Matrix), Error>;
}

/// CPU backend using nalgebra: solves F C = S C e via Cholesky reduction to a
/// standard symmetric eigenproblem.
pub struct CpuEigenSolver;

impl GeneralizedEigenSolver for CpuEigenSolver {
    fn solve(
        fock: &Matrix,
        overlap: &Matrix,
        n_bands: Option<usize>,
    ) -> Result<(Vec<f64>, Matrix), Error> {
        if fock.nrow != fock.ncol || overlap.nrow != overlap.ncol || fock.nrow != overlap.nrow
        {
            return Err(Error::EigenFailure {
                message: "non-square or dimension-mismatched matrices".into(),
            });
        }

        let dim = fock.nrow;
        let nb = n_bands.unwrap_or(dim).min(dim);

        let s = overlap.to_dmatrix();
        let chol = Cholesky::new(s).ok_or_else(|| Error::EigenFailure {
            message: "overlap not positive definite (Cholesky failed)".into(),
        })?;

        // Transform: F' = L^{-1} F L^{-T}
        let l = chol.l();
        let linv = l.clone().try_inverse().ok_or_else(|| Error::EigenFailure {
            message: "Cholesky inverse failed".into(),
        })?;
        let f = fock.to_dmatrix();
        let reduced = &linv * f * linv.transpose();

        let se = SymmetricEigen::new(reduced);

        // Sort eigenpairs ascending just in case backend changes ordering.
        let mut pairs: Vec<(f64, Vec<f64>)> = se
            .eigenvalues
            .iter()
            .copied()
            .zip(se.eigenvectors.column_iter())
            .map(|(val, vec)| (val, vec.iter().copied().collect()))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut eigvals = Vec::with_capacity(nb);
        let mut eigvecs_col_major = vec![0.0; dim * dim];

        for (idx, (val, vec)) in pairs.into_iter().take(nb).enumerate() {
            eigvals.push(val);
            // Back-transform eigenvectors: C = L^{-T} U
            let u = DMatrix::from_column_slice(dim, 1, &vec);
            let c = chol.solve(&u); // L L^T c = u => c = L^{-T} u
            for (row, &v) in c.column(0).iter().enumerate() {
                eigvecs_col_major[row + idx * dim] = v;
            }
        }

        Ok((eigvals, Matrix { nrow: dim, ncol: nb, data: eigvecs_col_major }))
    }
}

/// Placeholder for future cubecl GPU backend; currently routes to CPU.
pub struct CubeclEigenSolver;

impl GeneralizedEigenSolver for CubeclEigenSolver {
    fn solve(
        fock: &Matrix,
        overlap: &Matrix,
        n_bands: Option<usize>,
    ) -> Result<(Vec<f64>, Matrix), Error> {
        // Delegate to CPU until cubecl generalized eigen kernels are available.
        CpuEigenSolver::solve(fock, overlap, n_bands)
    }
}
