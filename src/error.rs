use thiserror::Error;

/// Public, semantically stable error type for the crate.
#[derive(Debug, Error)]
pub enum Error {
    /// Invalid user input or unsupported configuration.
    #[error("input error: {0}")]
    InputError(#[from] InputError),

    /// Self-consistent field (SCF) procedure failed to converge.
    #[error("SCF did not converge: {reason}")]
    ConvergenceError {
        reason: String,
        last_energy: Option<f64>,
        iterations: usize,
    },

    /// Insufficient resources (e.g., host or device memory).
    #[error("resource error: {message}")]
    ResourceError {
        message: String,
        requested_memory_mb: Option<usize>,
        limit_memory_mb: Option<usize>,
    },

    /// Internal bug or unexpected state.
    #[error("internal error: {0}")]
    InternalError(String),
}

/// Detailed input and validation errors.
#[derive(Debug, Error)]
pub enum InputError {
    #[error("unsupported element: {symbol}")]
    UnsupportedElement { symbol: String },

    #[error("unsupported basis set: {basis}")]
    UnsupportedBasis { basis: String },

    #[error("unsupported pseudo potential: {pseudo}")]
    UnsupportedPseudo { pseudo: String },

    #[error("unsupported dimension: {dimension} (only 3D is supported in v0.1.0)")]
    UnsupportedDimension { dimension: i32 },

    #[error("unsupported k-mesh: {kmesh:?} (only 4x4x4 is supported in v0.1.0)")]
    UnsupportedKMesh { kmesh: [usize; 3] },

    #[error("invalid atom specification: {message}")]
    InvalidAtomSpec { message: String },

    #[error("invalid parameter: {parameter} - {message}")]
    InvalidParameter { parameter: String, message: String },
}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;
