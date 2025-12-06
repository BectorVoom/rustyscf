use crate::error as public;

/// Backend-local error type to decouple runtime details from public errors.
#[derive(Debug)]
pub enum BackendError {
    /// Requested allocation exceeds the configured budget.
    OutOfMemory {
        requested_mb: Option<usize>,
        limit_mb: Option<usize>,
    },
    /// Backend (typically WGPU) is unavailable or not built.
    DeviceUnavailable { message: String },
    /// Kernel failed to compile or execute.
    KernelFailure { message: String },
}

pub type Result<T> = std::result::Result<T, BackendError>;

impl From<BackendError> for public::Error {
    fn from(err: BackendError) -> Self {
        match err {
            BackendError::OutOfMemory {
                requested_mb,
                limit_mb,
            } => public::Error::ResourceError {
                message: "out of memory while allocating backend buffer".into(),
                requested_memory_mb: requested_mb,
                limit_memory_mb: limit_mb,
            },
            BackendError::DeviceUnavailable { message } => public::Error::ResourceError {
                message,
                requested_memory_mb: None,
                limit_memory_mb: None,
            },
            BackendError::KernelFailure { message } => public::Error::InternalError(message),
        }
    }
}
