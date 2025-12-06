//! Minimal logging setup for binary/examples until a richer tracing pipeline
//! is wired. This keeps logging optional for library consumers.

use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::SubscriberBuilder;
use tracing_subscriber::prelude::*;

/// Initialise a simple `tracing_subscriber` subscriber. Idempotent courtesy of
/// `try_init` so repeated calls in tests won't panic.
pub fn init_tracing() {
    let _ = SubscriberBuilder::default()
        .with_env_filter(EnvFilter::from_default_env())
        .finish()
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_tracing_is_idempotent() {
        init_tracing();
        init_tracing();
    }
}
