//! Tokio runtime singletons for synchronous binding crates.
//!
//! PHP, Ruby, Elixir NIF, and C FFI all need a process-wide Tokio runtime
//! to bridge async Rust code into their synchronous calling conventions.

use std::sync::OnceLock;

use tokio::runtime::Runtime;

static CURRENT_THREAD_RT: OnceLock<Runtime> = OnceLock::new();

/// Get or create a single-threaded Tokio runtime.
///
/// Suitable for PHP (single-threaded), Ruby (GVL), and Elixir NIF (DirtyIo).
/// Uses `current_thread` scheduler to avoid spawning OS threads.
pub fn current_thread_runtime() -> &'static Runtime {
    CURRENT_THREAD_RT.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to create tokio current_thread runtime")
    })
}
