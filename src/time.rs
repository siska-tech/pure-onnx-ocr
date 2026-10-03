//! Clock abstraction.
//!
//! `std::time::Instant::now()` panics on `wasm32-unknown-unknown` (browsers),
//! so that target uses `web_time::Instant`, which is backed by
//! `performance.now()`. Every other target uses the standard library.

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub(crate) use web_time::Instant;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub(crate) use std::time::Instant;
