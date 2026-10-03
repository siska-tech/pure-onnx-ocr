//! Thread pool used for tract's matrix multiplications.
//!
//! With the `multithread` feature (on by default), each engine owns a rayon
//! thread pool and runs its ONNX sessions inside
//! `tract_linalg::multithread::multithread_tract_scope`, so the setting is
//! scoped to this crate's calls instead of changing tract's process-wide
//! default executor. WebAssembly targets cannot spawn threads and always run
//! single-threaded.

pub(crate) use tract_linalg::multithread::Executor;

/// Whether multi-threaded inference is available in this build.
pub const MULTITHREAD_SUPPORTED: bool =
    cfg!(all(feature = "multithread", not(target_arch = "wasm32")));

/// Default number of inference threads: the number of logical CPUs, capped
/// at 8. Larger pools gave no further speed-up in our measurements (see
/// `docs/devlog/perf/task-perf-001-multithread.md`).
pub fn default_inference_threads() -> usize {
    if !MULTITHREAD_SUPPORTED {
        return 1;
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 8)
}

/// Builds the executor for `threads` worker threads (`<= 1` is
/// single-threaded). Falls back to single-threaded execution when threads
/// are not supported or the pool cannot be created.
pub(crate) fn executor_for(threads: usize) -> Executor {
    #[cfg(all(feature = "multithread", not(target_arch = "wasm32")))]
    {
        if threads > 1 {
            let pool = std::panic::catch_unwind(|| {
                Executor::multithread_with_name(threads, "pure-onnx-ocr")
            });
            match pool {
                Ok(executor) => return executor,
                Err(_) => log::warn!(
                    "failed to create a {}-thread inference pool; running single-threaded",
                    threads
                ),
            }
        }
    }
    let _ = threads;
    Executor::SingleThread
}

/// Runs `f` with `executor` installed for tract's matrix multiplications.
pub(crate) fn run_with<R>(executor: &Executor, f: impl FnOnce() -> R) -> R {
    tract_linalg::multithread::multithread_tract_scope(executor.clone(), f)
}

/// Applies `f` to every item, in parallel on the executor's thread pool when
/// it has one. Results keep the input order.
///
/// Used for independent units of work such as recognition batches: each
/// call runs single-threaded tract plans, so the parallelism comes from
/// running several batches at once rather than from splitting one matrix
/// multiplication, which scales much better for the small CNNs used by
/// PaddleOCR.
pub(crate) fn parallel_map<T, R, F>(executor: &Executor, items: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync + Send,
{
    #[cfg(all(feature = "multithread", not(target_arch = "wasm32")))]
    {
        if let Executor::MultiThread(pool) = executor {
            use rayon::prelude::*;
            if items.len() > 1 {
                return pool.install(|| items.par_iter().map(&f).collect());
            }
        }
    }
    let _ = executor;
    items.iter().map(f).collect()
}
