//! Thread pool used for tract's matrix multiplications.
//!
//! With the `multithread` feature (on by default), each engine owns a rayon
//! thread pool and runs its ONNX sessions inside
//! `tract_linalg::multithread::multithread_tract_scope`, so the setting is
//! scoped to this crate's calls instead of changing tract's process-wide
//! default executor.
//!
//! WebAssembly cannot create rayon pools of its own (`std::thread::spawn` is
//! unsupported there). A build with the `atomics` target feature (see
//! `bindings/wasm/threads`) instead runs on rayon's global pool, which the
//! JavaScript side starts with wasm-bindgen-rayon's `initThreadPool` (tract's
//! `Executor::RayonGlobal`). Other WebAssembly builds run single-threaded.

pub(crate) use tract_linalg::multithread::Executor;

/// Whether multi-threaded inference is available in this build.
pub const MULTITHREAD_SUPPORTED: bool = cfg!(all(
    feature = "multithread",
    any(not(target_arch = "wasm32"), target_feature = "atomics")
));

/// Default number of inference threads: the number of logical CPUs, capped
/// at 16. Recognition runs batches in parallel, so it keeps scaling up to 16
/// threads (see `docs/devlog/perf/task-perf-003-default-threads.md`).
///
/// On WebAssembly with atomics, it is the size of rayon's global pool
/// started by wasm-bindgen-rayon's `initThreadPool` (1 if no pool was
/// started).
pub fn default_inference_threads() -> usize {
    if !MULTITHREAD_SUPPORTED {
        return 1;
    }
    if cfg!(target_arch = "wasm32") {
        return max_inference_threads();
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 16)
}

/// Upper bound for the number of inference threads, if the build has one.
///
/// On WebAssembly with atomics, inference runs on rayon's global pool, so it
/// cannot use more threads than that pool has. Querying the pool starts it:
/// if the JavaScript side has not called `initThreadPool` yet, rayon falls
/// back to a one-thread pool on the calling thread and a later
/// `initThreadPool` fails. Engines must therefore be built after
/// `initThreadPool`.
pub(crate) fn max_inference_threads() -> usize {
    #[cfg(all(
        feature = "multithread",
        target_arch = "wasm32",
        target_feature = "atomics"
    ))]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(all(
        feature = "multithread",
        target_arch = "wasm32",
        target_feature = "atomics"
    )))]
    {
        usize::MAX
    }
}

/// Builds the executor for `threads` worker threads (`<= 1` is
/// single-threaded). Falls back to single-threaded execution when threads
/// are not supported or the pool cannot be created.
///
/// On WebAssembly with atomics, the executor is rayon's global pool and
/// `threads` is not used to size it: rayon's global pool must already have
/// been started (wasm-bindgen-rayon's `initThreadPool`), otherwise rayon
/// falls back to running on the calling thread.
pub(crate) fn executor_for(threads: usize) -> Executor {
    #[cfg(all(
        feature = "multithread",
        target_arch = "wasm32",
        target_feature = "atomics"
    ))]
    {
        if threads > 1 {
            return Executor::RayonGlobal;
        }
    }
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
    #[cfg(all(
        feature = "multithread",
        target_arch = "wasm32",
        target_feature = "atomics"
    ))]
    {
        if let Executor::RayonGlobal = executor {
            use rayon::prelude::*;
            if items.len() > 1 {
                return items.par_iter().map(&f).collect();
            }
        }
    }
    let _ = executor;
    items.iter().map(f).collect()
}
