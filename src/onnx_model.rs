//! Shared helpers for loading PaddleOCR ONNX exports with `tract-onnx`.

use std::path::Path;
use tract_onnx::prelude::*;

/// Loads an ONNX model and discards the intermediate shape hints recorded in
/// the graph's `value_info` section.
///
/// PaddleOCR 3.x exports (PP-OCRv6 and later) annotate almost every tensor
/// with symbolic dimensions such as `DynamicDimension.0`. `tract` applies
/// those hints verbatim, and once a concrete input shape such as
/// `[1, 3, 736, 736]` is set, analysis fails with
/// `Impossible to unify Sym(DynamicDimension.0) with Val(1)`. Clearing the
/// facts of every non-input, non-constant outlet lets `tract` re-derive the
/// shapes from the concrete input instead. The same reset is applied to older
/// exports (PP-OCRv5), even when they do not contain conflicting shape hints.
pub(crate) fn load_paddle_onnx(model_path: &Path) -> TractResult<InferenceModel> {
    let mut model = tract_onnx::onnx()
        .with_ignore_output_shapes(true)
        .model_for_path(model_path)?;
    clear_intermediate_facts(&mut model)?;
    Ok(model)
}

/// Same as [`load_paddle_onnx`] for an ONNX graph held in memory (for
/// example fetched by a browser).
pub(crate) fn load_paddle_onnx_from_bytes(bytes: &[u8]) -> TractResult<InferenceModel> {
    let mut reader = std::io::Cursor::new(bytes);
    let mut model = tract_onnx::onnx()
        .with_ignore_output_shapes(true)
        .model_for_read(&mut reader)?;
    clear_intermediate_facts(&mut model)?;
    Ok(model)
}

/// Retains graph inputs and constant weights while resetting inferred facts.
/// Callers must bind concrete input dimensions before compiling a runnable plan.
fn clear_intermediate_facts(model: &mut InferenceModel) -> TractResult<()> {
    let inputs: Vec<OutletId> = model.input_outlets()?.to_vec();
    for node_id in 0..model.nodes().len() {
        if model.nodes()[node_id].op.name() == "Const" {
            continue;
        }
        for slot in 0..model.nodes()[node_id].outputs.len() {
            let outlet = OutletId::new(node_id, slot);
            if inputs.contains(&outlet) {
                continue;
            }
            model.set_outlet_fact(outlet, InferenceFact::default())?;
        }
    }
    Ok(())
}

/// Small least-recently-used cache of optimised inference plans keyed by
/// input shape.
///
/// `tract` produces the fastest plans when every dimension is concrete, so a
/// plan is compiled per input shape (100-400 ms each). Each plan holds its own
/// optimised copy of the weights, so the cache is bounded to keep memory in
/// check when many image sizes or text widths are seen.
#[derive(Debug)]
pub(crate) struct PlanCache<K> {
    capacity: usize,
    // Front = least recently used; back = most recently used. Arc lets an
    // in-flight inference retain its plan even after the cache evicts it.
    entries: std::collections::VecDeque<(K, std::sync::Arc<TypedRunnableModel>)>,
}

impl<K: PartialEq + Copy> PlanCache<K> {
    /// Creates an empty cache. Zero capacity is treated as one, not disabled.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: std::collections::VecDeque::new(),
        }
    }

    /// Applies a minimum capacity of one and immediately evicts excess entries.
    pub(crate) fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity.max(1);
        while self.entries.len() > self.capacity {
            self.entries.pop_front();
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns the plan for `key`, marking it as most recently used.
    pub(crate) fn get(&mut self, key: K) -> Option<std::sync::Arc<TypedRunnableModel>> {
        let index = self.entries.iter().position(|(k, _)| *k == key)?;
        let entry = self.entries.remove(index)?;
        let plan = std::sync::Arc::clone(&entry.1);
        self.entries.push_back(entry);
        Some(plan)
    }

    /// Inserts a plan, evicting the least recently used one when full.
    pub(crate) fn insert(&mut self, key: K, plan: std::sync::Arc<TypedRunnableModel>) {
        if let Some(index) = self.entries.iter().position(|(k, _)| *k == key) {
            self.entries.remove(index);
        }
        while self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back((key, plan));
    }
}

/// Locks a plan cache, ignoring poisoning (a panic while holding the lock
/// cannot leave the cache in an invalid state).
fn lock_cache<T>(cache: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Debug)]
struct SharedState<K> {
    plans: PlanCache<K>,
    /// Keys whose plan one thread is compiling right now.
    compiling: Vec<K>,
}

/// [`PlanCache`] shared between threads that compiles each shape only once.
///
/// Recognition runs batches in parallel and most crops share one width (the
/// 320-pixel minimum), so on a cold cache every worker used to miss on the
/// same key and compile its own copy of the plan: 13-16 identical
/// compilations per image with 16 threads, which slowed the first run down
/// and multiplied its peak memory. Now the first thread compiles while the
/// others wait for the result. Different shapes still compile concurrently,
/// and the lock is never held while compiling.
#[derive(Debug)]
pub(crate) struct SharedPlanCache<K> {
    state: std::sync::Mutex<SharedState<K>>,
    compiled: std::sync::Condvar,
}

impl<K: PartialEq + Copy> SharedPlanCache<K> {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            state: std::sync::Mutex::new(SharedState {
                plans: PlanCache::new(capacity),
                compiling: Vec::new(),
            }),
            compiled: std::sync::Condvar::new(),
        }
    }

    pub(crate) fn set_capacity(&self, capacity: usize) {
        lock_cache(&self.state).plans.set_capacity(capacity);
    }

    pub(crate) fn len(&self) -> usize {
        lock_cache(&self.state).plans.len()
    }

    /// Returns the cached plan for `key`, or compiles it with `compile`.
    ///
    /// While another thread compiles the same key, waits for it and reuses
    /// its plan. If that compilation fails, a waiting thread compiles again
    /// and reports its own error.
    pub(crate) fn get_or_compile(
        &self,
        key: K,
        compile: impl FnOnce() -> TractResult<std::sync::Arc<TypedRunnableModel>>,
    ) -> TractResult<std::sync::Arc<TypedRunnableModel>> {
        let mut state = lock_cache(&self.state);
        loop {
            if let Some(plan) = state.plans.get(key) {
                return Ok(plan);
            }
            if !state.compiling.contains(&key) {
                break;
            }
            state = self
                .compiled
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state.compiling.push(key);
        drop(state);

        // Clears the mark and wakes the waiters even if `compile` fails or
        // panics.
        let _in_flight = InFlight { cache: self, key };
        let plan = compile()?;
        lock_cache(&self.state)
            .plans
            .insert(key, std::sync::Arc::clone(&plan));
        Ok(plan)
    }
}

struct InFlight<'a, K: PartialEq + Copy> {
    cache: &'a SharedPlanCache<K>,
    key: K,
}

impl<K: PartialEq + Copy> Drop for InFlight<'_, K> {
    fn drop(&mut self) {
        lock_cache(&self.cache.state)
            .compiling
            .retain(|k| *k != self.key);
        self.cache.compiled.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_plan() -> std::sync::Arc<TypedRunnableModel> {
        let mut model = TypedModel::default();
        let source = model
            .add_source("x", f32::fact([1]))
            .expect("source should be added");
        model.outputs = vec![source];
        model.into_runnable().unwrap()
    }

    #[test]
    fn evicts_least_recently_used_plan() {
        let mut cache = PlanCache::new(2);
        cache.insert(1u32, dummy_plan());
        cache.insert(2u32, dummy_plan());
        assert!(cache.get(1).is_some()); // 1 becomes most recent
        cache.insert(3u32, dummy_plan()); // evicts 2
        assert!(cache.get(2).is_none());
        assert!(cache.get(1).is_some());
        assert!(cache.get(3).is_some());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn concurrent_misses_compile_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let cache = SharedPlanCache::new(4);
        let compiles = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    cache
                        .get_or_compile(320u32, || {
                            compiles.fetch_add(1, Ordering::SeqCst);
                            // Keep the key in flight while the others arrive.
                            std::thread::sleep(std::time::Duration::from_millis(50));
                            Ok(dummy_plan())
                        })
                        .expect("compile should succeed");
                });
            }
        });
        assert_eq!(compiles.load(Ordering::SeqCst), 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn failed_compile_lets_next_caller_retry() {
        let cache = SharedPlanCache::new(4);
        let failed = cache.get_or_compile(1u32, || {
            Err(tract_onnx::tract_core::internal::anyhow!("boom"))
        });
        assert!(failed.is_err());
        assert_eq!(cache.len(), 0);
        assert!(cache.get_or_compile(1u32, || Ok(dummy_plan())).is_ok());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn shrinking_capacity_drops_oldest_entries() {
        let mut cache = PlanCache::new(3);
        for key in 0..3u32 {
            cache.insert(key, dummy_plan());
        }
        cache.set_capacity(1);
        assert_eq!(cache.len(), 1);
        assert!(cache.get(2).is_some());
    }
}
