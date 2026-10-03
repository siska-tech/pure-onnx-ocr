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
/// shapes from the concrete input instead. Older exports (PP-OCRv5) carry no
/// such hints, so this is a no-op for them.
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
    entries: std::collections::VecDeque<(K, std::sync::Arc<TypedRunnableModel>)>,
}

impl<K: PartialEq + Copy> PlanCache<K> {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: std::collections::VecDeque::new(),
        }
    }

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
pub(crate) fn lock_cache<T>(cache: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
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
