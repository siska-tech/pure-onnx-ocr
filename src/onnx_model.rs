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
