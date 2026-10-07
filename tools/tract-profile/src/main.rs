//! Per-node timing of a DBNet detection plan, loaded and compiled the same
//! way as pure-onnx-ocr's `DetInferenceSession` (intermediate facts cleared
//! as in `onnx_model::load_paddle_onnx`, concrete `[1, 3, H, W]` input,
//! optimized plan, tract's executor installed with `multithread_tract_scope`).
//!
//! ```text
//! tract-profile <model.onnx> <height> <width> <threads> <runs>
//! ```
//!
//! Prints the time per (optimized tract op, original ONNX op) group, averaged
//! over `runs` after two warm-up runs. With `DETAIL=<substring>` (or an empty
//! `DETAIL=`), it also lists the slowest nodes whose op name contains the
//! substring, with their shapes. The input is synthetic: timings of these ops
//! do not depend on the pixel values.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tract_linalg::multithread::Executor;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::plan::{eval, SimpleState};

fn main() -> TractResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let h: usize = args[2].parse()?;
    let w: usize = args[3].parse()?;
    let threads: usize = args[4].parse()?;
    let runs: usize = args[5].parse()?;

    let mut model = tract_onnx::onnx()
        .with_ignore_output_shapes(true)
        .model_for_path(path)?;
    // original ONNX op type per node name
    let onnx_ops: HashMap<String, String> = model
        .nodes()
        .iter()
        .map(|n| (n.name.clone(), n.op.name().to_string()))
        .collect();
    let inputs: Vec<OutletId> = model.input_outlets()?.to_vec();
    for id in 0..model.nodes().len() {
        if model.nodes()[id].op.name() == "Const" {
            continue;
        }
        for slot in 0..model.nodes()[id].outputs.len() {
            let o = OutletId::new(id, slot);
            if !inputs.contains(&o) {
                model.set_outlet_fact(o, InferenceFact::default())?;
            }
        }
    }
    model.set_input_fact(0, f32::fact([1, 3, h, w]).into())?;
    let plan: Arc<TypedRunnableModel> = model
        .into_typed()?
        .into_decluttered()?
        .into_optimized()?
        .into_runnable()?;

    let executor = if threads > 1 {
        Executor::multithread_with_name(threads, "tract-profile")
    } else {
        Executor::SingleThread
    };

    let input: Tensor =
        tract_ndarray::Array4::<f32>::from_shape_fn((1, 3, h, w), |(_, c, y, x)| {
            (((x * 7 + y * 13 + c * 31) % 255) as f32 / 255.0 - 0.5) * 4.0
        })
        .into();

    let mut per_node: Vec<f64> = vec![0.0; plan.model().nodes().len()];
    let mut totals = vec![];
    let mut output_hash = 0u64;
    tract_linalg::multithread::multithread_tract_scope(executor, || -> TractResult<()> {
        for run in 0..runs + 2 {
            let mut state = SimpleState::new(&plan)?;
            let t0 = Instant::now();
            let measure = run >= 2;
            let outputs =
                state.run_plan_with_eval(tvec!(input.clone().into()), |ctx, st, node, inp| {
                    let t = Instant::now();
                    let out = eval(ctx, st, node, inp);
                    if measure {
                        per_node[node.id] += t.elapsed().as_secs_f64() * 1000.0;
                    }
                    out
                })?;
            if measure {
                totals.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            // FNV-1a over the output's bits: equal hashes mean bit-identical outputs, which
            // tells whether a tract change altered the arithmetic.
            output_hash = 0xcbf2_9ce4_8422_2325;
            for value in outputs[0].to_plain_array_view::<f32>()?.iter() {
                for byte in value.to_bits().to_le_bytes() {
                    output_hash = (output_hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
                }
            }
        }
        Ok(())
    })?;
    totals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total_med = totals[totals.len() / 2];

    // group by (optimized op name, original onnx op)
    let mut groups: HashMap<(String, String), (f64, usize)> = HashMap::new();
    for node in plan.model().nodes() {
        let t = per_node[node.id] / runs as f64;
        let base = node
            .name
            .split('.')
            .next()
            .unwrap_or(&node.name)
            .to_string();
        let onnx = onnx_ops
            .get(&node.name)
            .or_else(|| onnx_ops.get(&base))
            .cloned()
            .unwrap_or_else(|| "?".into());
        let e = groups
            .entry((node.op().name().to_string(), onnx))
            .or_insert((0.0, 0));
        e.0 += t;
        e.1 += 1;
    }
    if std::env::var("DETAIL").is_ok() {
        let filter = std::env::var("DETAIL").unwrap();
        let mut nodes: Vec<_> = plan.model().nodes().iter().collect();
        nodes.sort_by(|a, b| per_node[b.id].partial_cmp(&per_node[a.id]).unwrap());
        for node in nodes
            .iter()
            .filter(|n| filter.is_empty() || n.op().name().contains(filter.as_str()))
            .take(40)
        {
            let ins: Vec<String> = node
                .inputs
                .iter()
                .map(|i| {
                    format!(
                        "{:?}",
                        plan.model().outlet_fact(*i).map(|f| f.shape.to_tvec())
                    )
                })
                .collect();
            println!(
                "{:7.2}ms {:<18} {:<40} in={} out={:?} op={:?}",
                per_node[node.id] / runs as f64,
                node.op().name(),
                node.name,
                ins.join(","),
                node.outputs[0].fact.shape,
                format!("{:?}", node.op())
                    .chars()
                    .take(160)
                    .collect::<String>()
            );
        }
    }
    let sum: f64 = groups.values().map(|v| v.0).sum();
    let mut rows: Vec<_> = groups.into_iter().collect();
    rows.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
    println!("# {path} [1,3,{h},{w}] threads={threads} runs={runs} total_median={total_med:.1}ms sum_nodes={sum:.1}ms output_hash={output_hash:016x}");
    println!("tract_op\tonnx_op\tnodes\tms\tpct");
    for ((op, onnx), (ms, n)) in rows {
        if ms < 0.05 {
            continue;
        }
        println!("{op}\t{onnx}\t{n}\t{ms:.1}\t{:.1}", ms / sum * 100.0);
    }
    Ok(())
}
