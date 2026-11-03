// Fused Metal kernels (src/metal.metal). Candle's per-op path needs many small
// dispatches per layer and a dense score tensor plus mask in sliding layers. Here
// LayerNorm, RoPE, attention and the MLP gate take a few dispatches, and sliding
// layers only visit keys within the window. Matmuls stay Candle's. Masked keys
// have an exactly-zero softmax weight in the dense path, so only float summation
// order and Metal's exp/erf differ from the CPU path.

use candle_core::backend::BackendStorage;
use candle_core::{DType, Layout, MetalDevice, MetalError, MetalStorage, Result, Shape};
use candle_metal_kernels::metal::{Buffer, ComputePipeline};
use objc2_metal::MTLSize;
use std::sync::{Arc, OnceLock};

const D: usize = 64;

struct Pipelines {
    norm_rows: ComputePipeline,
    gelu_gate: ComputePipeline,
    rope: ComputePipeline,
    scores: ComputePipeline,
    softmax_pv: ComputePipeline,
}

static PIPELINES: OnceLock<Pipelines> = OnceLock::new();

fn pipelines(device: &MetalDevice) -> Result<&'static Pipelines> {
    if let Some(p) = PIPELINES.get() {
        return Ok(p);
    }
    let device = device.metal_device();
    let library = device.new_library_with_source(include_str!("metal.metal"), None).map_err(MetalError::from)?;
    let pipeline = |name: &str| -> Result<ComputePipeline> {
        let function = library.get_function(name, None).map_err(MetalError::from)?;
        Ok(device.new_compute_pipeline_state_with_function(&function).map_err(MetalError::from)?)
    };
    let loaded = Pipelines {
        norm_rows: pipeline("norm_rows")?,
        gelu_gate: pipeline("gelu_gate")?,
        rope: pipeline("rope")?,
        scores: pipeline("scores")?,
        softmax_pv: pipeline("softmax_pv")?,
    };
    Ok(PIPELINES.get_or_init(|| loaded))
}

fn offset(layout: &Layout, op: &str) -> Result<usize> {
    if !layout.is_contiguous() {
        candle_core::bail!("{op} expects contiguous input")
    }
    Ok(layout.start_offset() * DType::F32.size_in_bytes())
}

fn f32_only(storage: &MetalStorage, op: &str) -> Result<()> {
    if storage.dtype() != DType::F32 {
        candle_core::bail!("{op} expects f32 input")
    }
    Ok(())
}

fn buffer(device: &MetalDevice, elems: usize, label: &str) -> Result<Arc<Buffer>> {
    device.new_buffer_builder().with_size_for(elems, DType::F32).with_label(label).build()
}

fn grid(width: usize, height: usize, depth: usize) -> MTLSize {
    MTLSize { width, height, depth }
}

// Apple GPUs run SIMD groups of 32 threads.
fn group(pipeline: &ComputePipeline) -> MTLSize {
    grid(pipeline.max_total_threads_per_threadgroup().min(32), 1, 1)
}

/// norm_nobias(x, w, 1e-5) over the last dimension.
pub fn norm(xs: &MetalStorage, xl: &Layout, ws: &MetalStorage, wl: &Layout) -> Result<(MetalStorage, Shape)> {
    #[repr(C)]
    struct Params {
        n: u32,
        eps: f32,
    }
    f32_only(xs, "metal norm")?;
    f32_only(ws, "metal norm")?;
    let n = wl.shape().elem_count();
    if xl.dims().last() != Some(&n) {
        candle_core::bail!("norm input {:?} does not end in {n}", xl.dims())
    }
    let device = xs.device();
    let pipeline = &pipelines(device)?.norm_rows;
    let elems = xl.shape().elem_count();
    let out = buffer(device, elems, "acclass-norm")?;
    let encoder = device.command_encoder()?;
    let e = encoder.as_ref();
    e.set_compute_pipeline_state(pipeline);
    e.set_input_buffer(0, Some(xs.buffer()), offset(xl, "metal norm")?);
    e.set_input_buffer(1, Some(ws.buffer()), offset(wl, "metal norm")?);
    e.set_output_buffer(2, Some(&out), 0);
    e.set_bytes(3, &Params { n: n as u32, eps: 1e-5 });
    e.dispatch_thread_groups(grid(elems / n, 1, 1), group(pipeline));
    drop(encoder);
    Ok((MetalStorage::new(out, device.clone(), elems, DType::F32), xl.shape().clone()))
}

/// gate * gelu_erf(input) for rows of [input | gate].
pub fn gelu_gate(xs: &MetalStorage, xl: &Layout, n: usize) -> Result<(MetalStorage, Shape)> {
    f32_only(xs, "metal gelu-gate")?;
    let mut dims = xl.dims().to_vec();
    if dims.last() != Some(&(2 * n)) {
        candle_core::bail!("gelu-gate input {dims:?} does not end in {}", 2 * n)
    }
    let device = xs.device();
    let pipeline = &pipelines(device)?.gelu_gate;
    let elems = xl.shape().elem_count() / 2;
    let out = buffer(device, elems, "acclass-gelu-gate")?;
    let encoder = device.command_encoder()?;
    let e = encoder.as_ref();
    e.set_compute_pipeline_state(pipeline);
    e.set_input_buffer(0, Some(xs.buffer()), offset(xl, "metal gelu-gate")?);
    e.set_output_buffer(1, Some(&out), 0);
    e.set_bytes(2, &(n as u32));
    e.dispatch_threads(grid(elems, 1, 1), group(pipeline));
    drop(encoder);
    *dims.last_mut().unwrap() = n;
    Ok((MetalStorage::new(out, device.clone(), elems, DType::F32), Shape::from(dims)))
}

/// RoPE and attention from one projection: qkv is (1, s, 3 * heads * 64), cos and
/// sin are (at least s, 64). Returns (1, s, heads * 64).
pub struct Attention {
    pub window: Option<usize>, // None: full attention
}

impl candle_core::CustomOp3 for Attention {
    fn name(&self) -> &'static str {
        "acclass-metal-attention"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("acclass-metal-attention is Metal-only")
    }

    fn metal_fwd(
        &self,
        qkv: &MetalStorage,
        lq: &Layout,
        cos: &MetalStorage,
        lc: &Layout,
        sin: &MetalStorage,
        ls: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        #[repr(C)]
        struct Params {
            s: u32,
            heads: u32,
            width: u32,
            window: u32,
        }
        for storage in [qkv, cos, sin] {
            f32_only(storage, "metal attention")?;
        }
        let &[1, s, wide] = lq.dims() else { candle_core::bail!("attention expects (1, s, 3 * heads * 64)") };
        let heads = wide / (3 * D);
        if heads == 0 || wide != 3 * heads * D {
            candle_core::bail!("attention input width {wide} is not 3 * heads * 64")
        }
        if lc.dims() != ls.dims() || lc.dims().len() != 2 || lc.dims()[0] < s || lc.dims()[1] != D {
            candle_core::bail!("attention rope tables {:?} do not cover {s} tokens", lc.dims())
        }
        let window = self.window.map_or(s, |w| w.min(s));
        let width = (2 * window + 1).min(s);
        let params = Params { s: s as u32, heads: heads as u32, width: width as u32, window: window as u32 };
        let device = qkv.device();
        let p = pipelines(device)?;
        let rotated = buffer(device, 2 * heads * s * D, "acclass-rope")?;
        let scores = buffer(device, heads * s * width, "acclass-scores")?;
        let out = buffer(device, s * heads * D, "acclass-attention")?;
        let qkv_at = offset(lq, "metal attention")?;
        let encoder = device.command_encoder()?;
        let e = encoder.as_ref();

        e.set_compute_pipeline_state(&p.rope);
        e.set_input_buffer(0, Some(qkv.buffer()), qkv_at);
        e.set_input_buffer(1, Some(cos.buffer()), offset(lc, "metal attention")?);
        e.set_input_buffer(2, Some(sin.buffer()), offset(ls, "metal attention")?);
        e.set_output_buffer(3, Some(&rotated), 0);
        e.set_bytes(4, &params);
        e.dispatch_threads(grid(heads, s, 2), group(&p.rope));

        e.set_compute_pipeline_state(&p.scores);
        e.set_input_buffer(0, Some(&rotated), 0);
        e.set_output_buffer(1, Some(&scores), 0);
        e.set_bytes(2, &params);
        e.dispatch_threads(grid(width, s, heads), group(&p.scores));

        e.set_compute_pipeline_state(&p.softmax_pv);
        e.set_input_buffer(0, Some(&scores), 0);
        e.set_input_buffer(1, Some(qkv.buffer()), qkv_at);
        e.set_output_buffer(2, Some(&out), 0);
        e.set_bytes(3, &params);
        e.dispatch_threads(grid(heads, s, 1), group(&p.softmax_pv));
        drop(encoder);
        Ok((MetalStorage::new(out, device.clone(), s * heads * D, DType::F32), Shape::from((1, s, heads * D))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::test_ops;
    use candle_core::{Device, Tensor, D as Dim};

    fn metal() -> Option<Device> {
        Device::new_metal(0).ok()
    }

    fn close(got: &Tensor, want: &Tensor, tol: f32, what: &str) -> Result<()> {
        assert_eq!(got.dims(), want.dims(), "{what}");
        let got = got.to_device(&Device::Cpu)?;
        let diff = (got - want)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(diff < tol, "{what}: max diff {diff}");
        Ok(())
    }

    #[test]
    fn norm_and_gelu_gate_match_cpu() -> Result<()> {
        let Some(dev) = metal() else { return Ok(()) };
        let cpu = Device::Cpu;
        // Row width above, equal to and below the threadgroup width.
        for (s, n) in [(1, 768), (37, 768), (70, 768), (5, 32), (5, 9)] {
            let x = (Tensor::randn(0f32, 1.0, (1, s, n), &cpu)? * 4.0)?;
            let w = Tensor::randn(1f32, 0.3, n, &cpu)?;
            let want = test_ops::norm(&x, &w)?;
            let got = test_ops::norm(&x.to_device(&dev)?, &w.to_device(&dev)?)?;
            close(&got, &want, 2e-4, &format!("norm s={s} n={n}"))?;
        }
        for s in [1, 37, 70] {
            let x = (Tensor::randn(0f32, 1.0, (1, s, test_ops::GELU_ROW), &cpu)? * 3.0)?;
            let want = test_ops::gelu_gate(&x)?;
            let got = test_ops::gelu_gate(&x.to_device(&dev)?)?;
            close(&got, &want, 2e-4, &format!("gelu-gate s={s}"))?;
        }
        Ok(())
    }

    #[test]
    fn attention_matches_cpu_rope_and_masked_softmax() -> Result<()> {
        let Some(dev) = metal() else { return Ok(()) };
        let cpu = Device::Cpu;
        let (heads, tokens) = (3, [1, 5, 64, 65, 66, 70, 129, 130, 300]);
        for s in tokens {
            let qkv = Tensor::randn(0f32, 1.0, (1, s, 3 * heads * D), &cpu)?;
            let angle = Tensor::randn(0f32, 3.0, (s + 3, D), &cpu)?;
            let (cos, sin) = (angle.cos()?, angle.sin()?);
            let part = |i: usize| -> Result<Tensor> {
                qkv.narrow(2, i * heads * D, heads * D)?.reshape((1, s, heads, D))?.transpose(1, 2)?.contiguous()
            };
            let rope = |x: &Tensor| -> Result<Tensor> {
                let cos = cos.narrow(0, 0, s)?.reshape((1, 1, s, D))?;
                let sin = sin.narrow(0, 0, s)?.reshape((1, 1, s, D))?;
                let x1 = x.narrow(Dim::Minus1, 0, D / 2)?;
                let x2 = x.narrow(Dim::Minus1, D / 2, D / 2)?;
                let rot = Tensor::cat(&[&x2.neg()?, &x1], Dim::Minus1)?;
                x.broadcast_mul(&cos)?.broadcast_add(&rot.broadcast_mul(&sin)?)
            };
            let (q, k, v) = (rope(&part(0)?)?, rope(&part(1)?)?, part(2)?);
            for window in [None, Some(64)] {
                let mut scores = (q.matmul(&k.transpose(Dim::Minus2, Dim::Minus1)?)? * 0.125)?;
                if let Some(w) = window {
                    let m: Vec<f32> = (0..s * s).map(|x| if (x / s).abs_diff(x % s) > w { -65504.0 } else { 0.0 }).collect();
                    scores = scores.broadcast_add(&Tensor::from_vec(m, (1, 1, s, s), &cpu)?)?;
                }
                let want = candle_nn::ops::softmax_last_dim(&scores)?
                    .matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((1, s, heads * D))?;
                let got = qkv.to_device(&dev)?.apply_op3_no_bwd(&cos.to_device(&dev)?, &sin.to_device(&dev)?, &Attention { window })?;
                close(&got, &want, 5e-4, &format!("s={s} window={window:?}"))?;
            }
        }
        Ok(())
    }
}
