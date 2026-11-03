// ModernBERT-base, nine sigmoid heads, inference on CPU, Metal, or opt-in CUDA (candle).
// CUDA hardware parity tests need no model files: cargo test --features cuda -- --ignored
// Mirrors transformers 5.17 ModernBertForSequenceClassification:
// embeddings -> norm -> 22 encoder layers (rope, sliding/full attention) ->
// final_norm -> mask-aware mean pooling -> head(dense -> gelu -> norm) ->
// classifier. Optional Q8_0 (ggml-style blockwise int8, 32-wide blocks) or
// opt-in small Q5_0 (5-bit, 32-wide blocks) quantization of the large weight
// matrices and embedding at load; norms/classifier stay fp32.

use candle_core::quantized::{gguf_file, GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Linear, Module};

const LAYERS: usize = 22;
const HIDDEN: usize = 768;
const NUM_HEADS: usize = 12;
const HEAD_DIM: usize = 64;
const INTERMEDIATE: usize = 1152;
const WINDOW: i64 = 64; // sliding_window = local_attention(128) / 2, inclusive +-64
pub const MAX_LEN: usize = 2048;
pub const OUTPUT_HEADS: usize = 9;
const NEG: f32 = -65504.0; // matches torch.finfo(float16).min used by HF masks

enum Weight {
    F32(Linear),
    Quant(QMatMul),
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    Packed(crate::qgemm::PackedQ8),
}

impl Weight {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Weight::F32(l) => l.forward(x),
            Weight::Quant(q) => q.forward(x),
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
            Weight::Packed(p) => x.apply_op1_no_bwd(p),
        }
    }

    /// Quantized matrix weight. On aarch64/x86_64 CPUs it is repacked for qgemm. On CUDA
    /// it is dequantized once at load to F16 and
    /// run through cuBLAS: Candle's native CUDA Q8 kernels also quantize the
    /// activations (Q8_1) and drift further from FP32 than CPU Q8. F16 rounding of
    /// the weights (<=2.4e-4 relative) is far below the Q8 step; Candle casts the
    /// activations to F16 for the product and back to F32.
    fn quant(qt: QTensor) -> Result<Self> {
        if qt.device().is_cuda() {
            let device = qt.device();
            return Ok(Weight::Quant(QMatMul::TensorF16(qt.dequantize_f16(&device)?)));
        }
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        if qt.device().is_cpu() {
            if let Some(p) = crate::qgemm::PackedQ8::new(&qt)? {
                return Ok(Weight::Packed(p));
            }
        }
        Ok(Weight::Quant(QMatMul::from_qtensor(qt)?))
    }
}

/// Quantized CUDA models are approximate anyway, so their remaining F32 GEMMs
/// (attention) may use TF32; FP32 models stay exact. Candle's flag is process-wide.
fn set_cuda_tf32(device: &Device, quantized: bool) {
    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        candle_core::cuda_backend::set_gemm_reduced_precision_f32(quantized);
    }
    let _ = (device, quantized);
}

fn make_rope(base: f64, device: &Device) -> Result<(Tensor, Tensor)> {
    // HF: inv_freq = 1/base^(arange(0,dim,2)/dim); emb = cat(freqs, freqs)
    let inv_freq: Vec<f32> = (0..HEAD_DIM)
        .step_by(2)
        .map(|i| 1.0 / base.powf(i as f64 / HEAD_DIM as f64) as f32)
        .collect();
    let mut freqs = Vec::with_capacity(MAX_LEN * HEAD_DIM);
    for pos in 0..MAX_LEN {
        for j in 0..32 {
            freqs.push(pos as f32 * inv_freq[j]);
        }
        for j in 0..32 {
            freqs.push(pos as f32 * inv_freq[j]);
        }
    }
    let freqs = Tensor::from_vec(freqs, (MAX_LEN, HEAD_DIM), device)?;
    Ok((freqs.cos()?, freqs.sin()?))
}

fn sliding_mask(s: usize, device: &Device) -> Result<Tensor> {
    // additive mask (s, s): 0 where |i-j| <= 64, NEG elsewhere
    let mut m = vec![0.0f32; s * s];
    for i in 0..s {
        for j in 0..s {
            if (i as i64 - j as i64).abs() > WINDOW {
                m[i * s + j] = NEG;
            }
        }
    }
    Ok(Tensor::from_vec(m, (s, s), device)?)
}

// CUDA only: build the same mask on the device instead of copying s*s floats.
fn sliding_mask_on_device(s: usize, device: &Device) -> Result<Tensor> {
    let idx = Tensor::arange(0u32, s as u32, device)?.to_dtype(DType::F32)?;
    let dist = idx.unsqueeze(1)?.broadcast_sub(&idx.unsqueeze(0)?)?.abs()?;
    dist.gt(WINDOW as f64)?.to_dtype(DType::F32)? * NEG as f64
}

// CUDA only: sliding-window attention in query blocks against only the keys
// within +-64. Every skipped key would get mask NEG and an exactly-zero softmax
// weight, so this matches the dense masked path up to summation order but does
// ~s*384 instead of s*s work. Returns (1, H, s, 64). `q` is pre-scaled.
const QBLOCK: usize = 256;

fn local_blocks(s: usize, device: &Device) -> Result<Vec<(usize, usize, usize, usize, Tensor)>> {
    let w = WINDOW as usize;
    let mut blocks = Vec::new();
    for q0 in (0..s).step_by(QBLOCK) {
        let qn = QBLOCK.min(s - q0);
        let k0 = q0.saturating_sub(w);
        let kn = (q0 + qn + w).min(s) - k0;
        let qi = Tensor::arange(q0 as u32, (q0 + qn) as u32, device)?.to_dtype(DType::F32)?;
        let kj = Tensor::arange(k0 as u32, (k0 + kn) as u32, device)?.to_dtype(DType::F32)?;
        let dist = qi.unsqueeze(1)?.broadcast_sub(&kj.unsqueeze(0)?)?.abs()?;
        let mask = (dist.gt(WINDOW as f64)?.to_dtype(DType::F32)? * NEG as f64)?.reshape((1, 1, qn, kn))?;
        blocks.push((q0, qn, k0, kn, mask));
    }
    Ok(blocks)
}

fn local_attention(q: &Tensor, k: &Tensor, v: &Tensor, blocks: &[(usize, usize, usize, usize, Tensor)]) -> Result<Tensor> {
    let mut out = Vec::with_capacity(blocks.len());
    for (q0, qn, k0, kn, mask) in blocks {
        let qb = q.narrow(2, *q0, *qn)?;
        let kb = k.narrow(2, *k0, *kn)?;
        let vb = v.narrow(2, *k0, *kn)?;
        // Row-narrowed views keep unit inner strides, so cuBLAS needs no copies.
        let scores = qb.matmul(&kb.t()?)?.broadcast_add(mask)?;
        out.push(candle_nn::ops::softmax_last_dim(&scores)?.matmul(&vb)?);
    }
    Tensor::cat(&out, 2)
}

fn gelu(x: &Tensor) -> Result<Tensor> {
    // HF ACT2FN["gelu"] uses torch's erf form, not Candle's tanh approximation.
    x.gelu_erf()
}

#[cfg(test)]
fn norm_nobias(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    // LayerNorm without bias; f32 accumulation like torch
    let x32 = x.to_dtype(DType::F32)?;
    let mu = x32.mean(D::Minus1)?.unsqueeze(D::Minus1)?;
    let centered = x32.broadcast_sub(&mu)?;
    let var = centered.sqr()?.mean(D::Minus1)?.unsqueeze(D::Minus1)?;
    let out = centered.broadcast_div(&(var + eps)?.sqrt()?)?;
    out.broadcast_mul(&w.to_dtype(DType::F32)?)
}

#[cfg(test)]
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, s: usize) -> Result<Tensor> {
    // x: (1, H, s, 64); cos/sin: (MAX_LEN, 64) sliced to (1, 1, s, 64)
    let cos = cos.narrow(0, 0, s)?.reshape((1, 1, s, HEAD_DIM))?;
    let sin = sin.narrow(0, 0, s)?.reshape((1, 1, s, HEAD_DIM))?;
    let x32 = x.to_dtype(DType::F32)?;
    let x1 = x32.narrow(D::Minus1, 0, HEAD_DIM / 2)?;
    let x2 = x32.narrow(D::Minus1, HEAD_DIM / 2, HEAD_DIM / 2)?;
    let rot = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    x32.broadcast_mul(&cos)?.broadcast_add(&rot.broadcast_mul(&sin)?)
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn attend_fused(q: &Tensor, k: &Tensor, v: &Tensor, sliding: bool) -> Result<Tensor> {
    let window = sliding.then_some(WINDOW as usize);
    q.apply_op3_no_bwd(k, v, &crate::attention::Attention { window })
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn attend_fused(_: &Tensor, _: &Tensor, _: &Tensor, _: bool) -> Result<Tensor> {
    unreachable!("fused attention is aarch64/x86_64-only")
}

fn fused_attention_supported() -> bool {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    return crate::attention::supported();
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    false
}

#[cfg(target_os = "macos")]
fn attend_metal(qkv: &Tensor, rope: &(Tensor, Tensor), sliding: bool) -> Result<Tensor> {
    let window = sliding.then_some(WINDOW as usize);
    qkv.apply_op3_no_bwd(&rope.0, &rope.1, &crate::metal::Attention { window })
}

#[cfg(not(target_os = "macos"))]
fn attend_metal(_: &Tensor, _: &(Tensor, Tensor), _: bool) -> Result<Tensor> {
    unreachable!("Metal attention is macOS-only")
}

/// CPU and Metal op: norm_nobias(x, w, 1e-5) over the last dim, multithreaded over rows.
/// Bit-identical to the reference: same candle sum kernel and scalar ops in the
/// same order (mean = sum * (1/n), var = mean of squares of x - mean).
struct NormRows;

impl candle_core::CustomOp2 for NormRows {
    fn name(&self) -> &'static str {
        "acclass-norm-rows"
    }

    fn cpu_fwd(
        &self,
        xs: &candle_core::CpuStorage,
        xl: &candle_core::Layout,
        ws: &candle_core::CpuStorage,
        wl: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        use candle_core::cpu::kernels::VecOps;
        use candle_core::CpuStorage::F32;
        use rayon::prelude::*;
        let (F32(x), Some((a, b)), F32(w), Some((c, d))) = (xs, xl.contiguous_offsets(), ws, wl.contiguous_offsets()) else {
            candle_core::bail!("norm expects contiguous f32 inputs")
        };
        let (x, w) = (&x[a..b], &w[c..d]);
        let n = w.len();
        if xl.dims().last() != Some(&n) {
            candle_core::bail!("norm input {:?} does not end in {n}", xl.dims())
        }
        let scale = (1f64 / n as f64) as f32;
        let eps = 1e-5f64 as f32;
        let mut out = vec![0f32; x.len()];
        out.par_chunks_mut(n).zip(x.par_chunks(n)).with_min_len(8).for_each(|(o, x)| {
            let sum = |v: &[f32]| {
                let mut s = 0f32;
                unsafe { f32::vec_reduce_sum(v.as_ptr(), &mut s, v.len()) };
                s
            };
            let mu = sum(x) * scale + 0.0;
            for (o, &x) in o.iter_mut().zip(x) {
                *o = x - mu;
            }
            let sq: Vec<f32> = o.iter().map(|&c| c * c).collect();
            let den = ((sum(&sq) * scale + 0.0) * 1.0 + eps).sqrt();
            for (o, &w) in o.iter_mut().zip(w) {
                *o = *o / den * w;
            }
        });
        Ok((F32(out), xl.shape().clone()))
    }

    #[cfg(target_os = "macos")]
    fn metal_fwd(
        &self,
        xs: &candle_core::MetalStorage,
        xl: &candle_core::Layout,
        ws: &candle_core::MetalStorage,
        wl: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        crate::metal::norm(xs, xl, ws, wl)
    }
}

/// CPU and Metal op: gate * gelu_erf(inp) for (.., 2 * INTERMEDIATE) rows = [inp | gate].
/// The CPU path is multithreaded over rows and bit-identical to candle's gelu_erf +
/// broadcast_mul, which run single-threaded.
struct GeluGate;

impl candle_core::CustomOp1 for GeluGate {
    fn name(&self) -> &'static str {
        "acclass-gelu-gate"
    }

    fn cpu_fwd(&self, storage: &candle_core::CpuStorage, layout: &candle_core::Layout) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        use candle_core::op::{GeluErf, UnaryOpT};
        use rayon::prelude::*;
        let (candle_core::CpuStorage::F32(src), Some((a, b))) = (storage, layout.contiguous_offsets()) else {
            candle_core::bail!("gelu-gate expects contiguous f32 input")
        };
        let mut dims = layout.dims().to_vec();
        if dims.last() != Some(&(2 * INTERMEDIATE)) {
            candle_core::bail!("gelu-gate input {dims:?} does not end in {}", 2 * INTERMEDIATE)
        }
        let x = &src[a..b];
        let mut out = vec![0f32; x.len() / 2];
        out.par_chunks_mut(INTERMEDIATE).zip(x.par_chunks(2 * INTERMEDIATE)).with_min_len(8).for_each(|(o, x)| {
            let (inp, gate) = x.split_at(INTERMEDIATE);
            for ((o, &i), &g) in o.iter_mut().zip(inp).zip(gate) {
                *o = g * GeluErf::f32(i);
            }
        });
        *dims.last_mut().unwrap() = INTERMEDIATE;
        Ok((candle_core::CpuStorage::F32(out), candle_core::Shape::from(dims)))
    }

    #[cfg(target_os = "macos")]
    fn metal_fwd(
        &self,
        storage: &candle_core::MetalStorage,
        layout: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        crate::metal::gelu_gate(storage, layout, INTERMEDIATE)
    }
}

struct Layer {
    attn_norm: Option<Tensor>,
    wqkv: Weight,
    wo: Weight,
    mlp_norm: Tensor,
    wi: Weight,
    wo2: Weight,
    sliding: bool,
}

pub struct ModernBert9Head {
    tok_emb: Tensor, // (50368, 768) fp32 lookup table (dequantized once if quantized)
    embed_norm: Tensor,
    layers: Vec<Layer>,
    final_norm: Tensor,
    head_dense: Weight,
    head_norm: Tensor,
    classifier: Linear,
    rope: [(Tensor, Tensor); 2], // [full_attention(160000), sliding_attention(10000)]
    device: Device,
}

// layer types from config.json: full at idx 0,3,6,...; sliding elsewhere
const LAYER_IS_SLIDING: [bool; LAYERS] = {
    let mut a = [false; LAYERS];
    let mut i = 0;
    while i < LAYERS {
        a[i] = i % 3 != 0;
        i += 1;
    }
    a
};

impl ModernBert9Head {
    /// `quant`: None loads FP32; Some(Q8_0|Q5_0) quantizes at load like `convert`.
    pub fn load(dir: &std::path::Path, device: Device, quant: Option<GgmlDType>) -> Result<Self> {
        set_cuda_tf32(&device, quant.is_some());
        let path = dir.join("model.safetensors");
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device)?
        };

        let get_lin_in = |name: &str, out_dim: usize, in_dim: usize| -> Result<Weight> {
            let w = vb.get((out_dim, in_dim), &format!("{name}.weight"))?;
            match quant {
                Some(dtype) => Weight::quant(QTensor::quantize(&w, dtype)?),
                None => Ok(Weight::F32(Linear::new(w, None))),
            }
        };
        let get_lin = |name: &str, out_dim: usize| get_lin_in(name, out_dim, HIDDEN);

        // Keep the in-RAM quantized path bit-identical to the GGUF artifact: both
        // store the embedding quantized, so quantize+dequantize here too. The
        // forward needs a dense embedding table for index_select.
        let tok_emb = vb.get((50368, HIDDEN), "model.embeddings.tok_embeddings.weight")?;
        let tok_emb = match quant {
            Some(dtype) => QTensor::quantize(&tok_emb, dtype)?.dequantize(&device)?,
            None => tok_emb,
        };
        let embed_norm = vb.get((HIDDEN,), "model.embeddings.norm.weight")?;
        let final_norm = vb.get((HIDDEN,), "model.final_norm.weight")?;

        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("model.layers.{i}");
            let attn_norm = if i == 0 {
                None
            } else {
                Some(vb.get((HIDDEN,), &format!("{p}.attn_norm.weight"))?)
            };
            layers.push(Layer {
                attn_norm,
                wqkv: get_lin(&format!("{p}.attn.Wqkv"), 3 * HIDDEN)?,
                wo: get_lin(&format!("{p}.attn.Wo"), HIDDEN)?,
                mlp_norm: vb.get((HIDDEN,), &format!("{p}.mlp_norm.weight"))?,
                wi: get_lin(&format!("{p}.mlp.Wi"), 2 * INTERMEDIATE)?,
                wo2: get_lin_in(&format!("{p}.mlp.Wo"), HIDDEN, INTERMEDIATE)?,
                sliding: LAYER_IS_SLIDING[i],
            });
        }

        let head_dense = get_lin("head.dense", HIDDEN)?;
        let head_norm = vb.get((HIDDEN,), "head.norm.weight")?;
        let classifier = Linear::new(
            vb.get((OUTPUT_HEADS, HIDDEN), "classifier.weight")?,
            Some(vb.get(OUTPUT_HEADS, "classifier.bias")?),
        );

        let rope = [
            make_rope(160_000.0, &device)?,
            make_rope(10_000.0, &device)?,
        ];
        Ok(Self {
            tok_emb,
            embed_norm,
            layers,
            final_norm,
            head_dense,
            head_norm,
            classifier,
            rope,
            device,
        })
    }

    /// Load the pre-quantized GGUF written by `acclass convert`: the large matrices
    /// and embedding are Q8_0 (or Q5_0), norms/classifier are F32. No in-RAM quantization.
    pub fn load_gguf(path: &std::path::Path, device: Device) -> Result<Self> {
        set_cuda_tf32(&device, true);
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut get = |name: &str| content.tensor(&mut file, name, &device);

        let to_lin = Weight::quant;
        let to_vec = |qt: QTensor| -> Result<Tensor> { qt.dequantize(&device) };

        let tok_emb = to_vec(get("model.embeddings.tok_embeddings.weight")?)?;
        let embed_norm = to_vec(get("model.embeddings.norm.weight")?)?;
        let final_norm = to_vec(get("model.final_norm.weight")?)?;

        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("model.layers.{i}");
            let attn_norm = if i == 0 {
                None
            } else {
                Some(to_vec(get(&format!("{p}.attn_norm.weight"))?)?)
            };
            layers.push(Layer {
                attn_norm,
                wqkv: to_lin(get(&format!("{p}.attn.Wqkv.weight"))?)?,
                wo: to_lin(get(&format!("{p}.attn.Wo.weight"))?)?,
                mlp_norm: to_vec(get(&format!("{p}.mlp_norm.weight"))?)?,
                wi: to_lin(get(&format!("{p}.mlp.Wi.weight"))?)?,
                wo2: to_lin(get(&format!("{p}.mlp.Wo.weight"))?)?,
                sliding: LAYER_IS_SLIDING[i],
            });
        }

        let head_dense = to_lin(get("head.dense.weight")?)?;
        let head_norm = to_vec(get("head.norm.weight")?)?;
        let classifier_weight = to_vec(get("classifier.weight")?)?;
        let classifier_bias = to_vec(get("classifier.bias")?)?;
        if classifier_weight.dims() != [OUTPUT_HEADS, HIDDEN] || classifier_bias.dims() != [OUTPUT_HEADS] {
            candle_core::bail!("GGUF classifier must have exactly nine output heads");
        }
        let classifier = Linear::new(classifier_weight, Some(classifier_bias));

        let rope = [make_rope(160_000.0, &device)?, make_rope(10_000.0, &device)?];
        Ok(Self {
            tok_emb,
            embed_norm,
            layers,
            final_norm,
            head_dense,
            head_norm,
            classifier,
            rope,
            device,
        })
    }

    pub fn forward(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let s = ids.len();
        if s == 0 || s > MAX_LEN {
            candle_core::bail!("input must contain 1..={MAX_LEN} inclusive tokens; clipping is forbidden");
        }
        let dev = &self.device;
        let ids_t = Tensor::from_vec(ids.to_vec(), (1, s), dev)?;
        let h = self.tok_emb.index_select(&ids_t.flatten_all()?, 0)?;
        let h = h.reshape((1, s, HIDDEN))?;
        // CUDA uses fused LayerNorm/RoPE kernels (fewer launches). CPU uses
        // multithreaded ops bit-identical to the reference ops (NormRows, candle's
        // rope kernel). Metal runs NormRows, GeluGate and attention as fused kernels
        // (src/metal.rs). All compute in F32.
        let cuda = dev.is_cuda();
        let cpu = dev.is_cpu();
        let metal = dev.is_metal();
        // aarch64/x86_64 CPU: fused attention (src/attention.rs) needs no mask tensor.
        let fused = cpu && fused_attention_supported();
        let zero_bias = if cuda { Some(Tensor::zeros(HIDDEN, DType::F32, dev)?) } else { None };
        let norm = |x: &Tensor, w: &Tensor| -> Result<Tensor> {
            match &zero_bias {
                Some(b) => candle_nn::ops::layer_norm(&x.contiguous()?, w, b, 1e-5),
                None => x.contiguous()?.apply_op2_no_bwd(w, &NormRows),
            }
        };
        let half_rope = |t: &Tensor| t.narrow(0, 0, s)?.narrow(1, 0, HEAD_DIM / 2)?.contiguous();
        let half = if metal {
            None
        } else {
            Some([
                (half_rope(&self.rope[0].0)?, half_rope(&self.rope[0].1)?),
                (half_rope(&self.rope[1].0)?, half_rope(&self.rope[1].1)?),
            ])
        };
        let rope = |x: &Tensor, sliding: bool| -> Result<Tensor> {
            let h = half.as_ref().expect("Metal attention applies RoPE itself");
            let i = usize::from(sliding);
            candle_nn::rotary_emb::rope(x, &h[i].0, &h[i].1)
        };
        let mut h = norm(&h, &self.embed_norm)?;
        // Build the sliding-window mask once per forward, not once per sliding layer.
        let blocks = if cuda && s > 2 * QBLOCK { Some(local_blocks(s, dev)?) } else { None };
        let mask = if fused || metal || s <= WINDOW as usize + 1 || blocks.is_some() {
            None // fused, all zeros (every |i-j| <= 64), or CUDA local blocks carry their own
        } else if cuda {
            Some(sliding_mask_on_device(s, dev)?.reshape((1, 1, s, s))?)
        } else {
            Some(sliding_mask(s, dev)?.reshape((1, 1, s, s))?)
        };

        for layer in &self.layers {
            let attn_in = match &layer.attn_norm {
                Some(n) => norm(&h, n)?,
                None => h.clone(),
            };

            let qkv = layer.wqkv.forward(&attn_in)?; // (1, s, 2304)
            if metal {
                let attn = attend_metal(&qkv, &self.rope[usize::from(layer.sliding)], layer.sliding)?;
                h = h.broadcast_add(&layer.wo.forward(&attn)?)?;
                let h2 = layer.wi.forward(&norm(&h, &layer.mlp_norm)?)?;
                let mlp = layer.wo2.forward(&h2.apply_op1_no_bwd(&GeluGate)?)?;
                h = h.broadcast_add(&mlp)?;
                continue;
            }
            let q = qkv.narrow(D::Minus1, 0, HIDDEN)?.contiguous()?;
            let k = qkv.narrow(D::Minus1, HIDDEN, HIDDEN)?.contiguous()?;
            let v = qkv.narrow(D::Minus1, 2 * HIDDEN, HIDDEN)?.contiguous()?;
            let to4d = |t: &Tensor| -> Result<Tensor> {
                t.reshape((1, s, NUM_HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()
            };
            let (q, k, v) = (to4d(&q)?, to4d(&k)?, to4d(&v)?);

            let q = rope(&q, layer.sliding)?;
            let k = rope(&k, layer.sliding)?;

            let attn = if fused {
                attend_fused(&q, &k, &v, layer.sliding)?
            } else {
            let attn = match (&blocks, layer.sliding) {
                // Scaling q by 0.125 (a power of two) first is exact and saves an s*s pass.
                (Some(b), true) => local_attention(&(q * 0.125f64)?, &k, &v, b)?,
                _ => {
                    let scores = if cuda {
                        (q * 0.125f64)?.matmul(&k.t()?)?
                    } else {
                        (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * 0.125f64)?
                    };
                    let scores = match (&mask, layer.sliding) {
                        (Some(m), true) => scores.broadcast_add(m)?,
                        _ => scores,
                    };
                    candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)? // (1, H, s, 64)
                }
            };
            let attn = attn
                .transpose(1, 2)?
                .contiguous()?
                .reshape((1, s, HIDDEN))?;
            attn
            };
            let attn = layer.wo.forward(&attn)?;

            h = h.broadcast_add(&attn)?;
            let mlp_in = norm(&h, &layer.mlp_norm)?;
            let h2 = layer.wi.forward(&mlp_in)?; // (1, s, 2304)
            let out = if dev.is_cpu() {
                h2.apply_op1_no_bwd(&GeluGate)?
            } else {
                let inp = h2.narrow(D::Minus1, 0, INTERMEDIATE)?;
                let gate = h2.narrow(D::Minus1, INTERMEDIATE, INTERMEDIATE)?;
                gate.broadcast_mul(&gelu(&inp)?)?
            };
            let mlp = layer.wo2.forward(&out)?;
            h = h.broadcast_add(&mlp)?;
        }

        h = norm(&h, &self.final_norm)?;
        // classifier_pooling = mean, batch 1 unpadded -> mean over tokens
        let pooled = h.mean(D::Minus2)?; // (1, 768)
        let pooled = self.head_dense.forward(&pooled)?;
        let pooled = gelu(&pooled)?;
        let pooled = norm(&pooled, &self.head_norm)?;
        let logits = self.classifier.forward(&pooled)?; // (1, 9)
        let logits = logits.squeeze(0)?.to_vec1::<f32>()?;
        Ok(logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect())
    }
}

#[cfg(all(test, target_os = "macos"))]
pub mod test_ops {
    use super::*;

    pub const GELU_ROW: usize = 2 * INTERMEDIATE;

    pub fn norm(x: &Tensor, w: &Tensor) -> Result<Tensor> {
        x.apply_op2_no_bwd(w, &NormRows)
    }

    pub fn gelu_gate(x: &Tensor) -> Result<Tensor> {
        x.apply_op1_no_bwd(&GeluGate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_matches_transformers_erf_activation_not_tanh_approximation() -> Result<()> {
        let input = Tensor::new(&[-3.0f32, -1.0, 0.0, 1.0, 3.0], &Device::Cpu)?;
        let output = gelu(&input)?.to_vec1::<f32>()?;
        // x * 0.5 * (1 + erf(x / sqrt(2))), rounded to f32.
        let expected = [-0.004049694f32, -0.15865526, 0.0, 0.8413447, 2.9959502];
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "GELU mismatch: {actual} != {expected}");
        }
        assert!((output[0] - input.gelu()?.to_vec1::<f32>()?[0]).abs() > 1e-4);
        Ok(())
    }

    #[test]
    fn cpu_norm_and_rope_are_bit_identical_to_reference_ops() -> Result<()> {
        let bits = |t: &Tensor| t.flatten_all()?.to_vec1::<f32>().map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        for s in [1, 37] {
            let h = (Tensor::randn(0f32, 1.0, (1, s, HIDDEN), &Device::Cpu)? * 4.0)?;
            let w = Tensor::randn(1f32, 0.3, HIDDEN, &Device::Cpu)?;
            assert_eq!(bits(&h.apply_op2_no_bwd(&w, &NormRows)?)?, bits(&norm_nobias(&h, &w, 1e-5)?)?);
            let q = Tensor::randn(0f32, 1.0, (1, NUM_HEADS, s, HEAD_DIM), &Device::Cpu)?;
            let (cos, sin) = make_rope(10_000.0, &Device::Cpu)?;
            let half = |t: &Tensor| t.narrow(0, 0, s)?.narrow(1, 0, HEAD_DIM / 2)?.contiguous();
            let fast = candle_nn::rotary_emb::rope(&q, &half(&cos)?, &half(&sin)?)?;
            assert_eq!(bits(&fast)?, bits(&apply_rope(&q, &cos, &sin, s)?)?);
        }
        Ok(())
    }

    #[test]
    fn gelu_gate_is_bit_identical_to_candle_ops() -> Result<()> {
        let h2 = (Tensor::randn(0f32, 1.0, (1, 37, 2 * INTERMEDIATE), &Device::Cpu)? * 3.0)?;
        let inp = h2.narrow(D::Minus1, 0, INTERMEDIATE)?;
        let gate = h2.narrow(D::Minus1, INTERMEDIATE, INTERMEDIATE)?;
        let want = gate.broadcast_mul(&gelu(&inp)?)?.flatten_all()?.to_vec1::<f32>()?;
        let got = h2.apply_op1_no_bwd(&GeluGate)?;
        assert_eq!(got.dims(), &[1, 37, INTERMEDIATE]);
        let got = got.flatten_all()?.to_vec1::<f32>()?;
        assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()));
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;

    // Synthetic, bounded, reproducible on both devices; no RNG state/model files.
    fn synthetic(
        shape: impl Into<candle_core::Shape>,
        mut seed: u32,
        bound: f32,
        device: &Device,
    ) -> Result<Tensor> {
        let shape = shape.into();
        let values = (0..shape.elem_count())
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((seed >> 8) as f32 / 8_388_608.0 - 1.0) * bound
            })
            .collect();
        Tensor::from_vec(values, shape, device)
    }

    // Return normalized RMSE as well as checking every element, including zeros.
    fn assert_close(
        name: &str,
        cpu: &Tensor,
        cuda: &Tensor,
        atol: f64,
        rtol: f64,
    ) -> Result<f64> {
        assert_eq!(cpu.dims(), cuda.dims(), "{name}: shape mismatch");
        let expected = cpu.flatten_all()?.to_vec1::<f32>()?;
        let actual = cuda.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(!expected.is_empty(), "{name}: empty comparison");
        let (mut max_abs, mut error_sq, mut reference_sq, mut worst_ratio) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (i, (&a, &b)) in expected.iter().zip(&actual).enumerate() {
            assert!(a.is_finite() && b.is_finite(), "{name}[{i}]: non-finite {a}, {b}");
            let error = (a as f64 - b as f64).abs();
            max_abs = max_abs.max(error);
            error_sq += error * error;
            reference_sq += (a as f64).powi(2);
            worst_ratio = worst_ratio.max(error / (atol + rtol * (a as f64).abs()));
        }
        let rmse = (error_sq / expected.len() as f64).sqrt();
        let nrmse = (error_sq / reference_sq.max(1e-24)).sqrt();
        eprintln!("{name}: max_abs={max_abs:.6e}, rmse={rmse:.6e}, nrmse={nrmse:.6e}");
        assert!(
            worst_ratio <= 1.0,
            "{name}: max_abs={max_abs:.6e}, rmse={rmse:.6e}, tolerance={atol} + {rtol}*abs(reference), worst_ratio={worst_ratio:.3}"
        );
        Ok(nrmse)
    }

    fn dense(input: usize, output: usize, seed: u32, device: &Device) -> Result<Weight> {
        let weight = synthetic((output, input), seed, 1.0 / (input as f32).sqrt(), device)?;
        Ok(Weight::F32(Linear::new(weight, None)))
    }

    fn norm(seed: u32, device: &Device) -> Result<Tensor> {
        synthetic(HIDDEN, seed, 0.1, device)? + 1.0
    }

    fn synthetic_model(device: &Device) -> Result<ModernBert9Head> {
        // Two real-width layers cover both attention types and both attn_norm
        // branches without paying for 22 layers or loading any model artifacts.
        let mut layers = Vec::new();
        for i in 0..2 {
            let seed = 10 + i as u32 * 10;
            layers.push(Layer {
                attn_norm: if i == 0 { None } else { Some(norm(seed, device)?) },
                wqkv: dense(HIDDEN, 3 * HIDDEN, seed + 1, device)?,
                wo: dense(HIDDEN, HIDDEN, seed + 2, device)?,
                mlp_norm: norm(seed + 3, device)?,
                wi: dense(HIDDEN, 2 * INTERMEDIATE, seed + 4, device)?,
                wo2: dense(INTERMEDIATE, HIDDEN, seed + 5, device)?,
                sliding: i != 0,
            });
        }
        Ok(ModernBert9Head {
            tok_emb: synthetic((19, HIDDEN), 1, 1.0, device)?,
            embed_norm: norm(2, device)?,
            layers,
            final_norm: norm(3, device)?,
            head_dense: dense(HIDDEN, HIDDEN, 4, device)?,
            head_norm: norm(5, device)?,
            classifier: Linear::new(
                synthetic((OUTPUT_HEADS, HIDDEN), 6, 1.0 / (HIDDEN as f32).sqrt(), device)?,
                Some(synthetic(OUTPUT_HEADS, 7, 0.1, device)?),
            ),
            rope: [make_rope(160_000.0, device)?, make_rope(10_000.0, device)?],
            device: device.clone(),
        })
    }

    // Mirror forward's exact operations/layouts and retain checkpoints so that
    // a small final sigmoid difference cannot hide a broken intermediate op.
    // The test also compares this trace to the actual production forward below.
    fn f32_stages(model: &ModernBert9Head, ids: &[u32]) -> Result<Vec<(String, Tensor)>> {
        let s = ids.len();
        let dev = &model.device;
        let mut stages = Vec::new();
        let mut record = |name: String, x: &Tensor| stages.push((name, x.clone()));
        let ids = Tensor::from_vec(ids.to_vec(), (1, s), dev)?;
        let mut h = model.tok_emb.index_select(&ids.flatten_all()?, 0)?.reshape((1, s, HIDDEN))?;
        record("embedding/index_select".into(), &h);
        h = norm_nobias(&h, &model.embed_norm, 1e-5)?;
        record("embedding/norm_nobias".into(), &h);
        for (i, layer) in model.layers.iter().enumerate() {
            let attn_in = match &layer.attn_norm {
                Some(n) => norm_nobias(&h, n, 1e-5)?,
                None => h.clone(),
            };
            let qkv = layer.wqkv.forward(&attn_in)?;
            record(format!("layer{i}/qkv"), &qkv);
            let to4d = |offset| -> Result<Tensor> {
                qkv.narrow(D::Minus1, offset, HIDDEN)?.contiguous()?
                    .reshape((1, s, NUM_HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()
            };
            let (q, k, v) = (to4d(0)?, to4d(HIDDEN)?, to4d(2 * HIDDEN)?);
            let rope = &model.rope[usize::from(layer.sliding)];
            let q = apply_rope(&q, &rope.0, &rope.1, s)?;
            let k = apply_rope(&k, &rope.0, &rope.1, s)?;
            record(format!("layer{i}/rope_q"), &q);
            record(format!("layer{i}/rope_k"), &k);
            let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * 0.125f64)?;
            // Compare before masking too: NEG would make a relative comparison
            // of masked scores uninformative (F32 checks use absolute tolerance).
            record(format!("layer{i}/dense_scores"), &scores);
            let scores = if layer.sliding {
                let mask = sliding_mask(s, dev)?;
                record(format!("layer{i}/sliding_mask"), &mask);
                scores.broadcast_add(&mask.reshape((1, 1, s, s))?)?
            } else {
                scores
            };
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            record(format!("layer{i}/attention_probs"), &probs);
            let attn = probs.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((1, s, HIDDEN))?;
            record(format!("layer{i}/dense_attention"), &attn);
            h = h.broadcast_add(&layer.wo.forward(&attn)?)?;
            let mlp_in = norm_nobias(&h, &layer.mlp_norm, 1e-5)?;
            record(format!("layer{i}/mlp_norm"), &mlp_in);
            let h2 = layer.wi.forward(&mlp_in)?;
            let inp = h2.narrow(D::Minus1, 0, INTERMEDIATE)?;
            let gate = h2.narrow(D::Minus1, INTERMEDIATE, INTERMEDIATE)?;
            // Deliberately retain the same non-contiguous view used by forward.
            let activated = gelu(&inp)?;
            record(format!("layer{i}/gelu_erf"), &activated);
            let out = gate.broadcast_mul(&activated)?;
            h = h.broadcast_add(&layer.wo2.forward(&out)?)?;
            record(format!("layer{i}/residual"), &h);
        }
        h = norm_nobias(&h, &model.final_norm, 1e-5)?;
        record("final_norm".into(), &h);
        let pooled = h.mean(D::Minus2)?;
        record("mean_pooling".into(), &pooled);
        let pooled = gelu(&model.head_dense.forward(&pooled)?)?;
        record("head/gelu_erf".into(), &pooled);
        let pooled = norm_nobias(&pooled, &model.head_norm, 1e-5)?;
        record("head/norm_nobias".into(), &pooled);
        let logits = model.classifier.forward(&pooled)?;
        record("classifier/logits".into(), &logits);
        Ok(stages)
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU and CUDA; synthetic tensors only"]
    fn cuda_f32_model_chain_matches_cpu() -> Result<()> {
        // Failing to create CUDA is an error, never a silent CPU fallback/skip.
        let cuda = Device::new_cuda(0)?;
        let cpu_model = synthetic_model(&Device::Cpu)?;
        let cuda_model = synthetic_model(&cuda)?;
        // 129 tokens exercises both sides of the inclusive +/-64 mask boundary;
        // 1 and 9 also cover degenerate and multi-token layouts/pooling; 589
        // takes CUDA's blocked local attention with a partial last block.
        for s in [1, 9, 2 * WINDOW as usize + 1, 2 * QBLOCK + 77] {
            let ids: Vec<u32> = (0..s).map(|i| ((i * 7) % 19) as u32).collect();
            let cpu_stages = f32_stages(&cpu_model, &ids)?;
            let cuda_stages = f32_stages(&cuda_model, &ids)?;
            assert_eq!(cpu_stages.len(), cuda_stages.len());
            for ((name, cpu), (cuda_name, gpu)) in cpu_stages.iter().zip(&cuda_stages) {
                assert_eq!(name, cuda_name);
                // F32 arithmetic/reduction order may differ, but no activation
                // quantization occurs here: a strict absolute 1e-4 is appropriate.
                assert_close(&format!("tokens={s}/{name}"), cpu, gpu, 1e-4, 0.0)?;
            }
            let cpu = Tensor::new(cpu_model.forward(&ids)?, &Device::Cpu)?;
            let gpu = Tensor::new(cuda_model.forward(&ids)?, &Device::Cpu)?;
            assert_close(&format!("tokens={s}/production_forward"), &cpu, &gpu, 1e-4, 0.0)?;
            for (model, stages, output) in [
                (&cpu_model, &cpu_stages, &cpu),
                (&cuda_model, &cuda_stages, &gpu),
            ] {
                let logits = stages.last().unwrap().1.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
                let traced: Vec<f32> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
                assert_close(
                    &format!("tokens={s}/{:?}/trace_vs_forward", model.device),
                    &Tensor::new(traced, &Device::Cpu)?, output, 1e-6, 0.0,
                )?;
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU and CUDA; synthetic Q8_0 tensors only"]
    fn cuda_q8_matmul_matches_cpu_at_kernel_boundaries() -> Result<()> {
        let cuda = Device::new_cuda(0)?;
        // These are the actual model projection shapes, including the 1152-wide
        // MLP output input (not a multiple of CUDA's 512-element activation pad).
        for (input, output) in [
            (HIDDEN, 3 * HIDDEN),
            (HIDDEN, HIDDEN),
            (HIDDEN, 2 * INTERMEDIATE),
            (INTERMEDIATE, HIDDEN),
        ] {
            let weights = synthetic(
                (output, input), 101, 1.0 / (input as f32).sqrt(), &Device::Cpu,
            )?;
            // Quantize the identical CPU weights onto each device, as a GGUF
            // load would: compare kernels, not two different weight encodings.
            let cpu_qt = QTensor::quantize(&weights, GgmlDType::Q8_0)?;
            let reference = Linear::new(cpu_qt.dequantize(&Device::Cpu)?, None);
            let cpu_q = QMatMul::from_qtensor(cpu_qt)?;
            let cuda_q = QMatMul::from_qtensor(QTensor::quantize_onto(
                &weights, GgmlDType::Q8_0, &cuda,
            )?)?;
            assert!(matches!(&cpu_q, QMatMul::QTensor(_)) && matches!(&cuda_q, QMatMul::QTensor(_)),
                "Q8 test requires native QMatMul; unset CANDLE_DEQUANTIZE_ALL[_F16]");
            let cpu_q = Weight::Quant(cpu_q);
            let cuda_q = Weight::Quant(cuda_q);
            // The production CUDA path: dequantized F16 weights through cuBLAS.
            let cuda_f16 = Weight::quant(QTensor::quantize_onto(&weights, GgmlDType::Q8_0, &cuda)?)?;
            // Candle dispatches <=8 tokens to MMVQ and >8 to MMQ. Keep batch=1,
            // rank=3 and F32 activations exactly as ModernBert9Head::forward.
            for tokens in [1, 8, 9] {
                let x = synthetic((1, tokens, input), 202, 0.75, &Device::Cpu)?;
                let expected = reference.forward(&x)?;
                let cpu = cpu_q.forward(&x)?;
                let gpu = cuda_q.forward(&x.to_device(&cuda)?)?;
                let gpu_f16 = cuda_f16.forward(&x.to_device(&cuda)?)?;
                let case = format!("Q8 tokens={tokens}, {input}->{output}");
                // Both backends may quantize activations; CUDA uses 32-wide Q8_1
                // blocks with half-precision scales. Do not demand 1e-4/bitwise
                // parity. With |x|<=0.75, weights <=1/sqrt(input), the typical
                // output RMS is ~0.25 and activation-rounding noise is ~1e-3.
                // Allow 5e-3 absolute + 1% relative per element, and independently
                // cap aggregate error at 1% RMS. The dense dequantized reference
                // also prevents two identically wrong Q8 results from passing.
                for (label, a, b) in [
                    ("cpu/dense", &expected, &cpu),
                    ("cuda/dense", &expected, &gpu),
                    ("cuda/cpu", &cpu, &gpu),
                    ("cuda_f16/dense", &expected, &gpu_f16),
                ] {
                    let nrmse = assert_close(&format!("{case}/{label}"), a, b, 5e-3, 1e-2)?;
                    assert!(nrmse <= 1e-2, "{case}/{label}: normalized RMSE {nrmse:.6e} > 1%");
                }
            }
        }
        Ok(())
    }
}
