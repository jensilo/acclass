// Fused CPU attention (aarch64 NEON, x86_64 AVX2): scores, window, softmax and
// PV in one rayon pass over (head, query block), written directly as
// (1, s, heads * 64). Candle's path builds the full (heads, s, s) score tensor and
// adds the sliding mask with single-threaded element-wise ops. Here sliding layers
// only visit keys with |i - j| <= window; the masked entries are exact zeros after
// softmax in the unfused path too (exp(-65504 - max) underflows), so only
// summation order differs.

use candle_core::{CpuStorage, Layout, Result, Shape};
use rayon::prelude::*;

const D: usize = 64;
const QB: usize = 16; // query rows per task

pub struct Attention {
    pub window: Option<usize>, // None: full attention
}

/// Whether this CPU can run the fused kernel.
pub fn supported() -> bool {
    arch::supported()
}

impl candle_core::CustomOp3 for Attention {
    fn name(&self) -> &'static str {
        "acclass-attention"
    }

    fn cpu_fwd(
        &self,
        q: &CpuStorage,
        lq: &Layout,
        k: &CpuStorage,
        lk: &Layout,
        v: &CpuStorage,
        lv: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn slice<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
            let (CpuStorage::F32(x), Some((a, b))) = (s, l.contiguous_offsets()) else {
                candle_core::bail!("attention expects contiguous f32 inputs")
            };
            Ok(&x[a..b])
        }
        let &[1, h, s, D] = lq.dims() else { candle_core::bail!("attention expects (1, heads, s, 64)") };
        if lk.dims() != lq.dims() || lv.dims() != lq.dims() {
            candle_core::bail!("attention q/k/v shapes differ")
        }
        let (q, k, v) = (slice(q, lq)?, slice(k, lk)?, slice(v, lv)?);
        let mut out = vec![0f32; s * h * D];
        let outp = out.as_mut_ptr() as usize;
        let blocks = s.div_ceil(QB);
        let span = |i: usize| match self.window {
            Some(wd) => (i.saturating_sub(wd), (i + wd + 1).min(s)),
            None => (0, s),
        };
        (0..h * blocks).into_par_iter().for_each_init(Vec::new, |scores, task| unsafe {
            let (head, i0) = (task / blocks, task % blocks * QB);
            let rows = (s - i0).min(QB);
            let (lo, hi) = (span(i0).0, span(i0 + rows - 1).1);
            let w = hi - lo;
            scores.resize(rows * w, 0.0);
            let base = head * s * D;
            arch::qk_block(q.as_ptr().add(base + i0 * D), k.as_ptr().add(base), rows, lo, hi, w, scores);
            for r in 0..rows {
                let (a, b) = span(i0 + r);
                let row = &mut scores[r * w + a - lo..r * w + b - lo];
                let max = row.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
                let mut sum = 0f32;
                for x in row.iter_mut() {
                    *x = (*x - max).exp();
                    sum += *x;
                }
                let o = (outp as *mut f32).add((i0 + r) * h * D + head * D);
                arch::pv(row, sum, v.as_ptr().add(base + a * D), o);
            }
        });
        Ok((CpuStorage::F32(out), Shape::from((1, s, h * D))))
    }
}

#[cfg(target_arch = "aarch64")]
mod arch {
    use super::D;
    use std::arch::aarch64::*;

    pub fn supported() -> bool {
        true
    }

    #[inline(always)]
    unsafe fn hsum4(a: float32x4_t, b: float32x4_t, c: float32x4_t, d: float32x4_t) -> float32x4_t {
        unsafe { vpaddq_f32(vpaddq_f32(a, b), vpaddq_f32(c, d)) }
    }

    /// scores[r][j - lo] = (q[i0 + r] . k[j]) * 0.125 for j in lo..hi, rows 0..rows.
    pub unsafe fn qk_block(q: *const f32, k: *const f32, rows: usize, lo: usize, hi: usize, w: usize, out: &mut [f32]) {
        unsafe {
            let mut r = 0;
            while r < rows {
                let rn = (rows - r).min(4);
                let mut j = lo;
                while j < hi {
                    let jn = (hi - j).min(4);
                    let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
                    for c in (0..D).step_by(4) {
                        let mut kv = [vdupq_n_f32(0.0); 4];
                        for b in 0..jn {
                            kv[b] = vld1q_f32(k.add((j + b) * D + c));
                        }
                        for a in 0..rn {
                            let qv = vld1q_f32(q.add((r + a) * D + c));
                            for b in 0..4 {
                                acc[a][b] = vfmaq_f32(acc[a][b], qv, kv[b]);
                            }
                        }
                    }
                    for a in 0..rn {
                        let mut v = [0f32; 4];
                        vst1q_f32(v.as_mut_ptr(), vmulq_n_f32(hsum4(acc[a][0], acc[a][1], acc[a][2], acc[a][3]), 0.125));
                        out[(r + a) * w + (j - lo)..][..jn].copy_from_slice(&v[..jn]);
                    }
                    j += 4;
                }
                r += 4;
            }
        }
    }

    /// o = sum_j (p[j] / sum) * v[j], with v pointing at the span's first key.
    pub unsafe fn pv(p: &[f32], sum: f32, v: *const f32, o: *mut f32) {
        unsafe {
            let mut acc = [vdupq_n_f32(0.0); D / 4];
            for (j, &p) in p.iter().enumerate() {
                let p = p / sum;
                let vj = v.add(j * D);
                for (c, acc) in acc.iter_mut().enumerate() {
                    *acc = vfmaq_n_f32(*acc, vld1q_f32(vj.add(c * 4)), p);
                }
            }
            for (c, acc) in acc.iter().enumerate() {
                vst1q_f32(o.add(c * 4), *acc);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod arch {
    use super::D;
    use std::arch::x86_64::*;

    pub fn supported() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    /// Lane i of the result is the sum of v[i].
    #[inline(always)]
    unsafe fn hsum4(v: [__m256; 4]) -> __m128 {
        unsafe {
            let h = _mm256_hadd_ps(_mm256_hadd_ps(v[0], v[1]), _mm256_hadd_ps(v[2], v[3]));
            _mm_add_ps(_mm256_castps256_ps128(h), _mm256_extractf128_ps::<1>(h))
        }
    }

    /// scores[r][j - lo] = (q[i0 + r] . k[j]) * 0.125 for j in lo..hi, rows 0..rows.
    /// Tiles of 2 rows x 4 keys keep the accumulators in the 16 registers.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn qk_block(q: *const f32, k: *const f32, rows: usize, lo: usize, hi: usize, w: usize, out: &mut [f32]) {
        unsafe {
            let mut r = 0;
            while r < rows {
                let rn = (rows - r).min(2);
                let mut j = lo;
                while j < hi {
                    let jn = (hi - j).min(4);
                    let mut acc = [[_mm256_setzero_ps(); 4]; 2];
                    for c in (0..D).step_by(8) {
                        let mut kv = [_mm256_setzero_ps(); 4];
                        for b in 0..jn {
                            kv[b] = _mm256_loadu_ps(k.add((j + b) * D + c));
                        }
                        for a in 0..rn {
                            let qv = _mm256_loadu_ps(q.add((r + a) * D + c));
                            for b in 0..4 {
                                acc[a][b] = _mm256_fmadd_ps(qv, kv[b], acc[a][b]);
                            }
                        }
                    }
                    for a in 0..rn {
                        let mut v = [0f32; 4];
                        _mm_storeu_ps(v.as_mut_ptr(), _mm_mul_ps(hsum4(acc[a]), _mm_set1_ps(0.125)));
                        out[(r + a) * w + (j - lo)..][..jn].copy_from_slice(&v[..jn]);
                    }
                    j += 4;
                }
                r += 2;
            }
        }
    }

    /// o = sum_j (p[j] / sum) * v[j], with v pointing at the span's first key.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn pv(p: &[f32], sum: f32, v: *const f32, o: *mut f32) {
        unsafe {
            let mut acc = [_mm256_setzero_ps(); D / 8];
            for (j, &p) in p.iter().enumerate() {
                let p = _mm256_set1_ps(p / sum);
                let vj = v.add(j * D);
                for (c, acc) in acc.iter_mut().enumerate() {
                    *acc = _mm256_fmadd_ps(_mm256_loadu_ps(vj.add(c * 8)), p, *acc);
                }
            }
            for (c, acc) in acc.iter().enumerate() {
                _mm256_storeu_ps(o.add(c * 8), *acc);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor, D as Dim};

    #[test]
    fn fused_attention_matches_masked_softmax_reference() -> Result<()> {
        if !supported() {
            return Ok(());
        }
        let dev = Device::Cpu;
        for s in [1, 5, 70, 150] {
            let q = Tensor::randn(0f32, 1.0, (1, 3, s, D), &dev)?;
            let k = Tensor::randn(0f32, 1.0, (1, 3, s, D), &dev)?;
            let v = Tensor::randn(0f32, 1.0, (1, 3, s, D), &dev)?;
            for window in [None, Some(64)] {
                let mut scores = (q.matmul(&k.transpose(Dim::Minus2, Dim::Minus1)?)? * 0.125)?;
                if let Some(wd) = window {
                    let m: Vec<f32> = (0..s * s)
                        .map(|x| if (x / s).abs_diff(x % s) > wd { -65504.0 } else { 0.0 })
                        .collect();
                    scores = scores.broadcast_add(&Tensor::from_vec(m, (1, 1, s, s), &dev)?)?;
                }
                let want = candle_nn::ops::softmax_last_dim(&scores)?
                    .matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((1, s, 3 * D))?;
                let got = q.apply_op3_no_bwd(&k, &v, &Attention { window })?;
                let diff = (got - want)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
                assert!(diff < 1e-5, "s={s} window={window:?}: max diff {diff}");
            }
        }
        Ok(())
    }
}
