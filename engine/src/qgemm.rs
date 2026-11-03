// Blocked int8 GEMM for Q8_0/Q5_0 weights on CPUs: aarch64 (i8mm `smmla`, else
// `sdot`, selected at runtime) and x86_64 (AVX2). Same math as candle's QMatMul:
// activations are quantized to Q8_0 per 32-wide block (f16 scale), each block's
// int8 dot product is scaled by both f16 block scales and summed in f32. Only the
// float summation order differs. Candle's CPU path instead runs one row at a time
// over all weights; this kernel reuses each weight tile across 4 rows.
//
// Shared: weight repack (per group of 8 output columns and 32-wide block, 256
// bytes), activation quantization (per quad of 4 rows), threading and stores.
// Per arch: the byte order inside those chunks and the 4x8 tile kernel.
//
// aarch64 weights: four 8-byte k-chunks, each holding 4 column pairs as
// [col 2p: 8 bytes | col 2p+1: 8 bytes]. Activations: per row pair and block,
// four [row 2r: 8 bytes | row 2r+1: 8 bytes].
// x86_64 weights: 8 columns x 32 bytes. Activations: per row, 32 bytes per block.

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{CpuStorage, Layout, Result, Shape};
use rayon::prelude::*;

const QK: usize = 32;

pub struct PackedQ8 {
    n: usize,
    k: usize,
    qs: Vec<i8>,  // (n/8) groups x (k/32) blocks x 256 bytes
    wd: Vec<f32>, // (n/8) groups x (k/32) blocks x 8 column scales
    kind: arch::Kind,
}

fn f16_at(bytes: &[u8], at: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([bytes[at], bytes[at + 1]])).to_f32()
}

impl PackedQ8 {
    /// Repack a CPU (n, k) Q8_0 or Q5_0 tensor (5-bit values widen to int8); None if this CPU/shape/dtype is unsupported.
    pub fn new(qt: &QTensor) -> Result<Option<Self>> {
        match arch::kind() {
            Some(kind) => Self::with_kind(qt, kind),
            None => Ok(None),
        }
    }

    fn with_kind(qt: &QTensor, kind: arch::Kind) -> Result<Option<Self>> {
        let Ok((n, k)) = qt.shape().dims2() else { return Ok(None) };
        let block_bytes = match qt.dtype() {
            GgmlDType::Q8_0 => 34,
            GgmlDType::Q5_0 => 22,
            _ => return Ok(None),
        };
        if n % 8 != 0 || k % QK != 0 {
            return Ok(None);
        }
        let data = qt.data()?;
        let nb = k / QK;
        // Decode one weight row's block into 32 int8 values and its scale.
        let block = |row: usize, b: usize| -> ([i8; QK], f32) {
            let at = (row * nb + b) * block_bytes;
            let mut q = [0i8; QK];
            if block_bytes == 34 {
                for j in 0..QK {
                    q[j] = data[at + 2 + j] as i8;
                }
            } else {
                // ggml BlockQ5_0: f16 d, u32 qh (bit j = high bit of value j), 16 nibble bytes.
                let qh = u32::from_le_bytes(data[at + 2..at + 6].try_into().unwrap());
                for j in 0..QK / 2 {
                    let v = data[at + 6 + j];
                    let h0 = ((qh >> j) << 4) & 0x10;
                    let h1 = (qh >> (j + 12)) & 0x10;
                    q[j] = ((v & 0x0f) as u32 | h0) as i8 - 16;
                    q[j + QK / 2] = ((v >> 4) as u32 | h1) as i8 - 16;
                }
            }
            (q, f16_at(&data, at))
        };
        let mut qs = vec![0i8; n * k];
        let mut wd = vec![0f32; n / 8 * nb * 8];
        for g in 0..n / 8 {
            for b in 0..nb {
                let base = (g * nb + b) * 256;
                for c in 0..8 {
                    let (q, d) = block(8 * g + c, b);
                    if !arch::ALLOWS_MINUS_128 && q.contains(&i8::MIN) {
                        return Ok(None); // raw Q8_0 from another quantizer: candle's QMatMul
                    }
                    for (j, &v) in q.iter().enumerate() {
                        qs[base + arch::w_pos(c, j)] = v;
                    }
                    wd[(g * nb + b) * 8 + c] = d;
                }
            }
        }
        Ok(Some(Self { n, k, qs, wd, kind }))
    }
}

/// Quantize activations like candle's BlockQ8_0::from_float into the arch's
/// layout, rows padded to a multiple of 4 with zeros. Returns (qs, xd): per quad
/// of rows 4k bytes and 4 * nb * arch::XD scale slots.
fn quantize_rows(x: &[f32], m: usize, k: usize, kind: arch::Kind) -> (Vec<i8>, Vec<f32>) {
    let nb = k / QK;
    let quads = m.div_ceil(4);
    let mut qs = vec![0i8; quads * 4 * k];
    let mut xd = vec![0f32; quads * 4 * nb * arch::XD];
    qs.par_chunks_mut(4 * k).zip(xd.par_chunks_mut(4 * nb * arch::XD)).enumerate().for_each(|(rq, (q, d))| {
        for r in 0..4 {
            let row = 4 * rq + r;
            if row >= m {
                break;
            }
            for b in 0..nb {
                let xs = &x[row * k + b * QK..][..QK];
                let amax = xs.iter().fold(0f32, |a, &v| a.max(v.abs()));
                let scale = amax / 127.0;
                let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
                for (t, &v) in xs.iter().enumerate() {
                    q[arch::a_pos(k, r, b, t)] = (v * inv).round() as i8;
                }
                arch::put_scale(d, kind, nb, r, b, half::f16::from_f32(scale).to_f32());
            }
        }
    });
    (qs, xd)
}

impl candle_core::CustomOp1 for PackedQ8 {
    fn name(&self) -> &'static str {
        "acclass-qgemm"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(src) = storage else { candle_core::bail!("qgemm expects f32 input") };
        let Some(x) = layout.contiguous_offsets().map(|(a, b)| &src[a..b]) else {
            candle_core::bail!("qgemm expects contiguous input")
        };
        let mut dims = layout.dims().to_vec();
        if dims.last() != Some(&self.k) {
            candle_core::bail!("qgemm input {dims:?} does not end in k={}", self.k)
        }
        let (n, k, nb) = (self.n, self.k, self.k / QK);
        let m = x.len() / k;
        let (aq, ad) = quantize_rows(x, m, k, self.kind);
        let mut dst = vec![0f32; m * n];
        let out = dst.as_mut_ptr() as usize;
        let quads = m.div_ceil(4);
        let groups = n / 8;
        // Each task owns whole column groups of every row: disjoint writes.
        let per_task = groups.div_ceil(4 * rayon::current_num_threads()).max(1);
        (0..groups).into_par_iter().with_min_len(per_task).for_each(|g| unsafe {
            let bq = self.qs.as_ptr().add(g * nb * 256);
            let bd = self.wd.as_ptr().add(g * nb * 8);
            for rq in 0..quads {
                let aqp = aq.as_ptr().add(rq * 4 * k);
                let adp = ad.as_ptr().add(rq * 4 * nb * arch::XD);
                let tile = arch::tile(self.kind, nb, aqp, adp, bq, bd);
                for (r, vals) in tile.iter().enumerate().take(m - rq * 4) {
                    let o = (out as *mut f32).add((rq * 4 + r) * n + g * 8);
                    std::ptr::copy_nonoverlapping(vals.as_ptr(), o, 8);
                }
            }
        });
        *dims.last_mut().unwrap() = n;
        Ok((CpuStorage::F32(dst), Shape::from(dims)))
    }
}

#[cfg(target_arch = "aarch64")]
mod arch {
    use std::arch::aarch64::*;

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum Kind {
        I8mm,
        Dotprod,
    }

    pub fn kind() -> Option<Kind> {
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            Some(Kind::I8mm)
        } else if std::arch::is_aarch64_feature_detected!("dotprod") {
            Some(Kind::Dotprod)
        } else {
            None
        }
    }

    #[cfg(test)]
    pub fn kinds() -> Vec<Kind> {
        match kind() {
            Some(Kind::I8mm) => vec![Kind::I8mm, Kind::Dotprod],
            Some(Kind::Dotprod) => vec![Kind::Dotprod],
            None => vec![],
        }
    }

    /// Position of column c's value j inside a group's 256-byte block.
    pub fn w_pos(c: usize, j: usize) -> usize {
        ((j / 8) * 4 + c / 2) * 16 + (c % 2) * 8 + j % 8
    }

    pub const ALLOWS_MINUS_128: bool = true;

    /// Scale slots per row: the 2x2 tile lanes of a row pair hold 4 scales.
    pub const XD: usize = 2;

    /// Position of row r's (0..4) value t of block b inside a quad's 4k bytes.
    pub fn a_pos(k: usize, r: usize, b: usize, t: usize) -> usize {
        (r / 2) * 2 * k + b * 64 + (t / 8) * 16 + (r % 2) * 8 + t % 8
    }

    pub fn put_scale(d: &mut [f32], kind: Kind, nb: usize, r: usize, b: usize, s: f32) {
        // smmla lanes: [r0c0, r0c1, r1c0, r1c1]; sdot lanes: [r0c0, r1c1, r1c0, r0c1]
        let lanes: [usize; 2] = match (kind, r % 2) {
            (Kind::I8mm, 0) => [0, 1],
            (Kind::I8mm, _) => [2, 3],
            (Kind::Dotprod, 0) => [0, 3],
            (Kind::Dotprod, _) => [1, 2],
        };
        for l in lanes {
            d[(r / 2) * nb * 4 + b * 4 + l] = s;
        }
    }

    #[inline(always)]
    unsafe fn smmla(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut acc = acc;
        unsafe {
            core::arch::asm!("smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
                acc = inout(vreg) acc, a = in(vreg) a, b = in(vreg) b,
                options(pure, nomem, nostack));
        }
        acc
    }

    // One tile: 4 rows (2 row pairs) x 8 columns (4 column pairs), all k blocks.
    // `$mac(acc, a, b)` accumulates one 8-wide k-chunk of a 2x2 tile.
    macro_rules! tile_kernel {
        ($name:ident, $feature:literal, $mac:expr) => {
            #[target_feature(enable = $feature)]
            unsafe fn $name(nb: usize, aq: *const i8, ad: *const f32, bq: *const i8, bd: *const f32) -> [float32x4_t; 8] {
                unsafe {
                    let mut f = [vdupq_n_f32(0.0); 8];
                    for b in 0..nb {
                        let mut acc = [vdupq_n_s32(0); 8];
                        for c in 0..4 {
                            let a0 = vld1q_s8(aq.add(b * 64 + c * 16));
                            let a1 = vld1q_s8(aq.add(nb * 64 + b * 64 + c * 16));
                            for p in 0..4 {
                                let w = vld1q_s8(bq.add(b * 256 + (c * 4 + p) * 16));
                                acc[p] = $mac(acc[p], a0, w);
                                acc[4 + p] = $mac(acc[4 + p], a1, w);
                            }
                        }
                        let x0 = vld1q_f32(ad.add(b * 4));
                        let x1 = vld1q_f32(ad.add(nb * 4 + b * 4));
                        for p in 0..4 {
                            // [d(2p), d(2p+1), d(2p), d(2p+1)] in one ld1r
                            let w = vreinterpretq_f32_f64(vld1q_dup_f64(bd.add(b * 8 + p * 2) as *const f64));
                            f[p] = vfmaq_f32(f[p], vcvtq_f32_s32(acc[p]), vmulq_f32(x0, w));
                            f[4 + p] = vfmaq_f32(f[4 + p], vcvtq_f32_s32(acc[4 + p]), vmulq_f32(x1, w));
                        }
                    }
                    f
                }
            }
        };
    }

    tile_kernel!(tile_i8mm, "i8mm", |acc, a, w| smmla(acc, a, w));
    // Two sdot per 2x2 tile: the straight product gives (r0,c0),(r1,c1) lane pairs,
    // half-swapped rows give (r1,c0),(r0,c1). Accumulate in separate halves:
    // lanes of `acc` hold 4-byte partial sums; pairwise-add once per block below.
    tile_kernel!(tile_dot, "dotprod", |acc: int32x4_t, a: int8x16_t, w: int8x16_t| {
        let d = vdotq_s32(vdupq_n_s32(0), a, w);
        let x = vdotq_s32(vdupq_n_s32(0), vextq_s8::<8>(a, a), w);
        vaddq_s32(acc, vpaddq_s32(d, x))
    });

    /// 4 rows x 8 columns of output for one quad of rows and one column group.
    pub unsafe fn tile(kind: Kind, nb: usize, aq: *const i8, ad: *const f32, bq: *const i8, bd: *const f32) -> [[f32; 8]; 4] {
        let (f, lanes) = unsafe {
            match kind {
                Kind::I8mm => (tile_i8mm(nb, aq, ad, bq, bd), [(0, 0), (0, 1), (1, 0), (1, 1)]),
                Kind::Dotprod => (tile_dot(nb, aq, ad, bq, bd), [(0, 0), (1, 1), (1, 0), (0, 1)]),
            }
        };
        let mut out = [[0f32; 8]; 4];
        let mut v = [0f32; 4];
        for (t, acc) in f.iter().enumerate() {
            unsafe { vst1q_f32(v.as_mut_ptr(), *acc) };
            let (pair, p) = (t / 4, t % 4);
            for (l, &(r, c)) in lanes.iter().enumerate() {
                out[pair * 2 + r][2 * p + c] = v[l];
            }
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
mod arch {
    use std::arch::x86_64::*;

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum Kind {
        Avx2,
    }

    pub fn kind() -> Option<Kind> {
        (is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")).then_some(Kind::Avx2)
    }

    #[cfg(test)]
    pub fn kinds() -> Vec<Kind> {
        kind().into_iter().collect()
    }

    pub fn w_pos(c: usize, j: usize) -> usize {
        c * 32 + j
    }

    /// sign_epi8 cannot negate -128; candle's quantizer never writes it.
    pub const ALLOWS_MINUS_128: bool = false;

    pub const XD: usize = 1;

    pub fn a_pos(k: usize, r: usize, b: usize, t: usize) -> usize {
        r * k + b * 32 + t
    }

    pub fn put_scale(d: &mut [f32], _: Kind, nb: usize, r: usize, b: usize, s: f32) {
        d[r * nb + b] = s;
    }

    /// Sum each of 8 i32x8 vectors: lane c of the result is the sum of v[c].
    #[inline(always)]
    unsafe fn hsum8(v: [__m256i; 8]) -> __m256i {
        unsafe {
            let a = _mm256_hadd_epi32(_mm256_hadd_epi32(v[0], v[1]), _mm256_hadd_epi32(v[2], v[3]));
            let b = _mm256_hadd_epi32(_mm256_hadd_epi32(v[4], v[5]), _mm256_hadd_epi32(v[6], v[7]));
            // a, b hold per-128-bit-half sums of v[0..4], v[4..8]; add the halves.
            _mm256_add_epi32(_mm256_permute2x128_si256::<0x20>(a, b), _mm256_permute2x128_si256::<0x31>(a, b))
        }
    }

    /// 4 rows x 8 columns, all k blocks. Block dot of x and w: sign_epi8 makes x
    /// non-negative and moves its sign onto w, maddubs (u8 x i8 -> i16 pairs,
    /// 2 * 127 * 127 fits), madd with ones -> i32. Needs w, x within +-127: x is by
    /// construction, w is checked at repack (ALLOWS_MINUS_128).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn tile(_: Kind, nb: usize, aq: *const i8, ad: *const f32, bq: *const i8, bd: *const f32) -> [[f32; 8]; 4] {
        unsafe {
            let ones = _mm256_set1_epi16(1);
            let mut f = [_mm256_setzero_ps(); 4];
            for b in 0..nb {
                let wd = _mm256_loadu_ps(bd.add(b * 8));
                let mut w = [_mm256_setzero_si256(); 8];
                for (c, w) in w.iter_mut().enumerate() {
                    *w = _mm256_loadu_si256(bq.add(b * 256 + c * 32) as *const __m256i);
                }
                for (r, f) in f.iter_mut().enumerate() {
                    let x = _mm256_loadu_si256(aq.add(r * nb * 32 + b * 32) as *const __m256i);
                    let ax = _mm256_sign_epi8(x, x);
                    let mut d = [_mm256_setzero_si256(); 8];
                    for c in 0..8 {
                        d[c] = _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(w[c], x)), ones);
                    }
                    let scale = _mm256_mul_ps(_mm256_set1_ps(*ad.add(r * nb + b)), wd);
                    *f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(hsum8(d)), scale, *f);
                }
            }
            let mut out = [[0f32; 8]; 4];
            for (o, f) in out.iter_mut().zip(f) {
                _mm256_storeu_ps(o.as_mut_ptr(), f);
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::QMatMul;
    use candle_core::{Device, Module, Tensor};

    #[test]
    fn packed_gemm_matches_candle_qmatmul() -> Result<()> {
        let kinds = arch::kinds();
        let mut seed = 7u32;
        let mut rand = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 8) as f32 / 8_388_608.0 - 1.0
                })
                .collect()
        };
        for dtype in [GgmlDType::Q8_0, GgmlDType::Q5_0] {
            // Candle's x86 Q5_0 reference needs an even block count (k / 32).
            for (m, k, n) in [(1, 64, 8), (5, 192, 16), (9, 768, 24)] {
                let w = Tensor::from_vec(rand(n * k), (n, k), &Device::Cpu)?;
                let x = Tensor::from_vec(rand(m * k), (1, m, k), &Device::Cpu)?;
                let qt = QTensor::quantize(&w, dtype)?;
                let want = QMatMul::from_qtensor(QTensor::quantize(&w, dtype)?)?.forward(&x)?;
                for &kind in &kinds {
                    let packed = PackedQ8::with_kind(&qt, kind)?.expect("supported shape");
                    let got = x.apply_op1_no_bwd(&packed)?;
                    assert_eq!(got.dims(), want.dims());
                    let diff = (got - &want)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
                    assert!(diff < 1e-4, "{dtype:?} {kind:?} m={m} k={k} n={n}: max diff {diff}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn raw_q8_weight_minus_128_is_exact_or_declined() -> Result<()> {
        // Valid Q8_0 bytes that candle's quantizer never writes: every weight -128, d = 1/128.
        let (n, k) = (8, 64);
        let block: Vec<u8> = half::f16::from_f32(1.0 / 128.0).to_le_bytes().into_iter().chain([0x80; 32]).collect();
        let raw = block.repeat(n * k / QK);
        let qt = candle_core::quantized::ggml_file::qtensor_from_ggml(GgmlDType::Q8_0, &raw, vec![n, k], &Device::Cpu)?;
        let x = Tensor::from_vec((0..k).map(|i| -0.5 - i as f32 / 128.0).collect(), (1, 1, k), &Device::Cpu)?;
        let want = x.broadcast_matmul(&qt.dequantize(&Device::Cpu)?.t()?)?.flatten_all()?.to_vec1::<f32>()?;
        for kind in arch::kinds() {
            let Some(packed) = PackedQ8::with_kind(&qt, kind)? else {
                assert!(!arch::ALLOWS_MINUS_128, "{kind:?} declined a valid tensor");
                continue;
            };
            let got = x.apply_op1_no_bwd(&packed)?.flatten_all()?.to_vec1::<f32>()?;
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-2 * w.abs(), "{kind:?}: {g} vs {w}");
            }
        }
        Ok(())
    }
}
