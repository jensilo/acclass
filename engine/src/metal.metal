// Fused kernels for one ModernBERT forward (F32). Layouts match src/metal.rs.
#include <metal_stdlib>
using namespace metal;

constant uint D = 64;
constant uint BLOCKS = D / 4;

struct NormParams { uint n; float eps; };

// LayerNorm without bias over rows of n values. One threadgroup per row, so short
// inputs still spread over the GPU cores; each thread owns a slice of the row.
kernel void norm_rows(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant NormParams &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup float shared[32];
    device const float *r = x + row * p.n;
    uint lo = t * p.n / width, hi = (t + 1) * p.n / width;
    float part = 0;
    for (uint i = lo; i < hi; i++) part += r[i];
    shared[t] = part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0;
    for (uint i = 0; i < width; i++) sum += shared[i];
    float mean = sum / p.n;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    part = 0;
    for (uint i = lo; i < hi; i++) { float c = r[i] - mean; part += c * c; }
    shared[t] = part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sq = 0;
    for (uint i = 0; i < width; i++) sq += shared[i];
    float den = sqrt(sq / p.n + p.eps);
    device float *o = out + row * p.n;
    for (uint i = lo; i < hi; i++) o[i] = (r[i] - mean) / den * w[i];
}

// A&S 7.1.26, the same approximation as Candle's Metal gelu_erf.
static inline float gelu_erf(float v) {
    float x = fabs(v * M_SQRT1_2_F);
    float t = 1.0 / (1.0 + 0.3275911 * x);
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
    return v * (1 + (v < 0 ? -y : y)) / 2;
}

// Rows of [input | gate] (2n values) to gate * gelu_erf(input) (n values), per element.
kernel void gelu_gate(
    device const float *x [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint at [[thread_position_in_grid]])
{
    uint row = at / n, i = at % n;
    device const float *r = x + row * 2 * n;
    out[at] = r[n + i] * gelu_erf(r[i]);
}

struct AttentionParams {
    uint s;       // tokens
    uint heads;
    uint width;   // keys stored per query row
    uint window;  // inclusive radius; >= s means full attention
};

static inline uint span_lo(uint row, constant AttentionParams &p) {
    return row > p.window ? row - p.window : 0;
}

static inline uint span_hi(uint row, constant AttentionParams &p) {
    return min(p.s, row + p.window + 1);
}

// qkv rows hold [q | k | v], each heads * 64 wide. Writes rotated q and k as
// [part][head][row][64]. rot = [-x2 | x1].
kernel void rope(
    device const float *qkv [[buffer(0)]],
    device const float *cos_t [[buffer(1)]],
    device const float *sin_t [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant AttentionParams &p [[buffer(4)]],
    uint3 t [[thread_position_in_grid]])
{
    uint head = t.x, row = t.y, part = t.z;
    uint hidden = p.heads * D;
    device const float *x = qkv + (row * 3 + part) * hidden + head * D;
    device const float *c = cos_t + row * D;
    device const float *n = sin_t + row * D;
    device float *o = out + ((part * p.heads + head) * p.s + row) * D;
    for (uint i = 0; i < D / 2; i++) {
        o[i] = x[i] * c[i] - x[i + D / 2] * n[i];
        o[i + D / 2] = x[i + D / 2] * c[i + D / 2] + x[i] * n[i + D / 2];
    }
}

// scores[head][row][j - lo] = (q[row] . k[j]) / 8, only for keys inside the span.
kernel void scores(
    device const packed_float4 *qk [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant AttentionParams &p [[buffer(2)]],
    uint3 t [[thread_position_in_grid]])
{
    uint at = t.x, row = t.y, head = t.z;
    uint lo = span_lo(row, p);
    uint key = lo + at;
    if (key >= span_hi(row, p)) return;
    device const packed_float4 *q = qk + (head * p.s + row) * BLOCKS;
    device const packed_float4 *k = qk + ((p.heads + head) * p.s + key) * BLOCKS;
    float sum = 0;
    for (uint c = 0; c < BLOCKS; c++) sum += dot(float4(q[c]), float4(k[c]));
    out[((head * p.s + row) * p.width) + at] = sum * 0.125;
}

// Softmax over one query row's span, then the weighted sum of its values.
kernel void softmax_pv(
    device const float *scores [[buffer(0)]],
    device const float *qkv [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant AttentionParams &p [[buffer(3)]],
    uint2 t [[thread_position_in_grid]])
{
    uint head = t.x, row = t.y;
    uint hidden = p.heads * D;
    uint lo = span_lo(row, p);
    uint count = span_hi(row, p) - lo;
    device const float *r = scores + (head * p.s + row) * p.width;
    float top = r[0];
    for (uint j = 1; j < count; j++) top = max(top, r[j]);
    float total = 0;
    float4 acc[BLOCKS];
    for (uint c = 0; c < BLOCKS; c++) acc[c] = 0;
    for (uint j = 0; j < count; j++) {
        float e = exp(r[j] - top);
        total += e;
        device const packed_float4 *v = (device const packed_float4 *)(qkv + ((lo + j) * 3 + 2) * hidden + head * D);
        for (uint c = 0; c < BLOCKS; c++) acc[c] += e * float4(v[c]);
    }
    device packed_float4 *o = (device packed_float4 *)(out + row * hidden + head * D);
    for (uint c = 0; c < BLOCKS; c++) o[c] = packed_float4(acc[c] / total);
}
