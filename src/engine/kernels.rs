//! CPU kernels for the Laya engine (x86-64: AVX2 + FMA + F16C).
//!
//! Every public function here requires [`supported`] to be true; they assert it, so the safe API
//! cannot reach an unsupported instruction. Weights stay in f16 (as stored in the checkpoint) and
//! are widened to f32 inside the matmul, so the arithmetic is the same f32 arithmetic the reference
//! implementation performs on its up-cast copy, with only the summation order differing.
//!
//! All parallel kernels run on the *current* rayon pool (the engine wraps each forward in
//! `ThreadPool::install`).

use rayon::prelude::*;
use std::arch::x86_64::*;

/// True when the CPU has everything the kernels below need.
pub fn supported() -> bool {
    is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("f16c")
}

#[inline]
fn check() {
    assert!(supported(), "AVX2/FMA/F16C required");
}

/// A raw pointer that may be shared across rayon tasks. Each task writes a disjoint region.
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

// ---------------------------------------------------------------------------------------------
// f16
// ---------------------------------------------------------------------------------------------

/// IEEE binary16 -> binary32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // subnormal: renormalise
            let shift = m.leading_zeros() - 21; // bring the top set bit to bit 10
            let m = (m << shift) & 0x3ff;
            let e = 127 - 15 + 1 - shift;
            sign | (e << 23) | (m << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

pub fn f16_slice_to_f32(src: &[u16]) -> Vec<f32> {
    src.iter().map(|&h| f16_to_f32(h)).collect()
}

/// Vectorised f16 -> f32 for whole rows (`len % 8 == 0`).
pub fn widen_f16(src: &[u16], dst: &mut [f32]) {
    check();
    assert!(src.len() == dst.len() && src.len() % 8 == 0);
    unsafe { widen_f16_inner(src, dst) }
}

#[target_feature(enable = "avx2,f16c")]
unsafe fn widen_f16_inner(src: &[u16], dst: &mut [f32]) {
    let mut i = 0;
    while i < src.len() {
        let v = _mm256_cvtph_ps(_mm_loadu_si128(src.as_ptr().add(i) as *const __m128i));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), v);
        i += 8;
    }
}

// ---------------------------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------------------------

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps::<1>(s, s));
    _mm_cvtss_f32(s)
}

/// Vector `exp` (Cephes polynomial, relative error ~1e-7 on the softmax range).
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp256(x: __m256) -> __m256 {
    let x = _mm256_min_ps(x, _mm256_set1_ps(88.376_26));
    let x = _mm256_max_ps(x, _mm256_set1_ps(-88.376_26));
    let fx = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        x,
        _mm256_set1_ps(std::f32::consts::LOG2_E),
    ));
    let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693_359_375), x);
    let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.121_944_4e-4), r);
    let z = _mm256_mul_ps(r, r);
    let mut y = _mm256_set1_ps(1.987_569_15e-4);
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.398_199_950_7e-3));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(8.333_451_907_3e-3));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(4.166_579_589_4e-2));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.666_666_545_9e-1));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(5.000_000_120_1e-1));
    y = _mm256_fmadd_ps(y, z, r);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0));
    let n = _mm256_cvtps_epi32(fx);
    let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
        n,
        _mm256_set1_epi32(127),
    )));
    _mm256_mul_ps(y, pow2n)
}

// ---------------------------------------------------------------------------------------------
// matmul: out[m, n] = sum_k a[m, k] * w[n, k] (+ bias[n])      (a, out f32; w f16, row-major [n, k])
// ---------------------------------------------------------------------------------------------

/// Rows of `a` handled per micro-kernel call. 6x2 accumulators + 2 weight vectors + 1 activation
/// vector = 15 of the 16 ymm registers.
const MR: usize = 6;

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile<const R: usize>(
    a: *const f32,
    lda: usize,
    w0: *const u16,
    w1: *const u16,
    k: usize,
    out: *mut f32,
    ldo: usize,
    bias: [f32; 2],
) {
    let mut c0 = [_mm256_setzero_ps(); R];
    let mut c1 = [_mm256_setzero_ps(); R];
    let mut p = 0;
    while p < k {
        let b0 = _mm256_cvtph_ps(_mm_loadu_si128(w0.add(p) as *const __m128i));
        let b1 = _mm256_cvtph_ps(_mm_loadu_si128(w1.add(p) as *const __m128i));
        for i in 0..R {
            let av = _mm256_loadu_ps(a.add(i * lda + p));
            c0[i] = _mm256_fmadd_ps(av, b0, c0[i]);
            c1[i] = _mm256_fmadd_ps(av, b1, c1[i]);
        }
        p += 8;
    }
    for i in 0..R {
        *out.add(i * ldo) = hsum(c0[i]) + bias[0];
        *out.add(i * ldo + 1) = hsum(c1[i]) + bias[1];
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_dispatch(
    rows: usize,
    a: *const f32,
    lda: usize,
    w0: *const u16,
    w1: *const u16,
    k: usize,
    out: *mut f32,
    ldo: usize,
    bias: [f32; 2],
) {
    match rows {
        1 => tile::<1>(a, lda, w0, w1, k, out, ldo, bias),
        2 => tile::<2>(a, lda, w0, w1, k, out, ldo, bias),
        3 => tile::<3>(a, lda, w0, w1, k, out, ldo, bias),
        4 => tile::<4>(a, lda, w0, w1, k, out, ldo, bias),
        5 => tile::<5>(a, lda, w0, w1, k, out, ldo, bias),
        _ => tile::<6>(a, lda, w0, w1, k, out, ldo, bias),
    }
}

/// One column block `[n0, n1)` of the output, for all `m` rows.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn block(
    a: *const f32,
    m: usize,
    k: usize,
    w: *const u16,
    n0: usize,
    n1: usize,
    bias: Option<&[f32]>,
    out: *mut f32,
    n: usize,
) {
    let mut i = 0;
    while i < m {
        let rows = MR.min(m - i);
        let mut j = n0;
        while j < n1 {
            let b = match bias {
                Some(b) => [b[j], b[j + 1]],
                None => [0.0, 0.0],
            };
            tile_dispatch(
                rows,
                a.add(i * k),
                k,
                w.add(j * k),
                w.add((j + 1) * k),
                k,
                out.add(i * n + j),
                n,
                b,
            );
            j += 2;
        }
        i += rows;
    }
}

/// `out[m, n] = a[m, k] * w[n, k]^T (+ bias)`. `n` must be even and `k` a multiple of 8.
pub fn linear(
    a: &[f32],
    m: usize,
    k: usize,
    w: &[u16],
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    check();
    assert!(k % 8 == 0 && n % 2 == 0, "linear: k%8 and n%2 must be 0");
    assert!(a.len() >= m * k && w.len() >= n * k && out.len() >= m * n);
    if let Some(b) = bias {
        assert!(b.len() >= n);
    }
    // Column blocks small enough for good load balance across heterogeneous cores, large enough
    // that a block's weight rows are re-used from L2 by every row tile.
    let threads = rayon::current_num_threads().max(1);
    let nb = ((n / (threads * 4)).clamp(8, 64)) & !1;
    let nblocks = n.div_ceil(nb);
    let ap = SendPtr(a.as_ptr() as *mut f32);
    let wp = SendPtr(w.as_ptr() as *mut u16);
    let op = SendPtr(out.as_mut_ptr());
    (0..nblocks).into_par_iter().with_min_len(1).for_each(|bi| {
        let (ap, wp, op) = (ap, wp, op);
        let n0 = bi * nb;
        let n1 = (n0 + nb).min(n);
        // SAFETY: feature support asserted above; blocks write disjoint columns of `out`.
        unsafe { block(ap.0, m, k, wp.0, n0, n1, bias, op.0, n) }
    });
}

/// Single dot product of an f32 vector with an f16 row (`len % 8 == 0`).
pub fn dot_f16(a: &[f32], w: &[u16]) -> f32 {
    check();
    assert!(a.len() == w.len() && a.len() % 8 == 0);
    unsafe { dot_f16_inner(a, w) }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_f16_inner(a: &[f32], w: &[u16]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut p = 0;
    while p < a.len() {
        let b = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(p) as *const __m128i));
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(p)), b, acc);
        p += 8;
    }
    hsum(acc)
}

// ---------------------------------------------------------------------------------------------
// layer norm
// ---------------------------------------------------------------------------------------------

/// Row-wise `(x - mean) / sqrt(var + eps) * weight (+ bias)`, matching candle's formulation
/// (biased variance, division rather than reciprocal multiply).
pub fn layer_norm(
    x: &[f32],
    rows: usize,
    d: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
    out: &mut [f32],
) {
    check();
    assert!(d % 8 == 0 && x.len() >= rows * d && out.len() >= rows * d && weight.len() >= d);
    if let Some(b) = bias {
        assert!(b.len() >= d);
    }
    for r in 0..rows {
        unsafe {
            layer_norm_row(
                &x[r * d..(r + 1) * d],
                weight,
                bias,
                eps,
                &mut out[r * d..(r + 1) * d],
            )
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn layer_norm_row(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
    out: &mut [f32],
) {
    let d = x.len();
    let xp = x.as_ptr();
    let mut s = _mm256_setzero_ps();
    let mut i = 0;
    while i < d {
        s = _mm256_add_ps(s, _mm256_loadu_ps(xp.add(i)));
        i += 8;
    }
    let mean = _mm256_set1_ps(hsum(s) / d as f32);
    let mut v = _mm256_setzero_ps();
    i = 0;
    while i < d {
        let c = _mm256_sub_ps(_mm256_loadu_ps(xp.add(i)), mean);
        v = _mm256_fmadd_ps(c, c, v);
        i += 8;
    }
    let denom = _mm256_set1_ps((hsum(v) / d as f32 + eps).sqrt());
    let op = out.as_mut_ptr();
    i = 0;
    while i < d {
        let c = _mm256_sub_ps(_mm256_loadu_ps(xp.add(i)), mean);
        let mut y = _mm256_mul_ps(
            _mm256_div_ps(c, denom),
            _mm256_loadu_ps(weight.as_ptr().add(i)),
        );
        if let Some(b) = bias {
            y = _mm256_add_ps(y, _mm256_loadu_ps(b.as_ptr().add(i)));
        }
        _mm256_storeu_ps(op.add(i), y);
        i += 8;
    }
}

// ---------------------------------------------------------------------------------------------
// element-wise
// ---------------------------------------------------------------------------------------------

/// `dst += src`.
pub fn add_assign(dst: &mut [f32], src: &[f32]) {
    assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}

/// `x[r, :] += row` for every row.
pub fn add_row_broadcast(x: &mut [f32], d: usize, row: &[f32]) {
    assert_eq!(row.len(), d);
    for r in x.chunks_exact_mut(d) {
        for (v, b) in r.iter_mut().zip(row) {
            *v += *b;
        }
    }
}

/// `x = max(x, 0)`.
pub fn relu(x: &mut [f32]) {
    for v in x {
        *v = v.max(0.0);
    }
}

/// candle's `gelu_erf`: `(erf(v / sqrt 2) + 1) * 0.5 * v`, using the same `libm::erff`.
#[inline]
pub fn gelu_erf(v: f32) -> f32 {
    (libm::erff(v * std::f32::consts::FRAC_1_SQRT_2) + 1.0) * 0.5 * v
}

pub fn gelu_erf_inplace(x: &mut [f32]) {
    for v in x {
        *v = gelu_erf(*v);
    }
}

/// ModernBERT GeGLU: `wi` is `[rows, 2*inter]`; `out[r, j] = gelu(wi[r, j]) * wi[r, inter + j]`.
pub fn geglu(wi: &[f32], rows: usize, inter: usize, out: &mut [f32]) {
    assert!(wi.len() >= rows * 2 * inter && out.len() >= rows * inter);
    out[..rows * inter]
        .par_chunks_mut(inter)
        .zip(wi.par_chunks(2 * inter))
        .for_each(|(o, r)| {
            let (a, g) = r.split_at(inter);
            for j in 0..inter {
                o[j] = gelu_erf(a[j]) * g[j];
            }
        });
}

// ---------------------------------------------------------------------------------------------
// attention
// ---------------------------------------------------------------------------------------------

pub const HEAD_DIM: usize = 64;

/// Split a fused `[rows, 3*d]` projection into head-major `q`, `k`, `v` (`[nh, rows, 64]`), applying
/// rotate-half RoPE to q and k when `rope` is given (tables `[>=rows, 32]`) and `scale` to q.
/// Order of operations (rope, then scale) follows the reference.
pub fn split_qkv(
    qkv: &[f32],
    rows: usize,
    nh: usize,
    rope: Option<(&[f32], &[f32])>,
    scale: f32,
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
) {
    let d = nh * HEAD_DIM;
    let half = HEAD_DIM / 2;
    assert!(qkv.len() >= rows * 3 * d && q.len() >= nh * rows * HEAD_DIM);
    q[..nh * rows * HEAD_DIM]
        .par_chunks_mut(rows * HEAD_DIM)
        .zip(k[..nh * rows * HEAD_DIM].par_chunks_mut(rows * HEAD_DIM))
        .zip(v[..nh * rows * HEAD_DIM].par_chunks_mut(rows * HEAD_DIM))
        .enumerate()
        .for_each(|(h, ((qh, kh), vh))| {
            for i in 0..rows {
                let base = i * 3 * d + h * HEAD_DIM;
                let (qs, ks, vs) = (
                    &qkv[base..base + HEAD_DIM],
                    &qkv[base + d..base + d + HEAD_DIM],
                    &qkv[base + 2 * d..base + 2 * d + HEAD_DIM],
                );
                let o = i * HEAD_DIM;
                vh[o..o + HEAD_DIM].copy_from_slice(vs);
                match rope {
                    Some((cos, sin)) => {
                        let (c, s) = (
                            &cos[i * half..(i + 1) * half],
                            &sin[i * half..(i + 1) * half],
                        );
                        for j in 0..half {
                            qh[o + j] = (qs[j] * c[j] - qs[j + half] * s[j]) * scale;
                            qh[o + j + half] = (qs[j] * s[j] + qs[j + half] * c[j]) * scale;
                            kh[o + j] = ks[j] * c[j] - ks[j + half] * s[j];
                            kh[o + j + half] = ks[j] * s[j] + ks[j + half] * c[j];
                        }
                    }
                    None => {
                        for j in 0..HEAD_DIM {
                            qh[o + j] = qs[j] * scale;
                        }
                        kh[o..o + HEAD_DIM].copy_from_slice(ks);
                    }
                }
            }
        });
}

/// Copy `rows` rows of a `[rows, stride]` matrix (columns `col..col + nh*64`) into head-major
/// `[nh, rows, 64]`, multiplying by `scale`.
pub fn to_head_major(
    src: &[f32],
    stride: usize,
    col: usize,
    rows: usize,
    nh: usize,
    scale: f32,
    dst: &mut [f32],
) {
    assert!(
        src.len() >= rows * stride
            && dst.len() >= nh * rows * HEAD_DIM
            && col + nh * HEAD_DIM <= stride
    );
    dst[..nh * rows * HEAD_DIM]
        .par_chunks_mut(rows * HEAD_DIM)
        .enumerate()
        .for_each(|(h, dh)| {
            for i in 0..rows {
                let s = &src[i * stride + col + h * HEAD_DIM..][..HEAD_DIM];
                let o = &mut dh[i * HEAD_DIM..][..HEAD_DIM];
                for c in 0..HEAD_DIM {
                    o[c] = s[c] * scale;
                }
            }
        });
}

/// One query row against keys `[j0, j1)`: softmax(q.k) then the weighted sum of v, into `out[0..64]`.
#[target_feature(enable = "avx2,fma")]
unsafe fn attend_row(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    j0: usize,
    j1: usize,
    scores: &mut [f32],
    out: *mut f32,
) {
    let q0 = _mm256_loadu_ps(q);
    let q1 = _mm256_loadu_ps(q.add(8));
    let q2 = _mm256_loadu_ps(q.add(16));
    let q3 = _mm256_loadu_ps(q.add(24));
    let q4 = _mm256_loadu_ps(q.add(32));
    let q5 = _mm256_loadu_ps(q.add(40));
    let q6 = _mm256_loadu_ps(q.add(48));
    let q7 = _mm256_loadu_ps(q.add(56));

    let cnt = j1 - j0;
    let mut mx = f32::NEG_INFINITY;
    for t in 0..cnt {
        let kp = k.add((j0 + t) * HEAD_DIM);
        let mut a0 = _mm256_mul_ps(q0, _mm256_loadu_ps(kp));
        let mut a1 = _mm256_mul_ps(q1, _mm256_loadu_ps(kp.add(8)));
        let mut a2 = _mm256_mul_ps(q2, _mm256_loadu_ps(kp.add(16)));
        let mut a3 = _mm256_mul_ps(q3, _mm256_loadu_ps(kp.add(24)));
        a0 = _mm256_fmadd_ps(q4, _mm256_loadu_ps(kp.add(32)), a0);
        a1 = _mm256_fmadd_ps(q5, _mm256_loadu_ps(kp.add(40)), a1);
        a2 = _mm256_fmadd_ps(q6, _mm256_loadu_ps(kp.add(48)), a2);
        a3 = _mm256_fmadd_ps(q7, _mm256_loadu_ps(kp.add(56)), a3);
        let s = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        *scores.get_unchecked_mut(t) = s;
        mx = mx.max(s);
    }

    // exp(s - max), vectorised with a scalar tail
    let mxv = _mm256_set1_ps(mx);
    let sp = scores.as_mut_ptr();
    let mut sum = _mm256_setzero_ps();
    let mut t = 0;
    while t + 8 <= cnt {
        let e = exp256(_mm256_sub_ps(_mm256_loadu_ps(sp.add(t)), mxv));
        _mm256_storeu_ps(sp.add(t), e);
        sum = _mm256_add_ps(sum, e);
        t += 8;
    }
    let mut total = hsum(sum);
    while t < cnt {
        let e = (*sp.add(t) - mx).exp();
        *sp.add(t) = e;
        total += e;
        t += 1;
    }

    let mut o0 = _mm256_setzero_ps();
    let mut o1 = _mm256_setzero_ps();
    let mut o2 = _mm256_setzero_ps();
    let mut o3 = _mm256_setzero_ps();
    let mut o4 = _mm256_setzero_ps();
    let mut o5 = _mm256_setzero_ps();
    let mut o6 = _mm256_setzero_ps();
    let mut o7 = _mm256_setzero_ps();
    for t in 0..cnt {
        let p = _mm256_set1_ps(*sp.add(t));
        let vp = v.add((j0 + t) * HEAD_DIM);
        o0 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp), o0);
        o1 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp.add(8)), o1);
        o2 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp.add(16)), o2);
        o3 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp.add(24)), o3);
        o4 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp.add(32)), o4);
        o5 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp.add(40)), o5);
        o6 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp.add(48)), o6);
        o7 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp.add(56)), o7);
    }
    let inv = _mm256_set1_ps(1.0 / total);
    _mm256_storeu_ps(out, _mm256_mul_ps(o0, inv));
    _mm256_storeu_ps(out.add(8), _mm256_mul_ps(o1, inv));
    _mm256_storeu_ps(out.add(16), _mm256_mul_ps(o2, inv));
    _mm256_storeu_ps(out.add(24), _mm256_mul_ps(o3, inv));
    _mm256_storeu_ps(out.add(32), _mm256_mul_ps(o4, inv));
    _mm256_storeu_ps(out.add(40), _mm256_mul_ps(o5, inv));
    _mm256_storeu_ps(out.add(48), _mm256_mul_ps(o6, inv));
    _mm256_storeu_ps(out.add(56), _mm256_mul_ps(o7, inv));
}

/// Fused multi-head attention over head-major `q`/`k`/`v` (`[nh, rows, 64]`). `window` is the
/// one-sided reach of a sliding-window layer (`|i - j| <= window`); `None` is full attention.
/// Writes `out` as `[rows, nh * 64]`. Keys outside the window contribute exactly zero in the
/// reference (`-inf` before softmax), so skipping them is exact.
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: usize,
    nh: usize,
    window: Option<usize>,
    out: &mut [f32],
) {
    attention_rows(q, rows, k, v, rows, nh, window, out)
}

/// [`attention`] with `nq` query rows against `rows` keys. A sliding window is only meaningful when
/// queries and keys are the same positions (`nq == rows`).
pub fn attention_rows(
    q: &[f32],
    nq: usize,
    k: &[f32],
    v: &[f32],
    rows: usize,
    nh: usize,
    window: Option<usize>,
    out: &mut [f32],
) {
    check();
    let d = nh * HEAD_DIM;
    assert!(
        window.is_none() || nq == rows,
        "windowed attention needs nq == rows"
    );
    assert!(rows <= 512, "sequence longer than the score buffer");
    assert!(
        q.len() >= nh * nq * HEAD_DIM
            && k.len() >= nh * rows * HEAD_DIM
            && v.len() >= nh * rows * HEAD_DIM
            && out.len() >= nq * d
    );
    let op = SendPtr(out.as_mut_ptr());
    let chunk = 8usize;
    let per_head = nq.div_ceil(chunk);
    let tasks = nh * per_head;
    (0..tasks).into_par_iter().with_min_len(1).for_each(|t| {
        // Rebind so the closure captures the whole `Sync` wrapper, not its raw-pointer field.
        #[allow(clippy::redundant_locals)]
        let op = op;
        let (h, c) = (t / per_head, t % per_head);
        let (qbase, kbase) = (h * nq * HEAD_DIM, h * rows * HEAD_DIM);
        let mut scores = [0f32; 512];
        let scores = &mut scores[..rows];
        for i in c * chunk..((c + 1) * chunk).min(nq) {
            let (j0, j1) = match window {
                Some(w) => (i.saturating_sub(w), (i + w + 1).min(rows)),
                None => (0, rows),
            };
            // SAFETY: feature support asserted above; each (head, row) owns a disjoint 64-wide slice.
            unsafe {
                attend_row(
                    q.as_ptr().add(qbase + i * HEAD_DIM),
                    k.as_ptr().add(kbase),
                    v.as_ptr().add(kbase),
                    j0,
                    j1,
                    scores,
                    op.0.add(i * d + h * HEAD_DIM),
                );
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn f32_to_f16(v: f32) -> u16 {
        // round-to-nearest-even via the F16C instruction under test-only use
        unsafe { _mm_extract_epi16::<0>(_mm256_cvtps_ph::<0>(_mm256_set1_ps(v))) as u16 }
    }

    #[test]
    fn f16_conversion_matches_hardware() {
        assert!(supported());
        for h in (0u32..=0xffff).step_by(1) {
            let h = h as u16;
            let ours = f16_to_f32(h);
            let hw = unsafe { _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(h as i16))) };
            assert!(
                ours.to_bits() == hw.to_bits() || (ours.is_nan() && hw.is_nan()),
                "h={h:#06x} {ours} vs {hw}"
            );
        }
    }

    #[test]
    fn linear_matches_reference() {
        for &(m, k, n) in &[
            (1usize, 8usize, 2usize),
            (7, 64, 10),
            (108, 1024, 96),
            (13, 2624, 34),
            (6, 1024, 4),
        ] {
            let mut r = Rng(m as u64 * 31 + k as u64);
            let a: Vec<f32> = (0..m * k).map(|_| r.next()).collect();
            let w: Vec<u16> = (0..n * k).map(|_| f32_to_f16(r.next() * 0.1)).collect();
            let bias: Vec<f32> = (0..n).map(|_| r.next()).collect();
            let mut out = vec![0f32; m * n];
            linear(&a, m, k, &w, n, Some(&bias), &mut out);
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0f64;
                    for p in 0..k {
                        acc += a[i * k + p] as f64 * f16_to_f32(w[j * k + p]) as f64;
                    }
                    let want = acc + bias[j] as f64;
                    let got = out[i * n + j] as f64;
                    assert!(
                        (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                        "({m},{k},{n}) [{i},{j}] {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn exp_accuracy() {
        let mut worst = 0f32;
        let mut x = -80.0f32;
        while x <= 0.0 {
            let got = unsafe {
                let mut o = [0f32; 8];
                _mm256_storeu_ps(o.as_mut_ptr(), exp256(_mm256_set1_ps(x)));
                o[0]
            };
            let want = x.exp();
            if want > 1e-30 {
                worst = worst.max(((got - want) / want).abs());
            }
            x += 0.0137;
        }
        assert!(worst < 5e-7, "worst relative error {worst}");
    }

    #[test]
    fn layer_norm_matches_reference() {
        let (rows, d) = (5, 1024);
        let mut r = Rng(9);
        let x: Vec<f32> = (0..rows * d).map(|_| r.next() * 3.0 + 0.5).collect();
        let w: Vec<f32> = (0..d).map(|_| r.next() + 1.0).collect();
        let b: Vec<f32> = (0..d).map(|_| r.next()).collect();
        for bias in [None, Some(&b[..])] {
            let mut out = vec![0f32; rows * d];
            layer_norm(&x, rows, d, &w, bias, 1e-5, &mut out);
            for i in 0..rows {
                let row = &x[i * d..(i + 1) * d];
                let mean = row.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
                let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
                for j in 0..d {
                    let mut want = (row[j] as f64 - mean) / (var + 1e-5).sqrt() * w[j] as f64;
                    if let Some(b) = bias {
                        want += b[j] as f64;
                    }
                    assert!(
                        (out[i * d + j] as f64 - want).abs() < 2e-5,
                        "row {i} col {j}"
                    );
                }
            }
        }
    }

    #[test]
    fn attention_matches_reference() {
        let (rows, nh) = (37usize, 3usize);
        let mut r = Rng(5);
        let n = nh * rows * HEAD_DIM;
        let q: Vec<f32> = (0..n).map(|_| r.next()).collect();
        let k: Vec<f32> = (0..n).map(|_| r.next()).collect();
        let v: Vec<f32> = (0..n).map(|_| r.next()).collect();
        for window in [None, Some(5usize), Some(64)] {
            let mut out = vec![0f32; rows * nh * HEAD_DIM];
            attention(&q, &k, &v, rows, nh, window, &mut out);
            for h in 0..nh {
                for i in 0..rows {
                    let js: Vec<usize> = (0..rows)
                        .filter(|&j| {
                            window
                                .is_none_or(|w| (i as i64 - j as i64).unsigned_abs() as usize <= w)
                        })
                        .collect();
                    let s: Vec<f64> = js
                        .iter()
                        .map(|&j| {
                            (0..HEAD_DIM)
                                .map(|c| {
                                    q[(h * rows + i) * HEAD_DIM + c] as f64
                                        * k[(h * rows + j) * HEAD_DIM + c] as f64
                                })
                                .sum()
                        })
                        .collect();
                    let mx = s.iter().cloned().fold(f64::MIN, f64::max);
                    let e: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
                    let z: f64 = e.iter().sum();
                    for c in 0..HEAD_DIM {
                        let want: f64 = js
                            .iter()
                            .zip(&e)
                            .map(|(&j, &e)| e / z * v[(h * rows + j) * HEAD_DIM + c] as f64)
                            .sum();
                        let got = out[i * nh * HEAD_DIM + h * HEAD_DIM + c] as f64;
                        assert!(
                            (got - want).abs() < 1e-5,
                            "win {window:?} h{h} i{i} c{c}: {got} vs {want}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rectangular_attention_matches_square_rows() {
        let (rows, nh) = (29usize, 2usize);
        let mut r = Rng(11);
        let n = nh * rows * HEAD_DIM;
        let q: Vec<f32> = (0..n).map(|_| r.next()).collect();
        let k: Vec<f32> = (0..n).map(|_| r.next()).collect();
        let v: Vec<f32> = (0..n).map(|_| r.next()).collect();
        let mut full = vec![0f32; rows * nh * HEAD_DIM];
        attention(&q, &k, &v, rows, nh, None, &mut full);
        let picks = [3usize, 17, 28];
        let mut qs = vec![0f32; nh * picks.len() * HEAD_DIM];
        for h in 0..nh {
            for (pi, &i) in picks.iter().enumerate() {
                qs[(h * picks.len() + pi) * HEAD_DIM..][..HEAD_DIM]
                    .copy_from_slice(&q[(h * rows + i) * HEAD_DIM..][..HEAD_DIM]);
            }
        }
        let mut part = vec![0f32; picks.len() * nh * HEAD_DIM];
        attention_rows(&qs, picks.len(), &k, &v, rows, nh, None, &mut part);
        for (pi, &i) in picks.iter().enumerate() {
            assert_eq!(
                &part[pi * nh * HEAD_DIM..(pi + 1) * nh * HEAD_DIM],
                &full[i * nh * HEAD_DIM..(i + 1) * nh * HEAD_DIM]
            );
        }
    }

    #[test]
    fn geglu_matches_reference() {
        let (rows, inter) = (3, 40);
        let mut r = Rng(3);
        let wi: Vec<f32> = (0..rows * 2 * inter).map(|_| r.next() * 4.0).collect();
        let mut out = vec![0f32; rows * inter];
        geglu(&wi, rows, inter, &mut out);
        for i in 0..rows {
            for j in 0..inter {
                let a = wi[i * 2 * inter + j];
                let g = wi[i * 2 * inter + inter + j];
                assert_eq!(out[i * inter + j], gelu_erf(a) * g);
            }
        }
    }
}
