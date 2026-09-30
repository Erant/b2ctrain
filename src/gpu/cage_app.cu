#include "gpu/cage_app.h"
#include "gpu/splat_math.cuh"
#include "util/log.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>

namespace b2c {
namespace {

constexpr int O_W1 = 0, O_B1 = O_W1 + APP_HID * APP_IN, O_W2 = O_B1 + APP_HID, O_B2 = O_W2 + APP_HID * APP_HID,
              O_W3 = O_B2 + APP_HID, O_B3 = O_W3 + APP_OUT * APP_HID;
static_assert(O_B3 + APP_OUT == APP_NP, "parameter layout");
constexpr float LOG_S_MAX = 2.f;   // |log principal stretch| is clamped here (degenerate posed triangles)

__device__ __forceinline__ float silu(float a) { return a / (1.f + __expf(-a)); }
__device__ __forceinline__ float silu_d(float a) { float s = 1.f / (1.f + __expf(-a)); return s * (1.f + a * (1.f - s)); }

// Canonical-frame edge matrix of triangle (a, b, c): columns are the edges b - a, c - a in the frame
// (e1 = unit(b - a), e2 = n x e1), i.e. upper triangular [[|u|, w.e1], [0, w.e2]]. Returns false when degenerate.
__device__ __forceinline__ bool edge_matrix(float3 a, float3 b, float3 c, float& p, float& q, float& s) {
  float3 u = b - a, w = c - a;
  float3 nn = make_float3(u.y * w.z - u.z * w.y, u.z * w.x - u.x * w.z, u.x * w.y - u.y * w.x);
  float nl = len3(nn), ul = len3(u);
  if (!(nl > 1e-20f) || !(ul > 1e-12f)) return false;
  float3 e1 = u * (1.f / ul), n = nn * (1.f / nl);
  float3 e2 = make_float3(n.y * e1.z - n.z * e1.y, n.z * e1.x - n.x * e1.z, n.x * e1.y - n.y * e1.x);
  p = ul; q = dot3(w, e1); s = dot3(w, e2);
  return s > 1e-12f;
}

// Per face: log principal stretches (s1 >= s2) of the canonical -> posed deformation gradient F = E E0^-1.
__global__ void face_feat_kernel(int nf, const int3* __restrict__ faces, const float3* __restrict__ v, const float4* __restrict__ inv0,
                                 const float* __restrict__ area0, float2* __restrict__ out) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= nf) return;
  float2 r = make_float2(0.f, 0.f);
  int3 t = faces[f]; float p, q, s;
  if (area0[f] > 0.f && edge_matrix(v[t.x], v[t.y], v[t.z], p, q, s)) {
    float4 iv = inv0[f];   // [[i00, i01], [i10, i11]]
    // F = [[p, q], [0, s]] * inv0
    float f00 = p * iv.x + q * iv.z, f01 = p * iv.y + q * iv.w, f10 = s * iv.z, f11 = s * iv.w;
    float a = f00 * f00 + f10 * f10, b = f00 * f01 + f10 * f11, c = f01 * f01 + f11 * f11;   // F^T F
    float m = 0.5f * (a + c), d = sqrtf(fmaxf(0.25f * (a - c) * (a - c) + b * b, 0.f));
    float l1 = m + d, l2 = fmaxf(m - d, 1e-12f);
    r.x = fminf(fmaxf(0.5f * logf(l1), -LOG_S_MAX), LOG_S_MAX);
    r.y = fminf(fmaxf(0.5f * logf(l2), -LOG_S_MAX), LOG_S_MAX);
  }
  out[f] = r;
}

// Per vertex: canonical-area-weighted mean of its faces' values; optionally also written to feature slot `slot`.
__global__ void gather_kernel(int nv, const int* __restrict__ off, const int* __restrict__ idx, const float* __restrict__ area0,
                              const float* __restrict__ varea, const float2* __restrict__ fv, float2* __restrict__ out,
                              float* __restrict__ feat, int slot) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nv) return;
  float2 s = make_float2(0.f, 0.f);
  for (int j = off[k]; j < off[k + 1]; j++) { int f = idx[j]; float w = area0[f]; float2 x = fv[f]; s.x += w * x.x; s.y += w * x.y; }
  float inv = varea[k] > 0.f ? 1.f / varea[k] : 0.f;
  s.x *= inv; s.y *= inv;
  out[k] = s;
  if (feat) { feat[(size_t)k * APP_FEAT + slot] = s.x; feat[(size_t)k * APP_FEAT + slot + 1] = s.y; }
}

constexpr float OCC_R1 = 0.04f, OCC_R2 = 0.08f;
constexpr int OCC_TILE = 1024;

// Neighbour counts within OCC_R1 / OCC_R2 among the cage vertices that belong to a face (varea > 0). With cnt0:
// the log ratio to the canonical counts into feature slots 6, 7; without: the counts themselves into cnt_out.
__global__ void occ_kernel(int nv, const float3* __restrict__ v, const float* __restrict__ varea, const float2* __restrict__ cnt0,
                           float2* __restrict__ cnt_out, float* __restrict__ feat) {
  __shared__ float3 sv[OCC_TILE];
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  float3 p = k < nv ? v[k] : make_float3(0.f, 0.f, 0.f);
  float n1 = 0.f, n2 = 0.f;
  const float r1 = OCC_R1 * OCC_R1, r2 = OCC_R2 * OCC_R2;
  for (int base = 0; base < nv; base += OCC_TILE) {
    int m = min(OCC_TILE, nv - base);
    __syncthreads();
    for (int j = threadIdx.x; j < m; j += blockDim.x) sv[j] = varea[base + j] > 0.f ? v[base + j] : make_float3(1e9f, 1e9f, 1e9f);
    __syncthreads();
    if (k < nv)
      for (int j = 0; j < m; j++) { float3 d = sv[j] - p; float d2 = dot3(d, d); n1 += d2 < r1 ? 1.f : 0.f; n2 += d2 < r2 ? 1.f : 0.f; }
  }
  if (k >= nv) return;
  if (cnt_out) cnt_out[k] = make_float2(n1, n2);
  if (feat) {
    bool ok = varea[k] > 0.f;
    float2 c0 = cnt0[k];
    feat[(size_t)k * APP_FEAT + 6] = ok ? logf((n1 + 1.f) / (c0.x + 1.f)) : 0.f;
    feat[(size_t)k * APP_FEAT + 7] = ok ? logf((n2 + 1.f) / (c0.y + 1.f)) : 0.f;
  }
}

__global__ void face_mean_kernel(int nf, const int3* __restrict__ faces, const float2* __restrict__ vv, float2* __restrict__ out) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= nf) return;
  int3 t = faces[f]; float2 a = vv[t.x], b = vv[t.y], c = vv[t.z];
  out[f] = make_float2((a.x + b.x + c.x) * (1.f / 3.f), (a.y + b.y + c.y) * (1.f / 3.f));
}

// MLP forward: pre-activations a1, a2, activations h1, h2, output o.
__device__ __forceinline__ void mlp_fwd(const float* P, const float* x, float* a1, float* h1, float* a2, float* h2, float* o) {
#pragma unroll 4
  for (int h = 0; h < APP_HID; h++) {
    float s = P[O_B1 + h];
#pragma unroll
    for (int d = 0; d < APP_IN; d++) s += P[O_W1 + h * APP_IN + d] * x[d];
    a1[h] = s; h1[h] = silu(s);
  }
#pragma unroll 4
  for (int h = 0; h < APP_HID; h++) {
    float s = P[O_B2 + h];
#pragma unroll
    for (int k = 0; k < APP_HID; k++) s += P[O_W2 + h * APP_HID + k] * h1[k];
    a2[h] = s; h2[h] = silu(s);
  }
#pragma unroll
  for (int j = 0; j < APP_OUT; j++) {
    float s = P[O_B3 + j];
#pragma unroll
    for (int k = 0; k < APP_HID; k++) s += P[O_W3 + j * APP_HID + k] * h2[k];
    o[j] = s;
  }
}

__device__ __forceinline__ void warp_add(float v, float* dst) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0 && v != 0.f) atomicAdd(dst, v);
}

// MLP backward for one pass: parameter gradients reduced over the warp into gP, input gradient into gx.
// Every lane of the warp must call it (lanes without a vertex pass go = 0).
__device__ void mlp_bwd(const float* P, const float* x, const float* go, float* gP, float* gx) {
  float a1[APP_HID], h1[APP_HID], a2[APP_HID], h2[APP_HID], o[APP_OUT];
  mlp_fwd(P, x, a1, h1, a2, h2, o);
  float g2[APP_HID], g1[APP_HID];
#pragma unroll
  for (int j = 0; j < APP_OUT; j++) {
    warp_add(go[j], gP + O_B3 + j);
    for (int k = 0; k < APP_HID; k++) warp_add(go[j] * h2[k], gP + O_W3 + j * APP_HID + k);
  }
  for (int k = 0; k < APP_HID; k++) {
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < APP_OUT; j++) s += P[O_W3 + j * APP_HID + k] * go[j];
    g2[k] = s * silu_d(a2[k]);
  }
  for (int h = 0; h < APP_HID; h++) {
    warp_add(g2[h], gP + O_B2 + h);
    for (int k = 0; k < APP_HID; k++) warp_add(g2[h] * h1[k], gP + O_W2 + h * APP_HID + k);
  }
  for (int k = 0; k < APP_HID; k++) {
    float s = 0.f;
    for (int h = 0; h < APP_HID; h++) s += P[O_W2 + h * APP_HID + k] * g2[h];
    g1[k] = s * silu_d(a1[k]);
  }
  for (int h = 0; h < APP_HID; h++) {
    warp_add(g1[h], gP + O_B1 + h);
    for (int d = 0; d < APP_IN; d++) warp_add(g1[h] * x[d], gP + O_W1 + h * APP_IN + d);
  }
  for (int d = 0; d < APP_IN; d++) {
    float s = 0.f;
    for (int h = 0; h < APP_HID; h++) s += P[O_W1 + h * APP_IN + d] * g1[h];
    gx[d] = s;
  }
}

// R = [o0 - o0(rest), o1, o2, o3, o4 - o4(rest), o5 - o5(rest)]: the blend, opacity and scale channels are zero at rest.
__device__ __forceinline__ float dead(float f, float dz) { return copysignf(fmaxf(fabsf(f) - dz, 0.f), f); }

__global__ void vert_fwd_kernel(int nv, const float* __restrict__ Pg, const float* __restrict__ feat, const float* __restrict__ Z, float dz, float* __restrict__ out) {
  __shared__ float P[APP_NP];
  for (int k = threadIdx.x; k < APP_NP; k += blockDim.x) P[k] = Pg[k];
  __syncthreads();
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= nv) return;
  float x[APP_IN], a1[APP_HID], h1[APP_HID], a2[APP_HID], h2[APP_HID], o[APP_OUT], o0[APP_OUT];
  for (int d = 0; d < APP_LAT; d++) x[APP_FEAT + d] = Z[(size_t)v * APP_LAT + d];
  for (int d = 0; d < APP_FEAT; d++) x[d] = 0.f;
  mlp_fwd(P, x, a1, h1, a2, h2, o0);
  for (int d = 0; d < APP_FEAT; d++) x[d] = dead(feat[(size_t)v * APP_FEAT + d], dz);
  mlp_fwd(P, x, a1, h1, a2, h2, o);
  float* r = out + (size_t)v * APP_OUT;
  r[0] = o[0] - o0[0]; r[1] = o[1]; r[2] = o[2]; r[3] = o[3];
  for (int j = 4; j < APP_OUT; j++) r[j] = o[j] - o0[j];
}

__global__ void vert_bwd_kernel(int nv, const float* __restrict__ Pg, const float* __restrict__ feat, const float* __restrict__ Z,
                                float dz, const float* __restrict__ gout, float* __restrict__ gP, float* __restrict__ gZ) {
  __shared__ float P[APP_NP];
  for (int k = threadIdx.x; k < APP_NP; k += blockDim.x) P[k] = Pg[k];
  __syncthreads();
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  float g[APP_OUT];
  bool act = false;
  for (int j = 0; j < APP_OUT; j++) { g[j] = v < nv ? gout[(size_t)v * APP_OUT + j] : 0.f; act = act || g[j] != 0.f; }
  if (!__any_sync(0xffffffffu, act)) return;   // warp-uniform
  float x[APP_IN], gx[APP_IN], gz[APP_LAT];
  for (int d = 0; d < APP_LAT; d++) x[APP_FEAT + d] = v < nv ? Z[(size_t)v * APP_LAT + d] : 0.f;
  for (int d = 0; d < APP_FEAT; d++) x[d] = v < nv ? dead(feat[(size_t)v * APP_FEAT + d], dz) : 0.f;
  mlp_bwd(P, x, g, gP, gx);
  for (int d = 0; d < APP_LAT; d++) gz[d] = gx[APP_FEAT + d];
  float g0[APP_OUT];
  for (int j = 0; j < APP_OUT; j++) g0[j] = (j >= 1 && j <= 3) ? 0.f : -g[j];
  for (int d = 0; d < APP_FEAT; d++) x[d] = 0.f;
  mlp_bwd(P, x, g0, gP, gx);
  if (v < nv) for (int d = 0; d < APP_LAT; d++) gZ[(size_t)v * APP_LAT + d] = gz[d] + gx[APP_FEAT + d];
}

__global__ void adam_kernel(int n, float* __restrict__ p, float* __restrict__ m, float* __restrict__ v, float* __restrict__ g,
                            float lr, float bc1, float bc2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float gi = g[i]; g[i] = 0.f;
  if (!isfinite(gi)) gi = 0.f;
  float mi = 0.9f * m[i] + 0.1f * gi, vi = 0.999f * v[i] + 0.001f * gi * gi;
  m[i] = mi; v[i] = vi;
  p[i] -= lr * (mi * bc1) / (sqrtf(vi * bc2) + 1e-15f);
}

__global__ void splat_apply_kernel(int n, const int* __restrict__ bf, const float2* __restrict__ bb, const int3* __restrict__ faces,
                                   const float* __restrict__ vout, float max_do, float max_ds, float max_dp, const float* __restrict__ Rt,
                                   float4* __restrict__ pos, float4* __restrict__ ls,
                                   float4* __restrict__ app, float* __restrict__ dsv, float4* __restrict__ dpv, float* __restrict__ stats) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int f = bf[i];
  if (f < 0) { app[i] = make_float4(0.f, 0.f, 0.f, 0.f); dsv[i] = 0.f; dpv[i] = make_float4(0.f, 0.f, 0.f, 0.f); return; }
  int3 t = faces[f]; float2 b = bb[i];
  float w0 = 1.f - b.x - b.y, w1 = b.x, w2 = b.y;
  float R[APP_OUT];
#pragma unroll
  for (int j = 0; j < APP_OUT; j++) R[j] = w0 * vout[(size_t)t.x * APP_OUT + j] + w1 * vout[(size_t)t.y * APP_OUT + j] + w2 * vout[(size_t)t.z * APP_OUT + j];
  float al = tanhf(R[0]), th = max_ds > 0.f ? tanhf(R[5] / max_ds) : 0.f, ds = max_ds * th;
  float dO = fminf(R[4], max_do);
  app[i] = make_float4(sigmoidf_(R[1]), sigmoidf_(R[2]), sigmoidf_(R[3]), al);
  dsv[i] = th;   // d dS / d R5 = 1 - th^2
  float4 p = pos[i]; p.w += dO;
  if (max_dp > 0.f) {   // offset in the posed triangle frame (e1, e2, n), max_dp tanh(R / max_dp) metres per axis
    float t0 = tanhf(R[6] / max_dp), t1 = tanhf(R[7] / max_dp), t2 = tanhf(R[8] / max_dp);
    const float* Rf = Rt + (size_t)f * 9;
    float lx = max_dp * t0, ly = max_dp * t1, lz = max_dp * t2;
    p.x += Rf[0] * lx + Rf[1] * ly + Rf[2] * lz; p.y += Rf[3] * lx + Rf[4] * ly + Rf[5] * lz; p.z += Rf[6] * lx + Rf[7] * ly + Rf[8] * lz;
    dpv[i] = make_float4(t0, t1, t2, 0.f);
    if (stats) atomicAdd(stats + 3, max_dp * sqrtf(t0 * t0 + t1 * t1 + t2 * t2));
  } else dpv[i] = make_float4(0.f, 0.f, 0.f, 0.f);
  pos[i] = p;
  float4 l = ls[i]; l.x += ds; l.y += ds; l.z += ds; ls[i] = l;
  if (stats) { atomicAdd(stats, fabsf(al)); atomicAdd(stats + 1, fabsf(dO)); atomicAdd(stats + 2, fabsf(ds)); }
}

__global__ void pre_optim_kernel(int n, const uint32_t* __restrict__ vis, float* __restrict__ v_splat, const float4* __restrict__ app,
                                 const float4* __restrict__ cb, float4* __restrict__ gcol, float2* __restrict__ gapp, float3* __restrict__ gpos) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  gapp[i] = make_float2(0.f, 0.f); gpos[i] = make_float3(0.f, 0.f, 0.f);
  if (!vis[i]) { gcol[i] = make_float4(0.f, 0.f, 0.f, 0.f); return; }
  float* vs = v_splat + (size_t)i * GRAD_LANES;
  float4 a = app[i], c = cb[i];
  float3 vc = make_float3(vs[5], vs[6], vs[7]);
  float ga = vc.x * (a.x - c.x) + vc.y * (a.y - c.y) + vc.z * (a.z - c.z);
  gcol[i] = make_float4(a.w * vc.x, a.w * vc.y, a.w * vc.z, ga);
  float k = 1.f - a.w;
  vs[5] = vc.x * k; vs[6] = vc.y * k; vs[7] = vc.z * k;
}

__global__ void scatter_kernel(int n, const int* __restrict__ bf, const float2* __restrict__ bb, const int3* __restrict__ faces,
                               const float4* __restrict__ app, const float* __restrict__ dsv, const float4* __restrict__ gcol,
                               const float2* __restrict__ gapp, const float3* __restrict__ gpos, const float4* __restrict__ dpv, const float* __restrict__ Rt,
                               const float* __restrict__ vout, float max_do, bool ds_on, bool dp_on, float pen_do, float pen_ds, float* __restrict__ gout) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int f = bf[i];
  if (f < 0) return;
  float4 gc = gcol[i]; float2 ga = gapp[i]; float3 gp = dp_on ? gpos[i] : make_float3(0.f, 0.f, 0.f);
  if (gc.x == 0.f && gc.y == 0.f && gc.z == 0.f && gc.w == 0.f && ga.x == 0.f && ga.y == 0.f && gp.x == 0.f && gp.y == 0.f && gp.z == 0.f) return;
  float4 a = app[i]; float th = dsv[i];
  int3 t = faces[f]; float2 b = bb[i];
  float w[3] = {1.f - b.x - b.y, b.x, b.y}; int vv[3] = {t.x, t.y, t.z};
  float r4 = w[0] * vout[(size_t)t.x * APP_OUT + 4] + w[1] * vout[(size_t)t.y * APP_OUT + 4] + w[2] * vout[(size_t)t.z * APP_OUT + 4];
  float g[APP_OUT] = {gc.w * (1.f - a.w * a.w), gc.x * a.x * (1.f - a.x), gc.y * a.y * (1.f - a.y), gc.z * a.z * (1.f - a.z),
                      r4 < max_do ? ga.x : 0.f, ds_on ? ga.y * (1.f - th * th) : 0.f, 0.f, 0.f, 0.f};
  // Rise penalties (visible splats): a constant pull against any opacity / scale increase (the haze of faint grown splats).
  if (pen_do > 0.f && r4 > 0.f && r4 < max_do) g[4] += pen_do;
  if (pen_ds > 0.f && ds_on && th > 0.f) g[5] += pen_ds * (1.f - th * th);
  if (dp_on) {   // R^T g, through max_dp tanh(R / max_dp): d/dR = 1 - t^2
    const float* Rf = Rt + (size_t)f * 9; float4 d = dpv[i];
    g[6] = (Rf[0] * gp.x + Rf[3] * gp.y + Rf[6] * gp.z) * (1.f - d.x * d.x);
    g[7] = (Rf[1] * gp.x + Rf[4] * gp.y + Rf[7] * gp.z) * (1.f - d.y * d.y);
    g[8] = (Rf[2] * gp.x + Rf[5] * gp.y + Rf[8] * gp.z) * (1.f - d.z * d.z);
  }
  for (int j = 0; j < APP_OUT; j++) if (!isfinite(g[j])) return;
  for (int k = 0; k < 3; k++)
    for (int j = 0; j < APP_OUT; j++) atomicAdd(gout + (size_t)vv[k] * APP_OUT + j, w[k] * g[j]);
}

// Output L2 at the vertices: dL/dR += lam R on the rest-subtracted channels (blend, opacity, scale, offset).
__global__ void out_reg_kernel(int nv, const float* __restrict__ vout, float lam, float* __restrict__ gout) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nv * APP_OUT) return;
  int j = k % APP_OUT;
  if (j >= 1 && j <= 3) return;
  float r = vout[k];
  if (r != 0.f) gout[k] += lam * r;
}

__global__ void decay_kernel(int n, float* __restrict__ p, float f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] *= f;
}

__global__ void debug_kernel(int n, float4* __restrict__ app) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 a = app[i];
  app[i] = make_float4(1.f, 0.f, 1.f, fabsf(a.w));
}

}  // namespace

void CageApp::debug_alpha(int n, cudaStream_t stream) {
  debug_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, splat_app);
  CUDA_KERNEL_CHECK();
}

void CageApp::init(const CageRig& cage, uint32_t seed, cudaStream_t stream) {
  nv = cage.nv; nf = cage.nf;
  auto fc = cage.faces.download(nf, stream);
  auto v0 = cage.verts0.download(nv, stream);
  std::vector<int> off(nv + 1, 0);
  for (auto& t : fc) { off[t.x + 1]++; off[t.y + 1]++; off[t.z + 1]++; }
  for (int k = 0; k < nv; k++) off[k + 1] += off[k];
  std::vector<int> idx(off[nv]), fill(off.begin(), off.end() - 1);
  for (int f = 0; f < nf; f++) { const int3 t = fc[f]; idx[fill[t.x]++] = f; idx[fill[t.y]++] = f; idx[fill[t.z]++] = f; }
  std::vector<float4> inv(nf); std::vector<float> area(nf, 0.f), varea(nv, 0.f);
  for (int f = 0; f < nf; f++) {
    const int3 t = fc[f];
    const float3 a = v0[t.x], b = v0[t.y], c = v0[t.z];
    const double ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z, wx = c.x - a.x, wy = c.y - a.y, wz = c.z - a.z;
    const double nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
    const double nl = std::sqrt(nx * nx + ny * ny + nz * nz), ul = std::sqrt(ux * ux + uy * uy + uz * uz);
    inv[f] = make_float4(0.f, 0.f, 0.f, 0.f);
    if (!(nl > 1e-20) || !(ul > 1e-12)) continue;
    const double e1x = ux / ul, e1y = uy / ul, e1z = uz / ul, n0 = nx / nl, n1 = ny / nl, n2 = nz / nl;
    const double e2x = n1 * e1z - n2 * e1y, e2y = n2 * e1x - n0 * e1z, e2z = n0 * e1y - n1 * e1x;
    const double p = ul, q = wx * e1x + wy * e1y + wz * e1z, s = wx * e2x + wy * e2y + wz * e2z;
    if (!(s > 1e-12)) continue;
    inv[f] = make_float4((float)(1.0 / p), (float)(-q / (p * s)), 0.f, (float)(1.0 / s));
    area[f] = (float)(0.5 * nl);
    varea[t.x] += area[f]; varea[t.y] += area[f]; varea[t.z] += area[f];
  }
  vf_off.upload(off, stream); vf_idx.upload(idx, stream); tri_inv0.upload(inv, stream);
  tri_area0.upload(area, stream); vert_area0.upload(varea, stream);
  vcnt0.reserve(nv);
  occ_kernel<<<div_up(nv, 256), 256, 0, stream>>>(nv, cage.verts0, vert_area0, nullptr, vcnt0, nullptr);
  CUDA_KERNEL_CHECK();
  ffeat.reserve(nf); vtmp.reserve(nv); vfeat.reserve((size_t)nv * APP_FEAT); vfeat.zero(stream);
  vout.reserve((size_t)nv * APP_OUT); g_vout.reserve((size_t)nv * APP_OUT); g_vout.zero(stream);
  // Weights: N(0, 1/fan_in); the output layer small, so training starts from the plain splat.
  std::mt19937 rng(seed); std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> hp(APP_NP, 0.f), hz((size_t)nv * APP_LAT);
  for (int k = 0; k < APP_HID * APP_IN; k++) hp[O_W1 + k] = nd(rng) / std::sqrt((float)APP_IN);
  for (int k = 0; k < APP_HID * APP_HID; k++) hp[O_W2 + k] = nd(rng) / std::sqrt((float)APP_HID);
  for (int k = 0; k < APP_OUT * APP_HID; k++) hp[O_W3 + k] = 0.01f * nd(rng) / std::sqrt((float)APP_HID);
  for (auto& z : hz) z = 0.3f * nd(rng);
  P.upload(hp, stream); Z.upload(hz, stream);
  for (auto* b : {&mP, &vP, &gP}) { b->reserve(APP_NP); b->zero(stream); }
  for (auto* b : {&mZ, &vZ, &gZ}) { b->reserve((size_t)nv * APP_LAT); b->zero(stream); }
  adam_t = 0; on = true;
}

bool CageApp::load(const std::string& path, const CageRig& cage, cudaStream_t stream) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[8]; int32_t h[5];
  if (fread(magic, 1, 8, f) != 8 || (memcmp(magic, "B2CAPP01", 8) != 0 && memcmp(magic, "B2CAPP02", 8) != 0 && memcmp(magic, "B2CAPP03", 8) != 0) || fread(h, 4, 5, f) != 5) { fclose(f); throw std::runtime_error("'" + path + "' is not a b2ctrain cage appearance model"); }
  float lim[4] = {0.f, 1e30f, 1.f, 0.f};   // v1: no dead zone, unlimited opacity, max_ds 1; v1/v2: no position channel
  const int nlim = magic[7] == '3' ? 4 : magic[7] == '2' ? 3 : 0;
  if (nlim && fread(lim, 4, nlim, f) != (size_t)nlim) { fclose(f); throw std::runtime_error("'" + path + "' is truncated"); }
  const int nout = h[4];   // v1/v2 files carry 6 outputs; the missing rows are zero
  const int nfeat = h[1];   // older files carry fewer features (6: no occlusion); the missing inputs get zero weights
  if (h[0] != cage.nv || nfeat > APP_FEAT || nfeat < 6 || h[2] != APP_LAT || h[3] != APP_HID || nout > APP_OUT || nout < 6) {
    fclose(f); throw std::runtime_error(format("'%s': layout (nv %d, %d/%d/%d/%d) does not match the cage (nv %d) / this build", path.c_str(), h[0], h[1], h[2], h[3], h[4], cage.nv));
  }
  const int in_f = nfeat + APP_LAT;
  const size_t w1_f = (size_t)APP_HID * in_f, w3_f = w1_f + APP_HID + APP_HID * APP_HID + APP_HID;
  const size_t np_file = w3_f + (size_t)nout * APP_HID + nout;
  std::vector<float> fp(np_file), hp(APP_NP, 0.f), hz((size_t)cage.nv * APP_LAT);
  if (fread(fp.data(), 4, fp.size(), f) != fp.size() || fread(hz.data(), 4, hz.size(), f) != hz.size()) { fclose(f); throw std::runtime_error("'" + path + "' is truncated"); }
  for (int hh = 0; hh < APP_HID; hh++) {
    for (int d = 0; d < nfeat; d++) hp[O_W1 + hh * APP_IN + d] = fp[(size_t)hh * in_f + d];
    for (int d = 0; d < APP_LAT; d++) hp[O_W1 + hh * APP_IN + APP_FEAT + d] = fp[(size_t)hh * in_f + nfeat + d];
  }
  std::copy(fp.begin() + w1_f, fp.begin() + w3_f + (size_t)nout * APP_HID, hp.begin() + O_B1);
  std::copy(fp.begin() + w3_f + (size_t)nout * APP_HID, fp.end(), hp.begin() + O_B3);
  fclose(f);
  init(cage, 0, stream);
  P.upload(hp, stream); Z.upload(hz, stream);
  dz = lim[0]; max_do = lim[1]; max_ds = lim[2]; max_dp = lim[3];
  return true;
}

void CageApp::save(const std::string& path, cudaStream_t stream) const {
  auto hp = P.download(APP_NP, stream); auto hz = Z.download((size_t)nv * APP_LAT, stream);
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write '" + path + "'");
  int32_t h[5] = {nv, APP_FEAT, APP_LAT, APP_HID, APP_OUT};
  float lim[4] = {dz, max_do, max_ds, max_dp};
  fwrite("B2CAPP03", 1, 8, f); fwrite(h, 4, 5, f); fwrite(lim, 4, 4, f);
  fwrite(hp.data(), 4, hp.size(), f); fwrite(hz.data(), 4, hz.size(), f);
  fclose(f);
}

void CageApp::vert_forward(const CageRig& cage, cudaStream_t stream) {
  // Features of the posed frame (cage.verts_view).
  face_feat_kernel<<<div_up(nf, 256), 256, 0, stream>>>(nf, cage.faces, cage.verts_view, tri_inv0, tri_area0, ffeat);
  gather_kernel<<<div_up(nv, 256), 256, 0, stream>>>(nv, vf_off, vf_idx, tri_area0, vert_area0, ffeat, vtmp, vfeat, 0);
  for (int it = 1; it <= 6; it++) {
    face_mean_kernel<<<div_up(nf, 256), 256, 0, stream>>>(nf, cage.faces, vtmp, ffeat);
    gather_kernel<<<div_up(nv, 256), 256, 0, stream>>>(nv, vf_off, vf_idx, tri_area0, vert_area0, ffeat, vtmp,
                                                       it == 2 || it == 6 ? vfeat.ptr : nullptr, it == 2 ? 2 : 4);
  }
  occ_kernel<<<div_up(nv, 256), 256, 0, stream>>>(nv, cage.verts_view, vert_area0, vcnt0, nullptr, vfeat);
  CUDA_KERNEL_CHECK();
  vert_fwd_kernel<<<div_up(nv, 128), 128, 0, stream>>>(nv, P, vfeat, Z, dz, vout);
  CUDA_KERNEL_CHECK();
}

void CageApp::vert_backward(cudaStream_t stream) {
  vert_bwd_kernel<<<div_up(nv, 128), 128, 0, stream>>>(nv, P, vfeat, Z, dz, g_vout, gP, gZ);
  CUDA_KERNEL_CHECK();
  g_vout.zero(stream);
}

void CageApp::apply(CageRig& cage, const Model& m, cudaStream_t stream, bool stats) {
  vert_forward(cage, stream);
  const int n = m.n;
  splat_app.reserve(n); col_base.reserve(n); splat_ds.reserve(n); splat_dp.reserve(n); g_col.reserve(n); g_app.reserve(n); g_pos.reserve(n);
  DevBuf<float> st;
  if (stats) { st.reserve(4); st.zero(stream); }
  splat_apply_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, cage.bind_f, cage.bind_b, cage.faces, vout, max_do, max_ds, max_dp, cage.tri_R,
                                                          cage.pos_view, cage.lscale_view, splat_app, splat_ds, splat_dp, stats ? st.ptr : nullptr);
  CUDA_KERNEL_CHECK();
  if (stats) { auto h = st.download(4, stream); stat_alpha = h[0] / n; stat_do = h[1] / n; stat_ds = h[2] / n; stat_dp = h[3] / n; }
}

void CageApp::pre_optim(RenderCtx& ctx, const Model& m, cudaStream_t stream) {
  pre_optim_kernel<<<div_up(m.n, 256), 256, 0, stream>>>(m.n, ctx.vis_flag, ctx.v_splat, splat_app, col_base, g_col, g_app, g_pos);
  CUDA_KERNEL_CHECK();
}

void CageApp::step(const CageRig& cage, const Model& m, float lr, float lr_lat, cudaStream_t stream) {
  scatter_kernel<<<div_up(m.n, 256), 256, 0, stream>>>(m.n, cage.bind_f, cage.bind_b, cage.faces, splat_app, splat_ds, g_col, g_app, g_pos, splat_dp, cage.tri_R,
                                                        vout, max_do, max_ds > 0.f, max_dp > 0.f, reg_rise_do, reg_rise_ds, g_vout);
  CUDA_KERNEL_CHECK();
  if (stats_grad) {   // magnitudes of the data gradients, to set the regularisers against (host, rare)
    auto ga = g_app.download(m.n, stream); auto gv = g_vout.download((size_t)nv * APP_OUT, stream);
    double so = 0, ss = 0; long c = 0;
    for (auto& x : ga) if (x.x != 0.f || x.y != 0.f) { so += std::fabs(x.x); ss += std::fabs(x.y); c++; }
    double sv[APP_OUT] = {0}; long cv = 0;
    for (int k = 0; k < nv; k++) { bool any = false; for (int j = 0; j < APP_OUT; j++) any |= gv[(size_t)k * APP_OUT + j] != 0.f;
      if (any) { cv++; for (int j = 0; j < APP_OUT; j++) sv[j] += std::fabs(gv[(size_t)k * APP_OUT + j]); } }
    log_info("Cage appearance gradients: %ld visible splats, mean |dL/d opacity logit| %.3g, |dL/d log scale| %.3g; %ld vertices, mean |dL/dR| blend %.3g opacity %.3g scale %.3g offset %.3g",
             c, so / std::max(c, 1L), ss / std::max(c, 1L), cv, sv[0] / std::max(cv, 1L), sv[4] / std::max(cv, 1L), sv[5] / std::max(cv, 1L), sv[6] / std::max(cv, 1L));
    stats_grad = false;
  }
  if (reg_out > 0.f) { out_reg_kernel<<<div_up(nv * APP_OUT, 256), 256, 0, stream>>>(nv, vout, reg_out, g_vout); CUDA_KERNEL_CHECK(); }
  vert_backward(stream);
  adam_t++;
  const float bc1 = 1.f / (1.f - std::pow(0.9f, (float)adam_t)), bc2 = 1.f / (1.f - std::pow(0.999f, (float)adam_t));
  adam_kernel<<<div_up(APP_NP, 256), 256, 0, stream>>>(APP_NP, P, mP, vP, gP, lr, bc1, bc2);
  adam_kernel<<<div_up(nv * APP_LAT, 256), 256, 0, stream>>>(nv * APP_LAT, Z, mZ, vZ, gZ, lr_lat, bc1, bc2);
  if (latent_decay > 0.f) decay_kernel<<<div_up(nv * APP_LAT, 256), 256, 0, stream>>>(nv * APP_LAT, Z, 1.f - lr_lat * latent_decay);
  CUDA_KERNEL_CHECK();
}

}  // namespace b2c
