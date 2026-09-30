#include "gpu/cage_open.h"
#include "gpu/splat_math.cuh"
#include "util/log.h"
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace b2c {
namespace {

// Per vertex: the frame's gate angle theta (degrees, b2crig) -> g.
__global__ void vert_gate_kernel(int nv, const float* __restrict__ theta, float start, float end, float* __restrict__ g) {
  int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= nv) return;
  g[a] = end > start ? fminf(fmaxf((theta[a] - start) / (end - start), 0.f), 1.f) : 0.f;
}

// Per splat: g from its triangle's vertices, the blend (t, g) and the opacity shift.
__global__ void apply_kernel(int n, const int* __restrict__ bf, const float2* __restrict__ bb, const int3* __restrict__ faces, const float* __restrict__ gv,
                             const float4* __restrict__ P, float max_do, float4* __restrict__ pos, float* __restrict__ gs, float4* __restrict__ app, float* __restrict__ stats) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int f = bf[i];
  if (f < 0) { gs[i] = 0.f; app[i] = make_float4(0.f, 0.f, 0.f, 0.f); return; }
  int3 t = faces[f]; float2 b = bb[i];
  float g = (1.f - b.x - b.y) * gv[t.x] + b.x * gv[t.y] + b.y * gv[t.z];
  g = fminf(fmaxf(g, 0.f), 1.f);
  float4 p = P[i];
  app[i] = make_float4(p.x, p.y, p.z, g);
  gs[i] = g;
  float dO = fminf(p.w, max_do);
  if (g > 0.f) { float4 q = pos[i]; q.w += g * dO; pos[i] = q; }
  if (stats) { atomicAdd(stats, g); atomicAdd(stats + 1, fabsf(dO)); if (g > 0.f) atomicAdd(stats + 2, 1.f); }
}

__global__ void pre_optim_kernel(int n, const uint32_t* __restrict__ vis, float* __restrict__ v_splat, const float4* __restrict__ app,
                                 float4* __restrict__ gcol, float2* __restrict__ gapp) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  gapp[i] = make_float2(0.f, 0.f);
  float4 a = app[i];
  if (!vis[i] || a.w <= 0.f) { gcol[i] = make_float4(0.f, 0.f, 0.f, 0.f); return; }
  float* vs = v_splat + (size_t)i * GRAD_LANES;
  float3 vc = make_float3(vs[5], vs[6], vs[7]);
  gcol[i] = make_float4(a.w * vc.x, a.w * vc.y, a.w * vc.z, 0.f);
  float k = 1.f - a.w;
  vs[5] = vc.x * k; vs[6] = vc.y * k; vs[7] = vc.z * k;
}

__device__ __forceinline__ float adam1(float& m, float& v, float g, float lr, float b1, float bc1, float bc2, float snr) {
  m = b1 * m + (1.f - b1) * g; v = 0.999f * v + 0.001f * g * g;
  float r = (m * bc1) / (sqrtf(v * bc2) + 1e-15f);   // in [-1, 1]
  if (snr > 0.f) r = copysignf(fmaxf(fabsf(r) - snr, 0.f) / (1.f - snr), r);
  return lr * r;
}

// Lazy Adam on the open state of every splat that received a gradient this step.
__global__ void step_kernel(int n, const float* __restrict__ gs, const float4* __restrict__ app, const float4* __restrict__ gcol, const float2* __restrict__ gapp,
                            float max_do, float lr_col, float lr_do, float b1, float bc1, float bc2, float snr, const float* __restrict__ dc, size_t dc_stride, float reg,
                            float4* __restrict__ P, float4* __restrict__ mP, float4* __restrict__ vP) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = gs[i];
  if (g <= 0.f) return;
  float4 gc = gcol[i]; float ga = gapp[i].x;
  if (gc.x == 0.f && gc.y == 0.f && gc.z == 0.f && ga == 0.f) return;
  float4 p = P[i];
  if (reg > 0.f) { gc.x += reg * p.x; gc.y += reg * p.y; gc.z += reg * p.z; }   // pull towards no offset: only strong evidence (a crease that opens on skin) moves the open state
  float gr = gc.x, gg = gc.y, gb = gc.z;
  float gw = g * ga;
  if (p.w >= max_do && gw < 0.f) gw = 0.f;   // at the cap: no further rise
  if (!isfinite(gr) || !isfinite(gg) || !isfinite(gb) || !isfinite(gw)) return;
  float4 m = mP[i], v = vP[i];
  p.x -= adam1(m.x, v.x, gr, lr_col, b1, bc1, bc2, snr);
  p.y -= adam1(m.y, v.y, gg, lr_col, b1, bc1, bc2, snr);
  p.z -= adam1(m.z, v.z, gb, lr_col, b1, bc1, bc2, snr);
  p.w -= adam1(m.w, v.w, gw, lr_do, b1, bc1, bc2, snr);
  p.x = fminf(fmaxf(p.x, -1.f), 1.f); p.y = fminf(fmaxf(p.y, -1.f), 1.f); p.z = fminf(fmaxf(p.z, -1.f), 1.f);
  p.w = fminf(p.w, max_do);
  P[i] = p; mP[i] = m; vP[i] = v;
}

__global__ void inherit_kernel(int pairs, const uint32_t* __restrict__ par, const uint32_t* __restrict__ chi, int n,
                               float4* __restrict__ P, float4* __restrict__ mP, float4* __restrict__ vP) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= pairs) return;
  int c = (int)chi[k], p = (int)par[k];
  if (c < 0 || c >= n || p < 0 || p >= n || c == p) return;
  P[c] = P[p];
  mP[c] = make_float4(0.f, 0.f, 0.f, 0.f); vP[c] = make_float4(0.f, 0.f, 0.f, 0.f);
}

__global__ void fill_kernel(int n0, int n1, float4* __restrict__ P, float4* __restrict__ mP, float4* __restrict__ vP) {
  int i = n0 + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n1) return;
  P[i] = make_float4(0.f, 0.f, 0.f, 0.f); mP[i] = P[i]; vP[i] = P[i];
}

__global__ void debug_kernel(int n, float4* __restrict__ app) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 a = app[i];
  app[i] = make_float4(1.f, 0.f, 1.f, a.w);
}

}  // namespace

void CageOpen::ensure(int want, cudaStream_t stream) {
  if (want <= cap) return;
  int ncap = cap ? cap : 1024;
  while (ncap < want) ncap *= 2;
  P.reserve(ncap, true, stream); mP.reserve(ncap, true, stream); vP.reserve(ncap, true, stream);
  fill_kernel<<<div_up(ncap - cap, 256), 256, 0, stream>>>(cap, ncap, P, mP, vP);
  CUDA_KERNEL_CHECK();
  gsplat.reserve(ncap); splat_app.reserve(ncap); col_base.reserve(ncap); g_col.reserve(ncap); g_app.reserve(ncap);
  cap = ncap;
}

void CageOpen::init(const CageRig& cage, const SplatCloud& c, cudaStream_t stream) {
  nv = cage.nv; nf = cage.nf;
  if (!cage.has_open_theta) throw std::runtime_error("--cage-open: the cage carries no opening gate (its B2COPEN1 section; b2crig: python -m b2crig.rig.open_gate CAGE)");
  gvert.reserve(nv); gvert.zero(stream);
  // Open states.
  const int n = (int)c.n;
  std::vector<float4> hp(n, make_float4(0.f, 0.f, 0.f, 0.f));
  if (c.has_open) for (int i = 0; i < n; i++) hp[i] = make_float4(c.open[(size_t)i * 4], c.open[(size_t)i * 4 + 1], c.open[(size_t)i * 4 + 2], c.open[(size_t)i * 4 + 3]);
  cap = 0; ensure(n, stream);
  P.upload(hp, stream);
  mP.zero(stream); vP.zero(stream);
  adam_t = 0; on = true;
}

void CageOpen::apply(const CageRig& cage, const Model& m, cudaStream_t stream, bool stats) {
  ensure(m.cap, stream);
  if (cage.frame < 0) throw std::runtime_error("CageOpen::apply before CageRig::pose");
  vert_gate_kernel<<<div_up(nv, 128), 128, 0, stream>>>(nv, cage.open_theta.ptr + (size_t)cage.frame * nv, start, end, gvert);
  CUDA_KERNEL_CHECK();
  DevBuf<float> st;
  if (stats) { st.reserve(3); st.zero(stream); }
  apply_kernel<<<div_up(m.n, 256), 256, 0, stream>>>(m.n, cage.bind_f, cage.bind_b, cage.faces, gvert, P, max_do, cage.pos_view, gsplat, splat_app, stats ? st.ptr : nullptr);
  CUDA_KERNEL_CHECK();
  if (stats) { auto h = st.download(3, stream); int d = std::max(m.n, 1); stat_g = h[0] / d; stat_do = h[1] / d; stat_open = (int)h[2]; }
}

void CageOpen::pre_optim(RenderCtx& ctx, const Model& m, cudaStream_t stream) {
  pre_optim_kernel<<<div_up(m.n, 256), 256, 0, stream>>>(m.n, ctx.vis_flag, ctx.v_splat, splat_app, g_col, g_app);
  CUDA_KERNEL_CHECK();
}

void CageOpen::step(const Model& m, float lr_col, float lr_do, cudaStream_t stream) {
  if (stats_grad) {   // magnitudes of the data gradients, to set the regulariser against (host, rare)
    auto gc = g_col.download(m.n, stream); auto ga = g_app.download(m.n, stream); auto gs = gsplat.download(m.n, stream);
    double sc = 0, so = 0; long c = 0;
    for (int i = 0; i < m.n; i++) if (gs[i] > 0.f && (gc[i].x != 0.f || gc[i].y != 0.f || gc[i].z != 0.f || ga[i].x != 0.f)) { sc += (std::fabs(gc[i].x) + std::fabs(gc[i].y) + std::fabs(gc[i].z)) / 3.0; so += std::fabs(ga[i].x); c++; }
    log_info("Cage open gradients: %ld open splats with a gradient, mean |dL/dt| %.3g per channel, |dL/d opacity logit| %.3g", c, sc / std::max(c, 1L), so / std::max(c, 1L));
    stats_grad = false;
  }
  adam_t++;
  const float bc1 = 1.f / (1.f - std::pow(beta1, (float)adam_t)), bc2 = 1.f / (1.f - std::pow(0.999f, (float)adam_t));
  step_kernel<<<div_up(m.n, 256), 256, 0, stream>>>(m.n, gsplat, splat_app, g_col, g_app, max_do, lr_col, lr_do, beta1, bc1, bc2, snr, m.sh_dc, (size_t)m.cap, reg_col, P, mP, vP);
  CUDA_KERNEL_CHECK();
}

void CageOpen::inherit(const uint32_t* parents, const uint32_t* children, uint32_t pairs, int n, cudaStream_t stream) {
  ensure(n, stream);
  if (pairs == 0) return;
  inherit_kernel<<<div_up((int)pairs, 256), 256, 0, stream>>>((int)pairs, parents, children, n, P, mP, vP);
  CUDA_KERNEL_CHECK();
}

void CageOpen::debug_g(int n, cudaStream_t stream) {
  debug_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, splat_app);
  CUDA_KERNEL_CHECK();
}

void CageOpen::download(SplatCloud& c, cudaStream_t stream) const {
  const int n = (int)c.n;
  auto hp = P.download(n, stream);
  c.has_open = true; c.open.assign((size_t)n * 4, 0.f);
  for (int i = 0; i < n; i++) { c.open[(size_t)i * 4] = hp[i].x; c.open[(size_t)i * 4 + 1] = hp[i].y; c.open[(size_t)i * 4 + 2] = hp[i].z; c.open[(size_t)i * 4 + 3] = hp[i].w; }
}

std::string CageOpen::header_comment() const { return format("b2c.cage_open add %g %g %g", start, end, max_do); }

bool CageOpen::parse_header(const std::vector<std::string>& comments) {
  for (auto& c : comments) {
    if (c.rfind("b2c.cage_open ", 0) != 0) continue;
    std::istringstream ss(c.substr(14)); std::string mode; float s, e, d;
    if (!(ss >> mode)) return false;
    if (mode != "add") throw std::runtime_error("this ply's open_* states were trained with an earlier (blend) --cage-open; retrain it (header: " + c + ")");
    if (ss >> s >> e >> d) { start = s; end = e; max_do = d; return true; }   // older headers add radius, min_dist (b2crig's now)
  }
  return false;
}

}  // namespace b2c
