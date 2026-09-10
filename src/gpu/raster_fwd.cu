#include "gpu/render.h"
#include "gpu/splat_math.cuh"

namespace b2c {

namespace {

template <bool FEAT, bool BWD, bool HOLLOW>
__global__ void __launch_bounds__(RT_PX) raster_fwd_kernel(
    uint2* __restrict__ tile_ranges, const uint32_t* __restrict__ sorted_vals,
    const float4* __restrict__ proj0, const float4* __restrict__ proj1, const float4* __restrict__ proj2, const float2* __restrict__ proj3,
    int W, int H, int tiles_x, float3 bg,
    float4* __restrict__ out_rgba, float4* __restrict__ out_feat, uint32_t* __restrict__ last_idx,
    const float* __restrict__ hollow_z, float hollow_margin, float hollow_tau, float hollow_push_tau, float* __restrict__ hollow_pen, float* __restrict__ hollow_zfirst, float* __restrict__ hollow_zpush) {
  __shared__ float4 s0[RT_PX], s1[RT_PX], s2[RT_PX];
  __shared__ float2 s3[RT_PX];
  __shared__ uint32_t s_max_last;
  const int tile = blockIdx.x;
  const int tx = tile % tiles_x, ty = tile / tiles_x;
  const int lx = threadIdx.x % RT_W, ly = threadIdx.x / RT_W;
  const int px = tx * RT_W + lx, py = ty * RT_W + ly;
  const bool inside = px < W && py < H;
  const float pcx = px + 0.5f, pcy = py + 0.5f;
  uint2 range = tile_ranges[tile];
  if (threadIdx.x == 0) s_max_last = range.x;
  float T = 1.f, r = 0.f, g = 0.f, b = 0.f, f0 = 0.f, f1 = 0.f, f2 = 0.f;
  float pen = 0.f, z_mesh = INFINITY, z_first = INFINITY, z_push = hollow_push_tau > 0.f ? INFINITY : -INFINITY;
  if constexpr (HOLLOW) { if (inside) z_mesh = hollow_z[py * W + px]; }
  bool done = !inside;
  uint32_t last = range.x;
  for (uint32_t start = range.x; start < range.y; start += RT_PX) {
    if (__syncthreads_count(done) == RT_PX) break;
    uint32_t remaining = min((uint32_t)RT_PX, range.y - start);
    if (threadIdx.x < remaining) {
      uint32_t gid = sorted_vals[start + threadIdx.x];
      s0[threadIdx.x] = proj0[gid]; s1[threadIdx.x] = proj1[gid]; s2[threadIdx.x] = proj2[gid];
      if constexpr (FEAT) s3[threadIdx.x] = proj3[gid];
    }
    __syncthreads();
    for (uint32_t t = 0; t < remaining && !done; t++) {
      float4 a = s0[t]; float4 c = s1[t];
      float dx = pcx - a.x, dy = pcy - a.y;
      float sigma = 0.5f * (a.z * dx * dx + c.x * dy * dy) + a.w * dx * dy;
      if (sigma < 0.f || sigma > c.w) continue;  // alpha < 1/255: cannot contribute
      float alpha = fminf(ALPHA_MAX, c.y * __expf(-sigma));
      {
        float next_T = T * (1.f - alpha);
        if (next_T <= T_CUTOFF) { done = true; break; }
        float vis = alpha * T;
        float4 col = s2[t];
        r += fmaxf(col.x, 0.f) * vis; g += fmaxf(col.y, 0.f) * vis; b += fmaxf(col.z, 0.f) * vis;
        if constexpr (FEAT) { float2 fz = s3[t]; f0 += col.w * vis; f1 += fz.x * vis; f2 += fz.y * vis; }
        if constexpr (HOLLOW) {
          // Fragments up to and including the one that completes the first surface lie at or in front of it, so
          // they carry no penalty under the max(mesh, first) reference; only later ones do.
          if (isfinite(z_first)) pen += vis * hollow_h(c.z, fmaxf(z_mesh, z_first), hollow_margin);
          else if (hollow_tau <= 0.f) { z_first = -INFINITY; pen += vis * hollow_h(c.z, z_mesh, hollow_margin); }
          else if (1.f - next_T >= hollow_tau) z_first = c.z;
          if (hollow_push_tau > 0.f && !isfinite(z_push) && 1.f - next_T >= hollow_push_tau) z_push = c.z;
        }
        T = next_T;
        last = start + t + 1;
      }
    }
  }
  if (inside) {
    int pix = py * W + px;
    out_rgba[pix] = make_float4(r + T * bg.x, g + T * bg.y, b + T * bg.z, 1.f - T);
    out_feat[pix] = make_float4(f0, f1, f2, T);
    if constexpr (HOLLOW) { hollow_pen[pix] = pen; hollow_zfirst[pix] = z_first; hollow_zpush[pix] = z_push; }
    if constexpr (BWD) last_idx[pix] = last;
  }
  if constexpr (BWD) {
    atomicMax(&s_max_last, last);
    __syncthreads();
    if (threadIdx.x == 0) tile_ranges[tile].y = s_max_last;
  }
}

}  // namespace

void rasterize_forward(RenderCtx& ctx, const RenderParams& p, cudaStream_t stream) {
  float3 bg = make_float3(p.bg[0], p.bg[1], p.bg[2]);
  bool feat = p.feat != FeatureMode::None;
  dim3 grid(ctx.n_tiles), block(RT_PX);
  const bool hollow = p.hollow_z != nullptr;
#define L(F, B, HO) raster_fwd_kernel<F, B, HO><<<grid, block, 0, stream>>>(ctx.tile_ranges, ctx.vals_sorted, ctx.proj0, ctx.proj1, ctx.proj2, ctx.proj3, ctx.W, ctx.H, ctx.tiles_x, bg, ctx.out_rgba, ctx.out_feat, ctx.last_idx, p.hollow_z, p.hollow_margin, p.hollow_tau, p.hollow_push_tau, ctx.hollow_pen, ctx.hollow_zfirst, ctx.hollow_zpush)
  if (hollow) {
    if (feat) { if (p.bwd_info) L(true, true, true); else L(true, false, true); }
    else { if (p.bwd_info) L(false, true, true); else L(false, false, true); }
  } else {
    if (feat) { if (p.bwd_info) L(true, true, false); else L(true, false, false); }
    else { if (p.bwd_info) L(false, true, false); else L(false, false, false); }
  }
#undef L
  CUDA_KERNEL_CHECK();
}

}  // namespace b2c
