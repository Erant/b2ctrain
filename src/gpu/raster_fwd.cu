#include "gpu/render.h"
#include "gpu/splat_math.cuh"

namespace b2c {

namespace {

template <bool FEAT, bool BWD>
__global__ void __launch_bounds__(TILE_PX) raster_fwd_kernel(
    uint2* __restrict__ tile_ranges, const uint32_t* __restrict__ sorted_vals,
    const float4* __restrict__ proj0, const float4* __restrict__ proj1, const float4* __restrict__ proj2, const float2* __restrict__ proj3,
    int W, int H, int tiles_x, float3 bg,
    float4* __restrict__ out_rgba, float4* __restrict__ out_feat, uint32_t* __restrict__ last_idx) {
  __shared__ float4 s0[TILE_PX], s1[TILE_PX], s2[TILE_PX];
  __shared__ float2 s3[TILE_PX];
  __shared__ uint32_t s_max_last;
  const int tile = blockIdx.x;
  const int tx = tile % tiles_x, ty = tile / tiles_x;
  const int lx = threadIdx.x % TILE_W, ly = threadIdx.x / TILE_W;
  const int px = tx * TILE_W + lx, py = ty * TILE_W + ly;
  const bool inside = px < W && py < H;
  const float pcx = px + 0.5f, pcy = py + 0.5f;
  uint2 range = tile_ranges[tile];
  if (threadIdx.x == 0) s_max_last = range.x;
  float T = 1.f, r = 0.f, g = 0.f, b = 0.f, f0 = 0.f, f1 = 0.f, f2 = 0.f;
  bool done = !inside;
  uint32_t last = range.x;
  for (uint32_t start = range.x; start < range.y; start += TILE_PX) {
    if (__syncthreads_count(done) == TILE_PX) break;
    uint32_t remaining = min((uint32_t)TILE_PX, range.y - start);
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
      float alpha = fminf(ALPHA_MAX, c.y * __expf(-sigma));
      if (sigma >= 0.f && alpha >= ALPHA_CUTOFF) {
        float next_T = T * (1.f - alpha);
        if (next_T <= T_CUTOFF) { done = true; break; }
        float vis = alpha * T;
        float4 col = s2[t];
        r += fmaxf(col.x, 0.f) * vis; g += fmaxf(col.y, 0.f) * vis; b += fmaxf(col.z, 0.f) * vis;
        if constexpr (FEAT) { float2 fz = s3[t]; f0 += col.w * vis; f1 += fz.x * vis; f2 += fz.y * vis; }
        T = next_T;
        last = start + t + 1;
      }
    }
  }
  if (inside) {
    int pix = py * W + px;
    out_rgba[pix] = make_float4(r + T * bg.x, g + T * bg.y, b + T * bg.z, 1.f - T);
    out_feat[pix] = make_float4(f0, f1, f2, T);
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
  dim3 grid(ctx.n_tiles), block(TILE_PX);
#define L(F, B) raster_fwd_kernel<F, B><<<grid, block, 0, stream>>>(ctx.tile_ranges, ctx.vals_sorted, ctx.proj0, ctx.proj1, ctx.proj2, ctx.proj3, ctx.W, ctx.H, ctx.tiles_x, bg, ctx.out_rgba, ctx.out_feat, ctx.last_idx)
  if (feat) { if (p.bwd_info) L(true, true); else L(true, false); }
  else { if (p.bwd_info) L(false, true); else L(false, false); }
#undef L
  CUDA_KERNEL_CHECK();
}

}  // namespace b2c
