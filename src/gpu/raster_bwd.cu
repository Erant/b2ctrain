#include "gpu/render.h"
#include "gpu/splat_math.cuh"

namespace b2c {

namespace {

// Per-pixel-parallel backward: reverse replay of the forward compositing, per-splat gradients reduced across the
// tile with warp shuffles into shared accumulators, then one global atomicAdd per lane per splat per tile.
template <bool FEAT>
__global__ void __launch_bounds__(TILE_PX) raster_bwd_kernel(
    const uint2* __restrict__ tile_ranges, const uint32_t* __restrict__ sorted_vals,
    const float4* __restrict__ proj0, const float4* __restrict__ proj1, const float4* __restrict__ proj2, const float2* __restrict__ proj3,
    const float4* __restrict__ out_rgba, const float4* __restrict__ out_feat, const uint32_t* __restrict__ last_idx,
    const float4* __restrict__ v_out, const float4* __restrict__ v_feat_in,
    int W, int H, int tiles_x, float3 bg,
    float* __restrict__ v_splat, uint32_t* __restrict__ vis_flag) {
  __shared__ float4 s0[TILE_PX], s1[TILE_PX], s2[TILE_PX];
  __shared__ float2 s3[TILE_PX];
  __shared__ uint32_t s_gid[TILE_PX];
  __shared__ float s_grad[TILE_PX][GRAD_LANES];
  __shared__ uint32_t s_vis[TILE_PX];

  const int tile = blockIdx.x;
  const uint2 range = tile_ranges[tile];
  if (range.y <= range.x) return;
  const int tx = tile % tiles_x, ty = tile / tiles_x;
  const int lx = threadIdx.x % TILE_W, ly = threadIdx.x / TILE_W;
  const int px = tx * TILE_W + lx, py = ty * TILE_W + ly;
  const bool inside = px < W && py < H;
  const float pcx = px + 0.5f, pcy = py + 0.5f;
  const int pix = inside ? py * W + px : 0;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

  float final_a = 0.f, T = 0.f;
  uint32_t last = range.x;
  float4 vo = make_float4(0, 0, 0, 0), vf = make_float4(0, 0, 0, 0);
  float v_o_w = 0.f;
  if (inside) {
    final_a = out_rgba[pix].w; T = out_feat[pix].w; last = last_idx[pix];
    vo = v_out[pix];
    if constexpr (FEAT) vf = v_feat_in[pix];
    v_o_w = (vo.w - (bg.x * vo.x + bg.y * vo.y + bg.z * vo.z)) * T;
  }
  float3 S = make_float3(0, 0, 0), Sf = make_float3(0, 0, 0);
  const float inv_final_a = 1.f / fmaxf(final_a, 1e-5f);
  const float Wf = (float)W, Hf = (float)H;

  for (uint32_t batch_end = range.y; batch_end > range.x; ) {
    uint32_t batch_start = batch_end > range.x + TILE_PX ? batch_end - TILE_PX : range.x;
    uint32_t count = batch_end - batch_start;
    __syncthreads();  // previous batch's shared use complete
    if (threadIdx.x < count) {
      uint32_t gid = sorted_vals[batch_start + threadIdx.x];
      s_gid[threadIdx.x] = gid;
      s0[threadIdx.x] = proj0[gid]; s1[threadIdx.x] = proj1[gid]; s2[threadIdx.x] = proj2[gid];
      if constexpr (FEAT) s3[threadIdx.x] = proj3[gid];
#pragma unroll
      for (int k = 0; k < (int)GRAD_LANES; k++) s_grad[threadIdx.x][k] = 0.f;
      s_vis[threadIdx.x] = 0;
    }
    __syncthreads();
    for (int j = (int)count - 1; j >= 0; j--) {
      uint32_t idx = batch_start + j;
      float g[GRAD_LANES];
#pragma unroll
      for (int k = 0; k < (int)GRAD_LANES; k++) g[k] = 0.f;
      bool contrib = inside && idx < last;
      if (contrib) {
        float4 a = s0[j]; float4 c = s1[j];
        float dx = pcx - a.x, dy = pcy - a.y;
        float sigma = 0.5f * (a.z * dx * dx + c.x * dy * dy) + a.w * dx * dy;
        float gauss = __expf(-sigma);
        float alpha = fminf(ALPHA_MAX, c.y * gauss);
        contrib = sigma >= 0.f && alpha >= ALPHA_CUTOFF;
        if (contrib) {
          float ra = 1.f / (1.f - alpha);
          float T_before = T * ra;
          float vis = alpha * T_before;
          float4 col = s2[j];
          float cr = fmaxf(col.x, 0.f), cg = fmaxf(col.y, 0.f), cb = fmaxf(col.z, 0.f);
          g[5] = col.x >= 0.f ? vis * vo.x : 0.f;
          g[6] = col.y >= 0.f ? vis * vo.y : 0.f;
          g[7] = col.z >= 0.f ? vis * vo.z : 0.f;
          float v_alpha = (T_before * cr - S.x * ra) * vo.x + (T_before * cg - S.y * ra) * vo.y + (T_before * cb - S.z * ra) * vo.z + v_o_w * ra;
          float fx = 0.f, fy = 0.f, fz = 0.f;
          if constexpr (FEAT) {
            float2 f2 = s3[j]; fx = col.w; fy = f2.x; fz = f2.y;
            g[10] = vis * vf.x; g[11] = vis * vf.y; g[12] = vis * vf.z;
            v_alpha += (T_before * fx - Sf.x * ra) * vf.x + (T_before * fy - Sf.y * ra) * vf.y + (T_before * fz - Sf.z * ra) * vf.z;
          }
          float v_sigma = -alpha * v_alpha;
          // Note dx here is pixel - mean; brush uses mean - pixel with the same formulas, sign cancels in dx*dx
          // but not in the xy gradient: d sigma / d mean = -(conic * d) with d = pixel - mean.
          float vxy_x = -v_sigma * (a.z * dx + a.w * dy);
          float vxy_y = -v_sigma * (a.w * dx + c.x * dy);
          if (c.y * gauss <= ALPHA_MAX) {
            g[2] = 0.5f * v_sigma * dx * dx; g[3] = v_sigma * dx * dy; g[4] = 0.5f * v_sigma * dy * dy;
            g[0] = vxy_x; g[1] = vxy_y;
            g[8] = v_alpha * gauss;
            float len = sqrtf(vxy_x * Wf * vxy_x * Wf + vxy_y * Hf * vxy_y * Hf);
            g[9] = len * inv_final_a;
          }
          S.x += cr * vis; S.y += cg * vis; S.z += cb * vis;
          if constexpr (FEAT) { Sf.x += fx * vis; Sf.y += fy * vis; Sf.z += fz * vis; }
          T = T_before;
        }
      }
      unsigned mask = __ballot_sync(0xffffffffu, contrib);
      if (mask) {
        if (contrib && lane == __ffs(mask) - 1) s_vis[j] = 1;
#pragma unroll
        for (int k = 0; k < (int)GRAD_LANES; k++) {
          float v = warp_sum(g[k]);
          if (lane == 0 && v != 0.f) atomicAdd(&s_grad[j][k], v);
        }
      }
    }
    __syncthreads();
    if (threadIdx.x < count) {
      uint32_t gid = s_gid[threadIdx.x];
      if (s_vis[threadIdx.x]) {
        vis_flag[gid] = 1u;
        float* dst = v_splat + (size_t)gid * GRAD_LANES;
#pragma unroll
        for (int k = 0; k < (int)GRAD_LANES; k++) { float v = s_grad[threadIdx.x][k]; if (v != 0.f) atomicAdd(dst + k, v); }
      }
    }
    batch_end = batch_start;
  }
  (void)warp;
}

}  // namespace

void rasterize_backward(RenderCtx& ctx, const Model& m, const RenderParams& p, cudaStream_t stream) {
  float3 bg = make_float3(p.bg[0], p.bg[1], p.bg[2]);
  dim3 grid(ctx.n_tiles), block(TILE_PX);
  if (p.feat != FeatureMode::None)
    raster_bwd_kernel<true><<<grid, block, 0, stream>>>(ctx.tile_ranges, ctx.vals_sorted, ctx.proj0, ctx.proj1, ctx.proj2, ctx.proj3, ctx.out_rgba, ctx.out_feat, ctx.last_idx, ctx.v_out, ctx.v_feat, ctx.W, ctx.H, ctx.tiles_x, bg, ctx.v_splat, ctx.vis_flag);
  else
    raster_bwd_kernel<false><<<grid, block, 0, stream>>>(ctx.tile_ranges, ctx.vals_sorted, ctx.proj0, ctx.proj1, ctx.proj2, ctx.proj3, ctx.out_rgba, ctx.out_feat, ctx.last_idx, ctx.v_out, ctx.v_feat, ctx.W, ctx.H, ctx.tiles_x, bg, ctx.v_splat, ctx.vis_flag);
  CUDA_KERNEL_CHECK();
}

}  // namespace b2c
