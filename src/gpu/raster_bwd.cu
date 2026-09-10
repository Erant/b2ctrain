#include "gpu/render.h"
#include "gpu/splat_math.cuh"

namespace b2c {

namespace {

// Per-pixel-parallel backward: reverse replay of the forward compositing, per-splat gradients reduced across the
// tile with warp shuffles into shared accumulators, then one global atomicAdd per lane per splat per tile.
template <bool FEAT, bool HOLLOW>
__global__ void __launch_bounds__(RT_PX) raster_bwd_kernel(
    const uint2* __restrict__ tile_ranges, const uint32_t* __restrict__ sorted_vals,
    const float4* __restrict__ proj0, const float4* __restrict__ proj1, const float4* __restrict__ proj2, const float2* __restrict__ proj3,
    const float4* __restrict__ out_rgba, const float4* __restrict__ out_feat, const uint32_t* __restrict__ last_idx,
    const float4* __restrict__ v_out, const float4* __restrict__ v_feat_in,
    int W, int H, int tiles_x, float3 bg,
    const float* __restrict__ hollow_z, const float* __restrict__ hollow_zfirst, const float* __restrict__ hollow_zpush, float hollow_margin, float hollow_lam, float hollow_front_alpha,
    float* __restrict__ v_splat, uint32_t* __restrict__ vis_flag) {
  __shared__ float4 s0[RT_PX], s1[RT_PX], s2[RT_PX];
  __shared__ float2 s3[RT_PX];
  __shared__ uint32_t s_gid[RT_PX];
  __shared__ float s_grad[RT_PX][GRAD_LANES];
  __shared__ uint32_t s_vis[RT_PX];

  const int tile = blockIdx.x;
  const uint2 range = tile_ranges[tile];
  if (range.y <= range.x) return;
  const int tx = tile % tiles_x, ty = tile / tiles_x;
  const int lx = threadIdx.x % RT_W, ly = threadIdx.x / RT_W;
  const int px = tx * RT_W + lx, py = ty * RT_W + ly;
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
  float Sh = 0.f, z_ref = INFINITY, z_push = -INFINITY;
  if constexpr (HOLLOW) { if (inside) { float zf = hollow_zfirst[pix]; z_ref = isfinite(zf) ? fmaxf(hollow_z[pix], zf) : (zf < 0.f ? hollow_z[pix] : INFINITY); z_push = hollow_zpush[pix]; } }
  const float inv_final_a = 1.f / fmaxf(final_a, 1e-5f);
  const float Wf = (float)W, Hf = (float)H;

  for (uint32_t batch_end = range.y; batch_end > range.x; ) {
    uint32_t batch_start = batch_end > range.x + RT_PX ? batch_end - RT_PX : range.x;
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
        contrib = sigma >= 0.f && sigma <= c.w;
        if (contrib) {
          float gauss = __expf(-sigma);
          float alpha = fminf(ALPHA_MAX, c.y * gauss);
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
          float hh = 0.f, v_sigma_photo = -alpha * v_alpha;  // the growth statistic (g[9]) is photometric only
          if constexpr (HOLLOW) { hh = hollow_h(c.z, z_ref, hollow_margin); float gate = (hollow_front_alpha > 0.f ? fminf(alpha / hollow_front_alpha, 1.f) : 1.f) * (c.z >= z_push ? 1.f : 0.f); v_alpha += hollow_lam * (T_before * hh - Sh * ra * gate); }
          float v_sigma = -alpha * v_alpha;
          // Note dx here is pixel - mean; brush uses mean - pixel with the same formulas, sign cancels in dx*dx
          // but not in the xy gradient: d sigma / d mean = -(conic * d) with d = pixel - mean.
          float vxy_x = -v_sigma * (a.z * dx + a.w * dy);
          float vxy_y = -v_sigma * (a.w * dx + c.x * dy);
          if (c.y * gauss <= ALPHA_MAX) {
            g[2] = 0.5f * v_sigma * dx * dx; g[3] = v_sigma * dx * dy; g[4] = 0.5f * v_sigma * dy * dy;
            g[0] = vxy_x; g[1] = vxy_y;
            g[8] = v_alpha * gauss;
            float px_x = -v_sigma_photo * (a.z * dx + a.w * dy), px_y = -v_sigma_photo * (a.w * dx + c.x * dy);
            float len = sqrtf(px_x * Wf * px_x * Wf + px_y * Hf * px_y * Hf);
            g[9] = len * inv_final_a;
          }
          S.x += cr * vis; S.y += cg * vis; S.z += cb * vis;
          if constexpr (FEAT) { Sf.x += fx * vis; Sf.y += fy * vis; Sf.z += fz * vis; }
          if constexpr (HOLLOW) Sh += hh * vis;
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
  dim3 grid(ctx.n_tiles), block(RT_PX);
  const bool feat = p.feat != FeatureMode::None, hollow = p.hollow_z != nullptr && p.hollow_lam != 0.f;
#define L(F, HO) raster_bwd_kernel<F, HO><<<grid, block, 0, stream>>>(ctx.tile_ranges, ctx.vals_sorted, ctx.proj0, ctx.proj1, ctx.proj2, ctx.proj3, ctx.out_rgba, ctx.out_feat, ctx.last_idx, ctx.v_out, ctx.v_feat, ctx.W, ctx.H, ctx.tiles_x, bg, p.hollow_z, ctx.hollow_zfirst, ctx.hollow_zpush, p.hollow_margin, p.hollow_lam, p.hollow_front_alpha, ctx.v_splat, ctx.vis_flag)
  if (feat) { if (hollow) L(true, true); else L(true, false); }
  else { if (hollow) L(false, true); else L(false, false); }
#undef L
  CUDA_KERNEL_CHECK();
}

}  // namespace b2c
