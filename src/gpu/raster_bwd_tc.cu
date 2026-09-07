// Tensor-core backward: per-pixel reverse replay writes per-fragment scalars (fp16) for groups of 16 splats;
// WMMA reduces them over the tile's 256 pixels against a per-pixel basis matrix.
#include "gpu/render.h"
#include "gpu/splat_math.cuh"
#include <mma.h>
#include <cuda_fp16.h>

namespace b2c {
using namespace nvcuda;

namespace {

constexpr int GROUP = 16;          // splats per MMA group (M)
constexpr int BATCH = 64;          // splats loaded to shared per batch
constexpr int NTYPES = 4;          // vis, v_alpha*gauss, v_sigma, refine
constexpr int NB = 16;             // basis columns (N)
constexpr float R_SCALE = 1.f / 64.f;

struct SmemLayout {
  static constexpr size_t A = (size_t)NTYPES * GROUP * TILE_PX * sizeof(__half);  // 32768
  static constexpr size_t B = (size_t)TILE_PX * NB * sizeof(__half);              // 8192
  static constexpr size_t C = (size_t)8 * GROUP * NB * sizeof(float);             // 8192 (8 warp partials)
  static constexpr size_t S0 = (size_t)BATCH * sizeof(float4) * 3;                // 3072
  static constexpr size_t S3 = (size_t)BATCH * sizeof(float2);                    // 512
  static constexpr size_t G = (size_t)BATCH * sizeof(uint32_t);                   // 256
  static constexpr size_t total = A + B + C + S0 + S3 + G;
};

template <bool FEAT>
__global__ void __launch_bounds__(TILE_PX, 2) raster_bwd_tc_kernel(
    const uint2* __restrict__ tile_ranges, const uint32_t* __restrict__ sorted_vals,
    const float4* __restrict__ proj0, const float4* __restrict__ proj1, const float4* __restrict__ proj2, const float2* __restrict__ proj3,
    const float4* __restrict__ out_rgba, const float4* __restrict__ out_feat, const uint32_t* __restrict__ last_idx,
    const float4* __restrict__ v_out, const float4* __restrict__ v_feat_in,
    int W, int H, int tiles_x, float3 bg, float inv_gscale,
    float* __restrict__ v_splat, uint32_t* __restrict__ vis_flag) {
  extern __shared__ __align__(128) unsigned char smem[];
  __half* A = (__half*)smem;                                   // [NTYPES][GROUP][TILE_PX]
  __half* Bm = (__half*)(smem + SmemLayout::A);                // [TILE_PX][NB]
  float* Cp = (float*)(smem + SmemLayout::A + SmemLayout::B);  // [8][GROUP][NB]
  float4* s0 = (float4*)(smem + SmemLayout::A + SmemLayout::B + SmemLayout::C);
  float4* s1 = s0 + BATCH; float4* s2 = s1 + BATCH;
  float2* s3 = (float2*)(s2 + BATCH);
  uint32_t* s_gid = (uint32_t*)(s3 + BATCH);

  const int tile = blockIdx.x;
  const uint2 range = tile_ranges[tile];
  if (range.y <= range.x) return;
  const int tx = tile % tiles_x, ty = tile / tiles_x;
  const int lx = threadIdx.x % TILE_W, ly = threadIdx.x / TILE_W;
  const int px = tx * TILE_W + lx, py = ty * TILE_W + ly;
  const bool inside = px < W && py < H;
  const float ox = tx * TILE_W + 8.f, oy = ty * TILE_W + 8.f;   // local origin (tile centre)
  const float u = lx + 0.5f - 8.f, v = ly + 0.5f - 8.f;
  const int pix = inside ? py * W + px : 0;
  const int warp = threadIdx.x >> 5;

  float final_a = 0.f, T = 0.f; uint32_t last = range.x;
  float4 vo = make_float4(0, 0, 0, 0), vf = make_float4(0, 0, 0, 0); float v_o_w = 0.f;
  if (inside) {
    final_a = out_rgba[pix].w; T = out_feat[pix].w; last = last_idx[pix]; vo = v_out[pix];
    if constexpr (FEAT) vf = v_feat_in[pix];
    v_o_w = (vo.w - (bg.x * vo.x + bg.y * vo.y + bg.z * vo.z)) * T;
  }
  // Basis row for this pixel. Values are clamped to the fp16 range (a pixel with no coverage can carry an
  // ill-conditioned normal-loss gradient that never meets a fragment, but inf * 0 would poison the MMA).
  {
    auto h = [](float x) { x = isfinite(x) ? fminf(fmaxf(x, -60000.f), 60000.f) : 0.f; return __float2half(x); };
    __half* b = Bm + threadIdx.x * NB;
    b[0] = h(vo.x); b[1] = h(vo.y); b[2] = h(vo.z);
    b[3] = h(vf.x); b[4] = h(vf.y); b[5] = h(vf.z);
    b[6] = __float2half(1.f); b[7] = __float2half(u); b[8] = __float2half(v);
    b[9] = __float2half(u * u); b[10] = __float2half(u * v); b[11] = __float2half(v * v);
    b[12] = __float2half(1.f); b[13] = __float2half(0.f); b[14] = __float2half(0.f); b[15] = __float2half(0.f);
  }
  float3 S = make_float3(0, 0, 0), Sf = make_float3(0, 0, 0);
  const float inv_final_a = 1.f / fmaxf(final_a, 1e-5f);
  const float Wf = (float)W, Hf = (float)H;

  for (uint32_t batch_end = range.y; batch_end > range.x;) {
    uint32_t batch_start = batch_end > range.x + BATCH ? batch_end - BATCH : range.x;
    uint32_t count = batch_end - batch_start;
    __syncthreads();
    if (threadIdx.x < count) {
      uint32_t gid = sorted_vals[batch_start + threadIdx.x];
      s_gid[threadIdx.x] = gid; s0[threadIdx.x] = proj0[gid]; s1[threadIdx.x] = proj1[gid]; s2[threadIdx.x] = proj2[gid];
      if constexpr (FEAT) s3[threadIdx.x] = proj3[gid];
    }
    __syncthreads();
    int ngroups = (int)((count + GROUP - 1) / GROUP);
    for (int g = ngroups - 1; g >= 0; g--) {
      int gbase = g * GROUP;
      int gcount = min(GROUP, (int)count - gbase);
      // Per-pixel reverse replay over this group's splats.
      for (int j = GROUP - 1; j >= 0; j--) {
        float s_vis = 0.f, s_va = 0.f, s_vs = 0.f, s_r = 0.f;
        if (j < gcount) {
          uint32_t idx = batch_start + gbase + j;
          int sj = gbase + j;
          if (inside && idx < last) {
            float4 a = s0[sj]; float4 c = s1[sj];
            float dx = px + 0.5f - a.x, dy = py + 0.5f - a.y;
            float sigma = 0.5f * (a.z * dx * dx + c.x * dy * dy) + a.w * dx * dy;
            float gauss = __expf(-sigma);
            float alpha = fminf(ALPHA_MAX, c.y * gauss);
            if (sigma >= 0.f && alpha >= ALPHA_CUTOFF) {
              float ra = 1.f / (1.f - alpha);
              float T_before = T * ra;
              float vis = alpha * T_before;
              float4 col = s2[sj];
              float cr = fmaxf(col.x, 0.f), cg = fmaxf(col.y, 0.f), cb = fmaxf(col.z, 0.f);
              float v_alpha = (T_before * cr - S.x * ra) * vo.x + (T_before * cg - S.y * ra) * vo.y + (T_before * cb - S.z * ra) * vo.z + v_o_w * ra;
              float fx = 0.f, fy = 0.f, fz = 0.f;
              if constexpr (FEAT) {
                float2 f2 = s3[sj]; fx = col.w; fy = f2.x; fz = f2.y;
                v_alpha += (T_before * fx - Sf.x * ra) * vf.x + (T_before * fy - Sf.y * ra) * vf.y + (T_before * fz - Sf.z * ra) * vf.z;
              }
              float v_sigma = -alpha * v_alpha;
              s_vis = vis;
              if (c.y * gauss <= ALPHA_MAX) {
                s_va = v_alpha * gauss; s_vs = v_sigma;
                float vxy_x = -v_sigma * (a.z * dx + a.w * dy), vxy_y = -v_sigma * (a.w * dx + c.x * dy);
                s_r = sqrtf(vxy_x * Wf * vxy_x * Wf + vxy_y * Hf * vxy_y * Hf) * inv_final_a * R_SCALE;
              }
              S.x += cr * vis; S.y += cg * vis; S.z += cb * vis;
              if constexpr (FEAT) { Sf.x += fx * vis; Sf.y += fy * vis; Sf.z += fz * vis; }
              T = T_before;
            }
          }
        }
        auto h = [](float x) { x = isfinite(x) ? fminf(fmaxf(x, -60000.f), 60000.f) : 0.f; return __float2half(x); };
        __half* row = A + (size_t)j * TILE_PX + threadIdx.x;
        row[0] = h(s_vis);
        row[(size_t)1 * GROUP * TILE_PX] = h(s_va);
        row[(size_t)2 * GROUP * TILE_PX] = h(s_vs);
        row[(size_t)3 * GROUP * TILE_PX] = h(s_r);
      }
      __syncthreads();
      // MMA: warp w -> type (w & 3), k-half (w >> 2).
      {
        int type = warp & 3, khalf = warp >> 2;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.f);
        const __half* Ab = A + (size_t)type * GROUP * TILE_PX;
#pragma unroll
        for (int ks = 0; ks < 8; ks++) {
          int k0 = khalf * 128 + ks * 16;
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
          wmma::load_matrix_sync(fa, Ab + k0, TILE_PX);
          wmma::load_matrix_sync(fb, Bm + (size_t)k0 * NB, NB);
          wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(Cp + (size_t)warp * GROUP * NB, acc, NB, wmma::mem_row_major);
      }
      __syncthreads();
      if (threadIdx.x < (unsigned)gcount) {
        int j = threadIdx.x, sj = gbase + j;
        auto Cv = [&](int type, int col) { return Cp[(size_t)(type) * GROUP * NB + j * NB + col] + Cp[(size_t)(type + 4) * GROUP * NB + j * NB + col]; };
        float sum_vis = Cv(0, 12);
        if (sum_vis > 0.f) {
          float4 a = s0[sj]; float4 c = s1[sj]; float4 col = s2[sj];
          float mx = a.x - ox, my = a.y - oy;
          float g[GRAD_LANES];
          g[5] = col.x >= 0.f ? Cv(0, 0) : 0.f; g[6] = col.y >= 0.f ? Cv(0, 1) : 0.f; g[7] = col.z >= 0.f ? Cv(0, 2) : 0.f;
          g[10] = Cv(0, 3); g[11] = Cv(0, 4); g[12] = Cv(0, 5);
          g[8] = Cv(1, 6);
          float sv = Cv(2, 6), svu = Cv(2, 7), svv = Cv(2, 8), svuu = Cv(2, 9), svuv = Cv(2, 10), svvv = Cv(2, 11);
          float m1u = svu - mx * sv, m1v = svv - my * sv;              // sum v_sigma * (u - mx), (v - my)
          float m2uu = svuu - 2.f * mx * svu + mx * mx * sv;
          float m2uv = svuv - my * svu - mx * svv + mx * my * sv;
          float m2vv = svvv - 2.f * my * svv + my * my * sv;
          g[2] = 0.5f * m2uu; g[3] = m2uv; g[4] = 0.5f * m2vv;
          g[0] = -(a.z * m1u + a.w * m1v); g[1] = -(a.w * m1u + c.x * m1v);
          g[9] = Cv(3, 6) * (1.f / R_SCALE);
          uint32_t gid = s_gid[sj];
          vis_flag[gid] = 1u;
          float* dst = v_splat + (size_t)gid * GRAD_LANES;
#pragma unroll
          for (int k = 0; k < (int)GRAD_LANES; k++) { float val = g[k] * inv_gscale; if (val != 0.f) atomicAdd(dst + k, val); }
        }
      }
    }
    batch_end = batch_start;
  }
}

bool g_attr_set[2] = {false, false};

}  // namespace

void rasterize_backward_tc(RenderCtx& ctx, const Model& m, const RenderParams& p, float grad_scale, cudaStream_t stream) {
  float3 bg = make_float3(p.bg[0], p.bg[1], p.bg[2]);
  dim3 grid(ctx.n_tiles), block(TILE_PX);
  size_t smem = SmemLayout::total;
  float inv = 1.f / grad_scale;
  if (p.feat != FeatureMode::None) {
    if (!g_attr_set[1]) { CUDA_CHECK(cudaFuncSetAttribute(raster_bwd_tc_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); g_attr_set[1] = true; }
    raster_bwd_tc_kernel<true><<<grid, block, smem, stream>>>(ctx.tile_ranges, ctx.vals_sorted, ctx.proj0, ctx.proj1, ctx.proj2, ctx.proj3, ctx.out_rgba, ctx.out_feat, ctx.last_idx, ctx.v_out, ctx.v_feat, ctx.W, ctx.H, ctx.tiles_x, bg, inv, ctx.v_splat, ctx.vis_flag);
  } else {
    if (!g_attr_set[0]) { CUDA_CHECK(cudaFuncSetAttribute(raster_bwd_tc_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); g_attr_set[0] = true; }
    raster_bwd_tc_kernel<false><<<grid, block, smem, stream>>>(ctx.tile_ranges, ctx.vals_sorted, ctx.proj0, ctx.proj1, ctx.proj2, ctx.proj3, ctx.out_rgba, ctx.out_feat, ctx.last_idx, ctx.v_out, ctx.v_feat, ctx.W, ctx.H, ctx.tiles_x, bg, inv, ctx.v_splat, ctx.vis_flag);
  }
  CUDA_KERNEL_CHECK();
}

}  // namespace b2c
