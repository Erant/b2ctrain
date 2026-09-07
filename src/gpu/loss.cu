#include "gpu/loss.h"
#include "gpu/splat_math.cuh"

namespace b2c {

namespace {

constexpr int HALO = 5;
constexpr int STAT_W = TILE_W + 2 * HALO;   // 26: pixels whose stats are needed
constexpr int IN_W = STAT_W + 2 * HALO;     // 36: input footprint
constexpr float C1 = 0.01f * 0.01f, C2 = 0.03f * 0.03f;

__constant__ float c_gauss[11];

__device__ __forceinline__ float gt_channel(uint32_t packed, int c) { return (float)((packed >> (8 * c)) & 0xffu) * (1.f / 255.f); }

// One block per (tile, channel). Computes the loss map contribution and the per-pixel gradient.
__global__ void __launch_bounds__(TILE_PX) photometric_kernel(
    const float4* __restrict__ pred, const uint32_t* __restrict__ gt, const uint8_t* __restrict__ weights,
    int W, int H, int tiles_x, LossParams lp, float* __restrict__ v_out, float* __restrict__ loss_accum) {
  __shared__ float s_in[2][IN_W * IN_W];            // pred, gt_eff
  __shared__ float s_h[IN_W * STAT_W * 5];          // horizontal pass (rows 0..35, cols 0..25)
  __shared__ float s_v[STAT_W * STAT_W * 5];        // vertical pass / partial maps
  __shared__ float s_red[TILE_PX / 32];

  const int c = blockIdx.y;
  const int tile = blockIdx.x;
  const int tx0 = (tile % tiles_x) * TILE_W, ty0 = (tile / tiles_x) * TILE_W;
  const int tid = threadIdx.x;
  const float bg_c = lp.composite ? lp.bg[c] : 0.f;

  // Load 36x36 footprint (zero outside the image).
  for (int i = tid; i < IN_W * IN_W; i += TILE_PX) {
    int ly = i / IN_W, lx = i % IN_W;
    int gy = ty0 + ly - 2 * HALO, gx = tx0 + lx - 2 * HALO;
    float p = 0.f, g = 0.f;
    if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
      int pix = gy * W + gx;
      float4 pv = pred[pix];
      p = c == 0 ? pv.x : (c == 1 ? pv.y : pv.z);
      uint32_t gp = gt[pix];
      g = gt_channel(gp, c);
      if (lp.composite) g += (1.f - gt_channel(gp, 3)) * bg_c;
    }
    s_in[0][i] = p; s_in[1][i] = g;
  }
  __syncthreads();
  // Horizontal blur: for rows 0..35, output cols 0..25 (input cols lx+0..lx+10).
  for (int i = tid; i < IN_W * STAT_W; i += TILE_PX) {
    int ly = i / STAT_W, ox = i % STAT_W;
    float sx = 0, sx2 = 0, sy = 0, sy2 = 0, sxy = 0;
#pragma unroll
    for (int d = 0; d < 11; d++) {
      float w = c_gauss[d];
      float x = s_in[0][ly * IN_W + ox + d], y = s_in[1][ly * IN_W + ox + d];
      sx += w * x; sx2 += w * x * x; sy += w * y; sy2 += w * y * y; sxy += w * x * y;
    }
    float* o = s_h + i * 5; o[0] = sx; o[1] = sx2; o[2] = sy; o[3] = sy2; o[4] = sxy;
  }
  __syncthreads();
  // Vertical blur -> stats at 26x26, then SSIM and partial maps A, B, Bmu1, C, Cmu2 (scaled by M_p * ssim_w).
  const float norm = lp.scale / (3.f * (float)W * (float)H);
  float loss_local = 0.f;
  for (int i = tid; i < STAT_W * STAT_W; i += TILE_PX) {
    int oy = i / STAT_W, ox = i % STAT_W;
    float st[5] = {0, 0, 0, 0, 0};
#pragma unroll
    for (int d = 0; d < 11; d++) {
      float w = c_gauss[d];
      const float* h = s_h + ((oy + d) * STAT_W + ox) * 5;
#pragma unroll
      for (int k = 0; k < 5; k++) st[k] += w * h[k];
    }
    int gy = ty0 + oy - HALO, gx = tx0 + ox - HALO;
    float A = 0, B = 0, Bm = 0, Cc = 0, Cm = 0;
    if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
      int pix = gy * W + gx;
      uint32_t gp = gt[pix];
      float gt_a = gt_channel(gp, 3);
      float wp = weights ? weights[pix] * (1.f / 255.f) : 1.f;
      float M = wp * (lp.mask ? gt_a : 1.f) * norm;
      float mu1 = st[0], mu2 = st[2];
      float mu1_sq = mu1 * mu1, mu2_sq = mu2 * mu2;
      float s1_raw = st[1] - mu1_sq, s2_raw = st[3] - mu2_sq;
      float sigma1_sq = fmaxf(0.f, s1_raw), sigma2_sq = fmaxf(0.f, s2_raw);
      float sigma12 = st[4] - mu1 * mu2;
      float A1 = 2.f * mu1 * mu2 + C1, A2 = 2.f * sigma12 + C2, B1 = mu1_sq + mu2_sq + C1, B2 = sigma1_sq + sigma2_sq + C2;
      float raw = (A1 * A2) / (B1 * B2);
      float ssim = fminf(fmaxf(raw, -1.f), 1.f);
      // L1 at the centre pixel of this stat position.
      float p = s_in[0][(oy + HALO) * IN_W + ox + HALO], g = s_in[1][(oy + HALO) * IN_W + ox + HALO];
      // Only pixels inside the tile contribute to the loss value (each pixel is counted by exactly one block).
      if (oy >= HALO && oy < HALO + TILE_W && ox >= HALO && ox < HALO + TILE_W)
        loss_local += M * (lp.l1_w * fabsf(p - g) + lp.ssim_w * ssim);
      if (raw > -1.f && raw < 1.f) {
        float k = M * lp.ssim_w;
        float inv_BB = 1.f / (B1 * B2);
        float d_mu1 = 2.f * mu2 * A2 * inv_BB - 2.f * mu1 * ssim / B1;
        float d_s1 = s1_raw > 0.f ? -ssim / B2 : 0.f;
        float d_s12 = 2.f * A1 * inv_BB;
        A = k * d_mu1; B = k * d_s1; Bm = B * mu1; Cc = k * d_s12; Cm = Cc * mu2;
      }
    }
    float* o = s_v + i * 5; o[0] = A; o[1] = B; o[2] = Bm; o[3] = Cc; o[4] = Cm;
  }
  __syncthreads();
  // Second blur (horizontal): rows 0..25, output cols 0..15.
  for (int i = tid; i < STAT_W * TILE_W; i += TILE_PX) {
    int ly = i / TILE_W, ox = i % TILE_W;
    float acc[5] = {0, 0, 0, 0, 0};
#pragma unroll
    for (int d = 0; d < 11; d++) {
      float w = c_gauss[d];
      const float* v = s_v + (ly * STAT_W + ox + d) * 5;
#pragma unroll
      for (int k = 0; k < 5; k++) acc[k] += w * v[k];
    }
    float* o = s_h + i * 5;
#pragma unroll
    for (int k = 0; k < 5; k++) o[k] = acc[k];
  }
  __syncthreads();
  // Vertical: one output pixel per thread.
  {
    int ly = tid / TILE_W, lx = tid % TILE_W;
    int gy = ty0 + ly, gx = tx0 + lx;
    if (gy < H && gx < W) {
      float acc[5] = {0, 0, 0, 0, 0};
#pragma unroll
      for (int d = 0; d < 11; d++) {
        float w = c_gauss[d];
        const float* v = s_h + ((ly + d) * TILE_W + lx) * 5;
#pragma unroll
        for (int k = 0; k < 5; k++) acc[k] += w * v[k];
      }
      int pix = gy * W + gx;
      float p = s_in[0][(ly + 2 * HALO) * IN_W + lx + 2 * HALO], g = s_in[1][(ly + 2 * HALO) * IN_W + lx + 2 * HALO];
      uint32_t gp = gt[pix];
      float gt_a = gt_channel(gp, 3);
      float wp = weights ? weights[pix] * (1.f / 255.f) : 1.f;
      float M = wp * (lp.mask ? gt_a : 1.f) * norm;
      float sgn = p > g ? 1.f : (p < g ? -1.f : 0.f);
      float grad = M * lp.l1_w * sgn + acc[0] + 2.f * p * acc[1] - 2.f * acc[2] + g * acc[3] - acc[4];
      v_out[(size_t)pix * 4 + c] = grad * lp.grad_scale;
      if (c == 0) {
        float va = 0.f;
        if (lp.alpha_lane) {
          float4 pv = pred[pix];
          float da = pv.w - gt_a;
          float sa = da > 0.f ? 1.f : (da < 0.f ? -1.f : 0.f);
          float Ma = wp * lp.match_alpha_weight * lp.scale / ((float)W * (float)H);
          va = Ma * sa * lp.grad_scale;
          loss_local += Ma * fabsf(da);
        }
        v_out[(size_t)pix * 4 + 3] = va;
      }
    }
  }
  // Reduce loss.
  float s = warp_sum(loss_local);
  if ((tid & 31) == 0) s_red[tid >> 5] = s;
  __syncthreads();
  if (tid == 0) {
    float t = 0.f;
    for (int i = 0; i < TILE_PX / 32; i++) t += s_red[i];
    atomicAdd(loss_accum, t);
  }
}

__global__ void normal_loss_kernel(int npix, const float4* __restrict__ feat, const uint32_t* __restrict__ gtn, const uint8_t* __restrict__ weights,
                                   float scale, float grad_scale, float4* __restrict__ v_feat, float* __restrict__ loss_accum) {
  int pix = blockIdx.x * blockDim.x + threadIdx.x;
  float loss = 0.f;
  if (pix < npix) {
    uint32_t g = gtn[pix];
    float4 out = make_float4(0, 0, 0, 0);
    if (((g >> 24) & 0xffu) > 127u) {
      float wp = weights ? weights[pix] * (1.f / 255.f) : 1.f;
      float k = scale * wp;
      // GT decode with the [+X, -Y, -Z] convention.
      float gx = (float)(g & 0xffu) * (2.f / 255.f) - 1.f;
      float gy = 1.f - (float)((g >> 8) & 0xffu) * (2.f / 255.f);
      float gz = 1.f - (float)((g >> 16) & 0xffu) * (2.f / 255.f);
      float4 f = feat[pix];
      float px = f.x, py = f.y, pz = f.z;
      float pl = fmaxf(sqrtf(px * px + py * py + pz * pz), 1e-6f), gl = fmaxf(sqrtf(gx * gx + gy * gy + gz * gz), 1e-6f);
      float dotpg = px * gx + py * gy + pz * gz;
      float cosv = dotpg / (pl * gl);
      loss = k * (fabsf(px - gx) + fabsf(py - gy) + fabsf(pz - gz) + 1.f - cosv);
      float inv_plgl = 1.f / (pl * gl), pl3 = pl * pl * pl;
      auto sgn = [](float v) { return v > 0.f ? 1.f : (v < 0.f ? -1.f : 0.f); };
      float kg = k * grad_scale;
      out.x = kg * (sgn(px - gx) - gx * inv_plgl + px * dotpg / (pl3 * gl));
      out.y = kg * (sgn(py - gy) - gy * inv_plgl + py * dotpg / (pl3 * gl));
      out.z = kg * (sgn(pz - gz) - gz * inv_plgl + pz * dotpg / (pl3 * gl));
    }
    v_feat[pix] = out;
  }
  float s = warp_sum(loss);
  if ((threadIdx.x & 31) == 0 && s != 0.f) atomicAdd(loss_accum, s);
}

// PSNR/SSIM eval: pred quantised to 8 bits, on black background. Accumulates per-pixel-channel sums.
__global__ void __launch_bounds__(TILE_PX) eval_kernel(const float4* __restrict__ pred, const uint32_t* __restrict__ gt, int W, int H, int tiles_x, bool mask_w, float* __restrict__ accum) {
  __shared__ float s_in[2][IN_W * IN_W];
  __shared__ float s_h[IN_W * TILE_W * 5];
  __shared__ float s_red[2][TILE_PX / 32];
  const int c = blockIdx.y, tile = blockIdx.x;
  const int tx0 = (tile % tiles_x) * TILE_W, ty0 = (tile / tiles_x) * TILE_W;
  const int tid = threadIdx.x;
  for (int i = tid; i < IN_W * IN_W; i += TILE_PX) {
    int ly = i / IN_W, lx = i % IN_W;
    int gy = ty0 + ly - 2 * HALO, gx = tx0 + lx - 2 * HALO;
    float p = 0.f, g = 0.f;
    if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
      int pix = gy * W + gx; float4 pv = pred[pix];
      p = c == 0 ? pv.x : (c == 1 ? pv.y : pv.z);
      p = rintf(fminf(fmaxf(p, 0.f), 1.f) * 255.f) / 255.f;
      g = gt_channel(gt[pix], c);
    }
    s_in[0][i] = p; s_in[1][i] = g;
  }
  __syncthreads();
  // Horizontal over rows HALO..HALO+25 -> only rows needed for the 16 outputs: rows 5..30 (26 rows), cols 0..15 of the tile (+HALO offset).
  for (int i = tid; i < STAT_W * TILE_W; i += TILE_PX) {
    int ly = i / TILE_W + HALO, ox = i % TILE_W + HALO;
    float sx = 0, sx2 = 0, sy = 0, sy2 = 0, sxy = 0;
#pragma unroll
    for (int d = 0; d < 11; d++) {
      float w = c_gauss[d]; float x = s_in[0][ly * IN_W + ox + d], y = s_in[1][ly * IN_W + ox + d];
      sx += w * x; sx2 += w * x * x; sy += w * y; sy2 += w * y * y; sxy += w * x * y;
    }
    float* o = s_h + i * 5; o[0] = sx; o[1] = sx2; o[2] = sy; o[3] = sy2; o[4] = sxy;
  }
  __syncthreads();
  float mse = 0.f, ssim_v = 0.f;
  {
    int ly = tid / TILE_W, lx = tid % TILE_W;
    int gy = ty0 + ly, gx = tx0 + lx;
    if (gy < H && gx < W) {
      float st[5] = {0, 0, 0, 0, 0};
#pragma unroll
      for (int d = 0; d < 11; d++) { float w = c_gauss[d]; const float* h = s_h + ((ly + d) * TILE_W + lx) * 5; for (int k = 0; k < 5; k++) st[k] += w * h[k]; }
      float mu1 = st[0], mu2 = st[2], s1 = fmaxf(0.f, st[1] - mu1 * mu1), s2 = fmaxf(0.f, st[3] - mu2 * mu2), s12 = st[4] - mu1 * mu2;
      float ssim = fminf(fmaxf(((2.f * mu1 * mu2 + C1) * (2.f * s12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (s1 + s2 + C2)), -1.f), 1.f);
      float p = s_in[0][(ly + 2 * HALO) * IN_W + lx + 2 * HALO], g = s_in[1][(ly + 2 * HALO) * IN_W + lx + 2 * HALO];
      float d = fabsf(p - g);
      float m = mask_w ? gt_channel(gt[gy * W + gx], 3) : 1.f;
      mse = (d * m) * (d * m); ssim_v = ssim * m;
    }
  }
  float a = warp_sum(mse), b = warp_sum(ssim_v);
  if ((tid & 31) == 0) { s_red[0][tid >> 5] = a; s_red[1][tid >> 5] = b; }
  __syncthreads();
  if (tid == 0) { float ta = 0, tb = 0; for (int i = 0; i < TILE_PX / 32; i++) { ta += s_red[0][i]; tb += s_red[1][i]; } atomicAdd(accum + 4, ta); atomicAdd(accum + 5, tb); }
}

bool g_gauss_ready = false;
void ensure_gauss() {
  if (g_gauss_ready) return;
  float w[11]; float sum = 0;
  for (int i = 0; i < 11; i++) { float x = i - 5.f; w[i] = expf(-x * x / (2.f * 1.5f * 1.5f)); sum += w[i]; }
  for (int i = 0; i < 11; i++) w[i] /= sum;
  CUDA_CHECK(cudaMemcpyToSymbol(c_gauss, w, sizeof(w)));
  g_gauss_ready = true;
}

}  // namespace

void photometric_loss(RenderCtx& ctx, const ViewGPU& view, const LossParams& lp, cudaStream_t stream) {
  ensure_gauss();
  dim3 grid(ctx.n_tiles, 3), block(TILE_PX);
  photometric_kernel<<<grid, block, 0, stream>>>(ctx.out_rgba, view.rgba, view.weights, ctx.W, ctx.H, ctx.tiles_x, lp, (float*)ctx.v_out.ptr, ctx.loss_accum.ptr);
  CUDA_KERNEL_CHECK();
}

__global__ void accumulate_kernel(float* a) { a[2] += a[0] + a[1]; a[3] += 1.f; }
void accumulate_loss(RenderCtx& ctx, cudaStream_t stream) { accumulate_kernel<<<1, 1, 0, stream>>>(ctx.loss_accum.ptr); }

void normal_loss(RenderCtx& ctx, const ViewGPU& view, const LossParams& lp, cudaStream_t stream) {
  int npix = ctx.W * ctx.H;
  normal_loss_kernel<<<div_up(npix, 256), 256, 0, stream>>>(npix, ctx.out_feat, view.normals, view.weights, lp.normal_scale, lp.grad_scale, ctx.v_feat, ctx.loss_accum.ptr + 1);
  CUDA_KERNEL_CHECK();
}

void eval_metrics(RenderCtx& ctx, const ViewGPU& view, bool mask_weighted, cudaStream_t stream) {
  ensure_gauss();
  dim3 grid(ctx.n_tiles, 3), block(TILE_PX);
  eval_kernel<<<grid, block, 0, stream>>>(ctx.out_rgba, view.rgba, ctx.W, ctx.H, ctx.tiles_x, mask_weighted, ctx.loss_accum.ptr);
  CUDA_KERNEL_CHECK();
}

}  // namespace b2c
