#pragma once
#include "gpu/render.h"

namespace b2c {

// A training view resident on the GPU.
struct ViewGPU {
  const uint32_t* rgba = nullptr;     // packed RGBA8 (premultiplied for transparent views)
  const uint32_t* normals = nullptr;  // packed RGBA8 normal map or nullptr
  const uint8_t* weights = nullptr;   // per-pixel loss weight or nullptr
  int W = 0, H = 0;
  bool has_alpha = false, masked = false;
  float alpha_coverage = 1.f;
  float normal_count = 0.f;           // sum of weights where the normal mask is set
  const float2* warp = nullptr;       // render-side alignment field (see RenderParams::warp) or nullptr
  int warp_w = 0, warp_h = 0;
};

struct LossParams {
  float l1_w = 0.8f, ssim_w = -0.2f;  // brush: (1 - ssim_weight), -ssim_weight
  float bg[3] = {0, 0, 0};
  bool composite = false;             // gt_eff = gt + (1 - gt.a) * bg (transparent views with alpha)
  bool mask = false;                  // multiply the loss map by gt.a (masked views)
  bool alpha_lane = false;            // add match_alpha_weight * |pred.a - gt.a|
  float match_alpha_weight = 0.1f;
  float scale = 1.f;                  // extra multiplier on the whole photometric loss (1 / coverage)
  float normal_scale = 0.f;           // normal_loss_weight * every / count, 0 = off
  float grad_scale = 1.f;             // multiplies the per-pixel gradients (not the loss value)
};

// Fused L1 + SSIM (+ alpha lane) loss with per-pixel gradient into ctx.v_out. Adds loss to ctx.loss_accum[0].
void photometric_loss(RenderCtx& ctx, const ViewGPU& view, const LossParams& lp, cudaStream_t stream);
// Normal supervision loss (L1 + 1 - cos), gradient into ctx.v_feat, loss into ctx.loss_accum[1].
void normal_loss(RenderCtx& ctx, const ViewGPU& view, const LossParams& lp, cudaStream_t stream);
// Adds loss_accum[0]+[1] into the running sum loss_accum[2] and increments loss_accum[3].
void accumulate_loss(RenderCtx& ctx, cudaStream_t stream);
// Adds lam * sum(ctx.hollow_pen) (the hollow loss value; the gradient is taken in the rasteriser backward) to ctx.loss_accum[0].
void hollow_loss(RenderCtx& ctx, float lam, cudaStream_t stream);
// Eval metrics on the current render: writes [mse_sum, ssim_sum, count] into ctx.loss_accum[4..7].
void eval_metrics(RenderCtx& ctx, const ViewGPU& view, bool mask_weighted, cudaStream_t stream);

}  // namespace b2c
