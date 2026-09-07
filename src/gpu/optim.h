#pragma once
#include "gpu/render.h"

namespace b2c {

struct OptimParams {
  CameraGPU cam;
  bool mip = false;
  int active_sh_degree = 3;
  // Adam
  int t = 1;                 // 1-based step for bias correction
  float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-15f;
  float lr_mean = 0.f;       // already includes the schedule and median-scale factor
  float lr_rot = 2e-3f, lr_scale = 5e-3f, lr_opac = 0.012f, lr_dc = 2e-3f, lr_sh_rest = 2e-4f;
  // MCMC noise
  float noise_weight = 0.f;  // lr_mean * mean_noise_weight (0 disables)
  float noise_clamp = 0.f;   // median_scale
  uint32_t seed = 42, step = 0;
  bool sparse = false;       // skip Adam for splats without gradient
  float* grad_out = nullptr; // debug: write parameter gradients [n][11 + K*3] (pos3, opac, quat4, lscale3, sh) and skip the update
};

// Fused: projection backward (2D grads -> parameter grads), Adam update, noise, stats bookkeeping.
// Consumes and clears ctx.v_splat / ctx.vis_flag.
void optimizer_step(RenderCtx& ctx, Model& m, const OptimParams& op, cudaStream_t stream);

}  // namespace b2c
