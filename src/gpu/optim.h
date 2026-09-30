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
  // Articulated per-view deformation: the projection Jacobian and the SH view direction are taken at these posed
  // means (gpu/deform.h) while the update lands on the canonical ones; the gradient is not rotated back.
  const float4* pos_override = nullptr;
  float3* g_pos_out = nullptr;   // when set, receives dL/d(mean) per splat (zeros for invisible ones) and the update proceeds
  // Cage deformation (gpu/cage.h): the posed rotations, log scales and SH frames the view was rendered with (see
  // RenderParams); the projection VJP is taken with them. `frozen`: write g_pos_out and leave the model untouched
  // (a deformation-only fit, where the splats' appearance and canonical geometry are fixed).
  const float4* quat_override = nullptr;
  const float4* lscale_override = nullptr;
  const float4* sh_frame = nullptr;
  bool frozen = false;
  const uint8_t* update_mask = nullptr;
  // Cage deformation, learning the splats themselves (fit-cage --fit-splats): per splat the rotation r its triangle
  // applied (posed = r * canonical; CageRig::sh_frame). The mean and rotation gradients are turned back into the
  // canonical frame (mean: R(r)^T g; rotation: conj(r) * g) before the update; the scales' gradient needs nothing.
  const float4* canon_rot = nullptr;   // when set, only splats with a nonzero entry take the Adam update (fit-cage --fit-appearance)
  bool sparse = false;       // skip Adam for splats without gradient
  // GaussianSpa's opacity penalty 0.5 * rho * (o - z + u)^2 on every splat, visible or not (gpu/sparsify.h); null = off.
  const float* spa_z = nullptr; const float* spa_u = nullptr; float spa_rho = 0.f;
  // Stretch regulariser (cage training): per splat weight w (log of its triangle's largest growth over the poses,
  // times the global weight); every log-scale above stretch_log_tau gets +w on its gradient (a hinge), so splats bound
  // where the body stretches stay small.
  const float* stretch_w = nullptr; float stretch_log_tau = 0.f;
  // Pose-dependent appearance (gpu/cage_app.h): per splat (dL/d posed opacity logit, dL/d posed log scales summed
  // over the axes), written for splats with a gradient (others untouched), before the stretch regulariser.
  float2* g_app_out = nullptr;
  float* grad_out = nullptr; // debug: write parameter gradients [n][11 + K*3] (pos3, opac, quat4, lscale3, sh) and skip the update
};

// Fused: projection backward (2D grads -> parameter grads), Adam update, noise, stats bookkeeping.
// Consumes and clears ctx.v_splat / ctx.vis_flag.
void optimizer_step(RenderCtx& ctx, Model& m, const OptimParams& op, cudaStream_t stream);

}  // namespace b2c
