#pragma once
#include "gpu/render.h"
#include "ply.h"
#include <vector>

namespace b2c {

struct ConfidenceParams {
  float tau = 0.08f, min_views = 4.f, inmask_lo = 0.3f, inmask_hi = 0.8f;
  float angle_margin_deg = 30.f, angle_soft_deg = 15.f;
  bool facing = false; float graze_deg = 80.f;
};

// Per-splat multi-view confidence: view-independent part computed once on the CPU, per-camera part on the GPU.
struct ConfidenceModel {
  int n = 0;
  DevBuf<float> static_conf, cos_in, cos_out, gate;
  DevBuf<float4> mu, normal, normal_means;
  DevBuf<float> feature;  // [n][3] output for the rasterizer
  float cos_graze = 0.f;
  bool has_facing = false;
  void build(const SplatCloud& cloud, const ConfidenceParams& p);
  // Fills `feature` with [conf, 0, 0] per splat for a camera at `campos`.
  void for_camera(const float* campos, cudaStream_t stream);
  // Everything trusted: feature = [1, 0, 0].
  void build_trusting(int n);
};

}  // namespace b2c
