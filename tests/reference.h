// Double-precision CPU reference of the render + loss pipeline for gradient tests.
#pragma once
#include "ply.h"
#include "dataset/camera.h"
#include <vector>
#include <cstdint>

struct RefParams {
  int W, H;
  b2c::Camera cam;
  double bg[3];
  bool composite, mask, alpha_lane, normals;
  double l1_w, ssim_w, match_alpha_weight, scale, normal_scale;
  const std::vector<uint32_t>* gt; const std::vector<uint32_t>* gtn; const std::vector<uint8_t>* wts;
  const std::vector<float>* hollow_z = nullptr; double hollow_lam = 0, hollow_margin = 0.05;  // hollow loss (per-pixel reference depth)
};
// Returns the total loss (photometric + normal) for the cloud under the given params.
double reference_loss(const b2c::SplatCloud& c, const RefParams& p);
