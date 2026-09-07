#pragma once
#include "gpu/util.cuh"
#include "ply.h"

namespace b2c {

// GPU splat model in brush's parameterisation, structure-of-float4s.
//   pos_op : (mean x, y, z, raw opacity logit)
//   quat   : (w, x, y, z), unnormalised
//   lscale : (log sx, log sy, log sz, min-scale floor f)   -- f is a frozen constant, 0 = none
//   sh     : [n][K][3] coefficient-major, K = (degree+1)^2
struct Model {
  int n = 0, cap = 0, degree = 3;
  DevBuf<float4> pos_op, quat, lscale;
  DevBuf<float> sh;
  // Adam moments (same layouts); v_sh is one scalar per splat (second moment averaged over SH lanes).
  DevBuf<float4> m_pos_op, v_pos_op, m_quat, v_quat, m_lscale, v_lscale;
  DevBuf<float> m_sh, v_sh;
  // Refine statistics accumulated between refines.
  DevBuf<float> refine_norm, max_screen, vis_count;
  int adam_t = 0;

  int K() const { return sh_coeffs_for_degree(degree); }
  // Ensure capacity for `want` splats, preserving contents.
  void reserve(int want, cudaStream_t stream = 0);
  void upload(const SplatCloud& c, cudaStream_t stream = 0);
  SplatCloud download(cudaStream_t stream = 0) const;
  void zero_optimizer(cudaStream_t stream = 0);
  void zero_stats(cudaStream_t stream = 0);
};

}  // namespace b2c
