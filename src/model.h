#pragma once
#include "gpu/util.cuh"
#include "ply.h"

namespace b2c {

// GPU splat model in brush's parameterisation, structure-of-float4s.
//   pos_op : (mean x, y, z, raw opacity logit)
//   quat   : (w, x, y, z), unnormalised
//   lscale : (log sx, log sy, log sz, min-scale floor f)   -- f is a frozen constant, 0 = none
//   sh     : planar, lane (k*3+ch) of splat i, K = (degree+1)^2: the DC lanes 0..2 in sh_dc[lane*cap + i] (fp32);
//            lanes >= 3 in sh_hi[(lane-3)*cap + i] as fp16 when `sh_fp16`, else in sh_hi32 (see ShBuf)
struct Model {
  int n = 0, cap = 0, degree = 3;
  bool sh_fp16 = false;  // choose before upload(): fp16 storage (stochastically rounded updates) for SH bands >= 1
  DevBuf<float4> pos_op, quat, lscale;
  DevBuf<float> sh_dc, sh_hi32; DevBuf<__half> sh_hi;
  // Adam moments (same layouts); v_sh is one scalar per splat (second moment averaged over SH lanes).
  DevBuf<float4> m_pos_op, v_pos_op, m_quat, v_quat, m_lscale, v_lscale;
  DevBuf<float> m_sh_dc, m_sh_hi32, v_sh; DevBuf<__half> m_sh_hi;
  ShBuf sh() const { ShBuf b; b.dc = sh_dc.ptr; b.hi = sh_hi.ptr; b.hi32 = sh_hi32.ptr; b.stride = (size_t)cap; return b; }
  ShBuf m_sh() const { ShBuf b; b.dc = m_sh_dc.ptr; b.hi = m_sh_hi.ptr; b.hi32 = m_sh_hi32.ptr; b.stride = (size_t)cap; return b; }
  // Refine statistics accumulated between refines.
  DevBuf<float> refine_norm, max_screen, vis_count;
  DevBuf<uint32_t> last_step;  // Adam step at which the splat was last updated (lazy sparse Adam)
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
