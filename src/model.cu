#include "model.h"
#include <type_traits>

namespace b2c {

void Model::reserve(int want, cudaStream_t stream) {
  if (want <= cap) return;
  int ncap = cap ? cap : 1024;
  while (ncap < want) ncap *= 2;
  int K = this->K();
  auto grow4 = [&](DevBuf<float4>& b) { b.reserve(ncap, true, stream); };
  auto growf = [&](DevBuf<float>& b) { b.reserve((size_t)ncap, true, stream); };
  // Planar [lanes][cap] buffers: re-layout with a strided 2D copy.
  auto grow_planar = [&](auto& b, int lanes) {
    using T = std::remove_reference_t<decltype(*b.ptr)>;
    if (lanes == 0) return;
    DevBuf<T> nb; nb.reserve((size_t)ncap * lanes); nb.zero(stream);
    if (b.ptr && n > 0) CUDA_CHECK(cudaMemcpy2DAsync(nb.ptr, (size_t)ncap * sizeof(T), b.ptr, (size_t)cap * sizeof(T), (size_t)n * sizeof(T), lanes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    b = std::move(nb);
  };
  const int hi_lanes = (K - 1) * 3;
  grow4(pos_op); grow4(quat); grow4(lscale); grow_planar(sh_dc, 3);
  if (sh_fp16) grow_planar(sh_hi, hi_lanes); else grow_planar(sh_hi32, hi_lanes);
  grow4(m_pos_op); grow4(v_pos_op); grow4(m_quat); grow4(v_quat); grow4(m_lscale); grow4(v_lscale);
  grow_planar(m_sh_dc, 3);
  if (sh_fp16) grow_planar(m_sh_hi, hi_lanes); else grow_planar(m_sh_hi32, hi_lanes);
  growf(v_sh);
  growf(refine_norm); growf(max_screen); growf(vis_count); last_step.reserve((size_t)ncap, true, stream);
  size_t old = cap;
  auto zero_tail4 = [&](DevBuf<float4>& b) { CUDA_CHECK(cudaMemsetAsync(b.ptr + old, 0, (ncap - old) * sizeof(float4), stream)); };
  auto zero_tailf = [&](DevBuf<float>& b) { CUDA_CHECK(cudaMemsetAsync(b.ptr + old, 0, (ncap - old) * sizeof(float), stream)); };
  zero_tail4(m_pos_op); zero_tail4(v_pos_op); zero_tail4(m_quat); zero_tail4(v_quat); zero_tail4(m_lscale); zero_tail4(v_lscale);
  zero_tailf(v_sh); zero_tailf(refine_norm); zero_tailf(max_screen); zero_tailf(vis_count);
  CUDA_CHECK(cudaMemsetAsync(last_step.ptr + old, 0, (ncap - old) * sizeof(uint32_t), stream));
  cap = ncap;
}

void Model::upload(const SplatCloud& c, cudaStream_t stream) {
  degree = c.sh_degree;
  n = (int)c.n;
  cap = 0;
  pos_op.free(); quat.free(); lscale.free(); sh_dc.free(); sh_hi.free(); sh_hi32.free();
  m_pos_op.free(); v_pos_op.free(); m_quat.free(); v_quat.free(); m_lscale.free(); v_lscale.free(); m_sh_dc.free(); m_sh_hi.free(); m_sh_hi32.free(); v_sh.free();
  refine_norm.free(); max_screen.free(); vis_count.free(); last_step.free();
  reserve(n, stream);
  std::vector<float4> p(n), q(n), s(n);
  for (int i = 0; i < n; i++) {
    p[i] = make_float4(c.pos[i * 3], c.pos[i * 3 + 1], c.pos[i * 3 + 2], c.opacity[i]);
    q[i] = make_float4(c.quat[i * 4], c.quat[i * 4 + 1], c.quat[i * 4 + 2], c.quat[i * 4 + 3]);
    s[i] = make_float4(c.log_scale[i * 3], c.log_scale[i * 3 + 1], c.log_scale[i * 3 + 2], 0.f);
  }
  CUDA_CHECK(cudaMemcpyAsync(pos_op.ptr, p.data(), n * sizeof(float4), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(quat.ptr, q.data(), n * sizeof(float4), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(lscale.ptr, s.data(), n * sizeof(float4), cudaMemcpyHostToDevice, stream));
  {
    int K = this->K(); const int hi = (K - 1) * 3;
    std::vector<float> dc((size_t)cap * 3, 0.f), rest((size_t)cap * hi, 0.f);
    for (int i = 0; i < n; i++) {
      for (int k = 0; k < 3; k++) dc[(size_t)k * cap + i] = c.sh[(size_t)i * K * 3 + k];
      for (int k = 0; k < hi; k++) rest[(size_t)k * cap + i] = c.sh[(size_t)i * K * 3 + 3 + k];
    }
    CUDA_CHECK(cudaMemcpyAsync(sh_dc.ptr, dc.data(), dc.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    if (hi > 0) {
      if (sh_fp16) {
        std::vector<__half> h(rest.size());
        for (size_t j = 0; j < rest.size(); j++) h[j] = __float2half_rn(rest[j]);
        CUDA_CHECK(cudaMemcpyAsync(sh_hi.ptr, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
      } else {
        CUDA_CHECK(cudaMemcpyAsync(sh_hi32.ptr, rest.data(), rest.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  zero_optimizer(stream); zero_stats(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

SplatCloud Model::download(cudaStream_t stream) const {
  SplatCloud c; c.resize(n, degree);
  auto p = pos_op.download(n, stream); auto q = quat.download(n, stream); auto s = lscale.download(n, stream);
  const int K = this->K(), hi = (K - 1) * 3;
  auto dc = sh_dc.download((size_t)cap * 3, stream);
  std::vector<float> rest;
  if (hi > 0) {
    if (sh_fp16) { auto h = sh_hi.download((size_t)cap * hi, stream); rest.resize(h.size()); for (size_t j = 0; j < h.size(); j++) rest[j] = __half2float(h[j]); }
    else rest = sh_hi32.download((size_t)cap * hi, stream);
  }
  for (int i = 0; i < n; i++) {
    c.pos[i * 3] = p[i].x; c.pos[i * 3 + 1] = p[i].y; c.pos[i * 3 + 2] = p[i].z; c.opacity[i] = p[i].w;
    c.quat[i * 4] = q[i].x; c.quat[i * 4 + 1] = q[i].y; c.quat[i * 4 + 2] = q[i].z; c.quat[i * 4 + 3] = q[i].w;
    c.log_scale[i * 3] = s[i].x; c.log_scale[i * 3 + 1] = s[i].y; c.log_scale[i * 3 + 2] = s[i].z;
  }
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < 3; k++) c.sh[(size_t)i * K * 3 + k] = dc[(size_t)k * cap + i];
    for (int k = 0; k < hi; k++) c.sh[(size_t)i * K * 3 + 3 + k] = rest[(size_t)k * cap + i];
  }
  return c;
}

void Model::zero_optimizer(cudaStream_t stream) {
  m_pos_op.zero(stream); v_pos_op.zero(stream); m_quat.zero(stream); v_quat.zero(stream); m_lscale.zero(stream); v_lscale.zero(stream);
  m_sh_dc.zero(stream); m_sh_hi.zero(stream); m_sh_hi32.zero(stream); v_sh.zero(stream); adam_t = 0; last_step.zero(stream);
}
void Model::zero_stats(cudaStream_t stream) { refine_norm.zero(stream); max_screen.zero(stream); vis_count.zero(stream); }

}  // namespace b2c
