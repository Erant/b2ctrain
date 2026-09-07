#include "model.h"

namespace b2c {

void Model::reserve(int want, cudaStream_t stream) {
  if (want <= cap) return;
  int ncap = cap ? cap : 1024;
  while (ncap < want) ncap *= 2;
  int K = this->K();
  auto grow4 = [&](DevBuf<float4>& b) { b.reserve(ncap, true, stream); };
  auto growf = [&](DevBuf<float>& b) { b.reserve((size_t)ncap, true, stream); };
  // Planar [K*3][cap] buffers: re-layout with a strided 2D copy.
  auto grow_planar = [&](DevBuf<float>& b) {
    DevBuf<float> nb; nb.reserve((size_t)ncap * K * 3); nb.zero(stream);
    if (b.ptr && n > 0) CUDA_CHECK(cudaMemcpy2DAsync(nb.ptr, (size_t)ncap * sizeof(float), b.ptr, (size_t)cap * sizeof(float), (size_t)n * sizeof(float), K * 3, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    b = std::move(nb);
  };
  grow4(pos_op); grow4(quat); grow4(lscale); grow_planar(sh);
  grow4(m_pos_op); grow4(v_pos_op); grow4(m_quat); grow4(v_quat); grow4(m_lscale); grow4(v_lscale);
  grow_planar(m_sh); growf(v_sh);
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
  pos_op.free(); quat.free(); lscale.free(); sh.free();
  m_pos_op.free(); v_pos_op.free(); m_quat.free(); v_quat.free(); m_lscale.free(); v_lscale.free(); m_sh.free(); v_sh.free();
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
    int K = this->K();
    std::vector<float> planar((size_t)cap * K * 3, 0.f);
    for (int i = 0; i < n; i++) for (int k = 0; k < K * 3; k++) planar[(size_t)k * cap + i] = c.sh[(size_t)i * K * 3 + k];
    CUDA_CHECK(cudaMemcpyAsync(sh.ptr, planar.data(), planar.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  zero_optimizer(stream); zero_stats(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

SplatCloud Model::download(cudaStream_t stream) const {
  SplatCloud c; c.resize(n, degree);
  auto p = pos_op.download(n, stream); auto q = quat.download(n, stream); auto s = lscale.download(n, stream);
  auto shv = sh.download((size_t)cap * K() * 3, stream);
  for (int i = 0; i < n; i++) {
    c.pos[i * 3] = p[i].x; c.pos[i * 3 + 1] = p[i].y; c.pos[i * 3 + 2] = p[i].z; c.opacity[i] = p[i].w;
    c.quat[i * 4] = q[i].x; c.quat[i * 4 + 1] = q[i].y; c.quat[i * 4 + 2] = q[i].z; c.quat[i * 4 + 3] = q[i].w;
    c.log_scale[i * 3] = s[i].x; c.log_scale[i * 3 + 1] = s[i].y; c.log_scale[i * 3 + 2] = s[i].z;
  }
  int K = this->K();
  for (int i = 0; i < n; i++) for (int k = 0; k < K * 3; k++) c.sh[(size_t)i * K * 3 + k] = shv[(size_t)k * cap + i];
  return c;
}

void Model::zero_optimizer(cudaStream_t stream) {
  m_pos_op.zero(stream); v_pos_op.zero(stream); m_quat.zero(stream); v_quat.zero(stream); m_lscale.zero(stream); v_lscale.zero(stream);
  m_sh.zero(stream); v_sh.zero(stream); adam_t = 0; last_step.zero(stream);
}
void Model::zero_stats(cudaStream_t stream) { refine_norm.zero(stream); max_screen.zero(stream); vis_count.zero(stream); }

}  // namespace b2c
