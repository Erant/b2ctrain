#include "model.h"

namespace b2c {

void Model::reserve(int want, cudaStream_t stream) {
  if (want <= cap) return;
  int ncap = cap ? cap : 1024;
  while (ncap < want) ncap *= 2;
  int K = this->K();
  auto grow4 = [&](DevBuf<float4>& b) { b.reserve(ncap, true, stream); };
  auto growf = [&](DevBuf<float>& b, size_t per) { b.reserve((size_t)ncap * per, true, stream); };
  grow4(pos_op); grow4(quat); grow4(lscale); growf(sh, K * 3);
  grow4(m_pos_op); grow4(v_pos_op); grow4(m_quat); grow4(v_quat); grow4(m_lscale); grow4(v_lscale);
  growf(m_sh, K * 3); growf(v_sh, 1);
  growf(refine_norm, 1); growf(max_screen, 1); growf(vis_count, 1);
  // Newly reserved tail must be zero for moments/stats (contents beyond `n` are only read after being written by split).
  size_t old = cap;
  auto zero_tail4 = [&](DevBuf<float4>& b) { CUDA_CHECK(cudaMemsetAsync(b.ptr + old, 0, (ncap - old) * sizeof(float4), stream)); };
  auto zero_tailf = [&](DevBuf<float>& b, size_t per) { CUDA_CHECK(cudaMemsetAsync(b.ptr + old * per, 0, (ncap - old) * per * sizeof(float), stream)); };
  zero_tail4(m_pos_op); zero_tail4(v_pos_op); zero_tail4(m_quat); zero_tail4(v_quat); zero_tail4(m_lscale); zero_tail4(v_lscale);
  zero_tailf(m_sh, K * 3); zero_tailf(v_sh, 1); zero_tailf(refine_norm, 1); zero_tailf(max_screen, 1); zero_tailf(vis_count, 1);
  cap = ncap;
}

void Model::upload(const SplatCloud& c, cudaStream_t stream) {
  degree = c.sh_degree;
  n = (int)c.n;
  cap = 0;
  pos_op.free(); quat.free(); lscale.free(); sh.free();
  m_pos_op.free(); v_pos_op.free(); m_quat.free(); v_quat.free(); m_lscale.free(); v_lscale.free(); m_sh.free(); v_sh.free();
  refine_norm.free(); max_screen.free(); vis_count.free();
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
  CUDA_CHECK(cudaMemcpyAsync(sh.ptr, c.sh.data(), c.sh.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
  zero_optimizer(stream); zero_stats(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

SplatCloud Model::download(cudaStream_t stream) const {
  SplatCloud c; c.resize(n, degree);
  auto p = pos_op.download(n, stream); auto q = quat.download(n, stream); auto s = lscale.download(n, stream);
  auto shv = sh.download((size_t)n * K() * 3, stream);
  for (int i = 0; i < n; i++) {
    c.pos[i * 3] = p[i].x; c.pos[i * 3 + 1] = p[i].y; c.pos[i * 3 + 2] = p[i].z; c.opacity[i] = p[i].w;
    c.quat[i * 4] = q[i].x; c.quat[i * 4 + 1] = q[i].y; c.quat[i * 4 + 2] = q[i].z; c.quat[i * 4 + 3] = q[i].w;
    c.log_scale[i * 3] = s[i].x; c.log_scale[i * 3 + 1] = s[i].y; c.log_scale[i * 3 + 2] = s[i].z;
  }
  c.sh = std::move(shv);
  return c;
}

void Model::zero_optimizer(cudaStream_t stream) {
  m_pos_op.zero(stream); v_pos_op.zero(stream); m_quat.zero(stream); v_quat.zero(stream); m_lscale.zero(stream); v_lscale.zero(stream);
  m_sh.zero(stream); v_sh.zero(stream); adam_t = 0;
}
void Model::zero_stats(cudaStream_t stream) { refine_norm.zero(stream); max_screen.zero(stream); vis_count.zero(stream); }

}  // namespace b2c
