#include "gpu/deform.h"
#include "util/log.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace b2c {
namespace {

constexpr int BIND_TILE = 1024;
constexpr int NJ_MAX = 128;

__global__ void bind_kernel(int n, const float* __restrict__ pos, int stride, int nv, const float3* __restrict__ verts,
                            const int4* __restrict__ vj, const float4* __restrict__ vw, int4* __restrict__ bj, float4* __restrict__ bw, int* __restrict__ bv) {
  __shared__ float3 sv[BIND_TILE];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float3 p = make_float3(0.f, 0.f, 0.f);
  if (i < n) p = make_float3(pos[(size_t)i * stride], pos[(size_t)i * stride + 1], pos[(size_t)i * stride + 2]);
  float best = 3.4e38f; int bi = 0;
  for (int base = 0; base < nv; base += BIND_TILE) {
    int m = min(BIND_TILE, nv - base);
    __syncthreads();
    for (int k = threadIdx.x; k < m; k += blockDim.x) sv[k] = verts[base + k];
    __syncthreads();
    if (i < n) {
      for (int k = 0; k < m; k++) {
        float dx = sv[k].x - p.x, dy = sv[k].y - p.y, dz = sv[k].z - p.z;
        float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < best) { best = d2; bi = base + k; }
      }
    }
  }
  if (i < n) { bj[i] = vj[bi]; bw[i] = vw[bi]; bv[i] = bi; }
}

__device__ __forceinline__ float3 blend_apply(float3 x, int4 j, float4 w, const float* __restrict__ xf) {
  float3 out = make_float3(0.f, 0.f, 0.f);
  const int js[4] = {j.x, j.y, j.z, j.w}; const float ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
  for (int k = 0; k < 4; k++) {
    if (ws[k] == 0.f) continue;
    const float* t = xf + (size_t)js[k] * 12;
    out.x += ws[k] * (t[0] * x.x + t[1] * x.y + t[2] * x.z + t[9]);
    out.y += ws[k] * (t[3] * x.x + t[4] * x.y + t[5] * x.z + t[10]);
    out.z += ws[k] * (t[6] * x.x + t[7] * x.y + t[8] * x.z + t[11]);
  }
  return out;
}

// `delta` (v3 rigs, else nullptr): the view's displacement of the bound rig vertex, added before the blend.
__global__ void pose4_kernel(int n, const float4* __restrict__ pos, const int4* __restrict__ bj, const float4* __restrict__ bw, const int* __restrict__ bv,
                             const float3* __restrict__ delta, const float* __restrict__ xf, float4* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 p = pos[i]; float3 x = make_float3(p.x, p.y, p.z);
  if (delta) { float3 d = delta[bv[i]]; x.x += d.x; x.y += d.y; x.z += d.z; }
  float3 q = blend_apply(x, bj[i], bw[i], xf);
  out[i] = make_float4(q.x, q.y, q.z, p.w);
}
__global__ void pose3_kernel(int n, const float3* __restrict__ pos, const int4* __restrict__ bj, const float4* __restrict__ bw, const int* __restrict__ bv,
                             const float3* __restrict__ delta, const float* __restrict__ xf, float3* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float3 x = pos[i];
  if (delta) { float3 d = delta[bv[i]]; x.x += d.x; x.y += d.y; x.z += d.z; }
  out[i] = blend_apply(x, bj[i], bw[i], xf);
}

// Rodrigues: rotation matrix (row-major) of the axis-angle vector w.
__device__ __forceinline__ void rodrigues(float3 w, float* R) {
  float th = sqrtf(w.x * w.x + w.y * w.y + w.z * w.z);
  float kx = 0.f, ky = 0.f, kz = 0.f;
  if (th > 1e-9f) { kx = w.x / th; ky = w.y / th; kz = w.z / th; }
  float c = cosf(th), s = sinf(th), t = 1.f - c;
  R[0] = c + kx * kx * t;      R[1] = kx * ky * t - kz * s; R[2] = kx * kz * t + ky * s;
  R[3] = ky * kx * t + kz * s; R[4] = c + ky * ky * t;      R[5] = ky * kz * t - kx * s;
  R[6] = kz * kx * t - ky * s; R[7] = kz * ky * t + kx * s; R[8] = c + kz * kz * t;
}

// Sequential top-down composition (parents precede children); one thread, ~130 joints.
__global__ void fk_kernel(int nj, const int* __restrict__ parents, const int* __restrict__ active, const float3* __restrict__ jpos0,
                          const float3* __restrict__ omega, float* __restrict__ xf, float3* __restrict__ pivots) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  for (int j = 0; j < nj; j++) {
    float Rp[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}; float3 tp = make_float3(0.f, 0.f, 0.f);
    int p = parents[j];
    if (p >= 0) { const float* x = xf + (size_t)p * 12; for (int k = 0; k < 9; k++) Rp[k] = x[k]; tp = make_float3(x[9], x[10], x[11]); }
    float3 p0 = jpos0[j];
    float3 q = make_float3(Rp[0] * p0.x + Rp[1] * p0.y + Rp[2] * p0.z + tp.x, Rp[3] * p0.x + Rp[4] * p0.y + Rp[5] * p0.z + tp.y, Rp[6] * p0.x + Rp[7] * p0.y + Rp[8] * p0.z + tp.z);
    pivots[j] = q;
    float* out = xf + (size_t)j * 12;
    if (active[j]) {
      float Rw[9]; rodrigues(omega[j], Rw);
      // R = Rw Rp ; t = Rw (tp - q) + q
      for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) out[r * 3 + c] = Rw[r * 3] * Rp[c] + Rw[r * 3 + 1] * Rp[3 + c] + Rw[r * 3 + 2] * Rp[6 + c];
      float3 d = make_float3(tp.x - q.x, tp.y - q.y, tp.z - q.z);
      out[9] = Rw[0] * d.x + Rw[1] * d.y + Rw[2] * d.z + q.x;
      out[10] = Rw[3] * d.x + Rw[4] * d.y + Rw[5] * d.z + q.y;
      out[11] = Rw[6] * d.x + Rw[7] * d.y + Rw[8] * d.z + q.z;
    } else {
      for (int k = 0; k < 9; k++) out[k] = Rp[k];
      out[9] = tp.x; out[10] = tp.y; out[11] = tp.z;
    }
  }
}

// Torque of the positional gradients about every active ancestor's pivot, block-reduced then one atomic per joint.
__global__ void torque_kernel(int n, const float4* __restrict__ pos_view, const float3* __restrict__ g_pos, const int4* __restrict__ bj, const float4* __restrict__ bw,
                              const int* __restrict__ anc, int nj, const float3* __restrict__ pivots, float3* __restrict__ torque) {
  __shared__ float acc[NJ_MAX * 3];
  for (int k = threadIdx.x; k < NJ_MAX * 3; k += blockDim.x) acc[k] = 0.f;
  __syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float3 g = g_pos[i];
    if (g.x != 0.f || g.y != 0.f || g.z != 0.f) {
      float4 p = pos_view[i]; int4 j = bj[i]; float4 w = bw[i];
      const int js[4] = {j.x, j.y, j.z, j.w}; const float ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int k = 0; k < 4; k++) {
        if (ws[k] == 0.f) continue;
        const int* a = anc + (size_t)js[k] * BodyRig::MAX_ANC;
        for (int m = 0; m < BodyRig::MAX_ANC; m++) {
          int aj = a[m]; if (aj < 0) break;
          float3 r = make_float3(p.x - pivots[aj].x, p.y - pivots[aj].y, p.z - pivots[aj].z);
          atomicAdd(&acc[aj * 3 + 0], ws[k] * (r.y * g.z - r.z * g.y));
          atomicAdd(&acc[aj * 3 + 1], ws[k] * (r.z * g.x - r.x * g.z));
          atomicAdd(&acc[aj * 3 + 2], ws[k] * (r.x * g.y - r.y * g.x));
        }
      }
    }
  }
  __syncthreads();
  for (int k = threadIdx.x; k < nj * 3; k += blockDim.x) if (acc[k] != 0.f) atomicAdd(reinterpret_cast<float*>(torque) + k, acc[k]);
}

__global__ void update_kernel(int nj, int nviews, int v, const int* __restrict__ active, const float3* __restrict__ torque,
                              float3* __restrict__ omega, float3* __restrict__ m_om, float3* __restrict__ v_om, float lr, float smooth, float zero, bool global_v, int t) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nj || !active[j]) return;
  size_t idx = (size_t)v * nj + j;
  float3 w = omega[idx], g = torque[j];
  if (zero > 0.f) { g.x += zero * w.x; g.y += zero * w.y; g.z += zero * w.z; }
  if (smooth > 0.f && nviews > 2) {
    // the orbit closes on itself: the neighbours of the first view are the last and the second
    float3 a = omega[(size_t)((v + nviews - 1) % nviews) * nj + j], b = omega[(size_t)((v + 1) % nviews) * nj + j];
    g.x += smooth * (2.f * w.x - a.x - b.x); g.y += smooth * (2.f * w.y - a.y - b.y); g.z += smooth * (2.f * w.z - a.z - b.z);
  }
  const float b1 = 0.9f, b2 = 0.999f, eps = 1e-12f;
  // The second moment is shared by all views of a joint: a view in which the arm is hidden (a tiny, noisy torque)
  // then takes a small step instead of the full one per-parameter Adam would give it.
  // With `global_v` ONE second moment (v_om[0], the mean over the active joints' torques, kept by the caller) serves
  // every joint: a joint with little evidence (a finger, a hidden arm, a head that does not move) then takes a step
  // proportional to its own torque instead of the full one per-parameter Adam gives it.
  float3 m = m_om[idx]; float3 vv = global_v ? v_om[0] : v_om[j];
  m.x = b1 * m.x + (1.f - b1) * g.x; m.y = b1 * m.y + (1.f - b1) * g.y; m.z = b1 * m.z + (1.f - b1) * g.z;
  m_om[idx] = m;
  if (!global_v) {
    vv.x = b2 * vv.x + (1.f - b2) * g.x * g.x; vv.y = b2 * vv.y + (1.f - b2) * g.y * g.y; vv.z = b2 * vv.z + (1.f - b2) * g.z * g.z;
    v_om[j] = vv;
  }
  float bc1 = 1.f / (1.f - powf(b1, (float)t)), bc2 = 1.f / (1.f - powf(b2, (float)t));
  w.x -= lr * (m.x * bc1) / (sqrtf(vv.x * bc2) + eps);
  w.y -= lr * (m.y * bc1) / (sqrtf(vv.y * bc2) + eps);
  w.z -= lr * (m.z * bc1) / (sqrtf(vv.z * bc2) + eps);
  const float cap = 0.6f;  // radians: a safety clamp, well past any plausible per-view arm deviation
  w.x = fminf(fmaxf(w.x, -cap), cap); w.y = fminf(fmaxf(w.y, -cap), cap); w.z = fminf(fmaxf(w.z, -cap), cap);
  omega[idx] = w;
}

}  // namespace

bool BodyRig::load(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  auto need = [&](void* dst, size_t bytes) { if (fread(dst, 1, bytes, f) != bytes) { fclose(f); throw std::runtime_error("body rig '" + path + "' is truncated"); } };
  char magic[8]; need(magic, 8);
  const bool v3 = memcmp(magic, "B2CRIG3", 7) == 0;
  if (!v3 && memcmp(magic, "B2CRIG2", 7) != 0) { fclose(f); throw std::runtime_error("'" + path + "' is not a b2ctrain body rig v2/v3 (bad magic)"); }
  int32_t hdr[6]; need(hdr, sizeof(hdr));
  nv = hdr[0]; nj = hdr[1]; nviews = hdr[2]; n_active = hdr[5];
  if (hdr[3] != 4 || hdr[4] != 64 || nv <= 0 || nj <= 0 || nj > NJ_MAX || nviews <= 0 || n_active <= 0) { fclose(f); throw std::runtime_error("body rig '" + path + "': unsupported layout"); }
  std::vector<float3> v(nv); std::vector<int4> j(nv); std::vector<float4> w(nv);
  need(v.data(), (size_t)nv * sizeof(float3)); need(j.data(), (size_t)nv * sizeof(int4)); need(w.data(), (size_t)nv * sizeof(float4));
  std::vector<char> nm((size_t)nviews * 64); need(nm.data(), nm.size());
  parents_h.resize(nj); need(parents_h.data(), (size_t)nj * sizeof(int32_t));
  std::vector<float3> jp(nj); need(jp.data(), (size_t)nj * sizeof(float3));
  std::vector<int32_t> act(n_active); need(act.data(), (size_t)n_active * sizeof(int32_t));
  std::vector<float3> dl;
  if (v3) { dl.resize((size_t)nviews * nv); need(dl.data(), dl.size() * sizeof(float3)); }
  fclose(f);
  for (auto& jj : j) { const int js[4] = {jj.x, jj.y, jj.z, jj.w}; for (int k = 0; k < 4; k++) if (js[k] < 0 || js[k] >= nj) throw std::runtime_error("body rig '" + path + "': joint index out of range"); }
  for (int k = 0; k < nj; k++) if (parents_h[k] >= k) throw std::runtime_error("body rig '" + path + "': joints are not in parent-first order");
  active_h.assign(nj, 0);
  for (int a : act) { if (a < 0 || a >= nj) throw std::runtime_error("body rig '" + path + "': active joint out of range"); active_h[a] = 1; }
  // active ancestors-or-self per joint
  std::vector<int> anc_h((size_t)nj * MAX_ANC, -1);
  for (int k = 0; k < nj; k++) {
    int cnt = 0;
    for (int a = k; a >= 0; a = parents_h[a]) if (active_h[a]) { if (cnt >= MAX_ANC) throw std::runtime_error("body rig: more than 16 active joints on one chain"); anc_h[(size_t)k * MAX_ANC + cnt++] = a; }
  }
  names.clear(); for (int i = 0; i < nviews; i++) { std::string s(nm.data() + (size_t)i * 64, 64); s = s.c_str(); names.push_back(s); }
  verts.upload(v); vj.upload(j); vw.upload(w); parents.upload(parents_h); active.upload(active_h); anc.upload(anc_h); jpos0.upload(jp);
  has_delta = v3; if (v3) delta.upload(dl);
  xf.reserve((size_t)nviews * nj * 12); pivots.reserve(nj); torque.reserve(nj);
  omega.reserve((size_t)nviews * nj); m_om.reserve((size_t)nviews * nj); v_om.reserve((size_t)nviews * nj);
  omega.zero(); m_om.zero(); v_om.zero();
  CUDA_CHECK(cudaDeviceSynchronize());
  return true;
}

int BodyRig::view_index(const std::string& name) const {
  for (int i = 0; i < nviews; i++) if (names[i] == name) return i;
  return -1;
}

void BodyRig::bind_points(const float* pos, int n, int stride, DevBuf<int4>& bj, DevBuf<float4>& bw, DevBuf<int>& bv, cudaStream_t stream) const {
  if (n <= 0) return;
  if (bj.count < (size_t)n) bj.reserve(n); if (bw.count < (size_t)n) bw.reserve(n); if (bv.count < (size_t)n) bv.reserve(n);
  bind_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, pos, stride, nv, verts, vj, vw, bj, bw, bv);
  CUDA_KERNEL_CHECK();
}

void BodyRig::bind(const Model& m, cudaStream_t stream) {
  if (bind_j.count < (size_t)m.cap) { bind_j.reserve(m.cap); bind_w.reserve(m.cap); bind_v.reserve(m.cap); pos_view.reserve(m.cap); g_pos.reserve(m.cap); g_pos.zero(stream); }
  bind_points(reinterpret_cast<const float*>(m.pos_op.ptr), m.n, 4, bind_j, bind_w, bind_v, stream);
  bound_n = m.n;
}

void BodyRig::fk(int v, cudaStream_t stream) {
  if (v < 0 || v >= nviews) throw std::runtime_error("body rig: view index out of range");
  fk_kernel<<<1, 32, 0, stream>>>(nj, parents, active, jpos0, omega.ptr + (size_t)v * nj, xf.ptr + (size_t)v * nj * 12, pivots);
  CUDA_KERNEL_CHECK();
}

void BodyRig::pose(int v, const Model& m, cudaStream_t stream) {
  fk(v, stream);
  if (m.n > bound_n) throw std::runtime_error("body rig: the model grew since the last bind()");
  if (m.n == 0) return;
  pose4_kernel<<<div_up(m.n, 256), 256, 0, stream>>>(m.n, m.pos_op, bind_j, bind_w, bind_v, has_delta ? delta.ptr + (size_t)v * nv : nullptr, xf.ptr + (size_t)v * nj * 12, pos_view);
  CUDA_KERNEL_CHECK();
}

void BodyRig::pose_points(int v, const float3* src, const int4* bj, const float4* bw, const int* bv, int n, float3* dst, cudaStream_t stream) const {
  if (n <= 0) return;
  pose3_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, src, bj, bw, bv, has_delta ? delta.ptr + (size_t)v * nv : nullptr, xf.ptr + (size_t)v * nj * 12, dst);
  CUDA_KERNEL_CHECK();
}

__global__ void global_moment_kernel(int nj, const int* __restrict__ active, const float3* __restrict__ torque, float3* __restrict__ v_om) {
  // v_om[0] <- beta2 v + (1 - beta2) mean over active joints of torque^2 (per axis), one thread
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  float3 s = make_float3(0.f, 0.f, 0.f); int c = 0;
  for (int j = 0; j < nj; j++) if (active[j]) { s.x += torque[j].x * torque[j].x; s.y += torque[j].y * torque[j].y; s.z += torque[j].z * torque[j].z; c++; }
  const float b2 = 0.999f; float inv = c ? 1.f / c : 0.f;
  float3 vv = v_om[0];
  v_om[0] = make_float3(b2 * vv.x + (1.f - b2) * s.x * inv, b2 * vv.y + (1.f - b2) * s.y * inv, b2 * vv.z + (1.f - b2) * s.z * inv);
}

void BodyRig::update(int v, const Model& m, float lr, float smooth, float zero, bool global_v, cudaStream_t stream) {
  if (m.n == 0) return;
  torque.zero(stream);
  torque_kernel<<<div_up(m.n, 256), 256, 0, stream>>>(m.n, pos_view, g_pos, bind_j, bind_w, anc, nj, pivots, torque);
  CUDA_KERNEL_CHECK();
  adam_t++;
  if (global_v) { global_moment_kernel<<<1, 32, 0, stream>>>(nj, active, torque, v_om); CUDA_KERNEL_CHECK(); }
  update_kernel<<<div_up(nj, 128), 128, 0, stream>>>(nj, nviews, v, active, torque, omega, m_om, v_om, lr, smooth, zero, global_v, adam_t);
  CUDA_KERNEL_CHECK();
}

void BodyRig::sample_torque(cudaStream_t stream) {
  auto t = torque.download(nj, stream);
  double s = 0; int c = 0;
  for (int j = 0; j < nj; j++) if (active_h[j]) { s += std::sqrt((double)t[j].x * t[j].x + (double)t[j].y * t[j].y + (double)t[j].z * t[j].z); c++; }
  mean_abs_torque = c ? (float)(s / c) : 0.f;
}

std::vector<float3> BodyRig::download_omega(cudaStream_t stream) const { return omega.download((size_t)nviews * nj, stream); }

}  // namespace b2c
