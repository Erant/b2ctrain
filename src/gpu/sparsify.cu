#include "gpu/sparsify.h"
#include "gpu/splat_math.cuh"
#include <cub/cub.cuh>
#include <cmath>

namespace b2c {

namespace {

// The rasteriser's opacity without the view-dependent mip compensation: sigmoid times the floor's.
__device__ __forceinline__ float effective_opacity(float raw, float4 ls) {
  float sig = sigmoidf_(raw);
  if (ls.w <= 0.f) return sig;
  float f2 = ls.w * ls.w;
  float3 s2 = make_float3(__expf(2.f * ls.x), __expf(2.f * ls.y), __expf(2.f * ls.z));
  return sig * sqrtf((s2.x * s2.y * s2.z) / ((s2.x + f2) * (s2.y + f2) * (s2.z + f2)));
}

__global__ void strided_copy_kernel(int n, const float* __restrict__ acc, int stride, int offset, float* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = acc[(size_t)i * stride + offset];
}

// keys = the ranking: o + u, or the importance score.
__global__ void keys_kernel(int n, const float4* __restrict__ pos, const float4* __restrict__ ls, const float* __restrict__ u,
                            const float* __restrict__ score, float* __restrict__ keys) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float k = score ? score[i] : effective_opacity(pos[i].w, ls[i]) + (u ? u[i] : 0.f);
  keys[i] = isfinite(k) ? k : -INFINITY;
}

// z = keys > cut ? o + u : 0; then u += o - z.
__global__ void project_kernel(int n, const float4* __restrict__ pos, const float4* __restrict__ ls, const float* __restrict__ keys, float cut,
                               bool update_u, float* __restrict__ z, float* __restrict__ u) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float o = effective_opacity(pos[i].w, ls[i]);
  float t = o + u[i];
  float zi = keys[i] > cut ? t : 0.f;
  z[i] = zi;
  if (update_u) u[i] += o - zi;
}

__global__ void flags_kernel(int n, const float* __restrict__ keys, float cut, uint32_t* __restrict__ flags) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) flags[i] = keys[i] > cut ? 1u : 0u;
}

}  // namespace

void Sparsifier::set_importance(const float* acc, int stride, int offset, const Model& m, cudaStream_t stream) {
  score.reserve((size_t)m.cap, true, stream);
  if (m.n == 0) return;
  strided_copy_kernel<<<div_up(m.n, 256), 256, 0, stream>>>(m.n, acc, stride, offset, score);
  CUDA_KERNEL_CHECK();
}

float Sparsifier::cut(int n, cudaStream_t stream) {
  if (keep >= (uint32_t)n) return -INFINITY;
  size_t tmp = 0;
  cub::DeviceRadixSort::SortKeysDescending(nullptr, tmp, keys.ptr, keys_sorted.ptr, n, 0, 32, stream);
  cub_tmp.reserve(tmp);
  cub::DeviceRadixSort::SortKeysDescending(cub_tmp.ptr, tmp, keys.ptr, keys_sorted.ptr, n, 0, 32, stream);
  float c = 0.f;
  CUDA_CHECK(cudaMemcpyAsync(&c, keys_sorted.ptr + keep, sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return c;
}

void Sparsifier::project(const Model& m, bool update_u, cudaStream_t stream) {
  const int n = m.n;
  if (n == 0) return;
  keys.reserve((size_t)n); keys_sorted.reserve((size_t)n);
  keys_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, m.pos_op, m.lscale, u, by_importance ? score.ptr : nullptr, keys);
  CUDA_KERNEL_CHECK();
  float c = cut(n, stream);
  project_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, m.pos_op, m.lscale, keys, c, update_u, z, u);
  CUDA_KERNEL_CHECK();
}

void Sparsifier::begin(const Model& m, float ratio, cudaStream_t stream) {
  keep = (uint32_t)std::lround((1.0 - (double)ratio) * m.n);
  z.reserve((size_t)m.cap); u.reserve((size_t)m.cap); score.reserve((size_t)m.cap, true, stream);
  u.zero(stream);
  project(m, false, stream);
  active = true;
}

void Sparsifier::update(const Model& m, cudaStream_t stream) { project(m, true, stream); }

uint32_t Sparsifier::final_flags(const Model& m, cudaStream_t stream) {
  const int n = m.n;
  if (n == 0) return 0;
  keys.reserve((size_t)n); keys_sorted.reserve((size_t)n); flags.reserve((size_t)n); count.reserve(1);
  // The stop prunes by the model as it stands (o, not o + u), as GaussianSpa's train_op.py does.
  keys_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, m.pos_op, m.lscale, nullptr, by_importance ? score.ptr : nullptr, keys);
  CUDA_KERNEL_CHECK();
  float c = cut(n, stream);
  flags_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, keys, c, flags);
  CUDA_KERNEL_CHECK();
  size_t tmp = 0;
  cub::DeviceReduce::Sum(nullptr, tmp, flags.ptr, count.ptr, n, stream);
  cub_tmp.reserve(tmp);
  cub::DeviceReduce::Sum(cub_tmp.ptr, tmp, flags.ptr, count.ptr, n, stream);
  uint32_t kept = 0;
  CUDA_CHECK(cudaMemcpyAsync(&kept, count.ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return kept;
}

}  // namespace b2c
