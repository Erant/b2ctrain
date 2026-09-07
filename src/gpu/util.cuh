#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace b2c {

#define CUDA_CHECK(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) { \
  throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorString(_e) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } } while (0)
#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())

constexpr int TILE_W = 16;                 // loss / SSIM tile
constexpr int TILE_PX = TILE_W * TILE_W;
constexpr int RT_W = 8;                    // rasterizer tile
constexpr int RT_PX = RT_W * RT_W;
constexpr int PROJ_BLOCK = 256;

inline int div_up(size_t a, int b) { return (int)((a + b - 1) / b); }

// Simple owning device buffer.
template <typename T>
struct DevBuf {
  T* ptr = nullptr;
  size_t count = 0;
  DevBuf() = default;
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& o) noexcept : ptr(o.ptr), count(o.count) { o.ptr = nullptr; o.count = 0; }
  DevBuf& operator=(DevBuf&& o) noexcept { free(); ptr = o.ptr; count = o.count; o.ptr = nullptr; o.count = 0; return *this; }
  ~DevBuf() { free(); }
  void free() { if (ptr) cudaFree(ptr); ptr = nullptr; count = 0; }
  // Ensure capacity >= n (contents discarded when reallocating unless keep=true).
  void reserve(size_t n, bool keep = false, cudaStream_t stream = 0) {
    if (n <= count) return;
    T* np = nullptr;
    CUDA_CHECK(cudaMalloc(&np, n * sizeof(T)));
    if (keep && ptr && count) CUDA_CHECK(cudaMemcpyAsync(np, ptr, count * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    if (ptr) { CUDA_CHECK(cudaStreamSynchronize(stream)); cudaFree(ptr); }
    ptr = np; count = n;
  }
  void zero(cudaStream_t stream = 0, size_t n = (size_t)-1) { if (ptr) CUDA_CHECK(cudaMemsetAsync(ptr, 0, (n == (size_t)-1 ? count : n) * sizeof(T), stream)); }
  void upload(const T* src, size_t n, cudaStream_t stream = 0) { reserve(n); CUDA_CHECK(cudaMemcpyAsync(ptr, src, n * sizeof(T), cudaMemcpyHostToDevice, stream)); }
  void upload(const std::vector<T>& v, cudaStream_t stream = 0) { upload(v.data(), v.size(), stream); }
  std::vector<T> download(size_t n = (size_t)-1, cudaStream_t stream = 0) const {
    if (n == (size_t)-1) n = count;
    std::vector<T> v(n);
    if (n) { CUDA_CHECK(cudaMemcpyAsync(v.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost, stream)); CUDA_CHECK(cudaStreamSynchronize(stream)); }
    return v;
  }
  size_t bytes() const { return count * sizeof(T); }
  operator T*() const { return ptr; }
};

// Pinned host buffer for readbacks.
template <typename T>
struct PinnedBuf {
  T* ptr = nullptr; size_t count = 0;
  void reserve(size_t n) { if (n <= count) return; if (ptr) cudaFreeHost(ptr); CUDA_CHECK(cudaMallocHost(&ptr, n * sizeof(T))); count = n; }
  ~PinnedBuf() { if (ptr) cudaFreeHost(ptr); }
};

#ifdef __CUDACC__
// ---- device math helpers ----
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ float3 operator*(float s, float3 a) { return a * s; }
__device__ __forceinline__ float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float len3(float3 a) { return sqrtf(dot3(a, a)); }
__device__ __forceinline__ bool finite3(float3 a) { return isfinite(a.x) && isfinite(a.y) && isfinite(a.z); }

// Row-major 3x3.
struct Mat3 { float m[9]; };
__device__ __forceinline__ float3 mat3_mul(const Mat3& A, float3 v) {
  return make_float3(A.m[0] * v.x + A.m[1] * v.y + A.m[2] * v.z, A.m[3] * v.x + A.m[4] * v.y + A.m[5] * v.z, A.m[6] * v.x + A.m[7] * v.y + A.m[8] * v.z);
}
__device__ __forceinline__ float3 mat3_tmul(const Mat3& A, float3 v) {  // A^T v
  return make_float3(A.m[0] * v.x + A.m[3] * v.y + A.m[6] * v.z, A.m[1] * v.x + A.m[4] * v.y + A.m[7] * v.z, A.m[2] * v.x + A.m[5] * v.y + A.m[8] * v.z);
}
__device__ __forceinline__ Mat3 mat3_mul(const Mat3& A, const Mat3& B) {
  Mat3 C;
#pragma unroll
  for (int i = 0; i < 3; i++)
#pragma unroll
    for (int j = 0; j < 3; j++) C.m[i * 3 + j] = A.m[i * 3] * B.m[j] + A.m[i * 3 + 1] * B.m[3 + j] + A.m[i * 3 + 2] * B.m[6 + j];
  return C;
}
// Rotation matrix (row-major) from a normalised quaternion (w, x, y, z).
__device__ __forceinline__ Mat3 quat_to_mat(float4 q) {
  float w = q.x, x = q.y, y = q.z, z = q.w;
  float x2 = x * x, y2 = y * y, z2 = z * z, xy = x * y, xz = x * z, yz = y * z, wx = w * x, wy = w * y, wz = w * z;
  Mat3 R;
  R.m[0] = 1.f - 2.f * (y2 + z2); R.m[1] = 2.f * (xy - wz);       R.m[2] = 2.f * (xz + wy);
  R.m[3] = 2.f * (xy + wz);       R.m[4] = 1.f - 2.f * (x2 + z2); R.m[5] = 2.f * (yz - wx);
  R.m[6] = 2.f * (xz - wy);       R.m[7] = 2.f * (yz + wx);       R.m[8] = 1.f - 2.f * (x2 + y2);
  return R;
}

// Philox-free cheap counter hash RNG (PCG-style) for per-splat per-step noise.
__device__ __forceinline__ uint32_t hash_u32(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x;
}
__device__ __forceinline__ float hash_uniform(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = hash_u32(a * 0x9E3779B1U ^ hash_u32(b * 0x85EBCA77U ^ hash_u32(c * 0xC2B2AE3DU)));
  return (h >> 8) * (1.0f / 16777216.0f);
}
// Standard normal via Box-Muller from two hashes.
__device__ __forceinline__ float hash_normal(uint32_t a, uint32_t b, uint32_t c) {
  float u1 = fmaxf(hash_uniform(a, b, c), 1e-7f), u2 = hash_uniform(a, b, c + 0x1234567U);
  return sqrtf(-2.f * __logf(u1)) * __cosf(6.28318530718f * u2);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

#endif  // __CUDACC__

}  // namespace b2c
