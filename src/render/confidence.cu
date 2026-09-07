#include "render/confidence.h"
#include "gpu/splat_math.cuh"
#include <cmath>
#include <algorithm>

namespace b2c {

namespace {
float smoothstep_h(float e0, float e1, float x) { float t = std::min(std::max((x - e0) / std::max(e1 - e0, 1e-6f), 0.f), 1.f); return t * t * (3.f - 2.f * t); }

__device__ __forceinline__ float smoothstep_d(float e0, float e1, float x) { float t = fminf(fmaxf((x - e0) / fmaxf(e1 - e0, 1e-6f), 0.f), 1.f); return t * t * (3.f - 2.f * t); }

__global__ void conf_kernel2(int n, const float4* __restrict__ means, const float4* __restrict__ mu, const float* __restrict__ static_conf, const float* __restrict__ cos_in, const float* __restrict__ cos_out,
                             const float4* __restrict__ normal, const float* __restrict__ gate, float cos_graze, bool facing, float3 cam, float* __restrict__ feat) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 p = means[i];
  float3 v = make_float3(cam.x - p.x, cam.y - p.y, cam.z - p.z);
  float vl = fmaxf(len3(v), 1e-9f); v = v * (1.f / vl);
  float4 m = mu[i];
  float cos_theta = v.x * m.x + v.y * m.y + v.z * m.z;
  float coverage = smoothstep_d(cos_out[i], cos_in[i], cos_theta);
  float conf = static_conf[i] * coverage;
  if (facing) {
    float4 nn = normal[i];
    float dt = v.x * nn.x + v.y * nn.y + v.z * nn.z;
    float f = smoothstep_d(0.f, cos_graze, dt);
    conf = conf * (1.f - gate[i] * (1.f - f));
  }
  feat[i * 3] = conf; feat[i * 3 + 1] = 0.f; feat[i * 3 + 2] = 0.f;
}

__global__ void trust_kernel(int n, float* feat) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  feat[i * 3] = 1.f; feat[i * 3 + 1] = 0.f; feat[i * 3 + 2] = 0.f;
}
}  // namespace

void ConfidenceModel::build(const SplatCloud& c, const ConfidenceParams& p) {
  n = (int)c.n;
  const float eps = 1e-6f;
  const float m0 = p.angle_margin_deg * (float)M_PI / 180.f, m1 = (p.angle_margin_deg + p.angle_soft_deg) * (float)M_PI / 180.f;
  std::vector<float> sc(n), ci(n), co(n), gt(n, 0.f);
  std::vector<float4> mus(n), nrm(n), means(n);
  for (int i = 0; i < n; i++) {
    const float* e = &c.evidence[(size_t)i * 7];
    float w_in = e[0], w_all = e[1], err = e[2], views = e[3];
    float inmask = w_all > 0.f ? std::min(std::max(w_in / w_all, 0.f), 1.f) : 0.f;
    float agree = w_in > 0.f ? std::exp(-(err / std::max(w_in, eps)) / std::max(p.tau, eps)) : 0.f;
    float support = smoothstep_h(0.f, std::max(p.min_views, eps), views);
    float dx = e[4], dy = e[5], dz = e[6];
    float dl = std::sqrt(dx * dx + dy * dy + dz * dz);
    float conf = smoothstep_h(p.inmask_lo, p.inmask_hi, inmask) * agree * support;
    if (dl < eps || w_in <= 0.f) conf = 0.f;
    sc[i] = conf;
    float mx = dl > eps ? dx / dl : 0.f, my = dl > eps ? dy / dl : 0.f, mz = dl > eps ? dz / dl : 0.f;
    mus[i] = make_float4(mx, my, mz, 0.f);
    float kappa = std::min(std::max(dl / std::max(w_in, eps), 0.f), 1.f);
    float phi = std::acos(std::min(std::max(2.f * kappa - 1.f, -1.f), 1.f));
    float c_in = std::cos(std::min(phi + m0, (float)M_PI));
    float c_out = std::min(std::cos(std::min(phi + m1, (float)M_PI)), c_in - 1e-3f);
    ci[i] = c_in; co[i] = c_out;
    means[i] = make_float4(c.pos[i * 3], c.pos[i * 3 + 1], c.pos[i * 3 + 2], 0.f);
    if (p.facing) {
      const float* ls = &c.log_scale[i * 3];
      int order[3] = {0, 1, 2};
      std::sort(order, order + 3, [&](int a, int b) { return ls[a] < ls[b]; });
      float qw = c.quat[i * 4], qx = c.quat[i * 4 + 1], qy = c.quat[i * 4 + 2], qz = c.quat[i * 4 + 3];
      float qn = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz); if (qn > 0) { qw /= qn; qx /= qn; qy /= qn; qz /= qn; }
      float R[9] = {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
                    2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
                    2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)};
      int a = order[0];
      float ax = R[a], ay = R[3 + a], az = R[6 + a];
      if (ax * mx + ay * my + az * mz < 0.f) { ax = -ax; ay = -ay; az = -az; }
      nrm[i] = make_float4(ax, ay, az, 0.f);
      gt[i] = std::exp(ls[order[0]] - ls[order[1]]) < 0.5f ? 1.f : 0.f;
    }
  }
  static_conf.upload(sc); cos_in.upload(ci); cos_out.upload(co); mu.upload(mus); normal_means.upload(means);
  if (p.facing) { normal.upload(nrm); gate.upload(gt); }
  has_facing = p.facing; cos_graze = std::cos(p.graze_deg * (float)M_PI / 180.f);
  feature.reserve((size_t)n * 3);
  CUDA_CHECK(cudaDeviceSynchronize());
}

void ConfidenceModel::build_trusting(int count) {
  n = count; feature.reserve((size_t)n * 3);
  trust_kernel<<<div_up(n, 256), 256>>>(n, feature);
  CUDA_KERNEL_CHECK();
  static_conf.free();
}

void ConfidenceModel::for_camera(const float* campos, cudaStream_t stream) {
  if (!static_conf.ptr) return;  // trusting mode: feature already filled
  float3 cam = make_float3(campos[0], campos[1], campos[2]);
  conf_kernel2<<<div_up(n, 256), 256, 0, stream>>>(n, normal_means, mu, static_conf, cos_in, cos_out, has_facing ? normal.ptr : nullptr, has_facing ? gate.ptr : nullptr, cos_graze, has_facing, cam, feature);
  CUDA_KERNEL_CHECK();
}

}  // namespace b2c
