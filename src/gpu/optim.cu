#include "gpu/optim.h"
#include "gpu/splat_math.cuh"

namespace b2c {

CamDev to_camdev(const CameraGPU& c);

namespace {

__device__ __forceinline__ float4 apply_normalize_vjp(float4 q, float4 g) {
  float lsq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  float l = sqrtf(lsq);
  float inv = 1.f / (l * lsq);
  float qw = q.x, qx = q.y, qy = q.z, qz = q.w;
  float gw = g.x, gx = g.y, gy = g.z, gz = g.w;
  float cc0 = -qw * qx, cc1 = -qx * qy, cc2 = -qy * qw, cs0 = -qw * qz, cs1 = -qx * qz, cs2 = -qy * qz;
  return make_float4(((lsq - qw * qw) * gw + cc0 * gx + cc2 * gy + cs0 * gz) * inv,
                     (cc0 * gw + (lsq - qx * qx) * gx + cc1 * gy + cs1 * gz) * inv,
                     (cc2 * gw + cc1 * gx + (lsq - qy * qy) * gy + cs2 * gz) * inv,
                     (cs0 * gw + cs1 * gx + cs2 * gy + (lsq - qz * qz) * gz) * inv);
}

// VJP of quat_to_mat (row-major v_r). Returns (w, x, y, z) gradient for a normalised quaternion.
__device__ __forceinline__ float4 quat_to_mat_vjp(float4 q, const Mat3& v) {
  float qw = q.x, qx = q.y, qy = q.z, qz = q.w;
  const float* m = v.m;
  float v00 = m[0], v01 = m[1], v02 = m[2], v10 = m[3], v11 = m[4], v12 = m[5], v20 = m[6], v21 = m[7], v22 = m[8];
  float w_grad = qx * (v21 - v12) + qy * (v02 - v20) + qz * (v10 - v01);
  float x_grad = -2.f * qx * (v11 + v22) + qy * (v10 + v01) + qz * (v20 + v02) + qw * (v21 - v12);
  float y_grad = qx * (v10 + v01) - 2.f * qy * (v00 + v22) + qz * (v21 + v12) + qw * (v02 - v20);
  float z_grad = qx * (v20 + v02) + qy * (v21 + v12) - 2.f * qz * (v00 + v11) + qw * (v10 - v01);
  return make_float4(2.f * w_grad, 2.f * x_grad, 2.f * y_grad, 2.f * z_grad);
}

__device__ __forceinline__ Sym2 inverse2x2_vjp(Sym2 minv, Sym2 v) {
  float tmp00 = -minv.c00 * v.c00 + -minv.c01 * v.c01;
  float tmp01 = -minv.c01 * v.c00 + -minv.c11 * v.c01;
  float tmp10 = -minv.c00 * v.c01 + -minv.c01 * v.c11;
  float tmp11 = -minv.c01 * v.c01 + -minv.c11 * v.c11;
  Sym2 r;
  r.c00 = tmp00 * minv.c00 + tmp10 * minv.c01;
  r.c01 = tmp01 * minv.c00 + tmp11 * minv.c01;
  r.c11 = tmp01 * minv.c01 + tmp11 * minv.c11;
  return r;
}

// SH view-direction VJP (port of brush's sh_color_viewdir_vjp). `c` is [K][3], vc = dL/dcolour.
template <int DEG>
__device__ __forceinline__ float3 sh_viewdir_vjp(const ShBuf& sb, int i, float3 v, float3 vc, int active) {
  float gx = 0.f, gy = 0.f, gz = 0.f;
  auto dotc = [&](int k) { return sb.get(k * 3, i) * vc.x + sb.get(k * 3 + 1, i) * vc.y + sb.get(k * 3 + 2, i) * vc.z; };
  float x = v.x, y = v.y, z = v.z;
  if constexpr (DEG >= 1) if (active >= 1) {
    const float f0a = 0.4886025f;
    float s_n1 = dotc(1), s_z0 = dotc(2), s_p1 = dotc(3);
    gx += -f0a * s_p1; gy += -f0a * s_n1; gz += f0a * s_z0;
    if constexpr (DEG >= 2) if (active >= 2) {
      const float c2 = -1.0925485f, f1a = 0.54627424f;
      float s_n2 = dotc(4), s_n1b = dotc(5), s_z0b = dotc(6), s_p1b = dotc(7), s_p2 = dotc(8);
      gx += 2.f * f1a * y * s_n2 + c2 * z * s_p1b + 2.f * f1a * x * s_p2;
      gy += 2.f * f1a * x * s_n2 + c2 * z * s_n1b - 2.f * f1a * y * s_p2;
      gz += c2 * y * s_n1b + 2.f * 0.9461747f * z * s_z0b + c2 * x * s_p1b;
      if constexpr (DEG >= 3) if (active >= 3) {
        float z2 = z * z, x2 = x * x, y2 = y * y;
        const float f2a = -0.5900436f, c1b = 1.4453057f, c0c = -2.285229f;
        float f1b = c1b * z, f0c = c0c * z2 + 0.4570458f, f0c_dz = 2.f * c0c * z;
        float s_n3 = dotc(9), s_n2c = dotc(10), s_n1c = dotc(11), s_z0c = dotc(12), s_p1c = dotc(13), s_p2c = dotc(14), s_p3 = dotc(15);
        float d12_z = 3.f * 1.8658817f * z2 - 1.119529f;
        gx += f2a * 6.f * x * y * s_n3 + 2.f * f1b * y * s_n2c + f0c * s_p1c + 2.f * f1b * x * s_p2c + f2a * 3.f * (x2 - y2) * s_p3;
        gy += f2a * 3.f * (x2 - y2) * s_n3 + 2.f * f1b * x * s_n2c + f0c * s_n1c + (-2.f) * f1b * y * s_p2c + f2a * (-6.f) * x * y * s_p3;
        gz += 2.f * c1b * x * y * s_n2c + f0c_dz * y * s_n1c + d12_z * s_z0c + f0c_dz * x * s_p1c + c1b * (x2 - y2) * s_p2c;
        if constexpr (DEG >= 4) if (active >= 4) {
          float fc1 = x2 - y2, fs1 = 2.f * x * y, fc2 = x * fc1 - y * fs1, fs2 = x * fs1 + y * fc1;
          float f0d = z * (-4.683326f * z2 + 2.0071396f), f0d_dz = -14.049978f * z2 + 2.0071396f;
          float f1c = 3.3116114f * z2 - 0.47308735f, f1c_dz = 2.f * 3.3116114f * z;
          const float f2b_c = -1.7701308f; float f2b = f2b_c * z; const float f3a = 0.62583575f;
          float p12 = z * (1.8658817f * z2 - 1.119529f), dp12 = 3.f * 1.8658817f * z2 - 1.119529f, dp6 = 2.f * 0.9461747f * z;
          float dp20 = 1.9843135f * (p12 + z * dp12) - 1.0062306f * dp6;
          float s_n4 = dotc(16), s_n3d = dotc(17), s_n2d = dotc(18), s_n1d = dotc(19), s_z0d = dotc(20), s_p1d = dotc(21), s_p2d = dotc(22), s_p3d = dotc(23), s_p4 = dotc(24);
          gx += f3a * 4.f * fs2 * s_n4 + f2b * 3.f * fs1 * s_n3d + f1c * 2.f * y * s_n2d + f0d * s_p1d + f1c * 2.f * x * s_p2d + f2b * 3.f * fc1 * s_p3d + f3a * 4.f * fc2 * s_p4;
          gy += f3a * 4.f * fc2 * s_n4 + f2b * 3.f * fc1 * s_n3d + f1c * 2.f * x * s_n2d + f0d * s_n1d + f1c * (-2.f) * y * s_p2d + f2b * (-3.f) * fs1 * s_p3d + f3a * (-4.f) * fs2 * s_p4;
          gz += f2b_c * fs2 * s_n3d + f1c_dz * fs1 * s_n2d + f0d_dz * y * s_n1d + dp20 * s_z0d + f0d_dz * x * s_p1d + f1c_dz * fc1 * s_p2d + f2b_c * fc2 * s_p3d;
        }
      }
    }
  }
  return make_float3(gx, gy, gz);
}

__device__ __forceinline__ void adam4(float4& p, float4& m, float4& v, float4 g, float4 lr, float b1, float b2, float bc1, float bc2, float eps) {
  m.x = b1 * m.x + (1.f - b1) * g.x; m.y = b1 * m.y + (1.f - b1) * g.y; m.z = b1 * m.z + (1.f - b1) * g.z; m.w = b1 * m.w + (1.f - b1) * g.w;
  v.x = b2 * v.x + (1.f - b2) * g.x * g.x; v.y = b2 * v.y + (1.f - b2) * g.y * g.y; v.z = b2 * v.z + (1.f - b2) * g.z * g.z; v.w = b2 * v.w + (1.f - b2) * g.w * g.w;
  p.x -= lr.x * (m.x * bc1) / (sqrtf(v.x * bc2) + eps);
  p.y -= lr.y * (m.y * bc1) / (sqrtf(v.y * bc2) + eps);
  p.z -= lr.z * (m.z * bc1) / (sqrtf(v.z * bc2) + eps);
  p.w -= lr.w * (m.w * bc1) / (sqrtf(v.w * bc2) + eps);
}

template <int DEG>
__global__ void __launch_bounds__(128) optim_kernel(
    int n, float4* __restrict__ pos_op, float4* __restrict__ quat, float4* __restrict__ lscale, ShBuf shb,
    float4* __restrict__ m_pos, float4* __restrict__ v_pos, float4* __restrict__ m_q, float4* __restrict__ v_q, float4* __restrict__ m_ls, float4* __restrict__ v_ls,
    ShBuf msb, float* __restrict__ v_sh,
    float* __restrict__ v_splat, uint32_t* __restrict__ vis_flag, const uint32_t* __restrict__ tile_count,
    float* __restrict__ refine_norm, float* __restrict__ vis_count, uint32_t* __restrict__ last_step,
    CamDev cam, OptimParams op) {
  constexpr int K = (DEG + 1) * (DEG + 1);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g[GRAD_LANES];
  uint32_t vis = vis_flag[i];
  if (vis) vis_flag[i] = 0u;
  bool visible = vis != 0u && tile_count[i] != 0u;
  float* vs = v_splat + (size_t)i * GRAD_LANES;
  if (vis) {
#pragma unroll
    for (int k = 0; k < (int)GRAD_LANES; k++) { g[k] = vs[k]; vs[k] = 0.f; }
  } else {
#pragma unroll
    for (int k = 0; k < (int)GRAD_LANES; k++) g[k] = 0.f;
  }
  vis_count[i] += visible ? 1.f : 0.f;
  float rn = isfinite(g[9]) ? fminf(fmaxf(g[9], 0.f), 1e32f) : 0.f;
  refine_norm[i] = fmaxf(refine_norm[i], rn);
  bool any = false;
#pragma unroll
  for (int k = 0; k < (int)GRAD_LANES; k++) if (k != 9 && g[k] != 0.f) any = true;
  any = any && visible;
  bool finite = true;
#pragma unroll
  for (int k = 0; k < (int)GRAD_LANES; k++) finite = finite && isfinite(g[k]);
  if (!finite) any = false;
  if (op.sparse && !any) return;
  // Lazy sparse Adam: catch up on the steps skipped while invisible. With zero gradient the dense update would have
  // decayed the moments and applied a geometric series of momentum steps; both have closed forms (bias corrections and
  // the mean LR treated as constant over the gap).
  float skipped = 0.f;
  if (op.sparse) { uint32_t ls = last_step[i]; skipped = ls == 0u ? 0.f : fmaxf((float)op.t - (float)ls - 1.f, 0.f); last_step[i] = (uint32_t)op.t; }

  float4 po = pos_op[i], q = quat[i], ls = lscale[i];
  float3 g_pos = make_float3(0, 0, 0); float g_op = 0.f; float4 g_q = make_float4(0, 0, 0, 0); float3 g_ls = make_float3(0, 0, 0);
  float basis[K];
#pragma unroll
  for (int k = 0; k < K; k++) basis[k] = 0.f;
  float3 vc = make_float3(0.f, 0.f, 0.f);

  if (any) {
    ProjIntermediates pr = project_one(po, q, ls, cam, op.mip);
    if (pr.ok) {
      float3 mean = make_float3(po.x, po.y, po.z);
      float3 campos = make_float3(cam.pos[0], cam.pos[1], cam.pos[2]);
      // SH
      float3 u = mean - campos; float ul = len3(u); float3 v = u * (1.f / ul);
      sh_basis<DEG>(v, basis);
      vc = make_float3(g[5], g[6], g[7]);
      {
        int kmax = (op.active_sh_degree + 1) * (op.active_sh_degree + 1);
#pragma unroll
        for (int k = 0; k < K; k++) if (k >= kmax) basis[k] = 0.f;
      }
      float3 v_v = sh_viewdir_vjp<DEG>(shb, i, v, vc, op.active_sh_degree);
      float vdot = dot3(v, v_v);
      float3 v_mean_sh = (v_v - v * vdot) * (1.f / ul);
      // Opacity
      float v_alpha = g[8];
      g_op = v_alpha * pr.filter_comp * pr.comp_floor * pr.sig * (1.f - pr.sig);
      // Conic -> cov2d
      Sym2 v_inv{g[2], 0.5f * g[3], g[4]};
      Sym2 v_cov = inverse2x2_vjp(pr.conic, v_inv);
      // World covariance and camera-space covariance.
      Mat3 Rv; for (int k = 0; k < 9; k++) Rv.m[k] = cam.R[k];
      Mat3 M;  // Rq * diag(s)
      for (int r = 0; r < 3; r++) { M.m[r * 3] = pr.Rq.m[r * 3] * pr.s.x; M.m[r * 3 + 1] = pr.Rq.m[r * 3 + 1] * pr.s.y; M.m[r * 3 + 2] = pr.Rq.m[r * 3 + 2] * pr.s.z; }
      Mat3 covar;  // M M^T
      for (int r = 0; r < 3; r++) for (int cc = 0; cc < 3; cc++) covar.m[r * 3 + cc] = M.m[r * 3] * M.m[cc * 3] + M.m[r * 3 + 1] * M.m[cc * 3 + 1] + M.m[r * 3 + 2] * M.m[cc * 3 + 2];
      Mat3 RvC = mat3_mul(Rv, covar);
      Mat3 cov_c;  // Rv covar Rv^T
      for (int r = 0; r < 3; r++) for (int cc = 0; cc < 3; cc++) cov_c.m[r * 3 + cc] = RvC.m[r * 3] * Rv.m[cc * 3] + RvC.m[r * 3 + 1] * Rv.m[cc * 3 + 1] + RvC.m[r * 3 + 2] * Rv.m[cc * 3 + 2];
      // Projection VJP (pinhole) for mean_c.
      float3 mc = pr.mean_c;
      float inv_z = 1.f / mc.z, inv_z2 = inv_z * inv_z, inv_z3 = inv_z2 * inv_z;
      float v2x = g[0], v2y = g[1];
      float vcov[2][2] = {{v_cov.c00, v_cov.c01}, {v_cov.c01, v_cov.c11}};
      float tmp[2][3];
      for (int r = 0; r < 2; r++) for (int j = 0; j < 3; j++) tmp[r][j] = vcov[r][0] * pr.J[j] + vcov[r][1] * pr.J[3 + j];
      auto rowdot = [&](int r, int cr) { return tmp[r][0] * cov_c.m[cr * 3] + tmp[r][1] * cov_c.m[cr * 3 + 1] + tmp[r][2] * cov_c.m[cr * 3 + 2]; };
      float vj00 = 2.f * rowdot(0, 0), vj11 = 2.f * rowdot(1, 1), vj20 = 2.f * rowdot(0, 2), vj21 = 2.f * rowdot(1, 2);
      float rx = mc.x * inv_z, ry = mc.y * inv_z;
      float txc = mc.z * fminf(fmaxf(rx, cam.lim_neg_x), cam.lim_pos_x), tyc = mc.z * fminf(fmaxf(ry, cam.lim_neg_y), cam.lim_pos_y);
      float v_mx = cam.fx * inv_z * v2x, v_my = cam.fy * inv_z * v2y;
      float v_mz = -(cam.fx * mc.x * v2x + cam.fy * mc.y * v2y) * inv_z2;
      if (!pr.clamp_x) v_mx += -cam.fx * inv_z2 * vj20; else v_mz += -cam.fx * inv_z3 * vj20 * txc;
      if (!pr.clamp_y) v_my += -cam.fy * inv_z2 * vj21; else v_mz += -cam.fy * inv_z3 * vj21 * tyc;
      v_mz += -cam.fx * inv_z2 * vj00 - cam.fy * inv_z2 * vj11 + 2.f * cam.fx * txc * inv_z3 * vj20 + 2.f * cam.fy * tyc * inv_z3 * vj21;
      float3 v_mean_c = make_float3(v_mx, v_my, v_mz);
      // v_cov_c = J^T v_cov J  (3x3 sym)
      Mat3 vcc;
      for (int r = 0; r < 3; r++) for (int cc = 0; cc < 3; cc++) {
        float s = 0.f;
        for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) s += pr.J[a * 3 + r] * vcov[a][b] * pr.J[b * 3 + cc];
        vcc.m[r * 3 + cc] = s;
      }
      // v_covar (world) = Rv^T vcc Rv
      Mat3 t1;  // Rv^T vcc
      for (int r = 0; r < 3; r++) for (int cc = 0; cc < 3; cc++) t1.m[r * 3 + cc] = Rv.m[r] * vcc.m[cc] + Rv.m[3 + r] * vcc.m[3 + cc] + Rv.m[6 + r] * vcc.m[6 + cc];
      Mat3 v_covar = mat3_mul(t1, Rv);
      // v_M = 2 v_covar M
      Mat3 v_M = mat3_mul(v_covar, M);
      for (int k = 0; k < 9; k++) v_M.m[k] *= 2.f;
      g_pos = mat3_tmul(Rv, v_mean_c) + v_mean_sh;
      // scales: v_s_i = Rq.col_i . v_M.col_i ; chain through s_eff = sqrt(exp(2l) + f^2)
      float f2 = ls.w * ls.w;
      float vs0 = pr.Rq.m[0] * v_M.m[0] + pr.Rq.m[3] * v_M.m[3] + pr.Rq.m[6] * v_M.m[6];
      float vs1 = pr.Rq.m[1] * v_M.m[1] + pr.Rq.m[4] * v_M.m[4] + pr.Rq.m[7] * v_M.m[7];
      float vs2 = pr.Rq.m[2] * v_M.m[2] + pr.Rq.m[5] * v_M.m[5] + pr.Rq.m[8] * v_M.m[8];
      g_ls.x = vs0 * pr.s2_raw.x / pr.s.x + v_alpha * pr.opac * f2 / (pr.s2_raw.x + f2);
      g_ls.y = vs1 * pr.s2_raw.y / pr.s.y + v_alpha * pr.opac * f2 / (pr.s2_raw.y + f2);
      g_ls.z = vs2 * pr.s2_raw.z / pr.s.z + v_alpha * pr.opac * f2 / (pr.s2_raw.z + f2);
      // quaternion: v_R = v_M diag(s)
      Mat3 vR;
      for (int r = 0; r < 3; r++) { vR.m[r * 3] = v_M.m[r * 3] * pr.s.x; vR.m[r * 3 + 1] = v_M.m[r * 3 + 1] * pr.s.y; vR.m[r * 3 + 2] = v_M.m[r * 3 + 2] * pr.s.z; }
      float qn2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w; float inv_qn = rsqrtf(qn2);
      float4 qn = make_float4(q.x * inv_qn, q.y * inv_qn, q.z * inv_qn, q.w * inv_qn);
      float4 qg = quat_to_mat_vjp(qn, vR);
      g_q = apply_normalize_vjp(q, qg);
      // Feature (normal) gradient -> quaternion, when the feature was the pseudo-normal.
      // feat = Rv * (face * u), u = Rq.col(axis)/|.|; d feat / d Rq.col(axis) = Rv * face * (I - u u^T)/|a|.
      if (g[10] != 0.f || g[11] != 0.f || g[12] != 0.f) {
        int axis = (ls.x <= ls.y && ls.x <= ls.z) ? 0 : (ls.y <= ls.z ? 1 : 2);
        float3 a = axis == 0 ? make_float3(pr.Rq.m[0], pr.Rq.m[3], pr.Rq.m[6]) : (axis == 1 ? make_float3(pr.Rq.m[1], pr.Rq.m[4], pr.Rq.m[7]) : make_float3(pr.Rq.m[2], pr.Rq.m[5], pr.Rq.m[8]));
        float al = fmaxf(len3(a), 1e-12f); float3 uu = a * (1.f / al);
        float face = dot3(campos - mean, uu) >= 0.f ? 1.f : -1.f;
        float3 vf = make_float3(g[10], g[11], g[12]);
        float3 vu = mat3_tmul(Rv, vf) * face;          // d/d(u)
        float3 va = (vu - uu * dot3(uu, vu)) * (1.f / al);  // d/d(a)
        Mat3 vRn; for (int k = 0; k < 9; k++) vRn.m[k] = 0.f;
        if (axis == 0) { vRn.m[0] = va.x; vRn.m[3] = va.y; vRn.m[6] = va.z; }
        else if (axis == 1) { vRn.m[1] = va.x; vRn.m[4] = va.y; vRn.m[7] = va.z; }
        else { vRn.m[2] = va.x; vRn.m[5] = va.y; vRn.m[8] = va.z; }
        float4 qg2 = quat_to_mat_vjp(qn, vRn);
        float4 gq2 = apply_normalize_vjp(q, qg2);
        g_q.x += gq2.x; g_q.y += gq2.y; g_q.z += gq2.z; g_q.w += gq2.w;
      }
      bool ok = finite3(g_pos) && isfinite(g_op) && finite3(g_ls) && isfinite(g_q.x + g_q.y + g_q.z + g_q.w);
      if (!ok) { g_pos = make_float3(0, 0, 0); g_op = 0.f; g_ls = make_float3(0, 0, 0); g_q = make_float4(0, 0, 0, 0); }
    }
  }

  if (op.grad_out) {
    float* o = op.grad_out + (size_t)i * (11 + K * 3);
    o[0] = g_pos.x; o[1] = g_pos.y; o[2] = g_pos.z; o[3] = g_op; o[4] = g_q.x; o[5] = g_q.y; o[6] = g_q.z; o[7] = g_q.w; o[8] = g_ls.x; o[9] = g_ls.y; o[10] = g_ls.z;
    for (int k = 0; k < K; k++) { o[11 + k * 3] = basis[k] * vc.x; o[12 + k * 3] = basis[k] * vc.y; o[13 + k * 3] = basis[k] * vc.z; }
    return;
  }
  // ---- Adam ----
  float bc1 = 1.f / (1.f - powf(op.beta1, (float)op.t)), bc2 = 1.f / (1.f - powf(op.beta2, (float)op.t));
  float d1 = 1.f, d2 = 1.f, drift = 0.f;  // catch-up factors: m *= d1, v *= d2, p -= lr * drift * m_hat / (sqrt(v_hat) + eps)
  if (skipped > 0.f) {
    d1 = powf(op.beta1, skipped); d2 = powf(op.beta2, skipped);
    const float r = op.beta1 / sqrtf(op.beta2);
    drift = r * (1.f - powf(r, skipped)) / (1.f - r);
  }
  auto catch_up = [&](float4& p, float4& m, float4& v, float4 lr) {
    if (skipped <= 0.f) return;
    p.x -= lr.x * drift * (m.x * bc1) / (sqrtf(v.x * bc2) + op.eps);
    p.y -= lr.y * drift * (m.y * bc1) / (sqrtf(v.y * bc2) + op.eps);
    p.z -= lr.z * drift * (m.z * bc1) / (sqrtf(v.z * bc2) + op.eps);
    p.w -= lr.w * drift * (m.w * bc1) / (sqrtf(v.w * bc2) + op.eps);
    m.x *= d1; m.y *= d1; m.z *= d1; m.w *= d1; v.x *= d2; v.y *= d2; v.z *= d2; v.w *= d2;
  };
  {
    float4 m = m_pos[i], v = v_pos[i];
    float4 lr = make_float4(op.lr_mean, op.lr_mean, op.lr_mean, op.lr_opac);
    catch_up(po, m, v, lr);
    adam4(po, m, v, make_float4(g_pos.x, g_pos.y, g_pos.z, g_op), lr, op.beta1, op.beta2, bc1, bc2, op.eps);
    m_pos[i] = m; v_pos[i] = v;
  }
  {
    float4 m = m_q[i], v = v_q[i];
    float4 lr = make_float4(op.lr_rot, op.lr_rot, op.lr_rot, op.lr_rot);
    catch_up(q, m, v, lr);
    adam4(q, m, v, g_q, lr, op.beta1, op.beta2, bc1, bc2, op.eps);
    m_q[i] = m; v_q[i] = v; quat[i] = q;
  }
  {
    float4 m = m_ls[i], v = v_ls[i];
    float4 lsv = ls; float fkeep = ls.w;
    float4 lr = make_float4(op.lr_scale, op.lr_scale, op.lr_scale, 0.f);
    catch_up(lsv, m, v, lr);
    adam4(lsv, m, v, make_float4(g_ls.x, g_ls.y, g_ls.z, 0.f), lr, op.beta1, op.beta2, bc1, bc2, op.eps);
    lsv.w = fkeep; m.w = 0.f; v.w = 0.f;
    m_ls[i] = m; v_ls[i] = v; lscale[i] = lsv; ls = lsv;
  }
  {
    float bsq = 0.f;
#pragma unroll
    for (int k = 0; k < K; k++) bsq += basis[k] * basis[k];
    float gsq = bsq * (vc.x * vc.x + vc.y * vc.y + vc.z * vc.z) * (1.f / (float)(K * 3));
    float v_old = v_sh[i] * d2;
    float denom_old = skipped > 0.f ? drift * bc1 / (sqrtf(v_old * bc2) + op.eps) : 0.f;
    float v = op.beta2 * v_old + (1.f - op.beta2) * gsq; v_sh[i] = v;
    float denom = bc1 / (sqrtf(v * bc2) + op.eps);
    const uint32_t sr_seed = op.seed ^ (op.step * 0x9E3779B9u) ^ 0x5A5A5A5Au;
#pragma unroll
    for (int k = 0; k < K; k++) {
      float lrk = (k == 0 ? op.lr_dc : op.lr_sh_rest);
      float gk[3] = {basis[k] * vc.x, basis[k] * vc.y, basis[k] * vc.z};
#pragma unroll
      for (int ch = 0; ch < 3; ch++) {
        const int lane = k * 3 + ch;
        float m_old = msb.get(lane, i);
        float p = shb.get(lane, i);
        if (skipped > 0.f) { p -= lrk * denom_old * m_old; m_old *= d1; }
        float m = op.beta1 * m_old + (1.f - op.beta1) * gk[ch]; msb.set(lane, i, m);
        shb.set_sr(lane, i, p - lrk * denom * m, lane < 3 ? 0.f : hash_uniform(sr_seed, (uint32_t)i, (uint32_t)lane));
      }
    }
  }
  // ---- MCMC noise on means ----
  if (op.noise_weight > 0.f && visible) {
    float sig = sigmoidf_(po.w);
    float f2 = ls.w * ls.w;
    float3 s2 = make_float3(__expf(2.f * ls.x), __expf(2.f * ls.y), __expf(2.f * ls.z));
    float comp = ls.w > 0.f ? sqrtf((s2.x * s2.y * s2.z) / ((s2.x + f2) * (s2.y + f2) * (s2.z + f2))) : 1.f;
    float o = fminf(fmaxf(sig * comp, 0.f), 1.f);
    float w = fminf(fmaxf(powf(1.f - o, 150.f), 0.f), 1.f) * op.noise_weight;
    if (w > 0.f) {
      uint32_t s = op.seed ^ (op.step * 0x9E3779B9u);
      float nx = hash_normal(s, (uint32_t)i, 0u), ny = hash_normal(s, (uint32_t)i, 1u), nz = hash_normal(s, (uint32_t)i, 2u);
      float cl = op.noise_clamp;
      po.x += fminf(fmaxf(nx * w, -cl), cl); po.y += fminf(fmaxf(ny * w, -cl), cl); po.z += fminf(fmaxf(nz * w, -cl), cl);
    }
  }
  pos_op[i] = po;
}

}  // namespace

void optimizer_step(RenderCtx& ctx, Model& m, const OptimParams& op, cudaStream_t stream) {
  if (m.n == 0) return;
  CamDev cam = to_camdev(op.cam);
  int blocks = div_up(m.n, 128);
#define L(D) optim_kernel<D><<<blocks, 128, 0, stream>>>(m.n, m.pos_op, m.quat, m.lscale, m.sh(), m.m_pos_op, m.v_pos_op, m.m_quat, m.v_quat, m.m_lscale, m.v_lscale, m.m_sh(), m.v_sh, ctx.v_splat, ctx.vis_flag, ctx.tile_count, m.refine_norm, m.vis_count, m.last_step, cam, op)
  switch (m.degree) { case 0: L(0); break; case 1: L(1); break; case 2: L(2); break; case 3: L(3); break; case 4: L(4); break; default: throw std::runtime_error("bad degree"); }
#undef L
  CUDA_KERNEL_CHECK();
}

}  // namespace b2c
