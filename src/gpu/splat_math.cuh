#pragma once
#include "gpu/util.cuh"

namespace b2c {

constexpr float SH_C0_DEV = 0.28209479177387814f;
constexpr float ALPHA_CUTOFF = 1.0f / 255.0f;
constexpr float T_CUTOFF = 1e-4f;
constexpr float ALPHA_MAX = 0.999f;
constexpr uint32_t GRAD_LANES = 13;  // xy(2) conic(3) rgb(3) opac(1) refine(1) feat(3)

// Hollow-loss penalty of a fragment at camera depth z against the reference surface depth z_ref: 0 up to `margin`
// behind the surface, rising linearly to 1 at 2 * margin. z_ref = +inf (no surface) gives 0.
__device__ __forceinline__ float hollow_h(float z, float z_ref, float margin) {
  float d = (z - z_ref - margin) / fmaxf(margin, 1e-4f);
  return isfinite(d) ? fminf(fmaxf(d, 0.f), 1.f) : 0.f;
}

struct Sym2 { float c00, c01, c11; };

__device__ __forceinline__ float calc_sigma(float px, float py, Sym2 conic, float mx, float my) {
  float dx = px - mx, dy = py - my;
  return 0.5f * (conic.c00 * dx * dx + conic.c11 * dy * dy) + conic.c01 * dx * dy;
}

// Conservative ellipse-vs-tile test (StopThePop), identical to brush's `will_primitive_contribute`.
__device__ __forceinline__ bool tile_hit(float rmin_x, float rmin_y, float rmax_x, float rmax_y, float mx, float my, Sym2 conic, float power) {
  bool x_left = mx < rmin_x, x_right = mx > rmax_x, in_x = !(x_left || x_right);
  bool y_above = my < rmin_y, y_below = my > rmax_y, in_y = !(y_above || y_below);
  if (in_x && in_y) return true;
  float corner_x = x_left ? rmin_x : rmax_x, corner_y = y_above ? rmin_y : rmax_y;
  float width = rmax_x - rmin_x, height = rmax_y - rmin_y;
  float dxf = x_left ? width : -width, dyf = y_above ? height : -height;
  float diff_x = mx - corner_x, diff_y = my - corner_y;
  float tx_raw = (dxf * conic.c00 * diff_x + dxf * conic.c01 * diff_y) / (dxf * conic.c00 * dxf);
  float ty_raw = (dyf * conic.c01 * diff_x + dyf * conic.c11 * diff_y) / (dyf * conic.c11 * dyf);
  float tx = in_y ? 0.f : fminf(fmaxf(tx_raw, 0.f), 1.f);
  float ty = in_x ? 0.f : fminf(fmaxf(ty_raw, 0.f), 1.f);
  float px = corner_x + tx * dxf, py = corner_y + ty * dyf;
  return calc_sigma(px, py, conic, mx, my) <= power;
}

struct TileBox { int min_x, min_y, max_x, max_y; };
__device__ __forceinline__ TileBox tile_bbox(float mx, float my, float ex, float ey, int tiles_x, int tiles_y) {
  float tw = (float)RT_W;
  float cx = mx / tw, cy = my / tw, dx = ex / tw, dy = ey / tw;
  TileBox b;
  b.min_x = (int)fminf(fmaxf(cx - dx, 0.f), (float)tiles_x);
  b.min_y = (int)fminf(fmaxf(cy - dy, 0.f), (float)tiles_y);
  b.max_x = (int)fminf(fmaxf(cx + dx + 1.f, 0.f), (float)tiles_x);
  b.max_y = (int)fminf(fmaxf(cy + dy + 1.f, 0.f), (float)tiles_y);
  return b;
}

// Spherical harmonics (Sloan basis), coefficient-major [K][3]. Returns colour without the +0.5 offset.
#define SHC(k, ch) sb.get((k) * 3 + (ch), i)
#define SH3(k) make_float3(SHC(k, 0), SHC(k, 1), SHC(k, 2))
template <int DEG>
__device__ __forceinline__ float3 sh_eval(const ShBuf& sb, int i, float3 v, int active) {
  float3 col = SH3(0) * SH_C0_DEV;
  if constexpr (DEG >= 1) if (active >= 1) {
    const float f0a = 0.4886025f;
    col = col + SH3(1) * (-f0a * v.y);
    col = col + SH3(2) * (f0a * v.z);
    col = col + SH3(3) * (-f0a * v.x);
    if constexpr (DEG >= 2) if (active >= 2) {
      float z2 = v.z * v.z;
      float f0b = -1.0925485f * v.z, f1a = 0.54627424f;
      float fc1 = v.x * v.x - v.y * v.y, fs1 = 2.f * v.x * v.y;
      float p4 = f1a * fs1, p5 = f0b * v.y, p6 = 0.9461747f * z2 - 0.31539157f, p7 = f0b * v.x, p8 = f1a * fc1;
      col = col + SH3(4) * p4;
      col = col + SH3(5) * p5;
      col = col + SH3(6) * p6;
      col = col + SH3(7) * p7;
      col = col + SH3(8) * p8;
      if constexpr (DEG >= 3) if (active >= 3) {
        float f0c = -2.285229f * z2 + 0.4570458f, f1b = 1.4453057f * v.z, f2a = -0.5900436f;
        float fc2 = v.x * fc1 - v.y * fs1, fs2 = v.x * fs1 + v.y * fc1;
        float p12 = v.z * (1.8658817f * z2 - 1.119529f);
        float p9 = f2a * fs2, p10 = f1b * fs1, p11 = f0c * v.y, p13 = f0c * v.x, p14 = f1b * fc1, p15 = f2a * fc2;
        col = col + SH3(9) * p9;
        col = col + SH3(10) * p10;
        col = col + SH3(11) * p11;
        col = col + SH3(12) * p12;
        col = col + SH3(13) * p13;
        col = col + SH3(14) * p14;
        col = col + SH3(15) * p15;
        if constexpr (DEG >= 4) if (active >= 4) {
          float f0d = v.z * (-4.683326f * z2 + 2.0071396f), f1c = 3.3116114f * z2 - 0.47308735f, f2b = -1.7701308f * v.z, f3a = 0.62583575f;
          float fc3 = v.x * fc2 - v.y * fs2, fs3 = v.x * fs2 + v.y * fc2;
          float p20 = 1.9843135f * v.z * p12 - 1.0062306f * p6;
          float p16 = f3a * fs3, p17 = f2b * fs2, p18 = f1c * fs1, p19 = f0d * v.y, p21 = f0d * v.x, p22 = f1c * fc1, p23 = f2b * fc2, p24 = f3a * fc3;
          col = col + SH3(16) * p16;
          col = col + SH3(17) * p17;
          col = col + SH3(18) * p18;
          col = col + SH3(19) * p19;
          col = col + SH3(20) * p20;
          col = col + SH3(21) * p21;
          col = col + SH3(22) * p22;
          col = col + SH3(23) * p23;
          col = col + SH3(24) * p24;
        }
      }
    }
  }
  return col;
}
#undef SH3
#undef SHC

// Evaluate the SH basis values b[k] for direction v (so colour = sum_k b[k] * c[k]). Used by the backward.
template <int DEG>
__device__ __forceinline__ void sh_basis(float3 v, float* b) {
  b[0] = SH_C0_DEV;
  if constexpr (DEG >= 1) {
    const float f0a = 0.4886025f;
    b[1] = -f0a * v.y; b[2] = f0a * v.z; b[3] = -f0a * v.x;
    if constexpr (DEG >= 2) {
      float z2 = v.z * v.z, f0b = -1.0925485f * v.z, f1a = 0.54627424f;
      float fc1 = v.x * v.x - v.y * v.y, fs1 = 2.f * v.x * v.y;
      b[4] = f1a * fs1; b[5] = f0b * v.y; b[6] = 0.9461747f * z2 - 0.31539157f; b[7] = f0b * v.x; b[8] = f1a * fc1;
      if constexpr (DEG >= 3) {
        float f0c = -2.285229f * z2 + 0.4570458f, f1b = 1.4453057f * v.z, f2a = -0.5900436f;
        float fc2 = v.x * fc1 - v.y * fs1, fs2 = v.x * fs1 + v.y * fc1;
        b[9] = f2a * fs2; b[10] = f1b * fs1; b[11] = f0c * v.y; b[12] = v.z * (1.8658817f * z2 - 1.119529f); b[13] = f0c * v.x; b[14] = f1b * fc1; b[15] = f2a * fc2;
        if constexpr (DEG >= 4) {
          float f0d = v.z * (-4.683326f * z2 + 2.0071396f), f1c = 3.3116114f * z2 - 0.47308735f, f2b = -1.7701308f * v.z, f3a = 0.62583575f;
          float fc3 = v.x * fc2 - v.y * fs2, fs3 = v.x * fs2 + v.y * fc2;
          b[16] = f3a * fs3; b[17] = f2b * fs2; b[18] = f1c * fs1; b[19] = f0d * v.y; b[20] = 1.9843135f * v.z * b[12] - 1.0062306f * b[6];
          b[21] = f0d * v.x; b[22] = f1c * fc1; b[23] = f2b * fc2; b[24] = f3a * fc3;
        }
      }
    }
  }
}

// Single SH basis value for coefficient k (k uniform across the warp; deg gates the maximum).
__device__ __forceinline__ float sh_basis_k(float3 v, int k) {
  float x = v.x, y = v.y, z = v.z;
  if (k == 0) return SH_C0_DEV;
  if (k < 4) return k == 1 ? -0.4886025f * y : (k == 2 ? 0.4886025f * z : -0.4886025f * x);
  float z2 = z * z, fc1 = x * x - y * y, fs1 = 2.f * x * y;
  if (k < 9) {
    switch (k) {
      case 4: return 0.54627424f * fs1;
      case 5: return -1.0925485f * z * y;
      case 6: return 0.9461747f * z2 - 0.31539157f;
      case 7: return -1.0925485f * z * x;
      default: return 0.54627424f * fc1;
    }
  }
  float fc2 = x * fc1 - y * fs1, fs2 = x * fs1 + y * fc1;
  float f0c = -2.285229f * z2 + 0.4570458f, f1b = 1.4453057f * z;
  float p12 = z * (1.8658817f * z2 - 1.119529f), p6 = 0.9461747f * z2 - 0.31539157f;
  if (k < 16) {
    switch (k) {
      case 9: return -0.5900436f * fs2;
      case 10: return f1b * fs1;
      case 11: return f0c * y;
      case 12: return p12;
      case 13: return f0c * x;
      case 14: return f1b * fc1;
      default: return -0.5900436f * fc2;
    }
  }
  float fc3 = x * fc2 - y * fs2, fs3 = x * fs2 + y * fc2;
  float f0d = z * (-4.683326f * z2 + 2.0071396f), f1c = 3.3116114f * z2 - 0.47308735f, f2b = -1.7701308f * z, f3a = 0.62583575f;
  switch (k) {
    case 16: return f3a * fs3;
    case 17: return f2b * fs2;
    case 18: return f1c * fs1;
    case 19: return f0d * y;
    case 20: return 1.9843135f * z * p12 - 1.0062306f * p6;
    case 21: return f0d * x;
    case 22: return f1c * fc1;
    case 23: return f2b * fc2;
    default: return f3a * fc3;
  }
}

// Depth -> sort key bits: map z in [0, inf) to (2z+1)/(z+1) in [1,2) and keep the top mantissa bits.
__device__ __forceinline__ uint32_t depth_code(float z, int depth_bits) {
  float d = (2.f * z + 1.f) / (z + 1.f);
  uint32_t m = __float_as_uint(d) & 0x7FFFFFu;
  return m >> (23 - depth_bits);
}

// Everything the projection produces that the backward needs to re-derive gradients.
struct ProjIntermediates {
  float3 mean_c;      // view-space mean
  float3 s;           // effective world scales (with floor)
  float3 s2_raw;      // exp(2 log s)
  float comp_floor;   // min-scale opacity compensation
  Mat3 Rq;            // rotation from normalised quat
  Mat3 Mtot;          // R_view * Rq * diag(s)
  float J[6];         // 2x3 projection Jacobian, row-major
  bool clamp_x, clamp_y;
  Sym2 cov_raw;       // before blur
  Sym2 cov;           // blurred
  Sym2 conic;
  float opac;         // effective opacity
  float sig;          // sigmoid(raw)
  float filter_comp;  // mip compensation
  float2 mean2d;
  float ex, ey;
  bool ok;
};

struct CamDev {
  float R[9]; float t[3]; float pos[3]; float fx, fy, cx, cy; int W, H; float lim_pos_x, lim_pos_y, lim_neg_x, lim_neg_y;
};

// Shared projection math (forward). Returns ok=false when the splat is culled.
__device__ __forceinline__ ProjIntermediates project_one(float4 po, float4 q, float4 ls, const CamDev& cam, bool mip) {
  ProjIntermediates r; r.ok = false;
  float3 mean = make_float3(po.x, po.y, po.z);
  Mat3 Rv; for (int i = 0; i < 9; i++) Rv.m[i] = cam.R[i];
  float3 mc = mat3_mul(Rv, mean) + make_float3(cam.t[0], cam.t[1], cam.t[2]);
  r.mean_c = mc;
  if (!(finite3(mc) && mc.z <= 1e10f)) return r;
  if (!(mc.z >= 0.01f)) return r;
  float3 s2 = make_float3(__expf(2.f * ls.x), __expf(2.f * ls.y), __expf(2.f * ls.z));
  r.s2_raw = s2;
  float f2 = ls.w * ls.w;
  float3 s2f = make_float3(s2.x + f2, s2.y + f2, s2.z + f2);
  r.s = make_float3(sqrtf(s2f.x), sqrtf(s2f.y), sqrtf(s2f.z));
  r.comp_floor = ls.w > 0.f ? sqrtf((s2.x * s2.y * s2.z) / (s2f.x * s2f.y * s2f.z)) : 1.f;
  if (!finite3(r.s)) return r;
  float qn2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  if (!(qn2 >= 1e-6f && isfinite(qn2))) return r;
  if (!isfinite(po.w)) return r;
  float inv_qn = rsqrtf(qn2);
  float4 qn = make_float4(q.x * inv_qn, q.y * inv_qn, q.z * inv_qn, q.w * inv_qn);
  r.Rq = quat_to_mat(qn);
  Mat3 M = mat3_mul(Rv, r.Rq);
  // scale columns
  for (int i = 0; i < 3; i++) { M.m[i * 3] *= r.s.x; M.m[i * 3 + 1] *= r.s.y; M.m[i * 3 + 2] *= r.s.z; }
  r.Mtot = M;
  float inv_z = 1.f / mc.z;
  float rx = mc.x * inv_z, ry = mc.y * inv_z;
  float cxr = fminf(fmaxf(rx, cam.lim_neg_x), cam.lim_pos_x), cyr = fminf(fmaxf(ry, cam.lim_neg_y), cam.lim_pos_y);
  r.clamp_x = !(rx <= cam.lim_pos_x && rx >= cam.lim_neg_x);
  r.clamp_y = !(ry <= cam.lim_pos_y && ry >= cam.lim_neg_y);
  float dx = cam.fx * inv_z, dy = cam.fy * inv_z;
  r.J[0] = dx; r.J[1] = 0.f; r.J[2] = -dx * cxr;
  r.J[3] = 0.f; r.J[4] = dy; r.J[5] = -dy * cyr;
  // V = J * M (2x3)
  float V[6];
  for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) V[i * 3 + j] = r.J[i * 3] * M.m[j] + r.J[i * 3 + 1] * M.m[3 + j] + r.J[i * 3 + 2] * M.m[6 + j];
  Sym2 raw; raw.c00 = V[0] * V[0] + V[1] * V[1] + V[2] * V[2]; raw.c01 = V[0] * V[3] + V[1] * V[4] + V[2] * V[5]; raw.c11 = V[3] * V[3] + V[4] * V[4] + V[5] * V[5];
  float max_abs = fmaxf(fabsf(raw.c00), fmaxf(fabsf(raw.c01), fabsf(raw.c11)));
  if (max_abs > 1e18f) { float k = 1e18f / max_abs; raw.c00 *= k; raw.c01 *= k; raw.c11 *= k; }
  r.cov_raw = raw;
  float blur = mip ? 0.1f : 0.3f;
  Sym2 cov = raw; cov.c00 += blur; cov.c11 += blur;
  r.cov = cov;
  r.filter_comp = 1.f;
  if (mip) {
    float det_raw = fmaxf(raw.c00 * raw.c11 - raw.c01 * raw.c01, 0.f);
    float det_bl = cov.c00 * cov.c11 - cov.c01 * cov.c01;
    r.filter_comp = sqrtf(det_raw / det_bl);
  }
  r.sig = sigmoidf_(po.w);
  r.opac = r.sig * r.filter_comp * r.comp_floor;
  if (!(isfinite(cov.c00) && isfinite(cov.c01) && isfinite(cov.c11))) return r;
  float det = cov.c00 * cov.c11 - cov.c01 * cov.c01;
  if (!(det > 0.f)) return r;
  float inv_det = 1.f / det;
  r.conic.c00 = cov.c11 * inv_det; r.conic.c01 = -cov.c01 * inv_det; r.conic.c11 = cov.c00 * inv_det;
  r.mean2d = make_float2(cam.fx * rx + cam.cx, cam.fy * ry + cam.cy);
  if (!(r.opac >= ALPHA_CUTOFF)) return r;
  float power = __logf(r.opac * 255.f);
  float detc = r.conic.c00 * r.conic.c11 - r.conic.c01 * r.conic.c01;
  if (!(detc > 0.f)) return r;
  float inv_detc = 1.f / detc;
  r.ex = sqrtf(2.f * power * r.conic.c11 * inv_detc);
  r.ey = sqrtf(2.f * power * r.conic.c00 * inv_detc);
  if (!(r.ex >= 0.f && r.ey >= 0.f)) return r;
  bool on_screen = r.mean2d.x + r.ex > 0.f && r.mean2d.x - r.ex < (float)cam.W && r.mean2d.y + r.ey > 0.f && r.mean2d.y - r.ey < (float)cam.H;
  if (!on_screen) return r;
  r.ok = true;
  return r;
}

}  // namespace b2c
