#include "gpu/cage.h"
#include "util/log.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>

namespace b2c {
namespace {

constexpr int BIND_TILE = 1024;
constexpr int RING_MAX = 32;   // triangles tested around the nearest vertex (a vertex of a clean mesh has ~6)

// Triangle frame: e1 along the first edge, n the face normal, e2 = n x e1; R's columns are (e1, e2, n), row-major.
// Size k = sqrt(|(b - a) x (c - a)|), i.e. sqrt(2 * area). Degenerate triangles return k = 0 and the identity.
__device__ __forceinline__ float tri_frame(float3 a, float3 b, float3 c, float* R) {
  float3 u = b - a, w = c - a;
  float3 nn = make_float3(u.y * w.z - u.z * w.y, u.z * w.x - u.x * w.z, u.x * w.y - u.y * w.x);
  float nl = len3(nn), ul = len3(u);
  if (!(nl > 1e-20f) || !(ul > 1e-12f)) { for (int k = 0; k < 9; k++) R[k] = (k % 4 == 0) ? 1.f : 0.f; return 0.f; }
  float3 e1 = u * (1.f / ul), n = nn * (1.f / nl);
  float3 e2 = make_float3(n.y * e1.z - n.z * e1.y, n.z * e1.x - n.x * e1.z, n.x * e1.y - n.y * e1.x);
  R[0] = e1.x; R[1] = e2.x; R[2] = n.x;
  R[3] = e1.y; R[4] = e2.y; R[5] = n.y;
  R[6] = e1.z; R[7] = e2.z; R[8] = n.z;
  return sqrtf(nl);
}

// Rotation matrix (row-major) to a (w, x, y, z) quaternion, stored as float4(w, x, y, z) like Model::quat.
__device__ __forceinline__ float4 mat_to_quat(const float* m) {
  float tr = m[0] + m[4] + m[8], w, x, y, z;
  if (tr > 0.f) { float s = sqrtf(tr + 1.f) * 2.f; w = 0.25f * s; x = (m[7] - m[5]) / s; y = (m[2] - m[6]) / s; z = (m[3] - m[1]) / s; }
  else if (m[0] > m[4] && m[0] > m[8]) { float s = sqrtf(1.f + m[0] - m[4] - m[8]) * 2.f; w = (m[7] - m[5]) / s; x = 0.25f * s; y = (m[1] + m[3]) / s; z = (m[2] + m[6]) / s; }
  else if (m[4] > m[8]) { float s = sqrtf(1.f + m[4] - m[0] - m[8]) * 2.f; w = (m[2] - m[6]) / s; x = (m[1] + m[3]) / s; y = 0.25f * s; z = (m[5] + m[7]) / s; }
  else { float s = sqrtf(1.f + m[8] - m[0] - m[4]) * 2.f; w = (m[3] - m[1]) / s; x = (m[2] + m[6]) / s; y = (m[5] + m[7]) / s; z = 0.25f * s; }
  return make_float4(w, x, y, z);
}

// Hamilton product a * b of (w, x, y, z) quaternions held as float4(w, x, y, z).
__device__ __forceinline__ float4 quat_mul(float4 a, float4 b) {
  return make_float4(a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
                     a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
                     a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
                     a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x);
}

// Closest point on triangle (a, b, c) to p (Ericson, Real-Time Collision Detection 5.1.5): barycentrics (b1, b2) of
// the foot point, b0 = 1 - b1 - b2. Returns the squared distance.
__device__ __forceinline__ float closest_on_tri(float3 p, float3 a, float3 b, float3 c, float& b1, float& b2) {
  float3 ab = b - a, ac = c - a, ap = p - a;
  float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
  if (d1 <= 0.f && d2 <= 0.f) { b1 = 0.f; b2 = 0.f; }
  else {
    float3 bp = p - b; float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    float3 cp = p - c; float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    float vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
    if (d3 >= 0.f && d4 <= d3) { b1 = 1.f; b2 = 0.f; }
    else if (d6 >= 0.f && d5 <= d6) { b1 = 0.f; b2 = 1.f; }
    else if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) { b1 = d1 / (d1 - d3); b2 = 0.f; }
    else if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) { b1 = 0.f; b2 = d2 / (d2 - d6); }
    else if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) { float t = (d4 - d3) / ((d4 - d3) + (d5 - d6)); b1 = 1.f - t; b2 = t; }
    else { float den = 1.f / (va + vb + vc); b1 = vb * den; b2 = vc * den; }
  }
  float3 q = a + ab * b1 + ac * b2;
  float3 d = p - q;
  return dot3(d, d);
}

// Nearest vertex (restricted to one layer when want >= 0), then the nearest triangle among the ones around it.
__global__ void bind_kernel(int n, const float4* __restrict__ pos, const int* __restrict__ want_layer, int nv,
                            const float3* __restrict__ verts, const int* __restrict__ vert_layer,
                            const int* __restrict__ vf_off, const int* __restrict__ vf_idx, const int3* __restrict__ faces,
                            int* __restrict__ out_f, float2* __restrict__ out_b) {
  __shared__ float3 sv[BIND_TILE];
  __shared__ int sl[BIND_TILE];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float3 p = make_float3(0.f, 0.f, 0.f); int want = -1;
  if (i < n) { float4 q = pos[i]; p = make_float3(q.x, q.y, q.z); want = want_layer[i]; }
  float best = 3.4e38f; int bi = -1;
  for (int base = 0; base < nv; base += BIND_TILE) {
    int m = min(BIND_TILE, nv - base);
    __syncthreads();
    for (int k = threadIdx.x; k < m; k += blockDim.x) { sv[k] = verts[base + k]; sl[k] = vert_layer[base + k]; }
    __syncthreads();
    if (i < n) {
      for (int k = 0; k < m; k++) {
        if (sl[k] < 0 || (want >= 0 && sl[k] != want)) continue;
        float3 d = sv[k] - p; float d2 = dot3(d, d);
        if (d2 < best) { best = d2; bi = base + k; }
      }
    }
  }
  if (i >= n) return;
  int bf = -1; float bb1 = 0.f, bb2 = 0.f; float bd = 3.4e38f;
  if (bi >= 0) {
    int e = min(vf_off[bi + 1], vf_off[bi] + RING_MAX);
    for (int k = vf_off[bi]; k < e; k++) {
      int f = vf_idx[k]; int3 t = faces[f];
      float b1, b2; float d2 = closest_on_tri(p, verts[t.x], verts[t.y], verts[t.z], b1, b2);
      if (d2 < bd) { bd = d2; bf = f; bb1 = b1; bb2 = b2; }
    }
  }
  out_f[i] = bf; out_b[i] = make_float2(bb1, bb2);
}

__global__ void tri_canon_kernel(int nf, const int3* __restrict__ faces, const float3* __restrict__ v, float4* __restrict__ q0, float* __restrict__ k0) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= nf) return;
  int3 t = faces[f]; float R[9];
  k0[f] = tri_frame(v[t.x], v[t.y], v[t.z], R);
  q0[f] = mat_to_quat(R);
}

// Offset of each bound splat from its foot point, in the canonical triangle frame, divided by the triangle size.
__global__ void offset_kernel(int n, const float4* __restrict__ pos, const int* __restrict__ bf, const float2* __restrict__ bb,
                              const int3* __restrict__ faces, const float3* __restrict__ v, const float* __restrict__ k0, float3* __restrict__ off) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int f = bf[i]; if (f < 0) { off[i] = make_float3(0.f, 0.f, 0.f); return; }
  int3 t = faces[f]; float2 b = bb[i]; float R[9];
  float3 a = v[t.x], bv = v[t.y], c = v[t.z];
  tri_frame(a, bv, c, R);
  float3 foot = a * (1.f - b.x - b.y) + bv * b.x + c * b.y;
  float4 q = pos[i]; float3 d = make_float3(q.x, q.y, q.z) - foot;
  float kk = k0[f] > 0.f ? 1.f / k0[f] : 0.f;
  off[i] = make_float3((R[0] * d.x + R[3] * d.y + R[6] * d.z) * kk, (R[1] * d.x + R[4] * d.y + R[7] * d.z) * kk, (R[2] * d.x + R[5] * d.y + R[8] * d.z) * kk);
}

__global__ void verts_view_kernel(int nv, const float3* __restrict__ posed, const float3* __restrict__ delta, float3* __restrict__ out) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nv) return;
  float3 p = posed[k];
  if (delta) p = p + delta[k];
  out[k] = p;
}

__global__ void tri_pose_kernel(int nf, const int3* __restrict__ faces, const float3* __restrict__ v, const float4* __restrict__ q0, const float* __restrict__ k0,
                                float4* __restrict__ dq, float* __restrict__ kout, float* __restrict__ Rout, float max_growth,
                                float* __restrict__ ratio_out) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= nf) return;
  int3 t = faces[f]; float R[9];
  float k = tri_frame(v[t.x], v[t.y], v[t.z], R);
  for (int j = 0; j < 9; j++) Rout[(size_t)f * 9 + j] = R[j];
  if (ratio_out) ratio_out[f] = k0[f] > 0.f ? k / k0[f] : 1.f;   // unclamped: what the stretch fade reads
  // max_growth > 0: the size a bound splat's offset and scales follow stays within [k0 / g, k0 * g] of the canonical
  // triangle's. LBS stretches armpit / elbow triangles 2-4x; a splat sized and offset by that factor blooms off the skin.
  if (max_growth > 0.f && k0[f] > 0.f) k = fminf(fmaxf(k, k0[f] / max_growth), k0[f] * max_growth);
  kout[f] = k;
  float4 q = mat_to_quat(R), c = q0[f];
  dq[f] = quat_mul(q, make_float4(c.x, -c.y, -c.z, -c.w));   // R R0^T
}

// One splat's pose under binding (face f, barycentrics b, offset o): position, rotation, log-scales, SH frame.
__device__ __forceinline__ void pose_bound(int f, float2 b, float3 o, float4 p, float4 q0, float4 ls, const int3* __restrict__ faces,
                                           const float3* __restrict__ v, const float4* __restrict__ dq, const float* __restrict__ k,
                                           const float* __restrict__ k0, const float* __restrict__ Rt,
                                           const float* __restrict__ ratio, float fade0, float fade1, float fillv, int2 gate, const float3* __restrict__ v0, float fill0, float fill1,
                                           float4& pos_o, float4& quat_o, float4& ls_o, float4& shf_o) {
  int3 t = faces[f];
  float3 foot = v[t.x] * (1.f - b.x - b.y) + v[t.y] * b.x + v[t.z] * b.y;
  const float* R = Rt + (size_t)f * 9; float kk = k[f];
  float3 d = make_float3(R[0] * o.x + R[1] * o.y + R[2] * o.z, R[3] * o.x + R[4] * o.y + R[5] * o.z, R[6] * o.x + R[7] * o.y + R[8] * o.z) * kk;
  float op = p.w;
  if (ratio) {   // stretch fade: a splat riding a triangle LBS stretches into a membrane (an armpit web as the arm lifts) fades
    // out, linearly in opacity from fade0 to fade1 x the canonical triangle size; a filler splat (fillv > 0) fades IN instead
    float fa = 1.f;
    float rg = ratio[f];
    if (gate.x >= 0 && gate.y >= 0) { float d0 = len3(v0[gate.x] - v0[gate.y]); rg = d0 > 1e-6f ? len3(v[gate.x] - v[gate.y]) / d0 : 1.f; }
    if (fillv > 0.f) { if (fill1 > fill0) fa = fminf(fmaxf((rg - fill0) / (fill1 - fill0), 0.f), 1.f); }
    else if (fillv < 0.f) { if (fill1 > fill0) fa = fminf(fmaxf((fill1 - rg) / (fill1 - fill0), 0.f), 1.f); }
    else if (fade1 > fade0) fa = fminf(fmaxf((fade1 - ratio[f]) / (fade1 - fade0), 0.f), 1.f);
    if (fa < 1.f) {
      float sg = fa / (1.f + expf(-p.w));
      sg = fminf(fmaxf(sg, 1e-6f), 1.f - 1e-6f);
      op = logf(sg / (1.f - sg));
    }
  }
  pos_o = make_float4(foot.x + d.x, foot.y + d.y, foot.z + d.z, op);
  float4 r = dq[f];
  quat_o = quat_mul(r, q0);
  float g = (k0[f] > 0.f && kk > 0.f) ? logf(kk / k0[f]) : 0.f;
  ls_o = make_float4(ls.x + g, ls.y + g, ls.z + g, ls.w);
  shf_o = r;
}

__global__ void splat_pose_kernel(int n, const float4* __restrict__ pos, const float4* __restrict__ quat, const float4* __restrict__ lscale,
                                  const int* __restrict__ bf, const float2* __restrict__ bb, const float3* __restrict__ off,
                                  const int3* __restrict__ faces, const float3* __restrict__ v, const float4* __restrict__ dq,
                                  const float* __restrict__ k, const float* __restrict__ k0, const float* __restrict__ Rt,
                                  const float* __restrict__ ratio, float fade0, float fade1,
                                  const float* __restrict__ fill, const int2* __restrict__ gate, const float3* __restrict__ v0, float fill0, float fill1,
                                  float4* __restrict__ pos_out, float4* __restrict__ quat_out, float4* __restrict__ ls_out, float4* __restrict__ shf_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int f = bf[i];
  float4 p = pos[i];
  if (f < 0) { pos_out[i] = p; quat_out[i] = quat[i]; ls_out[i] = lscale[i]; shf_out[i] = make_float4(1.f, 0.f, 0.f, 0.f); return; }
  pose_bound(f, bb[i], off[i], p, quat[i], lscale[i], faces, v, dq, k, k0, Rt, ratio, fade0, fade1, fill ? fill[i] : 0.f, gate ? gate[i] : make_int2(-1, -1), v0, fill0, fill1, pos_out[i], quat_out[i], ls_out[i], shf_out[i]);
}

// Dual binding: triangle B's foot point and offset for listed splats (as bind_kernel + offset_kernel do for A).
__global__ void alt_bind_kernel(int na, const int* __restrict__ idx, const int* __restrict__ fb, const float4* __restrict__ pos,
                                const int3* __restrict__ faces, const float3* __restrict__ v, const float* __restrict__ k0,
                                float2* __restrict__ out_b, float3* __restrict__ out_off) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= na) return;
  int f = fb[j]; int3 t = faces[f]; float4 q = pos[idx[j]]; float3 p = make_float3(q.x, q.y, q.z);
  float b1, b2; closest_on_tri(p, v[t.x], v[t.y], v[t.z], b1, b2);
  out_b[j] = make_float2(b1, b2);
  float R[9]; tri_frame(v[t.x], v[t.y], v[t.z], R);
  float3 foot = v[t.x] * (1.f - b1 - b2) + v[t.y] * b1 + v[t.z] * b2, d = p - foot;
  float kk = k0[f] > 0.f ? 1.f / k0[f] : 0.f;
  out_off[j] = make_float3((R[0] * d.x + R[3] * d.y + R[6] * d.z) * kk, (R[1] * d.x + R[4] * d.y + R[7] * d.z) * kk, (R[2] * d.x + R[5] * d.y + R[8] * d.z) * kk);
}

// Dual binding: blend the listed splats' A pose (already in the *_out buffers) with their B pose by w.
__global__ void alt_pose_kernel(int na, const int* __restrict__ idx, const int* __restrict__ fb, const float2* __restrict__ bb,
                                const float3* __restrict__ off, const float* __restrict__ wv, const float4* __restrict__ pos,
                                const float4* __restrict__ quat, const float4* __restrict__ lscale, const int3* __restrict__ faces,
                                const float3* __restrict__ v, const float4* __restrict__ dq, const float* __restrict__ k,
                                const float* __restrict__ k0, const float* __restrict__ Rt, float4* __restrict__ pos_out,
                                float4* __restrict__ quat_out, float4* __restrict__ ls_out, float4* __restrict__ shf_out,
                                float3* __restrict__ dpos) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= na) return;
  int i = idx[j]; float w = wv[j];
  float4 pb, qb, lb, sb;
  pose_bound(fb[j], bb[j], off[j], pos[i], quat[i], lscale[i], faces, v, dq, k, k0, Rt, nullptr, 0.f, 0.f, 0.f, make_int2(-1, -1), nullptr, 0.f, 0.f, pb, qb, lb, sb);
  float4 pa = pos_out[i], la = ls_out[i];
  dpos[j] = make_float3(pb.x - pa.x, pb.y - pa.y, pb.z - pa.z);
  pos_out[i] = make_float4(pa.x + w * (pb.x - pa.x), pa.y + w * (pb.y - pa.y), pa.z + w * (pb.z - pa.z), pa.w);
  ls_out[i] = make_float4(la.x + w * (lb.x - la.x), la.y + w * (lb.y - la.y), la.z + w * (lb.z - la.z), la.w);
  if (w > 0.5f) { quat_out[i] = qb; shf_out[i] = sb; }
}

}  // namespace

bool CageRig::load(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  auto need = [&](void* dst, size_t bytes) { if (fread(dst, 1, bytes, f) != bytes) { fclose(f); throw std::runtime_error("cage '" + path + "' is truncated"); } };
  char magic[8]; need(magic, 8);
  if (memcmp(magic, "B2CCAGE1", 8) != 0) { fclose(f); throw std::runtime_error("'" + path + "' is not a b2ctrain cage v1 (bad magic)"); }
  int32_t hdr[4]; need(hdr, sizeof(hdr));
  const int nl = hdr[0]; nv = hdr[1]; nf = hdr[2]; nframes = hdr[3];
  if (nl <= 0 || nl > 32 || nv <= 0 || nf <= 0 || nframes <= 0) { fclose(f); throw std::runtime_error("cage '" + path + "': unsupported layout"); }
  layers.resize(nl);
  for (auto& L : layers) { int32_t a[4]; uint32_t bits; need(a, sizeof(a)); need(&bits, 4); L = {a[0], a[1], a[2], a[3], bits}; }
  std::vector<float3> v(nv); need(v.data(), (size_t)nv * sizeof(float3));
  std::vector<int3> fc(nf); need(fc.data(), (size_t)nf * sizeof(int3));
  std::vector<char> nm((size_t)nframes * 64); need(nm.data(), nm.size());
  std::vector<float3> ps((size_t)nframes * nv); need(ps.data(), ps.size() * sizeof(float3));
  std::vector<float> th;   // optional B2COPEN1 section (see cage.h)
  char tag[8];
  has_open_theta = false;
  if (fread(tag, 1, 8, f) == 8) {
    if (memcmp(tag, "B2COPEN1", 8) != 0) { fclose(f); throw std::runtime_error("cage '" + path + "': unknown section after the posed vertices"); }
    float rm[2]; need(rm, sizeof(rm)); open_radius = rm[0]; open_min_dist = rm[1];
    std::vector<__half> h((size_t)nframes * nv); need(h.data(), h.size() * sizeof(__half));
    th.resize(h.size());
    for (size_t i = 0; i < h.size(); i++) th[i] = __half2float(h[i]);
    has_open_theta = true;
  }
  fclose(f);
  face_layer_h.assign(nf, -1);
  for (int l = 0; l < nl; l++) {
    const Layer& L = layers[l];
    if (L.v_off < 0 || L.v_count <= 0 || L.v_off + L.v_count > nv || L.f_off < 0 || L.f_count <= 0 || L.f_off + L.f_count > nf)
      throw std::runtime_error("cage '" + path + "': layer range out of bounds");
    for (int k = L.f_off; k < L.f_off + L.f_count; k++) {
      const int3 t = fc[k];
      for (int x : {t.x, t.y, t.z}) if (x < L.v_off || x >= L.v_off + L.v_count) throw std::runtime_error("cage '" + path + "': a face indexes outside its layer");
      face_layer_h[k] = l;
    }
  }
  for (int k = 0; k < nf; k++) if (face_layer_h[k] < 0) throw std::runtime_error("cage '" + path + "': a face belongs to no layer");
  names.clear(); for (int i = 0; i < nframes; i++) { std::string s(nm.data() + (size_t)i * 64, 64); s = s.c_str(); names.push_back(s); }
  verts0.upload(v); faces.upload(fc); posed.upload(ps);
  if (has_open_theta) open_theta.upload(th);
  verts_view.reserve(nv); tri_q0.reserve(nf); tri_k0.reserve(nf); tri_dq.reserve(nf); tri_k.reserve(nf); tri_R.reserve((size_t)nf * 9);
  tri_canon_kernel<<<div_up(nf, 256), 256>>>(nf, faces, verts0, tri_q0, tri_k0);
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaDeviceSynchronize());
  return true;
}

int CageRig::frame_index(const std::string& name) const {
  for (int i = 0; i < nframes; i++) if (names[i] == name) return i;
  return -1;
}

void CageRig::bind(const Model& m, const std::vector<float>& labels, float min_conf, cudaStream_t stream) {
  const int n = m.n;
  // Which layer owns each splat's class (-1: any layer).
  std::vector<int> want(n, -1);
  n_fallback = 0;
  for (int i = 0; i < n; i++) {
    if (labels.empty()) { n_fallback++; continue; }
    int c = (int)std::lround(labels[(size_t)i * 2]); float conf = labels[(size_t)i * 2 + 1];
    int owner = -1;
    if (c >= 0 && c < 32 && conf >= min_conf)
      for (size_t l = 0; l < layers.size(); l++) if (layers[l].class_bits & (1u << c)) { owner = (int)l; break; }
    want[i] = owner;
    if (owner < 0) n_fallback++;
  }
  std::vector<int> vert_layer(nv, 0);
  for (size_t l = 0; l < layers.size(); l++) for (int k = layers[l].v_off; k < layers[l].v_off + layers[l].v_count; k++) vert_layer[k] = (int)l;
  // Vertex -> incident faces (CSR).
  auto fc = faces.download(nf, stream);
  std::vector<int> off(nv + 1, 0);
  for (auto& t : fc) { off[t.x + 1]++; off[t.y + 1]++; off[t.z + 1]++; }
  for (int k = 0; k < nv; k++) off[k + 1] += off[k];
  // A vertex no face uses (e.g. the body's lip seam once its expression-unstable faces are dropped) cannot bind a
  // splat: mark it layer -1 so the nearest-vertex search skips it (it would otherwise leave the splat unbound, static).
  for (int k = 0; k < nv; k++) if (off[k + 1] == off[k]) vert_layer[k] = -1;
  std::vector<int> idx(off[nv]), fill(off.begin(), off.end() - 1);
  for (int fi = 0; fi < nf; fi++) { const int3 t = fc[fi]; idx[fill[t.x]++] = fi; idx[fill[t.y]++] = fi; idx[fill[t.z]++] = fi; }
  DevBuf<int> d_want, d_vl, d_off, d_idx;
  d_want.upload(want, stream); d_vl.upload(vert_layer, stream); d_off.upload(off, stream); d_idx.upload(idx, stream);
  bind_f.reserve(n); bind_b.reserve(n); bind_off.reserve(n);
  bind_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, m.pos_op, d_want, nv, verts0, d_vl, d_off, d_idx, faces, bind_f, bind_b);
  CUDA_KERNEL_CHECK();
  offset_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, m.pos_op, bind_f, bind_b, faces, verts0, tri_k0, bind_off);
  CUDA_KERNEL_CHECK();
  pos_view.reserve(n); quat_view.reserve(n); lscale_view.reserve(n); sh_frame.reserve(n);
  bound_n = n;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void CageRig::set_fill(const std::vector<float>& f, const std::vector<float>& g, cudaStream_t stream) {
  if ((int)f.size() != bound_n) throw std::runtime_error("CageRig::set_fill: one value per bound splat expected");
  fill.upload(f, stream); has_fill = true;
  if (!g.empty()) {
    if ((int)g.size() != bound_n * 2) throw std::runtime_error("CageRig::set_fill: one vertex pair per bound splat expected");
    std::vector<int2> gi(g.size() / 2);
    for (size_t i = 0; i < gi.size(); i++) { int x = (int)std::lround(g[i * 2]), y = (int)std::lround(g[i * 2 + 1]); gi[i] = (x >= 0 && x < nv && y >= 0 && y < nv) ? make_int2(x, y) : make_int2(-1, -1); }
    gate.upload(gi, stream); has_gate = true;
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void CageRig::enable_delta(cudaStream_t stream) {
  delta.reserve((size_t)nframes * nv); delta.zero(stream); has_delta = true;
}

void CageRig::pose(int fr, const Model& m, cudaStream_t stream) {
  if (bound_n != m.n) throw std::runtime_error("CageRig::pose: the model changed size since bind()");
  frame = fr;
  verts_view_kernel<<<div_up(nv, 256), 256, 0, stream>>>(nv, posed.ptr + (size_t)fr * nv, has_delta ? delta.ptr + (size_t)fr * nv : nullptr, verts_view);
  CUDA_KERNEL_CHECK();
  const bool fade = (fade_end > fade_start && fade_start > 0.f) || (has_fill && fill_end > fill_start);
  if (fade) tri_ratio.reserve(nf);
  tri_pose_kernel<<<div_up(nf, 256), 256, 0, stream>>>(nf, faces, verts_view, tri_q0, tri_k0, tri_dq, tri_k, tri_R, max_growth,
                                                       fade ? tri_ratio.ptr : nullptr);
  CUDA_KERNEL_CHECK();
  splat_pose_kernel<<<div_up(m.n, 256), 256, 0, stream>>>(m.n, m.pos_op, m.quat, m.lscale, bind_f, bind_b, bind_off, faces, verts_view, tri_dq, tri_k, tri_k0, tri_R,
                                                           fade ? tri_ratio.ptr : nullptr, fade_start, fade_end,
                                                           has_fill ? fill.ptr : nullptr, has_gate ? gate.ptr : nullptr, verts0, fill_start, fill_end,
                                                           pos_view, quat_view, lscale_view, sh_frame);
  CUDA_KERNEL_CHECK();
  if (n_alt > 0) {
    alt_pose_kernel<<<div_up(n_alt, 256), 256, 0, stream>>>(n_alt, alt_idx, alt_f, alt_b, alt_off, alt_w, m.pos_op, m.quat, m.lscale, faces, verts_view,
                                                            tri_dq, tri_k, tri_k0, tri_R, pos_view, quat_view, lscale_view, sh_frame, alt_dpos);
    CUDA_KERNEL_CHECK();
  }
}

std::vector<float> CageRig::face_stretch(cudaStream_t stream) {
  std::vector<float> k0 = tri_k0.download(nf, stream), best(nf, 1.f);
  for (int fr = 0; fr < nframes; fr++) {
    verts_view_kernel<<<div_up(nv, 256), 256, 0, stream>>>(nv, posed.ptr + (size_t)fr * nv, nullptr, verts_view);
    tri_pose_kernel<<<div_up(nf, 256), 256, 0, stream>>>(nf, faces, verts_view, tri_q0, tri_k0, tri_dq, tri_k, tri_R, 0.f, nullptr);
    CUDA_KERNEL_CHECK();
    std::vector<float> k = tri_k.download(nf, stream);
    for (int f = 0; f < nf; f++) if (k0[f] > 0.f) best[f] = std::max(best[f], k[f] / k0[f]);
  }
  return best;
}

std::vector<float> CageRig::splat_stretch_weights(const std::vector<float>& stretch, float weight, cudaStream_t stream) {
  std::vector<int> bf = bind_f.download(bound_n, stream);
  std::vector<float> w(bound_n, 0.f);
  for (int i = 0; i < bound_n; i++) if (bf[i] >= 0) w[i] = weight * logf(std::max(stretch[bf[i]], 1.f));
  return w;
}

void CageRig::refresh_offsets(const Model& m, cudaStream_t stream) {
  offset_kernel<<<div_up(m.n, 256), 256, 0, stream>>>(m.n, m.pos_op, bind_f, bind_b, faces, verts0, tri_k0, bind_off);
  CUDA_KERNEL_CHECK();
}

void CageRig::bind_alt(const Model& m, const std::vector<int>& splat, const std::vector<int>& face_b, const std::vector<float>& w, cudaStream_t stream) {
  if (bound_n != m.n) throw std::runtime_error("CageRig::bind_alt: bind() first");
  n_alt = (int)splat.size();
  if (n_alt == 0) return;
  for (int j = 0; j < n_alt; j++) {
    if (splat[j] < 0 || splat[j] >= m.n) throw std::runtime_error("alt binding: splat index out of range");
    if (face_b[j] < 0 || face_b[j] >= nf) throw std::runtime_error("alt binding: face index out of range");
  }
  alt_idx.upload(splat, stream); alt_f.upload(face_b, stream); alt_w.upload(w, stream);
  alt_b.reserve(n_alt); alt_off.reserve(n_alt); alt_dpos.reserve(n_alt);
  alt_bind_kernel<<<div_up(n_alt, 256), 256, 0, stream>>>(n_alt, alt_idx, alt_f, m.pos_op, faces, verts0, tri_k0, alt_b, alt_off);
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool load_alt_binding(const std::string& path, std::vector<int>& splat, std::vector<int>& face_b, std::vector<float>& w) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[8]; int32_t n = 0;
  if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "B2CALT01", 8) != 0 || fread(&n, 4, 1, f) != 1 || n < 0) { fclose(f); throw std::runtime_error("'" + path + "' is not a b2ctrain alt binding"); }
  splat.resize(n); face_b.resize(n); w.resize(n);
  for (int j = 0; j < n; j++) {
    int32_t s2[2]; float ww;
    if (fread(s2, 4, 2, f) != 2 || fread(&ww, 4, 1, f) != 1) { fclose(f); throw std::runtime_error("alt binding '" + path + "' is truncated"); }
    splat[j] = s2[0]; face_b[j] = s2[1]; w[j] = ww;
  }
  fclose(f);
  return true;
}

void save_alt_binding(const std::string& path, const std::vector<int>& splat, const std::vector<int>& face_b, const std::vector<float>& w) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write '" + path + "'");
  int32_t n = (int32_t)splat.size();
  fwrite("B2CALT01", 1, 8, f); fwrite(&n, 4, 1, f);
  for (int j = 0; j < n; j++) { int32_t s2[2] = {splat[j], face_b[j]}; fwrite(s2, 4, 2, f); fwrite(&w[j], 4, 1, f); }
  fclose(f);
}

}  // namespace b2c
