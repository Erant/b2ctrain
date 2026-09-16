// `b2ctrain mesh-refine`: move vertices so face normals match the Sapiens normal maps; the cap's region stays as fused.
// Reference: out/mesh/tools/normal_refine.py (+ blend_face.py for the kept region).
#include "mesh/common.h"
#include "mesh/raster.h"
#include "mesh/geom.cuh"
#include "util/log.h"
#include <cmath>
#include <algorithm>

namespace b2c {

namespace {

// Per vertex: visibility against the mesh's own depth at this view, the normal map's world normal, facing weight.
__global__ void target_accum_kernel(int nv, const float3* __restrict__ v, const float3* __restrict__ vn, CameraGPU cam, const float* __restrict__ zbuf, const uint8_t* __restrict__ nmap,
                                    float tol, float power, float3* __restrict__ acc, float* __restrict__ wsum) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nv) return;
  float3 p = v[i]; float u, vv; float z = project_point(cam, p.x, p.y, p.z, u, vv);
  if (!(z > 0.f && u >= 1.f && u < cam.W - 2 && vv >= 1.f && vv < cam.H - 2)) return;
  int ui = min(max((int)floorf(u), 0), cam.W - 1), vi = min(max((int)floorf(vv), 0), cam.H - 1);
  size_t px = (size_t)vi * cam.W + ui;
  float d = zbuf[px];
  if (!(d > 0.f) || fabsf(d - z) >= tol) return;
  const uint8_t* q = nmap + px * 4;
  if (q[3] <= 127) return;
  // [+X, -Y, -Z] encoding in camera space, rotated to the world with the camera-to-world rotation (R^T).
  float nx = q[0] / 255.f * 2.f - 1.f, ny = 1.f - q[1] / 255.f * 2.f, nz = 1.f - q[2] / 255.f * 2.f;
  float3 nw = make_float3(cam.R[0] * nx + cam.R[3] * ny + cam.R[6] * nz, cam.R[1] * nx + cam.R[4] * ny + cam.R[7] * nz, cam.R[2] * nx + cam.R[5] * ny + cam.R[8] * nz);
  float3 dn = p - make_float3(cam.pos[0], cam.pos[1], cam.pos[2]); dn = dn * (1.f / fmaxf(len3(dn), 1e-9f));
  float facing = fminf(fmaxf(-dot3(vn[i], dn), 0.f), 1.f); float w = powf(facing, power);
  if (w <= 0.f) return;
  acc[i] = acc[i] + nw * w; wsum[i] += w;
}
__global__ void target_finish_kernel(int nv, const float3* __restrict__ acc, const float* __restrict__ wsum, float min_w, float3* __restrict__ tgt, uint8_t* __restrict__ ok) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nv) return;
  float3 t = acc[i] * (1.f / fmaxf(wsum[i], 1e-9f)); float l = len3(t);
  ok[i] = wsum[i] > min_w && l > 0.3f; tgt[i] = t * (1.f / fmaxf(l, 1e-9f));
}
__global__ void face_target_kernel(int nf, const uint3* __restrict__ f, const float3* __restrict__ tgt, const uint8_t* __restrict__ ok, float3* __restrict__ ftgt, uint8_t* __restrict__ fok) {
  int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= nf) return;
  uint3 fc = f[t]; float3 m = (tgt[fc.x] + tgt[fc.y] + tgt[fc.z]) * (1.f / 3.f); float l = len3(m);
  ftgt[t] = m * (1.f / fmaxf(l, 1e-6f)); fok[t] = ok[fc.x] && ok[fc.y] && ok[fc.z];
}
// Normal loss and its gradient: L = sum_f ||n_f - t_f||^2 / nF_ok.
__global__ void normal_grad_kernel(int nf, const float3* __restrict__ v, const uint3* __restrict__ f, const float3* __restrict__ ftgt, const uint8_t* __restrict__ fok, float scale, float3* __restrict__ grad, float* __restrict__ loss) {
  int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= nf || !fok[t]) return;
  uint3 fc = f[t]; float3 a = v[fc.x], b = v[fc.y], c = v[fc.z];
  float3 e1 = b - a, e2 = c - a;
  float3 cr = make_float3(e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x);
  float l = fmaxf(len3(cr), 1e-9f); float3 n = cr * (1.f / l);
  float3 d = n - ftgt[t];
  atomicAdd(loss, dot3(d, d) * scale);
  float3 gn = d * (2.f * scale);                                 // dL/dn
  float3 gc = (gn - n * dot3(n, gn)) * (1.f / l);                // dL/dcross
  float3 ge1 = make_float3(e2.y * gc.z - e2.z * gc.y, e2.z * gc.x - e2.x * gc.z, e2.x * gc.y - e2.y * gc.x);  // e2 x gc
  float3 ge2 = make_float3(gc.y * e1.z - gc.z * e1.y, gc.z * e1.x - gc.x * e1.z, gc.x * e1.y - gc.y * e1.x);  // gc x e1
  float3 ga = (ge1 + ge2) * -1.f;
  atomicAdd(&grad[fc.x].x, ga.x); atomicAdd(&grad[fc.x].y, ga.y); atomicAdd(&grad[fc.x].z, ga.z);
  atomicAdd(&grad[fc.y].x, ge1.x); atomicAdd(&grad[fc.y].y, ge1.y); atomicAdd(&grad[fc.y].z, ge1.z);
  atomicAdd(&grad[fc.z].x, ge2.x); atomicAdd(&grad[fc.z].y, ge2.y); atomicAdd(&grad[fc.z].z, ge2.z);
}
__global__ void laplacian_kernel(int nv, const float3* __restrict__ v, const int* __restrict__ start, const int* __restrict__ nbr, float3* __restrict__ lap) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nv) return;
  int s = start[i], e = start[i + 1]; float3 sum = make_float3(0, 0, 0);
  for (int k = s; k < e; k++) sum = sum + v[nbr[k]];
  lap[i] = e > s ? sum * (1.f / (e - s)) - v[i] : make_float3(0, 0, 0);
}
// Laplacian + positional terms: L = lam_lap * 1e4/nV sum |lap|^2 + lam_pos * 1e4/nV sum |V - V0|^2.
__global__ void reg_grad_kernel(int nv, const float3* __restrict__ v, const float3* __restrict__ v0, const float3* __restrict__ lap, const int* __restrict__ start, const int* __restrict__ nbr,
                                float k_lap, float k_pos, float3* __restrict__ grad, float* __restrict__ loss) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nv) return;
  float3 g = lap[i] * -1.f;
  for (int k = start[i]; k < start[i + 1]; k++) { int j = nbr[k]; int dj = start[j + 1] - start[j]; g = g + lap[j] * (1.f / dj); }
  float3 dp = v[i] - v0[i];
  grad[i] = grad[i] + g * (2.f * k_lap) + dp * (2.f * k_pos);
  atomicAdd(loss, k_lap * dot3(lap[i], lap[i]));
  atomicAdd(loss + 1, k_pos * dot3(dp, dp));
}
__global__ void adam_kernel(int nv, float3* __restrict__ v, float3* __restrict__ grad, float3* __restrict__ m, float3* __restrict__ s, float lr, float b1, float b2, float eps, int t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nv) return;
  float3 g = grad[i]; float3 mi = m[i] * b1 + g * (1.f - b1); float3 si = s[i] * b2 + make_float3(g.x * g.x, g.y * g.y, g.z * g.z) * (1.f - b2);
  m[i] = mi; s[i] = si;
  float c1 = 1.f - powf(b1, (float)t), c2 = 1.f - powf(b2, (float)t);
  float3 p = v[i];
  p.x -= lr * (mi.x / c1) / (sqrtf(si.x / c2) + eps); p.y -= lr * (mi.y / c1) / (sqrtf(si.y / c2) + eps); p.z -= lr * (mi.z / c1) / (sqrtf(si.z / c2) + eps);
  v[i] = p; grad[i] = make_float3(0, 0, 0);
}
__global__ void keep_init_kernel(int nv, const float* __restrict__ dist, float radius, float* __restrict__ w) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < nv) w[i] = dist[i] <= radius ? 1.f : 0.f; }
__global__ void keep_diffuse_kernel(int nv, const float* __restrict__ w, const int* __restrict__ start, const int* __restrict__ nbr, float* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nv) return;
  int s = start[i], e = start[i + 1]; float sum = 0.f; for (int k = s; k < e; k++) sum += w[nbr[k]];
  out[i] = fmaxf(w[i], e > s ? sum / (e - s) : 0.f);
}
__global__ void keep_blend_kernel(int nv, const float* __restrict__ w, const float3* __restrict__ v0, float3* __restrict__ v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nv) return;
  float a = fminf(fmaxf(w[i], 0.f), 1.f); v[i] = v0[i] * a + v[i] * (1.f - a);
}

}  // namespace

int mesh_refine_main(int argc, char** argv) {
  std::string in, out; std::vector<std::vector<std::string>> views, keep;
  int iters = 400, device = 0; float lr = 2e-4f, lam_lap = 2.f, lam_pos = 10.f, tol = 0.008f, power = 2.f, min_w = 0.05f, keep_feather = 0.015f, keep_step = 0.002f;
  ArgParser ap("Refine a fused mesh's vertices to the Sapiens normal maps (face normals -> map normals, Laplacian + positional regularisers, Adam)\n\nUsage: b2ctrain mesh-refine --input MESH.ply --output OUT.ply --views CAMS NORMALS_DIR [OPTIONS]");
  ap.s("input", "MESH.ply", "The fused mesh", in).s("output", "OUT.ply", "The refined mesh", out)
    .multi("views", "CAMS DIR", 2, "A camera list and a directory of RGBA normal maps named as the cameras ([+X,-Y,-Z] camera space, alpha = valid); repeatable", views)
    .i("iters", "N", "Adam iterations [default: 400]", iters).f("lr", "LR", "Adam step [default: 2e-4]", lr)
    .f("lam-lap", "W", "Weight of the uniform Laplacian term [default: 2]", lam_lap).f("lam-pos", "W", "Weight of the pull towards the input positions [default: 10]", lam_pos)
    .f("tol", "M", "A vertex is visible in a view when its depth agrees with the mesh's own depth there within this [default: 0.008]", tol)
    .f("power", "P", "Facing weight exponent of a view's normal sample [default: 2]", power)
    .f("min-w", "W", "Vertices with less total facing weight keep no target normal [default: 0.05]", min_w)
    .multi("keep", "PLY RADIUS", 2, "Vertices within RADIUS of the ply's points (the face cap) keep their input positions, feathered over --keep-feather", keep)
    .f("keep-feather", "M", "Blend width beyond the kept region [default: 0.015]", keep_feather)
    .f("keep-step", "M", "Mean edge length assumed for the feather's neighbour diffusion [default: 0.002]", keep_step)
    .i("device", "N", "CUDA device [default: 0]", device);
  if (!ap.parse(argc, argv)) return 0;
  if (in.empty() || out.empty() || views.empty()) fail("--input, --output and at least one --views group are required");
  CUDA_CHECK(cudaSetDevice(device)); cudaStream_t stream = 0; double t0 = now_seconds();
  TriMesh m = read_mesh(in); int nv = (int)m.nv(), nf = (int)m.nf();
  MeshRaster rast; rast.upload(m, stream);
  DevBuf<float3> vn; vn.reserve(nv); vertex_normals_gpu(rast.v, nv, rast.f, nf, vn, stream);
  DevBuf<float3> acc; DevBuf<float> wsum; acc.reserve(nv); wsum.reserve(nv); acc.zero(stream); wsum.zero(stream);
  DevBuf<uint8_t> nmap; int nviews = 0;
  for (auto& g : views) {
    CamSet cs = load_cams(g[0]);
    for (size_t i = 0; i < cs.size(); i++) {
      Image8 im; if (!try_load_image8(g[1] + "/" + cs.names[i], 4, im)) continue;
      if (im.W != cs.W || im.H != cs.H) fail("'%s': %d x %d does not match the camera list's %d x %d", (g[1] + "/" + cs.names[i]).c_str(), im.W, im.H, cs.W, cs.H);
      CameraGPU cam = CameraGPU::from(cs.cams[i], cs.W, cs.H);
      rast.raster(cam, 1, false, stream);
      nmap.upload(im.px, stream);
      target_accum_kernel<<<div_up(nv, 256), 256, 0, stream>>>(nv, rast.v, vn, cam, rast.z, nmap, tol, power, acc, wsum); CUDA_KERNEL_CHECK();
      CUDA_CHECK(cudaStreamSynchronize(stream)); nviews++;
    }
  }
  DevBuf<float3> tgt, ftgt; DevBuf<uint8_t> ok, fok; tgt.reserve(nv); ok.reserve(nv); ftgt.reserve(nf); fok.reserve(nf);
  target_finish_kernel<<<div_up(nv, 256), 256, 0, stream>>>(nv, acc, wsum, min_w, tgt, ok); CUDA_KERNEL_CHECK();
  face_target_kernel<<<div_up(nf, 256), 256, 0, stream>>>(nf, rast.f, tgt, ok, ftgt, fok); CUDA_KERNEL_CHECK();
  { std::vector<uint8_t> hok = ok.download(nv, stream), hfok = fok.download(nf, stream); size_t n1 = 0, n2 = 0; for (auto v : hok) n1 += v; for (auto v : hfok) n2 += v;
    log_info("%d views in %.1fs; targets for %.1f%% of vertices, %.1f%% of faces", nviews, now_seconds() - t0, 100.0 * n1 / nv, 100.0 * n2 / nf);
    if (n2 == 0) fail("no face has a target normal: check --views (the maps must be named as the cameras) and --tol");
    acc.free(); wsum.free();
    Adjacency adj; adj.build(m); adj.upload(stream);
    DevBuf<float3> v0, grad, mom, sec, lap; v0.reserve(nv); grad.reserve(nv); mom.reserve(nv); sec.reserve(nv); lap.reserve(nv);
    CUDA_CHECK(cudaMemcpyAsync(v0.ptr, rast.v.ptr, (size_t)nv * sizeof(float3), cudaMemcpyDeviceToDevice, stream));
    grad.zero(stream); mom.zero(stream); sec.zero(stream);
    DevBuf<float> loss; loss.reserve(3);
    float k_n = 1.f / (float)n2, k_lap = lam_lap * 1e4f / nv, k_pos = lam_pos * 1e4f / nv;
    for (int it = 0; it < iters; it++) {
      loss.zero(stream);
      normal_grad_kernel<<<div_up(nf, 256), 256, 0, stream>>>(nf, rast.v, rast.f, ftgt, fok, k_n, grad, loss.ptr + 2); CUDA_KERNEL_CHECK();
      laplacian_kernel<<<div_up(nv, 256), 256, 0, stream>>>(nv, rast.v, adj.d_start, adj.d_nbr, lap); CUDA_KERNEL_CHECK();
      reg_grad_kernel<<<div_up(nv, 256), 256, 0, stream>>>(nv, rast.v, v0, lap, adj.d_start, adj.d_nbr, k_lap, k_pos, grad, loss); CUDA_KERNEL_CHECK();
      adam_kernel<<<div_up(nv, 256), 256, 0, stream>>>(nv, rast.v, grad, mom, sec, lr, 0.9f, 0.999f, 1e-8f, it + 1); CUDA_KERNEL_CHECK();
      if (it % 50 == 0 || it == iters - 1) { std::vector<float> l = loss.download(3, stream); log_info("it %d normal %.4f lap %.4f pos %.4f", it, l[2], l[0] / std::max(lam_lap, 1e-9f), l[1] / std::max(lam_pos, 1e-9f)); }
    }
    if (!keep.empty()) {
      std::vector<float> pts = read_points(keep[0][0]); float radius = strtof(keep[0][1].c_str(), nullptr);
      PointGrid pg; pg.build(pts, std::max(radius, 1e-3f), stream);
      DevBuf<float> dist, w, w2; dist.reserve(nv); w.reserve(nv); w2.reserve(nv);
      point_query_gpu(pg.dev(), v0, nv, radius, dist, nullptr, stream);
      keep_init_kernel<<<div_up(nv, 256), 256, 0, stream>>>(nv, dist, radius, w); CUDA_KERNEL_CHECK();
      int steps = (int)(keep_feather / std::max(keep_step, 1e-6f));
      for (int k = 0; k < steps; k++) { keep_diffuse_kernel<<<div_up(nv, 256), 256, 0, stream>>>(nv, w, adj.d_start, adj.d_nbr, w2); CUDA_KERNEL_CHECK(); std::swap(w.ptr, w2.ptr); }
      keep_blend_kernel<<<div_up(nv, 256), 256, 0, stream>>>(nv, w, v0, rast.v); CUDA_KERNEL_CHECK();
      std::vector<float> hw = w.download(nv, stream); size_t nk = 0; for (float x : hw) nk += x >= 1.f;
      log_info("kept %.2f%% of vertices at their input positions (%zu cap points, radius %.3f, feather %d steps)", 100.0 * nk / nv, pts.size() / 3, radius, steps);
    }
    std::vector<float3> V = rast.v.download(nv, stream), V0 = v0.download(nv, stream);
    std::vector<float> mv(nv); double mean = 0;
    for (int i = 0; i < nv; i++) { float dx = V[i].x - V0[i].x, dy = V[i].y - V0[i].y, dz = V[i].z - V0[i].z; mv[i] = std::sqrt(dx * dx + dy * dy + dz * dz); mean += mv[i]; }
    std::sort(mv.begin(), mv.end());
    log_info("moved mean %.2f mm, p95 %.2f mm, max %.1f mm", mean / nv * 1000, mv[(size_t)(0.95 * (nv - 1))] * 1000, mv[nv - 1] * 1000);
    for (int i = 0; i < nv; i++) { m.v[i * 3] = V[i].x; m.v[i * 3 + 1] = V[i].y; m.v[i * 3 + 2] = V[i].z; }
  }
  vertex_normals_gpu(rast.v, nv, rast.f, nf, vn, stream);
  std::vector<float3> N = vn.download(nv, stream); m.nrm.resize((size_t)nv * 3);
  for (int i = 0; i < nv; i++) { m.nrm[i * 3] = N[i].x; m.nrm[i * 3 + 1] = N[i].y; m.nrm[i * 3 + 2] = N[i].z; }
  write_ply_mesh(out, m);
  log_info("wrote %s (%.1fs)", out.c_str(), now_seconds() - t0);
  return 0;
}

}  // namespace b2c
