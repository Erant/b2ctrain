// `b2ctrain mesh-unwrap`: quadric decimation (meshoptimizer), xatlas unwrap, atlas maps (position, normal, mask),
// colour transferred from the ORIGINAL mesh at the closest surface point, edge padding, and the protection masks
// for the texture refinement loop. Reference: out/mesh/tools/uv_bake.py, uv_protect.py, uv_protect_cap.py.
#include "mesh/common.h"
#include "mesh/raster.h"
#include "mesh/geom.cuh"
#include "util/log.h"
#include "xatlas/xatlas.h"
#include "meshoptimizer/meshoptimizer.h"
#include "json.hpp"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cfloat>
#include <cmath>

namespace b2c {
namespace fs = std::filesystem;

namespace {

// Texel (x, y) centre in atlas space: u = (x + 0.5) / R, v = 1 - (y + 0.5) / R (row 0 = top = v 1).
__global__ void atlas_raster_kernel(int nf, const float2* __restrict__ uv, const uint3* __restrict__ fuv, const uint3* __restrict__ f, const float3* __restrict__ v, const float3* __restrict__ vn, int R,
                                    float3* __restrict__ pos, float3* __restrict__ nrm, int* __restrict__ tri) {
  int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= nf) return;
  uint3 tu = fuv[t], tv = f[t];
  float2 a = uv[tu.x], b = uv[tu.y], c = uv[tu.z];
  float ax = a.x * R, ay = (1.f - a.y) * R, bx = b.x * R, by = (1.f - b.y) * R, cx = c.x * R, cy = (1.f - c.y) * R;
  float area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  if (fabsf(area) < 1e-12f) return;
  float inv = 1.f / area;
  int x0 = max((int)ceilf(fminf(ax, fminf(bx, cx)) - 0.5f), 0), x1 = min((int)floorf(fmaxf(ax, fmaxf(bx, cx)) - 0.5f), R - 1);
  int y0 = max((int)ceilf(fminf(ay, fminf(by, cy)) - 0.5f), 0), y1 = min((int)floorf(fmaxf(ay, fmaxf(by, cy)) - 0.5f), R - 1);
  float3 pa = v[tv.x], pb = v[tv.y], pc = v[tv.z], na = vn[tv.x], nb = vn[tv.y], nc = vn[tv.z];
  for (int y = y0; y <= y1; y++) for (int x = x0; x <= x1; x++) {
    float px = x + 0.5f, py = y + 0.5f;
    float w0 = ((bx - px) * (cy - py) - (cx - px) * (by - py)) * inv, w1 = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) * inv, w2 = 1.f - w0 - w1;
    if (w0 < -1e-6f || w1 < -1e-6f || w2 < -1e-6f) continue;
    size_t i = (size_t)y * R + x;
    pos[i] = pa * w0 + pb * w1 + pc * w2;
    float3 n = na * w0 + nb * w1 + nc * w2; nrm[i] = n * (1.f / fmaxf(len3(n), 1e-9f));
    tri[i] = t;
  }
}
// Colour of the original mesh at the closest point of every covered texel (barycentric on the found triangle).
__global__ void colour_kernel(int n, const int* __restrict__ tri, const float3* __restrict__ bary, const uint3* __restrict__ f, const uint8_t* __restrict__ col, uint8_t* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  int t = tri[i]; float3 b = bary[i]; float c[3] = {0, 0, 0};
  if (t >= 0) { uint3 fc = f[t]; for (int k = 0; k < 3; k++) c[k] = b.x * col[fc.x * 3 + k] + b.y * col[fc.y * 3 + k] + b.z * col[fc.z * 3 + k]; }
  for (int k = 0; k < 3; k++) out[i * 3 + k] = (uint8_t)fminf(fmaxf(c[k] + 0.5f, 0.f), 255.f);
}
// Jump flooding: nearest covered texel for every texel (seed = own index where covered, -1 elsewhere).
__global__ void jfa_init_kernel(int R, const int* __restrict__ tri, int* __restrict__ seed) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < (size_t)R * R) seed[i] = tri[i] >= 0 ? (int)i : -1; }
__global__ void jfa_step_kernel(int R, int step, const int* __restrict__ in, int* __restrict__ out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y; if (x >= R || y >= R) return;
  int best = in[(size_t)y * R + x]; float bd = best >= 0 ? (float)((best % R - x) * (best % R - x) + (best / R - y) * (best / R - y)) : INFINITY;
  for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
    int xx = x + dx * step, yy = y + dy * step; if (xx < 0 || yy < 0 || xx >= R || yy >= R) continue;
    int s = in[(size_t)yy * R + xx]; if (s < 0) continue;
    float d = (float)((s % R - x) * (s % R - x) + (s / R - y) * (s / R - y));
    if (d < bd) { bd = d; best = s; }
  }
  out[(size_t)y * R + x] = best;
}
template <typename T> __global__ void pad_kernel(size_t n, const int* __restrict__ seed, const T* __restrict__ in, T* __restrict__ out) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= n) return; int s = seed[i]; out[i] = s >= 0 ? in[s] : in[i];
}
__global__ void pad3_kernel(size_t n, const int* __restrict__ seed, const uint8_t* __restrict__ in, uint8_t* __restrict__ out) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= n) return; int s = seed[i] >= 0 ? seed[i] : (int)i; for (int k = 0; k < 3; k++) out[i * 3 + k] = in[(size_t)s * 3 + k];
}
__global__ void dist_kernel(int R, const int* __restrict__ seed, float* __restrict__ dist) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y; if (x >= R || y >= R) return;
  int s = seed[(size_t)y * R + x]; dist[(size_t)y * R + x] = s >= 0 ? sqrtf((float)((s % R - x) * (s % R - x) + (s / R - y) * (s / R - y))) : INFINITY;
}
// Nearest OTHER point for each cap point (its spacing).
__global__ void nn_spacing_kernel(PointGridDev g, float* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= g.n) return;
  float3 p = g.p[i]; int cx = (int)floorf((p.x - g.lo.x) / g.cell), cy = (int)floorf((p.y - g.lo.y) / g.cell), cz = (int)floorf((p.z - g.lo.z) / g.cell); float best = INFINITY;
  for (int x = max(cx - 1, 0); x <= min(cx + 1, g.dims.x - 1); x++) for (int y = max(cy - 1, 0); y <= min(cy + 1, g.dims.y - 1); y++) for (int z = max(cz - 1, 0); z <= min(cz + 1, g.dims.z - 1); z++) {
    int c = (x * g.dims.y + y) * g.dims.z + z;
    for (int k = g.cell_start[c]; k < g.cell_start[c + 1]; k++) { int j = g.items[k]; if (j == i) continue; float3 d = g.p[j] - p; best = fminf(best, dot3(d, d)); }
  }
  out[i] = sqrtf(best);
}

}  // namespace

int mesh_unwrap_main(int argc, char** argv) {
  std::string in, out, cap_path; int tris = 300000, res = 4096, pad = 6, device = 0, chart_smooth = 2; float max_cost = 8.f, normal_dev = 0.5f;
  float cap_near = 0.003f, cap_erode = 0.006f, cap_fill = 0.75f, cap_spacing = 0.f;
  bool protect_head = false; float centre[3] = {0, 0, 0}, facing[3] = {0, 0, 1}; bool have_centre = false; float head_radius = 0.12f, head_drop = 0.11f, head_cos = 0.15f, head_above = 0.07f, head_below = 0.10f;
  ArgParser ap("Decimate, unwrap (xatlas) and bake a vertex-coloured mesh into a texture atlas with position/normal/mask maps and protection masks\n\nUsage: b2ctrain mesh-unwrap --input MESH.ply --output DIR [OPTIONS]");
  ap.s("input", "MESH.ply", "The vertex-coloured mesh", in).s("output", "DIR", "Atlas directory: mesh_uv.obj/.mtl, texture.png, position.f32, normal.f32, mask.png, mask_dilated.png, atlas.json, protect_*.png", out)
    .i("tris", "N", "Decimate to this many triangles before unwrapping [default: 300000]", tris).i("res", "R", "Atlas resolution [default: 4096]", res).i("pad", "PX", "Chart padding [default: 6]", pad)
    .f("max-cost", "C", "xatlas chart max_cost (higher = fewer, larger charts) [default: 8]", max_cost).f("normal-dev", "W", "xatlas normal_deviation_weight [default: 0.5]", normal_dev)
    .i("chart-smooth", "N", "Laplacian smoothing passes on a COPY of the decimated mesh used only for charting (a noisy TSDF surface splits into many charts); the output keeps the unsmoothed positions [default: 2]", chart_smooth)
    .s("cap", "CAP.ply", "The face cap's Gaussians: writes protect_cap.png (texels within --cap-near of a cap point, kept where the cap fills --cap-fill of a --cap-erode disc)", cap_path)
    .f("cap-near", "M", "[default: 0.003]", cap_near).f("cap-erode", "M", "[default: 0.006]", cap_erode).f("cap-fill", "F", "[default: 0.75]", cap_fill)
    .f("cap-spacing", "M", "Cap point spacing (default: the measured median nearest-neighbour distance)", cap_spacing)
    .b("protect-head", "Also write protect_head.png: the face band within --head-radius of the head centre, facing --facing", protect_head)
    .f3("centre", "x,y,z", "Head centre for --protect-head (the SAM head centroid); default: estimated from the top 0.22 m of the mesh", centre)
    .f3("facing", "x,y,z", "Direction the protected face looks along (the photo camera's) [default: 0,0,1]", facing)
    .f("head-radius", "M", "[default: 0.12]", head_radius).f("head-drop", "M", "Centre estimate: this far below the top [default: 0.11]", head_drop).f("head-cos", "C", "Min normal . facing [default: 0.15]", head_cos)
    .f("head-above", "M", "[default: 0.07]", head_above).f("head-below", "M", "[default: 0.10]", head_below)
    .i("device", "N", "CUDA device [default: 0]", device);
  if (!ap.parse(argc, argv)) return 0;
  if (in.empty() || out.empty()) fail("--input and --output are required");
  have_centre = ap.given("centre");
  CUDA_CHECK(cudaSetDevice(device)); cudaStream_t stream = 0; double t0 = now_seconds();
  fs::create_directories(out);
  TriMesh src = read_mesh(in);
  if (!src.has_colour()) fail("'%s' has no vertex colours (run mesh-bake first)", in.c_str());
  log_info("source %zu v / %zu t", src.nv(), src.nf());

  // ---- decimation ----
  TriMesh dec;
  if ((int)src.nf() > tris) {
    std::vector<uint32_t> idx(src.f.size()); float err = 0.f;
    size_t n = meshopt_simplify(idx.data(), src.f.data(), src.f.size(), src.v.data(), src.nv(), 12, (size_t)tris * 3, FLT_MAX, 0, &err);
    if (n > (size_t)tris * 3 * 1.5) {  // topology-preserving collapse stalled: the sloppy simplifier reaches the count
      log_warn("decimation stalled at %zu triangles (error %.4f); using the sloppy simplifier", n / 3, err);
      n = meshopt_simplifySloppy(idx.data(), src.f.data(), src.f.size(), src.v.data(), src.nv(), 12, (size_t)tris * 3, FLT_MAX, &err);
    }
    idx.resize(n);
    std::vector<uint32_t> remap(src.nv(), UINT32_MAX); uint32_t nn = 0;
    for (uint32_t& i : idx) { if (remap[i] == UINT32_MAX) { remap[i] = nn++; dec.v.insert(dec.v.end(), &src.v[i * 3], &src.v[i * 3] + 3); } i = remap[i]; }
    dec.f = std::move(idx);
    log_info("decimated %zu v / %zu t (error %.5f, %.1fs)", dec.nv(), dec.nf(), err, now_seconds() - t0);
  } else dec = TriMesh{src.v, src.f};
  // Remove degenerate triangles (xatlas rejects them) and compute the decimated mesh's normals.
  { std::vector<uint32_t> nf; for (size_t t = 0; t < dec.nf(); t++) { uint32_t a = dec.f[t * 3], b = dec.f[t * 3 + 1], c = dec.f[t * 3 + 2]; if (a == b || b == c || a == c) continue; nf.push_back(a); nf.push_back(b); nf.push_back(c); } dec.f = std::move(nf); }
  MeshRaster dr; dr.upload(dec, stream);
  DevBuf<float3> dvn; dvn.reserve(dec.nv()); vertex_normals_gpu(dr.v, (int)dec.nv(), dr.f, (int)dec.nf(), dvn, stream);
  std::vector<float3> hvn = dvn.download(dec.nv(), stream);
  // The charting copy: smoothed positions and their normals when asked for.
  std::vector<float> chart_v = dec.v; std::vector<float3> chart_n = hvn;
  if (chart_smooth > 0) {
    Adjacency adj; adj.build(dec);
    std::vector<float> nv2(chart_v.size());
    for (int it = 0; it < chart_smooth; it++) {
      for (size_t i = 0; i < dec.nv(); i++) { int s0 = adj.start[i], s1 = adj.start[i + 1]; double acc[3] = {0, 0, 0}; for (int k = s0; k < s1; k++) for (int c = 0; c < 3; c++) acc[c] += chart_v[adj.nbr[k] * 3 + c]; for (int c = 0; c < 3; c++) nv2[i * 3 + c] = s1 > s0 ? (float)(0.5 * chart_v[i * 3 + c] + 0.5 * acc[c] / (s1 - s0)) : chart_v[i * 3 + c]; }
      chart_v.swap(nv2);
    }
    DevBuf<float3> sv; { std::vector<float3> h(dec.nv()); for (size_t i = 0; i < dec.nv(); i++) h[i] = make_float3(chart_v[i * 3], chart_v[i * 3 + 1], chart_v[i * 3 + 2]); sv.upload(h, stream); }
    vertex_normals_gpu(sv, (int)dec.nv(), dr.f, (int)dec.nf(), dvn, stream); chart_n = dvn.download(dec.nv(), stream);
    vertex_normals_gpu(dr.v, (int)dec.nv(), dr.f, (int)dec.nf(), dvn, stream);  // the atlas maps use the real normals
  }

  // ---- xatlas ----
  xatlas::Atlas* atlas = xatlas::Create();
  xatlas::MeshDecl md; md.vertexCount = (uint32_t)dec.nv(); md.vertexPositionData = chart_v.data(); md.vertexPositionStride = 12;
  md.vertexNormalData = chart_n.data(); md.vertexNormalStride = sizeof(float3); md.indexCount = (uint32_t)dec.f.size(); md.indexData = dec.f.data(); md.indexFormat = xatlas::IndexFormat::UInt32;
  xatlas::AddMeshError e = xatlas::AddMesh(atlas, md);
  if (e != xatlas::AddMeshError::Success) fail("xatlas: %s", xatlas::StringForEnum(e));
  xatlas::ChartOptions co; co.maxCost = max_cost; co.normalDeviationWeight = normal_dev;
  xatlas::PackOptions po; po.resolution = (uint32_t)res; po.padding = (uint32_t)pad; po.bilinear = true; po.bruteForce = false;
  xatlas::Generate(atlas, co, po);
  if (atlas->atlasCount != 1) fail("xatlas packed into %u atlases (raise --res or lower --pad)", atlas->atlasCount);
  const xatlas::Mesh& xm = atlas->meshes[0];
  TriMesh uvm; uvm.v = dec.v; uvm.uv.resize((size_t)xm.vertexCount * 2); uvm.f.resize(xm.indexCount); uvm.fuv.resize(xm.indexCount);
  for (uint32_t i = 0; i < xm.vertexCount; i++) { uvm.uv[i * 2] = xm.vertexArray[i].uv[0] / atlas->width; uvm.uv[i * 2 + 1] = xm.vertexArray[i].uv[1] / atlas->height; }
  for (uint32_t i = 0; i < xm.indexCount; i++) { uvm.fuv[i] = xm.indexArray[i]; uvm.f[i] = xm.vertexArray[xm.indexArray[i]].xref; }
  log_info("xatlas: %u charts, %u uv verts, utilisation %.2f, %u x %u (%.1fs)", atlas->chartCount, xm.vertexCount, atlas->utilization[0], atlas->width, atlas->height, now_seconds() - t0);
  xatlas::Destroy(atlas);
  int R = res;

  // ---- atlas maps ----
  size_t npx = (size_t)R * R; int nfu = (int)uvm.nf();
  DevBuf<float2> d_uv; { std::vector<float2> h(uvm.uv.size() / 2); for (size_t i = 0; i < h.size(); i++) h[i] = make_float2(uvm.uv[i * 2], uvm.uv[i * 2 + 1]); d_uv.upload(h, stream); }
  DevBuf<uint3> d_fuv, d_f; { std::vector<uint3> h(nfu), h2(nfu); for (int i = 0; i < nfu; i++) { h[i] = make_uint3(uvm.fuv[i * 3], uvm.fuv[i * 3 + 1], uvm.fuv[i * 3 + 2]); h2[i] = make_uint3(uvm.f[i * 3], uvm.f[i * 3 + 1], uvm.f[i * 3 + 2]); } d_fuv.upload(h, stream); d_f.upload(h2, stream); }
  DevBuf<float3> pos, nrm; DevBuf<int> tri; pos.reserve(npx); nrm.reserve(npx); tri.reserve(npx); pos.zero(stream); nrm.zero(stream);
  CUDA_CHECK(cudaMemsetAsync(tri.ptr, 0xff, npx * sizeof(int), stream));
  atlas_raster_kernel<<<div_up(nfu, 128), 128, 0, stream>>>(nfu, d_uv, d_fuv, d_f, dr.v, dvn, R, pos, nrm, tri); CUDA_KERNEL_CHECK();
  std::vector<int> htri = tri.download(npx, stream); size_t ncov = 0; for (int t : htri) ncov += t >= 0;
  log_info("rasterised: %.1f%% texels covered (%.1fs)", 100.0 * ncov / npx, now_seconds() - t0);
  // Colour from the original mesh at the closest surface point of every covered texel.
  std::vector<int> cov_idx; cov_idx.reserve(ncov); for (size_t i = 0; i < npx; i++) if (htri[i] >= 0) cov_idx.push_back((int)i);
  std::vector<uint8_t> tex(npx * 3, 0);
  {
    TriGrid tg; tg.build(src, 0.004f, false, stream);
    MeshRaster sr; sr.upload(src, stream); DevBuf<uint8_t> scol; scol.upload(src.col, stream);
    std::vector<float3> hpos = pos.download(npx, stream);
    const int CH = 1 << 20; DevBuf<float3> q, cp, qb; DevBuf<int> qt; DevBuf<float> qd; DevBuf<uint8_t> qc;
    q.reserve(CH); cp.reserve(CH); qb.reserve(CH); qt.reserve(CH); qd.reserve(CH); qc.reserve((size_t)CH * 3);
    std::vector<float3> hq(CH); std::vector<float> dists; dists.reserve(ncov);
    for (size_t off = 0; off < cov_idx.size(); off += CH) {
      int n = (int)std::min<size_t>(CH, cov_idx.size() - off);
      for (int k = 0; k < n; k++) hq[k] = hpos[cov_idx[off + k]];
      q.upload(hq.data(), n, stream);
      closest_points_gpu(tg.dev(), q, n, 12, qt, cp, qd, qb, nullptr, stream);
      colour_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, qt, qb, sr.f, scol, qc); CUDA_KERNEL_CHECK();
      std::vector<uint8_t> hc = qc.download((size_t)n * 3, stream); std::vector<float> hd = qd.download(n, stream);
      for (int k = 0; k < n; k++) { size_t i = cov_idx[off + k]; tex[i * 3] = hc[k * 3]; tex[i * 3 + 1] = hc[k * 3 + 1]; tex[i * 3 + 2] = hc[k * 3 + 2]; dists.push_back(hd[k]); }
    }
    std::sort(dists.begin(), dists.end());
    log_info("closest-point distance p50 %.2f mm p99 %.2f mm (%.1fs)", dists.empty() ? 0.0 : dists[dists.size() / 2] * 1000, dists.empty() ? 0.0 : dists[(size_t)(0.99 * (dists.size() - 1))] * 1000, now_seconds() - t0);
  }
  // Edge padding: every texel takes the nearest covered texel's values (jump flooding); mask_dilated = within `pad`.
  DevBuf<int> seed, seed2; seed.reserve(npx); seed2.reserve(npx);
  jfa_init_kernel<<<div_up(npx, 256), 256, 0, stream>>>(R, tri, seed); CUDA_KERNEL_CHECK();
  dim3 blk(16, 16), grd((R + 15) / 16, (R + 15) / 16);
  for (int step = R / 2; step >= 1; step /= 2) { jfa_step_kernel<<<grd, blk, 0, stream>>>(R, step, seed, seed2); CUDA_KERNEL_CHECK(); std::swap(seed.ptr, seed2.ptr); }
  jfa_step_kernel<<<grd, blk, 0, stream>>>(R, 1, seed, seed2); CUDA_KERNEL_CHECK(); std::swap(seed.ptr, seed2.ptr);
  DevBuf<float> dist; dist.reserve(npx); dist_kernel<<<grd, blk, 0, stream>>>(R, seed, dist); CUDA_KERNEL_CHECK();
  DevBuf<uint8_t> d_tex, d_tex2; d_tex.upload(tex, stream); d_tex2.reserve(npx * 3);
  pad3_kernel<<<div_up(npx, 256), 256, 0, stream>>>(npx, seed, d_tex, d_tex2); CUDA_KERNEL_CHECK();
  DevBuf<float3> pos2, nrm2; pos2.reserve(npx); nrm2.reserve(npx);
  pad_kernel<<<div_up(npx, 256), 256, 0, stream>>>(npx, seed, pos.ptr, pos2.ptr); CUDA_KERNEL_CHECK();
  pad_kernel<<<div_up(npx, 256), 256, 0, stream>>>(npx, seed, nrm.ptr, nrm2.ptr); CUDA_KERNEL_CHECK();
  std::vector<uint8_t> ptex = d_tex2.download(npx * 3, stream); std::vector<float3> ppos = pos2.download(npx, stream), pnrm = nrm2.download(npx, stream); std::vector<float> hdist = dist.download(npx, stream);
  std::vector<uint8_t> mask(npx), maskd(npx);
  for (size_t i = 0; i < npx; i++) { mask[i] = htri[i] >= 0 ? 255 : 0; maskd[i] = hdist[i] < (float)pad ? 255 : 0; }
  write_png8(out + "/texture.png", R, R, 3, ptex.data()); write_png8(out + "/mask.png", R, R, 1, mask.data()); write_png8(out + "/mask_dilated.png", R, R, 1, maskd.data());
  write_f32(out + "/position.f32", (const float*)ppos.data(), npx * 3); write_f32(out + "/normal.f32", (const float*)pnrm.data(), npx * 3);
  write_obj_mesh(out + "/mesh_uv.obj", uvm, "texture.png");

  // ---- protection masks ----
  size_t ncov_d = 0; for (auto m : maskd) ncov_d += m > 0;
  if (!cap_path.empty()) {
    std::vector<float> pts = read_points(cap_path);
    PointGrid pg; pg.build(pts, cap_erode, stream);
    if (cap_spacing <= 0.f) { DevBuf<float> sp; sp.reserve(pg.n); nn_spacing_kernel<<<div_up(pg.n, 256), 256, 0, stream>>>(pg.dev(), sp); CUDA_KERNEL_CHECK(); std::vector<float> h = sp.download(pg.n, stream); std::sort(h.begin(), h.end()); cap_spacing = h.empty() ? 1e-3f : h[h.size() / 2]; log_info("cap spacing %.2f mm", cap_spacing * 1000); }
    std::vector<int> cidx; for (size_t i = 0; i < npx; i++) if (maskd[i]) cidx.push_back((int)i);
    std::vector<uint8_t> prot(npx, 0); size_t non = 0, nprot = 0; float expected = (float)M_PI * cap_erode * cap_erode / (cap_spacing * cap_spacing);
    const int CH = 1 << 20; DevBuf<float3> q; DevBuf<float> qd; DevBuf<int> qc; q.reserve(CH); qd.reserve(CH); qc.reserve(CH); std::vector<float3> hq(CH);
    for (size_t off = 0; off < cidx.size(); off += CH) {
      int n = (int)std::min<size_t>(CH, cidx.size() - off);
      for (int k = 0; k < n; k++) hq[k] = ppos[cidx[off + k]];
      q.upload(hq.data(), n, stream); point_query_gpu(pg.dev(), q, n, cap_erode, qd, qc, stream);
      std::vector<float> hd = qd.download(n, stream); std::vector<int> hc = qc.download(n, stream);
      for (int k = 0; k < n; k++) { bool on = hd[k] <= cap_erode; non += on; if (on && hc[k] >= cap_fill * expected) { prot[cidx[off + k]] = 255; nprot++; } }
    }
    (void)cap_near;
    write_png8(out + "/protect_cap.png", R, R, 1, prot.data());
    log_info("%zu cap points; texels within %.0f mm of the cap %zu, protected %zu (%.0f%%)", pts.size() / 3, cap_erode * 1000, non, nprot, non ? 100.0 * nprot / non : 0.0);
  }
  if (protect_head) {
    float c[3];
    if (have_centre) { c[0] = centre[0]; c[1] = centre[1]; c[2] = centre[2]; }
    else {
      float hi = -INFINITY; for (size_t i = 0; i < dec.nv(); i++) hi = std::max(hi, dec.v[i * 3 + 1]);
      double sx = 0, sz = 0; size_t n = 0; for (size_t i = 0; i < dec.nv(); i++) if (dec.v[i * 3 + 1] > hi - 0.22f) { sx += dec.v[i * 3]; sz += dec.v[i * 3 + 2]; n++; }
      c[0] = (float)(sx / std::max<size_t>(n, 1)); c[1] = hi - head_drop; c[2] = (float)(sz / std::max<size_t>(n, 1));
    }
    float fl = std::sqrt(facing[0] * facing[0] + facing[1] * facing[1] + facing[2] * facing[2]); float fd[3] = {facing[0] / fl, facing[1] / fl, facing[2] / fl};
    std::vector<uint8_t> prot(npx, 0); size_t np = 0;
    for (size_t i = 0; i < npx; i++) {
      if (!maskd[i]) continue;
      float3 p = ppos[i], n = pnrm[i]; float dx = p.x - c[0], dy = p.y - c[1], dz = p.z - c[2];
      bool ok = std::sqrt(dx * dx + dy * dy + dz * dz) < head_radius && dy < head_above && dy > -head_below && (n.x * fd[0] + n.y * fd[1] + n.z * fd[2]) > head_cos && (dx * fd[0] + dy * fd[1] + dz * fd[2]) > -0.02f;
      if (ok) { prot[i] = 255; np++; }
    }
    write_png8(out + "/protect_head.png", R, R, 1, prot.data());
    log_info("head centre (%.4f, %.4f, %.4f), protected %zu texels (%.2f%% of the atlas)", c[0], c[1], c[2], np, 100.0 * np / npx);
  }
  nlohmann::json meta = {{"res", R}, {"pad", pad}, {"tris", uvm.nf()}, {"uv_verts", uvm.uv.size() / 2}, {"verts", uvm.nv()}, {"covered", ncov}, {"covered_dilated", ncov_d}, {"source", in},
                         {"files", {{"mesh", "mesh_uv.obj"}, {"texture", "texture.png"}, {"position", "position.f32"}, {"normal", "normal.f32"}, {"mask", "mask.png"}, {"mask_dilated", "mask_dilated.png"}}},
                         {"layout", "position.f32 / normal.f32: res x res x 3 float32 little-endian, row 0 = v 1 (the texture's top row)"}};
  { std::ofstream f(out + "/atlas.json"); f << meta.dump(1) << "\n"; }
  log_info("done %s (%.1fs)", out.c_str(), now_seconds() - t0);
  return 0;
}

}  // namespace b2c
