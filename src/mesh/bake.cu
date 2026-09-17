// `b2ctrain mesh-bake`: per-vertex colours from the splat's RGBA probes (visibility-tested, facing-weighted, trimmed
// median over the views), unseen vertices filled by diffusion, a 1-ring median pass, and the face cap's colours
// projected from the photo's camera onto the vertices it covers, seam-levelled against the surrounding bake.
// Reference: out/mesh/tools/texture.py, smooth_colours.py, cap_texture.py --project.
#include "mesh/common.h"
#include "mesh/raster.h"
#include "mesh/geom.cuh"
#include "mesh/cap_color.h"
#include "ply.h"
#include "util/log.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace b2c {

namespace {

// One (weight, r, g, b) sample per view and vertex, fp16.
struct Sample { __half w, r, g, b; };

__global__ void mask_erode_kernel(int W, int H, const uint8_t* __restrict__ rgba, int alpha_min, int r, uint8_t* __restrict__ ok) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y; if (x >= W || y >= H) return;
  uint8_t v = 1;
  for (int dy = -r; v && dy <= r; dy++) for (int dx = -r; v && dx <= r; dx++) { int xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue; v = rgba[((size_t)yy * W + xx) * 4 + 3] >= alpha_min; }
  ok[(size_t)y * W + x] = v;
}

__global__ void sample_kernel(int v0, int n, const float3* __restrict__ v, const float3* __restrict__ vn, CameraGPU cam, int ss, const float* __restrict__ zbuf, const uint8_t* __restrict__ rgba, const uint8_t* __restrict__ okmask,
                              float tol, float power, float factor, int view, int nviews, Sample* __restrict__ samples, float3* __restrict__ acc, float* __restrict__ wsum) {
  int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= n) return;
  int i = v0 + k; float3 p = v[i]; float u, vv; float z = project_point(cam, p.x, p.y, p.z, u, vv);
  float w = 0.f; float3 col = make_float3(0, 0, 0);
  if (z > 1e-3f && u >= 1.f && u < cam.W - 2 && vv >= 1.f && vv < cam.H - 2) {
    int pu = min(max((int)(u * ss), 0), cam.W * ss - 1), pv = min(max((int)(vv * ss), 0), cam.H * ss - 1);
    float zs = zbuf[(size_t)pv * (cam.W * ss) + pu];
    if (zs > 0.f && fabsf(zs - z) < tol) {
      float3 dn = p - make_float3(cam.pos[0], cam.pos[1], cam.pos[2]); dn = dn * (1.f / fmaxf(len3(dn), 1e-9f));
      float facing = fminf(fmaxf(-dot3(vn[i], dn), 0.f), 1.f);
      // Pixel centres are at (i + 0.5): the image is sampled at (u - 0.5, v - 0.5) in the integer-centre convention.
      float okv = sample_bilinear_u8(okmask, cam.W, cam.H, 1, 0, u - 0.5f, vv - 0.5f);
      if (okv > 0.999f) {
        w = powf(facing, power) * factor;
        col = make_float3(sample_bilinear_u8(rgba, cam.W, cam.H, 4, 0, u - 0.5f, vv - 0.5f), sample_bilinear_u8(rgba, cam.W, cam.H, 4, 1, u - 0.5f, vv - 0.5f), sample_bilinear_u8(rgba, cam.W, cam.H, 4, 2, u - 0.5f, vv - 0.5f)) * (1.f / 255.f);
      }
    }
  }
  if (samples) { Sample s; s.w = __float2half(w); s.r = __float2half(col.x); s.g = __float2half(col.y); s.b = __float2half(col.z); samples[(size_t)view * n + k] = s; }
  if (w > 0.f) { acc[i] = acc[i] + col * w; wsum[i] += w; }
}

// Weighted per-channel median of the samples (the first sorted sample whose cumulative weight reaches half; found
// through two levels of 256 bins and then the smallest value in the final bin), then the trimmed mean within `trim`.
__global__ void reduce_kernel(int n, int nviews, const Sample* __restrict__ samples, bool trimmed, float trim, float3* __restrict__ out) {
  int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= n) return;
  float hist[256]; float med[3];
  float total = 0.f;
  for (int v = 0; v < nviews; v++) total += __half2float(samples[(size_t)v * n + k].w);
  float half = 0.5f * total;
  for (int ch = 0; ch < 3; ch++) {
    if (total <= 0.f) { med[ch] = 0.f; continue; }
    for (int b = 0; b < 256; b++) hist[b] = 0.f;
    for (int v = 0; v < nviews; v++) { const Sample& s = samples[(size_t)v * n + k]; float w = __half2float(s.w); if (w <= 0.f) continue; float c = __half2float(ch == 0 ? s.r : ch == 1 ? s.g : s.b); int b = min(max((int)(c * 256.f), 0), 255); hist[b] += w; }
    float cum = 0.f; int b1 = 255; float before = 0.f;
    for (int b = 0; b < 256; b++) { before = cum; cum += hist[b]; if (cum >= half) { b1 = b; break; } }
    for (int b = 0; b < 256; b++) hist[b] = 0.f;
    for (int v = 0; v < nviews; v++) { const Sample& s = samples[(size_t)v * n + k]; float w = __half2float(s.w); if (w <= 0.f) continue; float c = __half2float(ch == 0 ? s.r : ch == 1 ? s.g : s.b); int b = min(max((int)(c * 256.f), 0), 255); if (b != b1) continue; int sb = min(max((int)((c * 256.f - b1) * 256.f), 0), 255); hist[sb] += w; }
    cum = before; int b2 = 255;
    for (int b = 0; b < 256; b++) { cum += hist[b]; if (cum >= half) { b2 = b; break; } }
    float m = 1.f;
    for (int v = 0; v < nviews; v++) { const Sample& s = samples[(size_t)v * n + k]; float w = __half2float(s.w); if (w <= 0.f) continue; float c = __half2float(ch == 0 ? s.r : ch == 1 ? s.g : s.b); int b = min(max((int)(c * 256.f), 0), 255); if (b != b1) continue; int sb = min(max((int)((c * 256.f - b1) * 256.f), 0), 255); if (sb == b2) m = fminf(m, c); }
    med[ch] = m;
  }
  float3 res = make_float3(med[0], med[1], med[2]);
  if (trimmed) {
    float3 sum = make_float3(0, 0, 0); float ws = 0.f;
    for (int v = 0; v < nviews; v++) {
      const Sample& s = samples[(size_t)v * n + k]; float w = __half2float(s.w); if (w <= 0.f) continue;
      float3 c = make_float3(__half2float(s.r), __half2float(s.g), __half2float(s.b)); float3 d = c - res;
      if (len3(d) < trim) { sum = sum + c * w; ws += w; }
    }
    res = sum * (1.f / fmaxf(ws, 1e-9f));
  }
  out[k] = res;
}
__global__ void mean_kernel(int nv, const float3* __restrict__ acc, const float* __restrict__ wsum, float3* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nv) return; out[i] = acc[i] * (1.f / fmaxf(wsum[i], 1e-9f));
}
// Jacobi fill of unknown vertices from known neighbours; one round per launch.
__global__ void fill_kernel(int nv, const float3* __restrict__ in, const uint8_t* __restrict__ known, const int* __restrict__ start, const int* __restrict__ nbr, float3* __restrict__ out, uint8_t* __restrict__ known_out, int* __restrict__ changed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nv) return;
  if (known[i]) { out[i] = in[i]; known_out[i] = 1; return; }
  float3 sum = make_float3(0, 0, 0); int cnt = 0;
  for (int k = start[i]; k < start[i + 1]; k++) { int j = nbr[k]; if (known[j]) { sum = sum + in[j]; cnt++; } }
  if (cnt) { out[i] = sum * (1.f / cnt); known_out[i] = 1; atomicAdd(changed, 1); } else { out[i] = in[i]; known_out[i] = 0; }
}
// 1-ring median: the vertex plus its first K neighbours (padded with itself), per-channel median (mean of the two
// middle values for an even count, as np.median).
__global__ void ring_median_kernel(int nv, const float3* __restrict__ in, const int* __restrict__ start, const int* __restrict__ nbr, int K, float3* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nv) return;
  float vals[64]; int n = K + 1; float3 self = in[i]; float res[3];
  for (int ch = 0; ch < 3; ch++) {
    for (int k = 0; k < n; k++) { int j = k == 0 ? i : (start[i] + k - 1 < start[i + 1] ? nbr[start[i] + k - 1] : i); float3 c = j == i ? self : in[j]; vals[k] = ch == 0 ? c.x : ch == 1 ? c.y : c.z; }
    for (int a = 1; a < n; a++) { float x = vals[a]; int b = a - 1; while (b >= 0 && vals[b] > x) { vals[b + 1] = vals[b]; b--; } vals[b + 1] = x; }
    res[ch] = (n & 1) ? vals[n / 2] : 0.5f * (vals[n / 2 - 1] + vals[n / 2]);
  }
  out[i] = make_float3(res[0], res[1], res[2]);
}
// Cap projection: per vertex the cap image's colour and soft alpha under the photo camera where the mesh is visible.
__global__ void cap_project_kernel(int nv, const float3* __restrict__ v, const float3* __restrict__ vn, CameraGPU cam, int ss, const float* __restrict__ zbuf, float3 lo, float3 hi,
                                   const float* __restrict__ img, const float* __restrict__ soft, int u0, int v0, int Wc, int Hc, const uint8_t* __restrict__ photo, float vis_tol,
                                   float3* __restrict__ cc, float* __restrict__ alpha, uint8_t* __restrict__ cand) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nv) return;
  float3 p = v[i]; alpha[i] = 0.f; cc[i] = make_float3(0, 0, 0);
  bool c = p.x > lo.x && p.y > lo.y && p.z > lo.z && p.x < hi.x && p.y < hi.y && p.z < hi.z; cand[i] = c;
  if (!c) return;
  float u, vv; float z = project_point(cam, p.x, p.y, p.z, u, vv);
  int pu = min(max((int)(u * ss), 0), cam.W * ss - 1), pv = min(max((int)(vv * ss), 0), cam.H * ss - 1);
  float zs = zbuf[(size_t)pv * (cam.W * ss) + pu];
  bool vis = zs > 0.f && fabsf(zs - z) < vis_tol;
  float ux = u - u0, vx = vv - v0;
  bool ok = vis && ux >= 1.f && ux < Wc - 2 && vx >= 1.f && vx < Hc - 2 && z > 0.f;
  if (!ok) return;
  float3 dn = p - make_float3(cam.pos[0], cam.pos[1], cam.pos[2]); dn = dn * (1.f / fmaxf(len3(dn), 1e-9f));
  if (-dot3(vn[i], dn) < 0.1f) return;
  float a = sample_bilinear_f(soft, Wc, Hc, ux - 0.5f, vx - 0.5f);
  float3 col;
  if (photo) col = make_float3(sample_bilinear_u8(photo, cam.W, cam.H, 4, 0, u - 0.5f, vv - 0.5f), sample_bilinear_u8(photo, cam.W, cam.H, 4, 1, u - 0.5f, vv - 0.5f), sample_bilinear_u8(photo, cam.W, cam.H, 4, 2, u - 0.5f, vv - 0.5f)) * (1.f / 255.f);
  else col = make_float3(sample_bilinear_f(img, Wc, Hc, ux - 0.5f, vx - 0.5f), sample_bilinear_f(img + (size_t)Wc * Hc, Wc, Hc, ux - 0.5f, vx - 0.5f), sample_bilinear_f(img + 2 * (size_t)Wc * Hc, Wc, Hc, ux - 0.5f, vx - 0.5f));
  alpha[i] = a; cc[i] = col;
}
// One masked blur round over the candidate subset: x <- (x + sum of candidate neighbours) / (1 + their count); 7 lanes.
__global__ void level_blur_kernel(int n, const int* __restrict__ idx, const uint8_t* __restrict__ cand, const int* __restrict__ start, const int* __restrict__ nbr, const float* __restrict__ in, float* __restrict__ out, int nv) {
  int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= n) return;
  int i = idx[k]; float s[7]; for (int l = 0; l < 7; l++) s[l] = in[(size_t)l * nv + i]; int cnt = 1;
  for (int e = start[i]; e < start[i + 1]; e++) { int j = nbr[e]; if (!cand[j]) continue; cnt++; for (int l = 0; l < 7; l++) s[l] += in[(size_t)l * nv + j]; }
  for (int l = 0; l < 7; l++) out[(size_t)l * nv + i] = s[l] / cnt;
}

}  // namespace

int mesh_bake_main(int argc, char** argv) {
  std::string in, out, mode = "trimmed", cap_path, photo_path; std::vector<std::vector<std::string>> views, factors, project;
  float trim = 0.12f, power = 6.f, tol = 0.006f, min_weight = 1e-3f, cap_radius = 0.003f, cap_feather = 0.008f, cap_vis_tol = 0.004f;
  int alpha_min = 128, erode = 3, ss = 2, smooth = 0, chunk = 0, lowpass = 300, device = 0; bool no_level = false;
  ArgParser ap("Bake per-vertex colours from RGBA views, smooth them, and project the face cap's colours from the photo's camera\n\nUsage: b2ctrain mesh-bake --input MESH.ply --output OUT.ply --views CAMS DIR [--views ...] [OPTIONS]");
  ap.s("input", "MESH.ply", "The mesh to colour", in).s("output", "OUT.ply", "The coloured mesh", out)
    .multi("views", "CAMS DIR", 2, "A camera list and a directory of RGBA images named as the cameras (alpha = subject); repeatable", views)
    .multi("factor", "F", 1, "Weight multiplier of the --views group in the same position (default 1)", factors)
    .s("mode", "MODE", "mean | median | trimmed (weighted mean of the samples within --trim of the per-channel weighted median) [default: trimmed]", mode)
    .f("trim", "D", "Colour distance of the trimmed mean [default: 0.12]", trim).f("power", "P", "Facing weight exponent [default: 6]", power)
    .f("tol", "M", "Visibility: the vertex's depth must agree with the mesh's depth at its sub-pixel within this [default: 0.006]", tol)
    .i("alpha-min", "A", "Alpha threshold of the subject mask [default: 128]", alpha_min).i("erode", "PX", "Erosion of the subject mask [default: 3]", erode)
    .i("ss", "N", "Supersampling of the visibility depth buffer [default: 2]", ss)
    .f("min-weight", "W", "Vertices with less total weight are unseen and filled from their neighbours [default: 1e-3]", min_weight)
    .i("chunk", "N", "Vertices per reduction chunk (0 = sized to ~1 GB of samples) [default: 0]", chunk)
    .i("smooth", "N", "1-ring median passes on the colours [default: 0]", smooth)
    .multi("project", "CAMS INDEX", 2, "Project the face cap's colours (--cap) from camera INDEX of CAMS (the photo's) onto the vertices it covers", project)
    .s("cap", "CAP.ply", "The face cap's Gaussians (one per photo pixel): their rasterisation at the photo camera is the face image, their footprint the region", cap_path)
    .s("photo", "IMG", "Take the colours from this RGBA image in the photo camera's pixel grid instead of the cap's own colours (the cap still defines the region)", photo_path)
    .f("cap-feather", "M", "Ramp of the cap's alpha beyond its coverage [default: 0.008]", cap_feather)
    .f("cap-vis-tol", "M", "Visibility tolerance at the photo camera [default: 0.004]", cap_vis_tol)
    .i("lowpass", "N", "Seam levelling: neighbour-averaging rounds of the low band swapped between the cap and the bake (0 = none) [default: 300]", lowpass)
    .b("no-level", "Paste the cap's colours without levelling", no_level)
    .i("device", "N", "CUDA device [default: 0]", device);
  if (!ap.parse(argc, argv)) return 0;
  if (in.empty() || out.empty() || views.empty()) fail("--input, --output and at least one --views group are required");
  if (mode != "mean" && mode != "median" && mode != "trimmed") fail("--mode must be mean, median or trimmed");
  if (!project.empty() && cap_path.empty()) fail("--project needs --cap");
  (void)cap_radius;
  CUDA_CHECK(cudaSetDevice(device)); cudaStream_t stream = 0; double t0 = now_seconds();
  TriMesh m = read_mesh(in); int nv = (int)m.nv(), nf = (int)m.nf();
  MeshRaster rast; rast.upload(m, stream);
  DevBuf<float3> vn; vn.reserve(nv); vertex_normals_gpu(rast.v, nv, rast.f, nf, vn, stream);

  // Every view's image decoded once (host memory), and its group's factor.
  struct View { CameraGPU cam; std::string path; float factor; Image8 img; };
  std::vector<View> vw;
  for (size_t gi = 0; gi < views.size(); gi++) {
    CamSet cs = load_cams(views[gi][0]); float fac = gi < factors.size() ? strtof(factors[gi][0].c_str(), nullptr) : 1.f;
    for (size_t i = 0; i < cs.size(); i++) vw.push_back({CameraGPU::from(cs.cams[i], cs.W, cs.H), views[gi][1] + "/" + cs.names[i], fac, {}});
  }
  std::vector<uint8_t> present(vw.size(), 0);
  parallel_for(vw.size(), [&](size_t i) { present[i] = try_load_image8(vw[i].path, 4, vw[i].img); if (present[i] && (vw[i].img.W != vw[i].cam.W || vw[i].img.H != vw[i].cam.H)) fail("'%s': %d x %d does not match its camera list", vw[i].path.c_str(), vw[i].img.W, vw[i].img.H); });
  { std::vector<View> kept; for (size_t i = 0; i < vw.size(); i++) if (present[i]) kept.push_back(std::move(vw[i])); else log_warn("missing %s", vw[i].path.c_str()); vw = std::move(kept); }
  int nviews = (int)vw.size();
  if (nviews == 0) fail("no view image found");
  log_info("%d views decoded (%.1fs)", nviews, now_seconds() - t0);

  bool need_samples = mode != "mean";
  if (chunk <= 0) chunk = need_samples ? (int)std::max<size_t>(4096, ((size_t)1 << 30) / ((size_t)nviews * sizeof(Sample))) : nv;  // ~1 GB of samples
  chunk = std::min(chunk, nv);
  DevBuf<float3> acc, colours; DevBuf<float> wsum; acc.reserve(nv); wsum.reserve(nv); colours.reserve(nv); acc.zero(stream); wsum.zero(stream);
  DevBuf<Sample> samples; if (need_samples) samples.reserve((size_t)nviews * chunk);
  DevBuf<uint8_t> rgba, okmask;
  for (int v0 = 0; v0 < nv; v0 += chunk) {
    int n = std::min(chunk, nv - v0);
    for (int vi = 0; vi < nviews; vi++) {
      View& w = vw[vi]; size_t npx = (size_t)w.cam.W * w.cam.H;
      rgba.upload(w.img.px, stream); okmask.reserve(npx);
      dim3 blk(16, 16), grd((w.cam.W + 15) / 16, (w.cam.H + 15) / 16);
      mask_erode_kernel<<<grd, blk, 0, stream>>>(w.cam.W, w.cam.H, rgba, alpha_min, erode, okmask); CUDA_KERNEL_CHECK();
      rast.raster(w.cam, ss, false, stream);
      sample_kernel<<<div_up(n, 256), 256, 0, stream>>>(v0, n, rast.v, vn, w.cam, ss, rast.z, rgba, okmask, tol, power, w.factor, vi, nviews, need_samples ? samples.ptr : nullptr, acc, wsum); CUDA_KERNEL_CHECK();
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    if (need_samples) { reduce_kernel<<<div_up(n, 128), 128, 0, stream>>>(n, nviews, samples, mode == "trimmed", trim, colours.ptr + v0); CUDA_KERNEL_CHECK(); }
  }
  if (!need_samples) { mean_kernel<<<div_up(nv, 256), 256, 0, stream>>>(nv, acc, wsum, colours); CUDA_KERNEL_CHECK(); }
  samples.free();
  std::vector<float> hw = wsum.download(nv, stream); size_t nbad = 0; for (float x : hw) nbad += x < min_weight;
  log_info("%d views baked in %.1fs; unseen vertices %.2f%%", nviews, now_seconds() - t0, 100.0 * nbad / nv);
  Adjacency adj; adj.build(m); adj.upload(stream);
  if (nbad) {
    std::vector<uint8_t> hk(nv); for (int i = 0; i < nv; i++) hk[i] = hw[i] >= min_weight;
    DevBuf<uint8_t> known, known2; known.upload(hk, stream); known2.reserve(nv);
    DevBuf<float3> c2; c2.reserve(nv); DevBuf<int> changed; changed.reserve(1);
    for (int round = 0; round < 200; round++) {
      changed.zero(stream);
      fill_kernel<<<div_up(nv, 256), 256, 0, stream>>>(nv, colours, known, adj.d_start, adj.d_nbr, c2, known2, changed); CUDA_KERNEL_CHECK();
      std::swap(colours.ptr, c2.ptr); std::swap(known.ptr, known2.ptr);
      if (changed.download(1, stream)[0] == 0) break;
    }
  }
  if (smooth > 0) {
    std::vector<int> deg(nv); for (int i = 0; i < nv; i++) deg[i] = adj.start[i + 1] - adj.start[i];
    std::vector<int> sd = deg; std::sort(sd.begin(), sd.end()); int K = std::min(63, sd[(size_t)(0.99 * (nv - 1))]);
    DevBuf<float3> c2; c2.reserve(nv);
    for (int s = 0; s < smooth; s++) { ring_median_kernel<<<div_up(nv, 256), 256, 0, stream>>>(nv, colours, adj.d_start, adj.d_nbr, K, c2); CUDA_KERNEL_CHECK(); std::swap(colours.ptr, c2.ptr); }
    log_info("smoothed %d pass(es), ring K %d", smooth, K);
  }
  if (!project.empty()) {
    // The cap's Gaussians rasterised at the photo camera: one per photo pixel, nearest wins.
    SplatCloud cap = read_ply(cap_path); int nc = (int)cap.n; int K = cap.K();
    CamSet cs = load_cams(project[0][0]); int idx = atoi(project[0][1].c_str());
    if (idx < 0 || idx >= (int)cs.size()) fail("--project: index %d out of range (%zu cameras)", idx, cs.size());
    CameraGPU cam = CameraGPU::from(cs.cams[idx], cs.W, cs.H);
    std::vector<float> ug(nc), vg(nc), zg(nc); float umin = INFINITY, umax = -INFINITY, vmin = INFINITY, vmax = -INFINITY;
    float3 lo = make_float3(INFINITY, INFINITY, INFINITY), hi = make_float3(-INFINITY, -INFINITY, -INFINITY);
    for (int i = 0; i < nc; i++) {
      float x = cap.pos[i * 3], y = cap.pos[i * 3 + 1], z = cap.pos[i * 3 + 2];
      lo.x = std::min(lo.x, x); lo.y = std::min(lo.y, y); lo.z = std::min(lo.z, z); hi.x = std::max(hi.x, x); hi.y = std::max(hi.y, y); hi.z = std::max(hi.z, z);
      zg[i] = project_point(cam, x, y, z, ug[i], vg[i]);
      umin = std::min(umin, ug[i]); umax = std::max(umax, ug[i]); vmin = std::min(vmin, vg[i]); vmax = std::max(vmax, vg[i]);
    }
    lo = make_float3(lo.x - 0.03f, lo.y - 0.03f, lo.z - 0.03f); hi = make_float3(hi.x + 0.03f, hi.y + 0.03f, hi.z + 0.03f);
    int u0 = (int)std::floor(umin) - 4, v0 = (int)std::floor(vmin) - 4, Wc = (int)std::ceil(umax) - u0 + 8, Hc = (int)std::ceil(vmax) - v0 + 8;
    if (Wc <= 0 || Hc <= 0 || (size_t)Wc * Hc > (size_t)64 << 20) fail("--project: the cap projects to a %d x %d image at this camera", Wc, Hc);
    std::vector<float> img((size_t)3 * Wc * Hc, 0.f), cov((size_t)Wc * Hc, 0.f);
    std::vector<int> order(nc); std::iota(order.begin(), order.end(), 0); std::sort(order.begin(), order.end(), [&](int a, int b) { return zg[a] > zg[b]; });
    for (int i : order) {
      int px = (int)std::lround(ug[i] - 0.5f) - u0, py = (int)std::lround(vg[i] - 0.5f) - v0; if (px < 0 || py < 0 || px >= Wc || py >= Hc) continue;
      float op = 1.f / (1.f + std::exp(-cap.opacity[i]));
      for (int ch = 0; ch < 3; ch++) img[(size_t)ch * Wc * Hc + (size_t)py * Wc + px] = std::min(std::max(0.5f + SH_C0 * cap.sh[(size_t)i * K * 3 + ch], 0.f), 1.f) * op;
      cov[(size_t)py * Wc + px] = op;
    }
    extend_cap_colors(img, cov, Wc, Hc);
    // inside = erode(cov > 0.5, 3x3); soft = the Gaussian blur of it, kept at 1 inside.
    std::vector<float> inside((size_t)Wc * Hc, 0.f), soft((size_t)Wc * Hc, 0.f);
    for (int y = 0; y < Hc; y++) for (int x = 0; x < Wc; x++) {
      bool in = true; for (int dy = -1; dy <= 1 && in; dy++) for (int dx = -1; dx <= 1 && in; dx++) { int xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= Wc || yy >= Hc) continue; in = cov[(size_t)yy * Wc + xx] > 0.5f; }
      inside[(size_t)y * Wc + x] = in ? 1.f : 0.f;
    }
    {
      float sigma = std::max(cap_feather / 0.0013f, 0.5f); int R = (int)std::lround(sigma * 4); std::vector<float> ker(2 * R + 1); float ks = 0;
      for (int k = -R; k <= R; k++) { ker[k + R] = std::exp(-0.5f * k * k / (sigma * sigma)); ks += ker[k + R]; }
      for (auto& k : ker) k /= ks;
      auto refl = [](int i, int n) { if (i < 0) i = -i; if (i >= n) i = 2 * n - 2 - i; return std::min(std::max(i, 0), n - 1); };
      std::vector<float> tmp((size_t)Wc * Hc);
      for (int y = 0; y < Hc; y++) for (int x = 0; x < Wc; x++) { float s = 0; for (int k = -R; k <= R; k++) s += ker[k + R] * inside[(size_t)y * Wc + refl(x + k, Wc)]; tmp[(size_t)y * Wc + x] = s; }
      for (int y = 0; y < Hc; y++) for (int x = 0; x < Wc; x++) { float s = 0; for (int k = -R; k <= R; k++) s += ker[k + R] * tmp[(size_t)refl(y + k, Hc) * Wc + x]; soft[(size_t)y * Wc + x] = inside[(size_t)y * Wc + x] > 0.f ? 1.f : s; }
    }
    DevBuf<float> d_img, d_soft; d_img.upload(img, stream); d_soft.upload(soft, stream);
    DevBuf<uint8_t> d_photo; if (!photo_path.empty()) { Image8 ph = load_image8(photo_path, 4); if (ph.W != cs.W || ph.H != cs.H) fail("--photo: %d x %d, the camera list is %d x %d", ph.W, ph.H, cs.W, cs.H); d_photo.upload(ph.px, stream); }
    rast.raster(cam, 2, false, stream);
    DevBuf<float3> cc; DevBuf<float> alpha; DevBuf<uint8_t> cand; cc.reserve(nv); alpha.reserve(nv); cand.reserve(nv);
    cap_project_kernel<<<div_up(nv, 256), 256, 0, stream>>>(nv, rast.v, vn, cam, 2, rast.z, lo, hi, d_img, d_soft, u0, v0, Wc, Hc, d_photo.ptr, cap_vis_tol, cc, alpha, cand); CUDA_KERNEL_CHECK();
    std::vector<float> ha = alpha.download(nv, stream); std::vector<uint8_t> hc = cand.download(nv, stream);
    size_t nfull = 0, ntouch = 0, ncand = 0; for (int i = 0; i < nv; i++) { nfull += ha[i] >= 1.f; ntouch += ha[i] > 0.f; ncand += hc[i]; }
    log_info("cap projected from camera %d: image %d x %d, %zu vertices full / %zu touched of %zu candidates", idx, Wc, Hc, nfull, ntouch, ncand);
    std::vector<float3> Cf = colours.download(nv, stream), Cc = cc.download(nv, stream);
    if (no_level || lowpass <= 0) {
      for (int i = 0; i < nv; i++) if (ha[i] > 0.f) { float a = ha[i]; Cf[i] = make_float3(Cc[i].x * a + Cf[i].x * (1.f - a), Cc[i].y * a + Cf[i].y * (1.f - a), Cc[i].z * a + Cf[i].z * (1.f - a)); }
    } else {
      // Low band on the candidate sub-mesh: masked blur of the cap colours, plain blur of the bake; the cap keeps its
      // detail (high band) over the bake's tone (low band).
      std::vector<int> sub; for (int i = 0; i < nv; i++) if (hc[i]) sub.push_back(i);
      std::vector<float> lanes((size_t)7 * nv, 0.f);
      for (int i : sub) { lanes[i] = ha[i] * Cc[i].x; lanes[(size_t)nv + i] = ha[i] * Cc[i].y; lanes[2 * (size_t)nv + i] = ha[i] * Cc[i].z; lanes[3 * (size_t)nv + i] = ha[i]; lanes[4 * (size_t)nv + i] = Cf[i].x; lanes[5 * (size_t)nv + i] = Cf[i].y; lanes[6 * (size_t)nv + i] = Cf[i].z; }
      DevBuf<int> d_sub; d_sub.upload(sub, stream); DevBuf<float> a, b; a.upload(lanes, stream); b.upload(lanes, stream);
      for (int r = 0; r < lowpass; r++) { level_blur_kernel<<<div_up((int)sub.size(), 256), 256, 0, stream>>>((int)sub.size(), d_sub, cand, adj.d_start, adj.d_nbr, a, b, nv); CUDA_KERNEL_CHECK(); std::swap(a.ptr, b.ptr); }
      std::vector<float> L = a.download((size_t)7 * nv, stream); double shift[3] = {0, 0, 0}; size_t nsh = 0;
      for (int i : sub) {
        float den = std::max(L[3 * (size_t)nv + i], 1e-6f);
        float3 low_c = make_float3(L[i] / den, L[(size_t)nv + i] / den, L[2 * (size_t)nv + i] / den), low_f = make_float3(L[4 * (size_t)nv + i], L[5 * (size_t)nv + i], L[6 * (size_t)nv + i]);
        float al = ha[i]; if (al <= 0.f) continue;
        float3 lev = make_float3(Cc[i].x - low_c.x + low_f.x, Cc[i].y - low_c.y + low_f.y, Cc[i].z - low_c.z + low_f.z);
        lev.x = std::min(std::max(lev.x, 0.f), 1.f); lev.y = std::min(std::max(lev.y, 0.f), 1.f); lev.z = std::min(std::max(lev.z, 0.f), 1.f);
        Cf[i] = make_float3(al * lev.x + (1 - al) * Cf[i].x, al * lev.y + (1 - al) * Cf[i].y, al * lev.z + (1 - al) * Cf[i].z);
        if (al >= 1.f) { shift[0] += std::abs(low_f.x - low_c.x); shift[1] += std::abs(low_f.y - low_c.y); shift[2] += std::abs(low_f.z - low_c.z); nsh++; }
      }
      log_info("levelled: mean low-band shift on the cap (%.3f, %.3f, %.3f) over %zu vertices", shift[0] / std::max<size_t>(nsh, 1), shift[1] / std::max<size_t>(nsh, 1), shift[2] / std::max<size_t>(nsh, 1), nsh);
    }
    colours.upload(Cf, stream);
    // The cap's coverage travels with the mesh as the vertex alpha: mesh-unwrap's protect_cap.png is the texels whose colour came from the photograph.
    m.alpha.resize(nv); for (int i = 0; i < nv; i++) m.alpha[i] = (uint8_t)std::min(std::max(ha[i] * 255.f + 0.5f, 0.f), 255.f);
  }
  std::vector<float3> C = colours.download(nv, stream), N = vn.download(nv, stream);
  m.col.resize((size_t)nv * 3); m.nrm.resize((size_t)nv * 3);
  auto u8 = [](float x) { return (uint8_t)std::min(std::max(x * 255.f + 0.5f, 0.f), 255.f); };
  for (int i = 0; i < nv; i++) { m.col[i * 3] = u8(C[i].x); m.col[i * 3 + 1] = u8(C[i].y); m.col[i * 3 + 2] = u8(C[i].z); m.nrm[i * 3] = N[i].x; m.nrm[i * 3 + 1] = N[i].y; m.nrm[i * 3 + 2] = N[i].z; }
  write_ply_mesh(out, m);
  log_info("wrote %s (%.1fs)", out.c_str(), now_seconds() - t0);
  return 0;
}

}  // namespace b2c
