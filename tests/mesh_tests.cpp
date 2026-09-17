// Tests of the meshification kernels (docs/mesh-plan.md): marching cubes on an analytic sphere, TSDF fusion of a
// synthetic sphere from rendered depth maps, closest-triangle queries against brute force, and a textured cube's
// round trip through unwrap -> render -> backproject.
#include "mesh/fuse.h"
#include "mesh/raster.h"
#include "mesh/common.h"
#include "mesh/mesh_main.h"
#include "mesh/cap_color.h"
#include "util/log.h"
#include <filesystem>
#include <random>
#include <cmath>
#include <cstdio>
#include <map>

using namespace b2c;
namespace fs = std::filesystem;

namespace {

int check(bool ok, const char* what) { printf("mesh: %s: %s\n", what, ok ? "ok" : "FAIL"); return ok ? 0 : 1; }

int test_cap_color() {
  const int W = 9, H = 9, N = W * H;
  std::vector<float> rgb(3 * N, 0.f), cov(N, 0.f);
  const float color[3] = {.72f, .43f, .26f};
  // A constant foreground with varying opacity and a missing centre sample.
  for (int y = 2; y <= 6; ++y) for (int x = 2; x <= 6; ++x) {
    if (x == 4 && y == 4) continue;
    int i = y * W + x;
    cov[i] = .1f + .02f * i;
    cov[i] = std::min(cov[i], 1.f);
    for (int c = 0; c < 3; ++c) rgb[c * N + i] = color[c] * cov[i];
  }
  extend_cap_colors(rgb, cov, W, H);
  float error = 0.f;
  for (int i = 0; i < N; ++i) for (int c = 0; c < 3; ++c)
    error = std::max(error, std::abs(rgb[c * N + i] - color[c]));
  int fails = check(error < 1e-6f && cov[0] == 0.f && cov[4 * W + 4] > 0.f,
                    "cap color: soft alpha and feather preserve constant foreground; extension adds no coverage");
  rgb.assign(3 * N, 0.f); cov.assign(N, 0.f); cov[4 * W + 4] = .2f;
  extend_cap_colors(rgb, cov, W, H);
  fails += check(*std::max_element(rgb.begin(), rgb.end()) == 0.f && cov[4 * W + 4] == .2f && cov[4 * W + 5] < .2f,
                 "cap color: observed black is valid and weak coverage is not promoted");
  rgb.assign(3 * N, 0.f); cov.assign(N, 0.f);
  extend_cap_colors(rgb, cov, W, H);
  fails += check(*std::max_element(cov.begin(), cov.end()) == 0.f, "cap color: empty input stays empty");
  return fails;
}

// A cube [-h, h]^3 with per-face vertices (24) and a colour per vertex.
TriMesh make_cube(float h) {
  TriMesh m;
  const float n[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  for (int f = 0; f < 6; f++) {
    float a[3] = {n[f][1], n[f][2], n[f][0]}, b[3] = {n[f][1] * a[2] - n[f][2] * a[1], n[f][2] * a[0] - n[f][0] * a[2], n[f][0] * a[1] - n[f][1] * a[0]};
    uint32_t base = (uint32_t)m.nv();
    const float s[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    for (int k = 0; k < 4; k++) {
      float p[3]; for (int i = 0; i < 3; i++) p[i] = h * (n[f][i] + s[k][0] * a[i] + s[k][1] * b[i]);
      m.v.insert(m.v.end(), p, p + 3);
      for (int i = 0; i < 3; i++) m.col.push_back((uint8_t)std::lround(127.5f + 120.f * p[i] / h));
    }
    // wind so the normal points along n
    float e1[3], e2[3]; for (int i = 0; i < 3; i++) { e1[i] = m.v[(base + 1) * 3 + i] - m.v[base * 3 + i]; e2[i] = m.v[(base + 2) * 3 + i] - m.v[base * 3 + i]; }
    float c[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    bool flip = c[0] * n[f][0] + c[1] * n[f][1] + c[2] * n[f][2] < 0;
    uint32_t q[4] = {base, base + 1, base + 2, base + 3}; if (flip) std::swap(q[1], q[3]);
    m.f.insert(m.f.end(), {q[0], q[1], q[2], q[0], q[2], q[3]});
  }
  return m;
}

// An icosphere-like closed mesh: a subdivided octahedron projected onto the sphere.
TriMesh make_sphere(float r, int sub) {
  TriMesh m;
  std::vector<float> v = {1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1};
  std::vector<uint32_t> f = {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
  for (int s = 0; s < sub; s++) {
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> mid; std::vector<uint32_t> nf;
    auto midpoint = [&](uint32_t a, uint32_t b) {
      auto key = std::make_pair(std::min(a, b), std::max(a, b)); auto it = mid.find(key); if (it != mid.end()) return it->second;
      float p[3]; for (int i = 0; i < 3; i++) p[i] = 0.5f * (v[a * 3 + i] + v[b * 3 + i]); float l = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
      uint32_t id = (uint32_t)(v.size() / 3); for (int i = 0; i < 3; i++) v.push_back(p[i] / l); mid[key] = id; return id;
    };
    for (size_t t = 0; t < f.size() / 3; t++) {
      uint32_t a = f[t * 3], b = f[t * 3 + 1], c = f[t * 3 + 2], ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
      nf.insert(nf.end(), {a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca});
    }
    f = std::move(nf);
  }
  for (auto& x : v) x *= r;
  m.v = std::move(v); m.f = std::move(f);
  return m;
}

int test_marching_cubes() {
  float r = 0.1f, vox = 0.004f, lo[3] = {-0.16f, -0.16f, -0.16f}, hi[3] = {0.16f, 0.16f, 0.16f};
  VolumeGrid g = VolumeGrid::from_bbox(lo, hi, vox);
  std::vector<float> vol(g.n());
  for (int x = 0; x < g.dims.x; x++) for (int y = 0; y < g.dims.y; y++) for (int z = 0; z < g.dims.z; z++) {
    float px = g.lo.x + x * vox, py = g.lo.y + y * vox, pz = g.lo.z + z * vox;
    vol[((size_t)x * g.dims.y + y) * g.dims.z + z] = std::sqrt(px * px + py * py + pz * pz) - r;
  }
  DevBuf<float> d; d.upload(vol);
  TriMesh m = marching_cubes(g, d);
  double max_err = 0, mean_err = 0;
  for (size_t i = 0; i < m.nv(); i++) { double rr = std::sqrt(m.v[i * 3] * m.v[i * 3] + m.v[i * 3 + 1] * m.v[i * 3 + 1] + m.v[i * 3 + 2] * m.v[i * 3 + 2]); double e = std::abs(rr - r); max_err = std::max(max_err, e); mean_err += e; }
  mean_err /= std::max<size_t>(m.nv(), 1);
  // Every edge shared by exactly two faces (closed surface) and outward winding.
  std::map<std::pair<uint32_t, uint32_t>, int> edges; size_t outward = 0, degenerate = 0;
  for (size_t t = 0; t < m.nf(); t++) {
    uint32_t a = m.f[t * 3], b = m.f[t * 3 + 1], c = m.f[t * 3 + 2];
    edges[{std::min(a, b), std::max(a, b)}]++; edges[{std::min(b, c), std::max(b, c)}]++; edges[{std::min(c, a), std::max(c, a)}]++;
    const float* pa = &m.v[a * 3]; const float* pb = &m.v[b * 3]; const float* pc = &m.v[c * 3];
    float e1[3] = {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]}, e2[3] = {pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]};
    float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    float cen[3] = {(pa[0] + pb[0] + pc[0]) / 3, (pa[1] + pb[1] + pc[1]) / 3, (pa[2] + pb[2] + pc[2]) / 3};
    if (n[0] * n[0] + n[1] * n[1] + n[2] * n[2] < 1e-20f) degenerate++;  // an edge vertex exactly on a grid corner: zero area
    else if (n[0] * cen[0] + n[1] * cen[1] + n[2] * cen[2] > 0) outward++;
  }
  size_t bad_edges = 0; for (auto& [k, c] : edges) if (c != 2) bad_edges++;
  printf("mesh: marching cubes sphere: V %zu F %zu, radius error mean %.4f max %.4f voxels, non-manifold edges %zu, outward %zu + degenerate %zu of %zu\n", m.nv(), m.nf(), mean_err / vox, max_err / vox, bad_edges, outward, degenerate, m.nf());
  return check(m.nv() > 1000 && max_err < 0.5 * vox && bad_edges == 0 && outward + degenerate == m.nf() && degenerate < m.nf() * 3 / 100, "marching cubes");
}

int test_tsdf_sphere() {
  float r = 0.12f, vox = 0.004f, lo[3] = {-0.2f, -0.2f, -0.2f}, hi[3] = {0.2f, 0.2f, 0.2f};
  VolumeGrid g = VolumeGrid::from_bbox(lo, hi, vox);
  DevBuf<float> sdf, wgt; sdf.reserve(g.n()); wgt.reserve(g.n()); sdf.zero(); wgt.zero();
  int W = 160, H = 160; std::vector<DevBuf<float>> maps(24); std::vector<FuseView> views;
  for (int k = 0; k < 24; k++) {
    // a camera on a tilted ring, looking at the origin
    float az = k * 2.f * (float)M_PI / 24, el = (k % 2 ? 0.5f : -0.5f);
    float pos[3] = {0.8f * std::cos(el) * std::sin(az), 0.8f * std::sin(el), 0.8f * std::cos(el) * std::cos(az)};
    float f[3] = {-pos[0], -pos[1], -pos[2]}; float fl = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]); for (auto& x : f) x /= fl;
    float up[3] = {0, 1, 0}; float rgt[3] = {f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]}; float rl = std::sqrt(rgt[0] * rgt[0] + rgt[1] * rgt[1] + rgt[2] * rgt[2]); for (auto& x : rgt) x /= rl;
    float dn[3] = {f[1] * rgt[2] - f[2] * rgt[1], f[2] * rgt[0] - f[0] * rgt[2], f[0] * rgt[1] - f[1] * rgt[0]};  // OpenCV: y down = f x right
    Camera cam; cam.width = W; cam.height = H; cam.fx = cam.fy = 200.f; cam.cx = W / 2.f; cam.cy = H / 2.f;
    for (int i = 0; i < 3; i++) { cam.R[i] = rgt[i]; cam.R[3 + i] = dn[i]; cam.R[6 + i] = f[i]; }
    for (int i = 0; i < 3; i++) cam.t[i] = -(cam.R[i * 3] * pos[0] + cam.R[i * 3 + 1] * pos[1] + cam.R[i * 3 + 2] * pos[2]);
    cam.update_pos();
    CameraGPU cg = CameraGPU::from(cam, W, H);
    std::vector<float> depth((size_t)W * H, 0.f);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
      float dx = (x + 0.5f - cam.cx) / cam.fx, dy = (y + 0.5f - cam.cy) / cam.fy;
      float d[3] = {cam.R[0] * dx + cam.R[3] * dy + cam.R[6], cam.R[1] * dx + cam.R[4] * dy + cam.R[7], cam.R[2] * dx + cam.R[5] * dy + cam.R[8]};
      // ray o + t d hits |p| = r: t^2 |d|^2 + 2 t o.d + |o|^2 - r^2 = 0
      float A = d[0] * d[0] + d[1] * d[1] + d[2] * d[2], B = 2 * (pos[0] * d[0] + pos[1] * d[1] + pos[2] * d[2]), C = pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2] - r * r;
      float disc = B * B - 4 * A * C; if (disc < 0) continue;
      float t = (-B - std::sqrt(disc)) / (2 * A); if (t <= 0) continue;
      depth[(size_t)y * W + x] = t;  // camera z of the hit: t * (unit z component of d) = t since d.z_cam = 1
    }
    // the edge rejection mesh-fuse applies (grazing pixels bias a TSDF at silhouettes): drop steep depth gradients
    std::vector<float> eroded = depth;
    for (int y = 1; y < H - 1; y++) for (int x = 1; x < W - 1; x++) {
      float gx = std::abs(depth[(size_t)y * W + x + 1] - depth[(size_t)y * W + x - 1]) / 2, gy = std::abs(depth[(size_t)(y + 1) * W + x] - depth[(size_t)(y - 1) * W + x]) / 2;
      if (std::max(gx, gy) > 0.01f || depth[(size_t)y * W + x] <= 0.f) eroded[(size_t)y * W + x] = 0.f;
    }
    maps[k].upload(eroded);
    views.push_back({cg, maps[k].ptr, 1.f, false});
  }
  tsdf_integrate(g, sdf, wgt, nullptr, views, 4 * vox);
  std::vector<float> hs = sdf.download(), hw = wgt.download();
  for (size_t i = 0; i < g.n(); i++) if (hw[i] < 1.f) hs[i] = 1.f;
  sdf.upload(hs);
  size_t filled = fill_cavities(g, sdf);  // the unobserved interior would otherwise grow an inner shell at the truncation distance
  TriMesh m = marching_cubes(g, sdf);
  double max_err = 0, mean_err = 0;
  for (size_t i = 0; i < m.nv(); i++) { double rr = std::sqrt(m.v[i * 3] * m.v[i * 3] + m.v[i * 3 + 1] * m.v[i * 3 + 1] + m.v[i * 3 + 2] * m.v[i * 3 + 2]); double e = std::abs(rr - r); max_err = std::max(max_err, e); mean_err += e; }
  mean_err /= std::max<size_t>(m.nv(), 1);
  printf("mesh: tsdf sphere from 24 depth maps: V %zu F %zu, radius error mean %.3f max %.3f voxels, %zu interior voxels filled\n", m.nv(), m.nf(), mean_err / vox, max_err / vox, filled);
  return check(m.nv() > 1000 && mean_err < 0.5 * vox && max_err < 1.5 * vox, "tsdf fusion");
}

int test_closest() {
  TriMesh m = make_sphere(0.1f, 3);
  TriGrid tg; tg.build(m, 0.02f, true);
  std::mt19937 rng(7); std::uniform_real_distribution<float> u(-0.2f, 0.2f);
  int n = 2000; std::vector<float3> q(n); for (auto& p : q) p = make_float3(u(rng), u(rng), u(rng));
  DevBuf<float3> dq, cp, bary; DevBuf<int> tri; DevBuf<float> dist, sign; dq.upload(q); cp.reserve(n); bary.reserve(n); tri.reserve(n); dist.reserve(n); sign.reserve(n);
  closest_points_gpu(tg.dev(), dq, n, 32, tri, cp, dist, bary, sign);
  std::vector<float> hd = dist.download(n), hsg = sign.download(n); std::vector<int> ht = tri.download(n);
  // brute force
  double max_err = 0; int sign_bad = 0;
  for (int i = 0; i < n; i++) {
    float best = INFINITY;
    for (size_t t = 0; t < m.nf(); t++) {
      const float* a = &m.v[m.f[t * 3] * 3]; const float* b = &m.v[m.f[t * 3 + 1] * 3]; const float* c = &m.v[m.f[t * 3 + 2] * 3];
      // sample the triangle densely (barycentric grid) for a robust brute-force bound
      for (int s = 0; s <= 20; s++) for (int r = 0; r <= 20 - s; r++) {
        float w0 = s / 20.f, w1 = r / 20.f, w2 = 1 - w0 - w1;
        float px = w0 * a[0] + w1 * b[0] + w2 * c[0], py = w0 * a[1] + w1 * b[1] + w2 * c[1], pz = w0 * a[2] + w1 * b[2] + w2 * c[2];
        float d = std::sqrt((px - q[i].x) * (px - q[i].x) + (py - q[i].y) * (py - q[i].y) + (pz - q[i].z) * (pz - q[i].z));
        best = std::min(best, d);
      }
    }
    // the sampled brute force is an upper bound within one sample step (~edge / 20)
    if (hd[i] > best + 1e-6f) max_err = std::max(max_err, (double)(hd[i] - best));
    float rr = std::sqrt(q[i].x * q[i].x + q[i].y * q[i].y + q[i].z * q[i].z);
    if (ht[i] >= 0 && std::abs(rr - 0.1f) > 0.003f && ((rr > 0.1f) != (hsg[i] > 0))) sign_bad++;
  }
  printf("mesh: closest triangle on a %zu-triangle sphere: max excess over brute force %.2e, wrong signs %d / %d\n", m.nf(), max_err, sign_bad, n);
  int fails = check(max_err < 1e-6 && sign_bad == 0, "closest point");
  // The volume SDF of the same sphere: exact band + winding sign against the analytic distance.
  float vox = 0.005f, lo[3] = {-0.15f, -0.15f, -0.15f}, hi[3] = {0.15f, 0.15f, 0.15f};
  VolumeGrid g = VolumeGrid::from_bbox(lo, hi, vox); DevBuf<float> out; out.reserve(g.n());
  mesh_signed_distance(g, m, 0.03f, out);
  std::vector<float> h = out.download(g.n()); double band_err = 0; size_t nb = 0, sign_wrong = 0;
  for (int x = 0; x < g.dims.x; x++) for (int y = 0; y < g.dims.y; y++) for (int z = 0; z < g.dims.z; z++) {
    float px = g.lo.x + x * vox, py = g.lo.y + y * vox, pz = g.lo.z + z * vox; float rr = std::sqrt(px * px + py * py + pz * pz); float v = h[((size_t)x * g.dims.y + y) * g.dims.z + z];
    if (std::abs(rr - 0.1f) > 0.002f && ((rr > 0.1f) != (v > 0))) sign_wrong++;
    if (std::abs(v) < 900.f) { band_err += std::abs(std::abs(v) - std::abs(rr - 0.1f)); nb++; }
  }
  printf("mesh: mesh sdf on the grid: band mean error %.2e (%zu voxels; the mesh is a %d-subdivision sphere), wrong signs %zu\n", band_err / std::max<size_t>(nb, 1), nb, 3, sign_wrong);
  fails += check(sign_wrong == 0 && band_err / std::max<size_t>(nb, 1) < 1e-3, "mesh sdf");
  return fails;
}

int run(std::vector<std::string> args) {
  std::vector<char*> argv; argv.push_back((char*)"b2ctrain"); for (auto& a : args) argv.push_back(a.data());
  return mesh_main((int)argv.size(), argv.data());
}

int test_cube_round_trip() {
  fs::path dir = fs::temp_directory_path() / "b2c_mesh_test"; fs::remove_all(dir); fs::create_directories(dir);
  TriMesh cube = make_cube(0.2f); write_ply_mesh((dir / "cube.ply").string(), cube);
  std::string atlas = (dir / "atlas").string();
  if (run({"mesh-unwrap", "--input", (dir / "cube.ply").string(), "--output", atlas, "--tris", "100", "--res", "512", "--pad", "4"})) return check(false, "unwrap");
  Image8 tex = load_image8(atlas + "/texture.png", 3), mask = load_image8(atlas + "/mask.png", 1);
  // cameras: 6 axis views + 8 corners, 512 px
  std::string cams = (dir / "cams.json").string();
  if (run({"mesh-render", "--make-cams", "--atlas", atlas, "--output", cams, "--width", "512", "--height", "512", "--azims", "0,45,90,135,180,225,270,315", "--elevs", "-60,-20,20,60", "--radius", "1.2", "--fov", "0.8"})) return check(false, "make-cams");
  std::string renders = (dir / "renders").string();
  if (run({"mesh-render", "--atlas", atlas, "--cameras", cams, "--output", renders, "--depth"})) return check(false, "render");
  // grey texture as the start, back-project the renders: the covered texels must come back
  std::vector<uint8_t> grey(tex.px.size(), 128); write_png8((dir / "grey.png").string(), tex.W, tex.H, 3, grey.data());
  std::string outtex = (dir / "back.png").string();
  if (run({"mesh-backproject", "--atlas", atlas, "--cameras", cams, "--images", renders, "--renders", renders, "--output", outtex, "--texture", (dir / "grey.png").string(), "--erode", "2"})) return check(false, "backproject");
  Image8 back = load_image8(outtex, 3);
  double sum = 0; size_t n = 0, unseen = 0;
  for (size_t i = 0; i < (size_t)tex.W * tex.H; i++) {
    if (!mask.px[i]) continue;
    // texels the views never reached stay grey; count them separately
    if (back.px[i * 3] == 128 && back.px[i * 3 + 1] == 128 && back.px[i * 3 + 2] == 128) { unseen++; continue; }
    for (int k = 0; k < 3; k++) sum += std::abs((int)back.px[i * 3 + k] - (int)tex.px[i * 3 + k]);
    n++;
  }
  double mean = n ? sum / (3.0 * n) : 999;
  printf("mesh: cube round trip (unwrap -> render -> backproject): %zu covered texels compared, mean |diff| %.3f/255, %zu unseen\n", n, mean, unseen);
  int fails = check(n > 10000 && mean < 1.0 && unseen < n / 20, "cube round trip");
  // the renders themselves: alpha covers the cube, colours are the cube's
  Image8 r0 = load_image8(renders + "/body_e-20_a045.png", 4); size_t hit = 0; for (size_t i = 0; i < (size_t)r0.W * r0.H; i++) hit += r0.px[i * 4 + 3] > 200;
  fails += check(hit > 20000 && hit < 200000, "render coverage");
  fs::remove_all(dir);
  return fails;
}

}  // namespace

int test_mesh() {
  int fails = 0;
  fails += test_cap_color();
  fails += test_marching_cubes();
  fails += test_tsdf_sphere();
  fails += test_closest();
  fails += test_cube_round_trip();
  return fails;
}
