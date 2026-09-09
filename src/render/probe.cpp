// `b2ctrain probe`: measure false transparency of a trained splat. For every camera it composites the splat and
// reports, per pixel, how much of the pixel's weight arrives from well behind the first surface it hits (and, with a
// proxy mesh, from behind the body surface). Writes probe.json and per-view heat maps.
#include "render/probe.h"
#include "render/renderer.h"
#include "gpu/render.h"
#include "gpu/probe.h"
#include "gpu/meshdepth.h"
#include "dataset/mesh.h"
#include "dataset/colmap.h"
#include "train/init.h"
#include "ply.h"
#include "util/log.h"
#include "stb_image_write.h"
#include "json.hpp"
#include <fstream>
#include <filesystem>
#include <cmath>
#include <cstring>

namespace b2c {
namespace fs = std::filesystem;

namespace {
const char* HELP =
"Measure false transparency of a trained splat against a camera list\n\n"
"Usage: b2ctrain probe [OPTIONS] --splat <SPLAT> --cameras <CAMERAS>\n\n"
"Options:\n"
"      --splat <SPLAT>          Trained gaussian splat scene, as .ply\n"
"      --cameras <CAMERAS>      Camera list (body2colmap cameras.json)\n"
"      --output-dir <DIR>       Where probe.json and the heat maps go [default: probe]\n"
"      --every <N>              Probe every N-th camera [default: 1]\n"
"      --tau <TAU>              Accumulated alpha that marks the first surface [default: 0.1]\n"
"      --delta <DIST>           Weight arriving more than this behind the first surface counts as 'deep' (scene units) [default: 0.03]\n"
"      --mesh <PATH>            Body proxy mesh; adds the 'behind' measure (weight from behind the mesh surface)\n"
"      --points <COLMAP_DIR>    Instead of (or as well as) a mesh: the model's points3D.txt splatted as discs, as the trainer's fallback does\n"
"      --points-radius <DIST>   Surfel radius for --points, 0 = four times the median point spacing [default: 0]\n"
"      --margin <DIST>          Depth behind the mesh surface where 'behind' starts [default: 0.05]\n"
"      --dilate <PX>            Mesh reference depth is the farthest surface within this radius [default: 2]\n"
"      --images                 Also write the RGB render of every probed camera\n"
"      --depth                  Also write the first-surface depth as <name>.zfirst.png (16-bit, millimetres, 0 = none)\n"
"      --device <N>             CUDA device [default: 0]\n";

unsigned char u8(float v) { return (unsigned char)std::lround(std::min(std::max(v, 0.f), 1.f) * 255.f); }

uint32_t crc32_of(const unsigned char* d, size_t n, uint32_t c = 0xffffffffu) {
  for (size_t i = 0; i < n; i++) { c ^= d[i]; for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u))); }
  return c;
}
uint32_t adler32_of(const unsigned char* d, size_t n) { uint32_t a = 1, b = 0; for (size_t i = 0; i < n; i++) { a = (a + d[i]) % 65521u; b = (b + a) % 65521u; } return (b << 16) | a; }
void put32(std::vector<unsigned char>& v, uint32_t x) { v.push_back(x >> 24); v.push_back(x >> 16); v.push_back(x >> 8); v.push_back(x); }
void chunk(std::vector<unsigned char>& out, const char* type, const std::vector<unsigned char>& data) {
  put32(out, (uint32_t)data.size());
  std::vector<unsigned char> td(type, type + 4); td.insert(td.end(), data.begin(), data.end());
  out.insert(out.end(), td.begin(), td.end());
  put32(out, crc32_of(td.data(), td.size()) ^ 0xffffffffu);
}
// 16-bit greyscale PNG with stored (uncompressed) deflate blocks: stb has no 16-bit writer.
void write_png16(const std::string& path, int W, int H, const unsigned short* px) {
  std::vector<unsigned char> raw; raw.reserve((size_t)H * (1 + 2 * W));
  for (int y = 0; y < H; y++) { raw.push_back(0); for (int x = 0; x < W; x++) { unsigned short v = px[(size_t)y * W + x]; raw.push_back(v >> 8); raw.push_back(v & 0xff); } }
  std::vector<unsigned char> z = {0x78, 0x01};
  for (size_t off = 0; off < raw.size();) {
    size_t n = std::min<size_t>(65535, raw.size() - off); bool last = off + n == raw.size();
    z.push_back(last ? 1 : 0); z.push_back(n & 0xff); z.push_back(n >> 8); z.push_back(~n & 0xff); z.push_back((~n >> 8) & 0xff);
    z.insert(z.end(), raw.begin() + off, raw.begin() + off + n); off += n;
  }
  put32(z, adler32_of(raw.data(), raw.size()));
  std::vector<unsigned char> out = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
  std::vector<unsigned char> ihdr; put32(ihdr, W); put32(ihdr, H); ihdr.push_back(16); ihdr.push_back(0); ihdr.push_back(0); ihdr.push_back(0); ihdr.push_back(0);
  chunk(out, "IHDR", ihdr); chunk(out, "IDAT", z); chunk(out, "IEND", {});
  FILE* f = fopen(path.c_str(), "wb"); if (!f) fail("failed to write '%s'", path.c_str()); fwrite(out.data(), 1, out.size(), f); fclose(f);
}
}  // namespace

int probe_main(int argc, char** argv) {
  std::string splat, cameras, output_dir = "probe", mesh_path, points_dir;
  int every = 1, device = 0, dilate = 2; float tau = 0.1f, delta = 0.03f, margin = 0.05f, points_radius = 0.f; bool images = false, depth = false;
  for (int i = 2; i < argc; i++) {
    std::string k = argv[i];
    auto val = [&]() -> std::string { if (i + 1 >= argc) fail("a value is required for '%s'", k.c_str()); return argv[++i]; };
    if (k == "--splat") splat = val(); else if (k == "--cameras") cameras = val(); else if (k == "--output-dir") output_dir = val();
    else if (k == "--every") every = std::max(1, atoi(val().c_str())); else if (k == "--tau") tau = strtof(val().c_str(), nullptr);
    else if (k == "--delta") delta = strtof(val().c_str(), nullptr); else if (k == "--mesh") mesh_path = val();
    else if (k == "--margin") margin = strtof(val().c_str(), nullptr); else if (k == "--dilate") dilate = atoi(val().c_str());
    else if (k == "--points") points_dir = val(); else if (k == "--points-radius") points_radius = strtof(val().c_str(), nullptr);
    else if (k == "--images") images = true; else if (k == "--depth") depth = true; else if (k == "--device") device = atoi(val().c_str());
    else if (k == "-h" || k == "--help") { fputs(HELP, stdout); return 0; }
    else fail("unexpected argument '%s' found", k.c_str());
  }
  if (splat.empty() || cameras.empty()) fail("the following required arguments were not provided: --splat <SPLAT> --cameras <CAMERAS>");
  CUDA_CHECK(cudaSetDevice(device));
  cudaStream_t stream = 0;
  SplatCloud cloud = read_ply(splat);
  log_info("Loaded %zu splats (SH degree %d) from %s", cloud.n, cloud.sh_degree, splat.c_str());
  std::ifstream cf(cameras); if (!cf) fail("failed to open cameras '%s'", cameras.c_str());
  nlohmann::json j; cf >> j;
  int W = j.at("width").get<int>(), H = j.at("height").get<int>();
  fs::create_directories(output_dir);
  Model model; model.upload(cloud);
  RenderCtx ctx; ctx.setup(W, H, model.cap, stream);
  MeshGPU mesh; bool have_mesh = false;
  if (!mesh_path.empty()) { TriMesh tm = read_mesh(mesh_path); mesh.upload(tm, stream); have_mesh = true; log_info("Loaded proxy mesh %s: %zu vertices, %zu triangles", mesh_path.c_str(), tm.nv(), tm.nf()); }
  if (!points_dir.empty()) {
    ColmapModel cm = read_colmap_text(points_dir);
    if (cm.points.size() < 100) fail("--points: only %zu points in %s", cm.points.size(), points_dir.c_str());
    std::vector<float> xyz(cm.points.size() * 3);
    for (size_t i = 0; i < cm.points.size(); i++) { xyz[i * 3] = cm.points[i].x; xyz[i * 3 + 1] = cm.points[i].y; xyz[i * 3 + 2] = cm.points[i].z; }
    float spacing = median_nn_distance(xyz.data(), cm.points.size());
    float radius = points_radius > 0.f ? points_radius : 4.f * spacing;
    mesh.upload_points(xyz.data(), cm.points.size(), radius, stream); have_mesh = true;
    log_info("Points proxy: %zu points as surfels of radius %.4f (median spacing %.4f)", cm.points.size(), radius, spacing);
  }
  DevBuf<float4> d_out; d_out.reserve((size_t)W * H);
  std::vector<unsigned char> img((size_t)W * H), rgb((size_t)W * H * 4);
  nlohmann::json views = nlohmann::json::array();
  double sum_deep = 0, sum_deep_frac = 0, sum_deep_in = 0, sum_deep_in_frac = 0, sum_behind = 0, sum_behind_frac = 0, sum_cov = 0; int nviews = 0;
  int idx = -1;
  for (auto& c : j.at("cameras")) {
    if (++idx % every) continue;
    Camera cam = camera_from_json(c, W, H);
    RenderParams p; p.cam = CameraGPU::from(cam, W, H); p.sh_degree = model.degree; p.bwd_info = false;
    p.bg[0] = p.bg[1] = p.bg[2] = 0.5f;
    render_forward(ctx, model, p, stream);
    const float* ref = nullptr;
    if (have_mesh) { mesh.rasterize(p.cam, W, H, dilate, stream); ref = mesh.depth; }
    probe_depth(ctx, tau, delta, ref, margin, d_out, stream);
    std::vector<float4> out = d_out.download((size_t)W * H, stream);
    std::vector<float> mz; if (have_mesh) mz = mesh.depth.download((size_t)W * H, stream);
    // Statistics over covered pixels (A > 0.5); fractions are relative to the pixel's own alpha.
    double deep_sum = 0, deep_hi = 0, behind_sum = 0, behind_hi = 0, deep_in = 0, deep_in_hi = 0; size_t covered = 0, interior = 0, mesh_hit = 0, both = 0, either = 0;
    float zmin = INFINITY, zmax = 0.f;
    // Interior pixels: every pixel within 4 px is covered, so a grazing first surface at a silhouette (where weight
    // legitimately arrives from the surface behind) is excluded.
    const int R = 4;
    auto is_interior = [&](size_t i) {
      int x = (int)(i % W), y = (int)(i / W);
      if (x < R || y < R || x >= W - R || y >= H - R) return false;
      for (int dy = -R; dy <= R; dy++) for (int dx = -R; dx <= R; dx++) if (out[(size_t)(y + dy) * W + (x + dx)].x <= 0.5f) return false;
      return true;
    };
    for (size_t i = 0; i < out.size(); i++) {
      float4 v = out[i];
      bool cov = v.x > 0.5f;
      bool mh = have_mesh && std::isfinite(mz[i]);
      if (mh) { mesh_hit++; zmin = std::min(zmin, mz[i]); zmax = std::max(zmax, mz[i]); }
      if (cov || mh) either++;
      if (cov && mh) both++;
      if (!cov) { img[i] = 0; continue; }
      covered++;
      float df = v.z / v.x, bf = v.w / v.x;
      deep_sum += df; if (df > 0.2f) deep_hi++;
      if (is_interior(i)) { interior++; deep_in += df; if (df > 0.2f) deep_in_hi++; }
      behind_sum += bf; if (bf > 0.2f) behind_hi++;
    }
    double n = std::max<double>((double)covered, 1.0), ni = std::max<double>((double)interior, 1.0);
    std::string name = c.at("name").get<std::string>(); auto dot = name.rfind('.'); if (dot != std::string::npos) name = name.substr(0, dot);
    nlohmann::json vj = {{"name", name}, {"covered_px", covered}, {"deep_mean", deep_sum / n}, {"deep_frac_gt_0.2", deep_hi / n}, {"interior_px", interior}, {"deep_interior_mean", deep_in / ni}, {"deep_interior_frac_gt_0.2", deep_in_hi / ni}};
    if (have_mesh) { vj["behind_mean"] = behind_sum / n; vj["behind_frac_gt_0.2"] = behind_hi / n; vj["mesh_iou"] = either ? (double)both / (double)either : 0.0; vj["mesh_px"] = mesh_hit; }
    (void)mesh_path;
    views.push_back(vj);
    sum_deep_in += deep_in / ni; sum_deep_in_frac += deep_in_hi / ni;
    sum_deep += deep_sum / n; sum_deep_frac += deep_hi / n; sum_behind += behind_sum / n; sum_behind_frac += behind_hi / n; sum_cov += either ? (double)both / (double)either : 0.0; nviews++;
    // Heat maps: deep fraction and (with a mesh) behind fraction and the mesh depth itself.
    for (size_t i = 0; i < out.size(); i++) img[i] = out[i].x > 0.5f ? u8(out[i].z / out[i].x) : 0;
    stbi_write_png((fs::path(output_dir) / (name + ".deep.png")).string().c_str(), W, H, 1, img.data(), W);
    if (have_mesh) {
      for (size_t i = 0; i < out.size(); i++) img[i] = out[i].x > 0.5f ? u8(out[i].w / out[i].x) : 0;
      stbi_write_png((fs::path(output_dir) / (name + ".behind.png")).string().c_str(), W, H, 1, img.data(), W);
      for (size_t i = 0; i < out.size(); i++) img[i] = std::isfinite(mz[i]) ? (unsigned char)(40 + 215 * (1.f - (mz[i] - zmin) / std::max(zmax - zmin, 1e-6f))) : 0;
      stbi_write_png((fs::path(output_dir) / (name + ".meshz.png")).string().c_str(), W, H, 1, img.data(), W);
    }
    if (depth) {
      std::vector<unsigned short> z16((size_t)W * H);
      for (size_t i = 0; i < out.size(); i++) { float zf = out[i].y; z16[i] = std::isfinite(zf) ? (unsigned short)std::min(std::max(zf * 1000.f, 0.f), 65535.f) : 0; }
      write_png16((fs::path(output_dir) / (name + ".zfirst.png")).string(), W, H, z16.data());
    }
    if (images) {
      std::vector<float4> o = ctx.out_rgba.download((size_t)W * H, stream);
      for (size_t i = 0; i < o.size(); i++) { rgb[i * 4] = u8(o[i].x); rgb[i * 4 + 1] = u8(o[i].y); rgb[i * 4 + 2] = u8(o[i].z); rgb[i * 4 + 3] = 255; }
      stbi_write_png((fs::path(output_dir) / (name + ".png")).string().c_str(), W, H, 4, rgb.data(), W * 4);
    }
    log_info("%s: deep %.4f (>0.2: %.2f%%), interior deep %.4f (>0.2: %.2f%%)%s", name.c_str(), deep_sum / n, 100.0 * deep_hi / n, deep_in / ni, 100.0 * deep_in_hi / ni,
             have_mesh ? format(", behind %.4f (>0.2: %.2f%%), mesh IoU %.3f", behind_sum / n, 100.0 * behind_hi / n, either ? (double)both / (double)either : 0.0).c_str() : "");
  }
  double nv = std::max(nviews, 1);
  nlohmann::json summary = {{"splat", splat}, {"tau", tau}, {"delta", delta}, {"views", views}, {"n_views", nviews},
                            {"deep_mean", sum_deep / nv}, {"deep_frac_gt_0.2", sum_deep_frac / nv}, {"deep_interior_mean", sum_deep_in / nv}, {"deep_interior_frac_gt_0.2", sum_deep_in_frac / nv}};
  if (have_mesh) { summary["mesh"] = mesh_path; summary["margin"] = margin; summary["behind_mean"] = sum_behind / nv; summary["behind_frac_gt_0.2"] = sum_behind_frac / nv; summary["mesh_iou"] = sum_cov / nv; }
  { std::ofstream f(fs::path(output_dir) / "probe.json"); f << summary.dump(1) << "\n"; }
  log_info("Summary over %d views: deep %.4f (>0.2: %.2f%% of covered pixels), interior deep %.4f (>0.2: %.2f%%)%s", nviews, sum_deep / nv, 100.0 * sum_deep_frac / nv, sum_deep_in / nv, 100.0 * sum_deep_in_frac / nv,
           have_mesh ? format(", behind %.4f (>0.2: %.2f%%), mesh IoU %.3f", sum_behind / nv, 100.0 * sum_behind_frac / nv, sum_cov / nv).c_str() : "");
  return 0;
}

}  // namespace b2c
