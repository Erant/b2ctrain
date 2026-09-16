// `b2ctrain mesh-render`: render the UV-textured mesh of a mesh-unwrap atlas at cameras (RGBA on grey, and the
// loop's per-pixel depth / facing / texel-density / best-so-far maps), or generate the orbit and head camera sets.
// Reference: out/mesh/tools/uv_render.py.
#include "mesh/common.h"
#include "mesh/raster.h"
#include "mesh/geom.cuh"
#include "util/log.h"
#include "json.hpp"
#include <filesystem>
#include <fstream>
#include <cmath>

namespace b2c {
namespace fs = std::filesystem;

namespace {

// Per output pixel: box-filter the ss x ss sub-samples of the raster (tri id + barycentrics) through the texture.
__global__ void shade_kernel(int W, int H, int ss, CameraGPU cam, const int* __restrict__ tri, const float2* __restrict__ bary, const float* __restrict__ zs,
                             const uint3* __restrict__ fuv, const float2* __restrict__ uv, const float3* __restrict__ fn, const uint8_t* __restrict__ tex, int R, const float* __restrict__ aux, float bg,
                             uint8_t* __restrict__ rgba, float* __restrict__ depth, float* __restrict__ cosv, float* __restrict__ dens, float* __restrict__ auxo) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y; if (x >= W || y >= H) return;
  int Ws = W * ss; float col[3] = {0, 0, 0}; float hit = 0.f, cs = 0.f, ds = 0.f; float z0 = 0.f, a0 = 0.f;
  for (int sy = 0; sy < ss; sy++) for (int sx = 0; sx < ss; sx++) {
    int px = x * ss + sx, py = y * ss + sy; size_t i = (size_t)py * Ws + px; int t = tri[i];
    if (t < 0) { for (int k = 0; k < 3; k++) col[k] += bg; continue; }
    hit += 1.f; float2 b = bary[i]; float b0 = 1.f - b.x - b.y; uint3 tu = fuv[t];
    float u = b0 * uv[tu.x].x + b.x * uv[tu.y].x + b.y * uv[tu.z].x, v = b0 * uv[tu.x].y + b.x * uv[tu.y].y + b.y * uv[tu.z].y;
    // bilinear with replicated borders at (u R - 0.5, (1 - v) R - 0.5)
    float tx = fminf(fmaxf(u * R - 0.5f, 0.f), R - 1.f), ty = fminf(fmaxf((1.f - v) * R - 0.5f, 0.f), R - 1.f);
    int x0 = (int)tx, y0 = (int)ty, x1 = min(x0 + 1, R - 1), y1 = min(y0 + 1, R - 1); float fx = tx - x0, fy = ty - y0;
    for (int k = 0; k < 3; k++) {
      float c00 = tex[((size_t)y0 * R + x0) * 3 + k], c10 = tex[((size_t)y0 * R + x1) * 3 + k], c01 = tex[((size_t)y1 * R + x0) * 3 + k], c11 = tex[((size_t)y1 * R + x1) * 3 + k];
      col[k] += ((1 - fx) * (1 - fy) * c00 + fx * (1 - fy) * c10 + (1 - fx) * fy * c01 + fx * fy * c11) / 255.f;
    }
    float z = zs[i];
    // the sub-pixel ray direction (unit) for the facing term
    float dx = ((px + 0.5f) / ss - cam.cx) / cam.fx, dy = ((py + 0.5f) / ss - cam.cy) / cam.fy;
    float3 d = make_float3(cam.R[0] * dx + cam.R[3] * dy + cam.R[6], cam.R[1] * dx + cam.R[4] * dy + cam.R[7], cam.R[2] * dx + cam.R[5] * dy + cam.R[8]);
    d = d * (1.f / fmaxf(len3(d), 1e-9f));
    cs += fabsf(dot3(fn[t], d)); ds += (cam.fx / fmaxf(z, 1e-3f)) * (cam.fx / fmaxf(z, 1e-3f)) * 1e-6f;
    if (sx == 0 && sy == 0) { z0 = z; if (aux) a0 = aux[(size_t)min(max((int)((1.f - v) * R), 0), R - 1) * R + min(max((int)(u * R), 0), R - 1)]; }
  }
  float n = (float)(ss * ss); size_t o = (size_t)y * W + x;
  for (int k = 0; k < 3; k++) rgba[o * 4 + k] = (uint8_t)fminf(fmaxf(col[k] / n * 255.f + 0.5f, 0.f), 255.f);
  rgba[o * 4 + 3] = (uint8_t)fminf(fmaxf(hit / n * 255.f + 0.5f, 0.f), 255.f);
  if (depth) { depth[o] = z0; cosv[o] = cs / n; dens[o] = ds / n; }
  if (auxo) auxo[o] = a0;
}

void look_at(const float* pos, const float* target, float rot[3][3]) {
  float f[3] = {target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]}; float fl = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]); for (auto& v : f) v /= fl;
  float up[3] = {0, 1, 0}; float r[3] = {f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]}; float rl = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]); for (auto& v : r) v /= rl;
  float u[3] = {r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]};
  for (int i = 0; i < 3; i++) { rot[i][0] = r[i]; rot[i][1] = u[i]; rot[i][2] = -f[i]; }  // OpenGL c2w: columns = right, up, back
}

std::vector<float> parse_list(const std::string& s) { std::vector<float> out; size_t p = 0; while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); if (q > p) out.push_back(strtof(s.substr(p, q - p).c_str(), nullptr)); p = q + 1; } return out; }

}  // namespace

int mesh_render_main(int argc, char** argv) {
  std::string atlas, cams, out, texture, aux_path, azims = "0,45,90,135,180,225,270,315", elevs = "0"; float bg = 0.5f; int ss = 2, device = 0, width = 768, height = 1536;
  bool depth = false, make_cams = false, head = false; float centre[3] = {0, 0, 0}; float radius = 0.f, fov = 0.f;
  ArgParser ap("Render a mesh-unwrap atlas at cameras, or generate orbit / head cameras for it\n\nUsage: b2ctrain mesh-render --atlas DIR --cameras CAMS.json --output DIR [--texture PNG] [--depth] [--aux MAP.f32]\n       b2ctrain mesh-render --make-cams --atlas DIR --output CAMS.json [--width W --height H --azims A,B,.. --elevs E,.. --head --centre x,y,z]");
  ap.s("atlas", "DIR", "mesh-unwrap output directory (mesh_uv.obj, texture.png)", atlas).s("cameras", "CAMS.json", "Camera list", cams).s("output", "PATH", "Output directory (renders) or camera file (--make-cams)", out)
    .s("texture", "PNG", "Texture to render instead of the atlas's texture.png", texture).f("bg", "G", "Background grey [default: 0.5]", bg).i("ss", "N", "Supersampling [default: 2]", ss)
    .b("depth", "Also write <stem>.depth.f32 (camera z, 0 = miss), <stem>.cos.f32 (|n . view|), <stem>.dens.f32 ((fx / z)^2 1e-6), W x H float32", depth)
    .s("aux", "MAP.f32", "An R x R float32 texture-space map sampled per pixel (nearest) -> <stem>.aux.f32 (the loop's best-so-far weights)", aux_path)
    .b("make-cams", "Generate cameras about the mesh instead of rendering", make_cams)
    .i("width", "W", "[default: 768]", width).i("height", "H", "[default: 1536]", height).s("azims", "LIST", "Azimuths in degrees [default: 0,45,..,315]", azims).s("elevs", "LIST", "Elevations in degrees [default: 0]", elevs)
    .b("head", "Head cameras: 0.9 m from --centre (or the top of the mesh minus 0.13 m), a 0.36 m field of view", head)
    .f3("centre", "x,y,z", "Look-at centre for --head (the SAM head centroid; the bbox centre drifts with a ponytail)", centre)
    .f("radius", "M", "Camera distance (default: 2.2 body / 0.9 head)", radius).f("fov", "M", "Vertical field of view in metres at the centre (default: 1.15 x the height / 0.36 head)", fov)
    .i("device", "N", "CUDA device [default: 0]", device);
  if (!ap.parse(argc, argv)) return 0;
  if (atlas.empty() || out.empty()) fail("--atlas and --output are required");
  double t0 = now_seconds();
  TriMesh m = read_mesh(atlas + "/mesh_uv.obj");
  if (make_cams) {
    float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (size_t i = 0; i < m.nv(); i++) for (int k = 0; k < 3; k++) { lo[k] = std::min(lo[k], m.v[i * 3 + k]); hi[k] = std::max(hi[k], m.v[i * 3 + k]); }
    float c[3] = {(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2};
    if (head) { if (ap.given("centre")) { c[0] = centre[0]; c[1] = centre[1]; c[2] = centre[2]; } else c[1] = hi[1] - 0.13f; if (radius <= 0) radius = 0.9f; if (fov <= 0) fov = 0.36f; }
    else { if (radius <= 0) radius = 2.2f; if (fov <= 0) fov = (hi[1] - lo[1]) * 1.15f; }
    float fx = radius * height / fov;
    nlohmann::json j; j["width"] = width; j["height"] = height; j["cameras"] = nlohmann::json::array();
    for (float e : parse_list(elevs)) for (float az : parse_list(azims)) {
      float ar = az * (float)M_PI / 180.f, er = e * (float)M_PI / 180.f;
      float pos[3] = {c[0] + radius * std::sin(ar) * std::cos(er), c[1] + radius * std::sin(er), c[2] + radius * std::cos(ar) * std::cos(er)};
      float rot[3][3]; look_at(pos, c, rot);
      nlohmann::json cj; cj["name"] = format("%s_e%+03d_a%03d.png", head ? "head" : "body", (int)e, (int)az); cj["fx"] = fx; cj["fy"] = fx; cj["cx"] = width / 2.0; cj["cy"] = height / 2.0;
      cj["position"] = {pos[0], pos[1], pos[2]}; cj["rotation"] = {{rot[0][0], rot[0][1], rot[0][2]}, {rot[1][0], rot[1][1], rot[1][2]}, {rot[2][0], rot[2][1], rot[2][2]}};
      j["cameras"].push_back(cj);
    }
    std::ofstream f(out); f << j.dump(1) << "\n";
    log_info("%zu cameras -> %s (fx %.0f)", j["cameras"].size(), out.c_str(), fx);
    return 0;
  }
  if (cams.empty()) fail("--cameras is required");
  if (!m.has_uv()) fail("'%s/mesh_uv.obj' has no UVs", atlas.c_str());
  CUDA_CHECK(cudaSetDevice(device)); cudaStream_t stream = 0;
  Image8 tex = load_image8(texture.empty() ? atlas + "/texture.png" : texture, 3);
  if (tex.W != tex.H) fail("the texture must be square (%d x %d)", tex.W, tex.H);
  int R = tex.W;
  CamSet cs = load_cams(cams); fs::create_directories(out);
  MeshRaster rast; rast.upload(m, stream);
  DevBuf<float3> fn; fn.reserve(m.nf()); face_normals_gpu(rast.v, rast.f, (int)m.nf(), fn, stream);
  DevBuf<float2> d_uv; { std::vector<float2> h(m.uv.size() / 2); for (size_t i = 0; i < h.size(); i++) h[i] = make_float2(m.uv[i * 2], m.uv[i * 2 + 1]); d_uv.upload(h, stream); }
  DevBuf<uint3> d_fuv; { std::vector<uint3> h(m.nf()); for (size_t i = 0; i < m.nf(); i++) h[i] = make_uint3(m.fuv[i * 3], m.fuv[i * 3 + 1], m.fuv[i * 3 + 2]); d_fuv.upload(h, stream); }
  DevBuf<uint8_t> d_tex; d_tex.upload(tex.px, stream);
  DevBuf<float> d_aux; if (!aux_path.empty()) { std::vector<float> a = read_f32(aux_path, (size_t)R * R); d_aux.upload(a, stream); }
  size_t npx = (size_t)cs.W * cs.H;
  DevBuf<uint8_t> d_rgba; DevBuf<float> d_depth, d_cos, d_dens, d_auxo; d_rgba.reserve(npx * 4);
  if (depth) { d_depth.reserve(npx); d_cos.reserve(npx); d_dens.reserve(npx); }
  if (d_aux.ptr) d_auxo.reserve(npx);
  for (size_t i = 0; i < cs.size(); i++) {
    CameraGPU cam = CameraGPU::from(cs.cams[i], cs.W, cs.H);
    rast.raster(cam, ss, true, stream);
    dim3 blk(16, 16), grd((cs.W + 15) / 16, (cs.H + 15) / 16);
    shade_kernel<<<grd, blk, 0, stream>>>(cs.W, cs.H, ss, cam, rast.tri, rast.bary, rast.z, d_fuv, d_uv, fn, d_tex, R, d_aux.ptr, bg, d_rgba, depth ? d_depth.ptr : nullptr, d_cos.ptr, d_dens.ptr, d_auxo.ptr);
    CUDA_KERNEL_CHECK();
    std::vector<uint8_t> rgba = d_rgba.download(npx * 4, stream);
    std::string stem = file_stem(cs.names[i]);
    write_png8(out + "/" + stem + ".png", cs.W, cs.H, 4, rgba.data());
    if (depth) {
      std::vector<float> z = d_depth.download(npx, stream), c = d_cos.download(npx, stream), d = d_dens.download(npx, stream);
      write_f32(out + "/" + stem + ".depth.f32", z.data(), npx); write_f32(out + "/" + stem + ".cos.f32", c.data(), npx); write_f32(out + "/" + stem + ".dens.f32", d.data(), npx);
    }
    if (d_auxo.ptr) { std::vector<float> a = d_auxo.download(npx, stream); write_f32(out + "/" + stem + ".aux.f32", a.data(), npx); }
  }
  log_info("rendered %zu views -> %s (%.1fs)", cs.size(), out.c_str(), now_seconds() - t0);
  return 0;
}

}  // namespace b2c
