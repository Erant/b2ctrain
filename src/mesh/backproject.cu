// `b2ctrain mesh-backproject`: bake refined view images back into the atlas texture where the view sees the texel
// (depth test against the mesh-render depth, inside the eroded mask), weighted by facing^power x texel density and,
// with --best, only where this view beats the best weight so far (updated in place). Reference: uv_backproject.py.
#include "mesh/common.h"
#include "mesh/raster.h"
#include "mesh/geom.cuh"
#include "util/log.h"
#include <cmath>

namespace b2c {

namespace {

__global__ void erode_kernel(int W, int H, const uint8_t* __restrict__ rgba, const uint8_t* __restrict__ mask, int r, uint8_t* __restrict__ out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y; if (x >= W || y >= H) return;
  uint8_t v = 1;
  for (int dy = -r; v && dy <= r; dy++) for (int dx = -r; v && dx <= r; dx++) {
    int xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
    size_t i = (size_t)yy * W + xx; v = rgba[i * 4 + 3] > 127 && (!mask || mask[i] > 127);
  }
  out[(size_t)y * W + x] = v;
}

__global__ void gather_kernel(int n, const float3* __restrict__ P, const float3* __restrict__ N, CameraGPU cam, const float* __restrict__ depth, const uint8_t* __restrict__ rgba, const uint8_t* __restrict__ ok,
                              float tol, float power, float min_cos, float3* __restrict__ acc, float* __restrict__ wsum) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float3 p = P[i]; float u, v; float z = project_point(cam, p.x, p.y, p.z, u, v);
  if (!(z > 1e-3f) || !(u >= 0.f && u < cam.W - 1 && v >= 0.f && v < cam.H - 1)) return;
  int ui = min(max((int)floorf(u), 0), cam.W - 1), vi = min(max((int)floorf(v), 0), cam.H - 1); size_t px = (size_t)vi * cam.W + ui;
  float dz = depth[px];
  if (fabsf(dz - z) >= tol || !ok[px]) return;
  float3 view = make_float3(cam.pos[0], cam.pos[1], cam.pos[2]) - p; view = view * (1.f / fmaxf(len3(view), 1e-9f));
  float cosn = dot3(N[i], view);
  if (!(cosn > min_cos)) return;
  float dens = (cam.fx / fmaxf(z, 1e-3f)) * (cam.fx / fmaxf(z, 1e-3f)) * 1e-6f;
  float w = powf(fmaxf(cosn, 0.f), power) * dens;
  float3 col = make_float3(sample_bilinear_u8(rgba, cam.W, cam.H, 4, 0, u - 0.5f, v - 0.5f), sample_bilinear_u8(rgba, cam.W, cam.H, 4, 1, u - 0.5f, v - 0.5f), sample_bilinear_u8(rgba, cam.W, cam.H, 4, 2, u - 0.5f, v - 0.5f)) * (1.f / 255.f);
  acc[i] = acc[i] + col * w; wsum[i] += w;
}

}  // namespace

int mesh_backproject_main(int argc, char** argv) {
  std::string atlas, cams, images, renders, out, texture, mask_dir, best_path; float power = 4.f, tol = 0.008f, blend = 1.f, min_cos = 0.2f; int erode = 4, device = 0;
  ArgParser ap("Back-project refined views into the atlas texture\n\nUsage: b2ctrain mesh-backproject --atlas DIR --cameras CAMS.json --images DIR --renders DIR --output OUT.png [--texture PNG] [--mask-dir DIR] [--best MAP.f32]");
  ap.s("atlas", "DIR", "mesh-unwrap output directory (position.f32, normal.f32, mask_dilated.png)", atlas).s("cameras", "CAMS.json", "Camera list", cams)
    .s("images", "DIR", "Refined RGBA views named as the cameras", images).s("renders", "DIR", "mesh-render --depth output of the same cameras (<stem>.depth.f32)", renders)
    .s("output", "OUT.png", "The updated texture", out).s("texture", "PNG", "The texture to update (default: the atlas's texture.png)", texture)
    .s("mask-dir", "DIR", "<stem>.mask.png per view: bake only where > 127 (the repainted pixels)", mask_dir)
    .s("best", "MAP.f32", "R x R float32 of the best weight so far: only texels this view sees better are written, and the map is updated in place", best_path)
    .f("power", "P", "Facing exponent [default: 4]", power).f("tol", "M", "Depth agreement [default: 0.008]", tol).i("erode", "PX", "Erosion of the view's alpha (and mask) [default: 4]", erode)
    .f("blend", "B", "new = B * baked + (1 - B) * old where seen [default: 1]", blend).f("min-cos", "C", "Minimum facing [default: 0.2]", min_cos)
    .i("device", "N", "CUDA device [default: 0]", device);
  if (!ap.parse(argc, argv)) return 0;
  if (atlas.empty() || cams.empty() || images.empty() || renders.empty() || out.empty()) fail("--atlas, --cameras, --images, --renders and --output are required");
  CUDA_CHECK(cudaSetDevice(device)); cudaStream_t stream = 0; double t0 = now_seconds();
  Image8 tex = load_image8(texture.empty() ? atlas + "/texture.png" : texture, 3); int R = tex.W; size_t npx = (size_t)R * R;
  Image8 covm = load_image8(atlas + "/mask_dilated.png", 1); if (covm.W != R || covm.H != R) fail("mask_dilated.png does not match the texture size");
  std::vector<float> pos = read_f32(atlas + "/position.f32", npx * 3), nrm = read_f32(atlas + "/normal.f32", npx * 3);
  std::vector<int> cov; for (size_t i = 0; i < npx; i++) if (covm.px[i] > 0) cov.push_back((int)i);
  int n = (int)cov.size();
  std::vector<float3> hp(n), hn(n); for (int k = 0; k < n; k++) { size_t i = cov[k]; hp[k] = make_float3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]); hn[k] = make_float3(nrm[i * 3], nrm[i * 3 + 1], nrm[i * 3 + 2]); }
  DevBuf<float3> P, N, acc; DevBuf<float> wsum; P.upload(hp, stream); N.upload(hn, stream); acc.reserve(n); wsum.reserve(n); acc.zero(stream); wsum.zero(stream);
  CamSet cs = load_cams(cams); int used = 0;
  DevBuf<uint8_t> d_rgba, d_mask, d_ok; DevBuf<float> d_depth;
  for (size_t i = 0; i < cs.size(); i++) {
    std::string stem = file_stem(cs.names[i]); Image8 im;
    if (!try_load_image8(images + "/" + cs.names[i], 4, im)) continue;
    if (im.W != cs.W || im.H != cs.H) fail("'%s': %d x %d does not match the camera list", (images + "/" + cs.names[i]).c_str(), im.W, im.H);
    size_t vpx = (size_t)cs.W * cs.H;
    std::vector<float> depth = read_f32(renders + "/" + stem + ".depth.f32", vpx);
    d_rgba.upload(im.px, stream); d_depth.upload(depth, stream); d_ok.reserve(vpx);
    bool have_mask = false;
    if (!mask_dir.empty()) { Image8 mk = load_image8(mask_dir + "/" + stem + ".mask.png", 1); if (mk.W != cs.W || mk.H != cs.H) fail("mask '%s' size mismatch", stem.c_str()); d_mask.upload(mk.px, stream); have_mask = true; }
    dim3 blk(16, 16), grd((cs.W + 15) / 16, (cs.H + 15) / 16);
    erode_kernel<<<grd, blk, 0, stream>>>(cs.W, cs.H, d_rgba, have_mask ? d_mask.ptr : nullptr, erode, d_ok); CUDA_KERNEL_CHECK();
    CameraGPU cam = CameraGPU::from(cs.cams[i], cs.W, cs.H);
    gather_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, P, N, cam, d_depth, d_rgba, d_ok, tol, power, min_cos, acc, wsum); CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaStreamSynchronize(stream)); used++;
  }
  std::vector<float3> ha = acc.download(n, stream); std::vector<float> hw = wsum.download(n, stream);
  std::vector<float> best; if (!best_path.empty()) best = read_f32(best_path, npx);
  size_t seen = 0;
  for (int k = 0; k < n; k++) {
    size_t i = cov[k]; float w = hw[k];
    bool s = w > 1e-6f; if (!best.empty()) s = s && w > best[i] + 1e-6f;
    if (!s) continue;
    seen++; if (!best.empty()) best[i] = w;
    float c[3] = {ha[k].x / w, ha[k].y / w, ha[k].z / w};
    for (int ch = 0; ch < 3; ch++) { float v = blend * c[ch] + (1.f - blend) * tex.px[i * 3 + ch] / 255.f; tex.px[i * 3 + ch] = (uint8_t)std::min(std::max(v * 255.f + 0.5f, 0.f), 255.f); }
  }
  write_png8(out, R, R, 3, tex.px.data());
  if (!best.empty()) write_f32(best_path, best.data(), npx);
  log_info("%d views, %.1f%% of covered texels updated -> %s (%.1fs)", used, 100.0 * seen / std::max(n, 1), out.c_str(), now_seconds() - t0);
  return 0;
}

}  // namespace b2c
