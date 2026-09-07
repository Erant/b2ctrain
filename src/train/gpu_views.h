#pragma once
#include "gpu/loss.h"
#include "dataset/views.h"
#include "gpu/images.h"
#include <vector>
#include <algorithm>

namespace b2c {

// All views of a dataset resident on the GPU (packed RGBA8, normals, weights) plus their cameras.
struct GpuViews {
  DevBuf<uint32_t> rgba, normals;
  DevBuf<uint8_t> weights;
  std::vector<ViewGPU> views;
  std::vector<Camera> cams;
  int max_w = 0, max_h = 0;
  // Downsampled pyramid levels (1 = half, 2 = quarter) for the progressive resolution schedule.
  DevBuf<uint32_t> lvl_rgba[3], lvl_normals[3];
  DevBuf<uint8_t> lvl_weights[3];
  std::vector<ViewGPU> lvl_views[3];
  int n_levels = 1;
  void build_pyramid(int levels, cudaStream_t stream) {
    n_levels = levels;
    lvl_views[0] = views;
    DevBuf<float> counts; counts.reserve(views.size());
    for (int l = 1; l < levels; l++) {
      const std::vector<ViewGPU>& src = lvl_views[l - 1];
      size_t n_px = 0, n_npx = 0, n_wpx = 0;
      std::vector<int> dws, dhs;
      for (auto& v : src) { int dw = std::max(1, v.W / 2), dh = std::max(1, v.H / 2); dws.push_back(dw); dhs.push_back(dh); n_px += (size_t)dw * dh; if (v.normals) n_npx += (size_t)dw * dh; if (v.weights) n_wpx += (size_t)dw * dh; }
      lvl_rgba[l].reserve(n_px); if (n_npx) lvl_normals[l].reserve(n_npx); if (n_wpx) lvl_weights[l].reserve(n_wpx);
      counts.zero(stream);
      size_t o = 0, on = 0, ow = 0;
      std::vector<ViewGPU> out;
      for (size_t i = 0; i < src.size(); i++) {
        ViewGPU g = src[i]; g.W = dws[i]; g.H = dhs[i];
        DownsampleOut d; d.rgba = lvl_rgba[l].ptr + o; d.normals = src[i].normals ? lvl_normals[l].ptr + on : nullptr; d.weights = src[i].weights ? lvl_weights[l].ptr + ow : nullptr;
        downsample_view(src[i], d, g.W, g.H, counts.ptr + i, stream);
        g.rgba = d.rgba; g.normals = d.normals; g.weights = d.weights;
        o += (size_t)g.W * g.H; if (d.normals) on += (size_t)g.W * g.H; if (d.weights) ow += (size_t)g.W * g.H;
        out.push_back(g);
      }
      auto cnt = counts.download(src.size(), stream);
      for (size_t i = 0; i < out.size(); i++) out[i].normal_count = cnt[i];
      lvl_views[l] = out;
    }
  }
  void upload(const std::vector<View>& src) {
    size_t n_px = 0, n_npx = 0, n_wpx = 0;
    for (auto& v : src) { n_px += v.rgba.size(); n_npx += v.normals.size(); n_wpx += v.weights.size(); max_w = std::max(max_w, v.w); max_h = std::max(max_h, v.h); }
    rgba.reserve(n_px); if (n_npx) normals.reserve(n_npx); if (n_wpx) weights.reserve(n_wpx);
    size_t o = 0, on = 0, ow = 0;
    for (auto& v : src) {
      ViewGPU g; g.W = v.w; g.H = v.h; g.has_alpha = v.has_alpha; g.masked = v.mode == AlphaMode::Masked; g.alpha_coverage = v.alpha_coverage; g.normal_count = (float)v.normal_mask_count;
      CUDA_CHECK(cudaMemcpy(rgba.ptr + o, v.rgba.data(), v.rgba.size() * 4, cudaMemcpyHostToDevice)); g.rgba = rgba.ptr + o; o += v.rgba.size();
      if (!v.normals.empty()) { CUDA_CHECK(cudaMemcpy(normals.ptr + on, v.normals.data(), v.normals.size() * 4, cudaMemcpyHostToDevice)); g.normals = normals.ptr + on; on += v.normals.size(); }
      if (!v.weights.empty()) { CUDA_CHECK(cudaMemcpy(weights.ptr + ow, v.weights.data(), v.weights.size(), cudaMemcpyHostToDevice)); g.weights = weights.ptr + ow; ow += v.weights.size(); }
      views.push_back(g); cams.push_back(v.cam);
    }
  }
};


}  // namespace b2c
