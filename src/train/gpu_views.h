#pragma once
#include "gpu/loss.h"
#include "dataset/views.h"
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
