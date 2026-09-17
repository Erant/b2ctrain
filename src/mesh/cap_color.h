#pragma once
#include <algorithm>
#include <vector>

namespace b2c {

// The projected splats supply premultiplied RGB, but mesh blending needs straight
// RGB independently of coverage. Fill small raster holes using only observed
// neighbours, then extend COLOR (not coverage) into the feather's support.
// Planar RGB: three W*H lanes. A valid black sample is not a hole.
inline void extend_cap_colors(std::vector<float>& rgb, std::vector<float>& coverage, int W, int H) {
  const size_t npx = (size_t)W * H;
  for (size_t i = 0; i < npx; ++i) if (coverage[i] > 0.f)
    for (int ch = 0; ch < 3; ++ch) rgb[ch * npx + i] /= coverage[i];
  auto filled = rgb;
  auto cov = coverage;
  for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
    size_t i = (size_t)y * W + x;
    if (coverage[i] > 0.f) continue;
    float vals[3][8], confidence = 0.f;
    int count = 0;
    for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
      int xx = x + dx, yy = y + dy;
      if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;
      size_t j = (size_t)yy * W + xx;
      if (coverage[j] <= 0.f) continue;
      for (int ch = 0; ch < 3; ++ch) vals[ch][count] = rgb[ch * npx + j];
      confidence = std::max(confidence, coverage[j]);
      ++count;
    }
    if (!count) continue;
    for (int ch = 0; ch < 3; ++ch) {
      std::sort(vals[ch], vals[ch] + count);
      filled[ch * npx + i] = .5f * (vals[ch][(count - 1) / 2] + vals[ch][count / 2]);
    }
    cov[i] = confidence * .999f;
  }
  rgb.swap(filled);
  coverage.swap(cov);
  // Multi-source breadth-first extension. The distance metric only selects an
  // edge colour; the separate soft coverage controls how far it is used.
  std::vector<int> queue;
  std::vector<unsigned char> known(npx, 0);
  queue.reserve(npx);
  for (size_t i = 0; i < npx; ++i) if (coverage[i] > 0.f) { known[i] = 1; queue.push_back((int)i); }
  for (size_t q = 0; q < queue.size(); ++q) {
    int i = queue[q], x = i % W, y = i / W;
    const int next[4] = {x > 0 ? i - 1 : -1, x + 1 < W ? i + 1 : -1,
                         y > 0 ? i - W : -1, y + 1 < H ? i + W : -1};
    for (int j : next) if (j >= 0 && !known[j]) {
      for (int ch = 0; ch < 3; ++ch) rgb[ch * npx + j] = rgb[ch * npx + i];
      known[j] = 1;
      queue.push_back(j);
    }
  }
}

}  // namespace b2c
