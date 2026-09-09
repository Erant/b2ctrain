#pragma once
#include "gpu/util.cuh"
#include <vector>

namespace b2c {

// In-trainer port of b2crunner's pipeline/align.py: dense flow from a training frame to the splat's render at the same
// camera, smoothed, zeroed outside the subject, capped, applied to the pristine frame by a Lanczos-4 backward warp.

struct AlignStats { float mean = 0.f, p90 = 0.f; };  // smoothed, uncapped magnitude over the subject, in pixels

constexpr int ALIGN_LEVELS = 4;

struct AlignScratch {
  DevBuf<float> ga[ALIGN_LEVELS], gb[ALIGN_LEVELS], gbx[ALIGN_LEVELS], gby[ALIGN_LEVELS];  // gray pyramids + render gradients
  DevBuf<float2> flow, flow_prev, tmp;
  DevBuf<float> hist;        // [bins + 2]: magnitude histogram, sum, count
  PinnedBuf<float> h_hist;
  DevBuf<float> smooth_k; int smooth_radius = 0;   // inter-iteration flow smoothing kernel
  int lw[ALIGN_LEVELS], lh[ALIGN_LEVELS], levels = 0;
  void setup(int W, int H);
};

constexpr int ALIGN_WARP_DOWN = 4;  // the render-side field is kept at 1/4 resolution (it is a sigma >= 6 px field)
inline int align_warp_dim(int n) { return (n + ALIGN_WARP_DOWN - 1) / ALIGN_WARP_DOWN; }

// `frame`: packed premultiplied RGBA8 (alpha = subject); `render`: float4 rgb composited on grey 0.5 (ctx.out_rgba).
// The flow is smoothed by `sigma` px and capped at `cap` px. With `dst`, writes the warped frame (same packing) there;
// with `warp_out`, writes the capped field box-averaged to align_warp_dim(W) x align_warp_dim(H) (full-res px units)
// for RenderParams::warp. Either may be null.
AlignStats align_view_gpu(AlignScratch& s, const uint32_t* frame, const float4* render, int W, int H, float sigma, float cap,
                          uint32_t* dst, cudaStream_t stream, float2* warp_out = nullptr);

}  // namespace b2c
