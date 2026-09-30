#pragma once
// Pose containment loss (--pose-contain-weight, cage training): the splat, posed by the cage's frames, must not render
// outside the silhouette its own surface draws in that pose.
//
// At the start of training the warm-start splat is posed and rendered once per containment camera (a cameras.json that
// b2crig writes, each camera named after the cage frame it poses), with the splats of an exclusion mask hidden (b2crig:
// those deep inside the body). That alpha, dilated by `dilate` px, is the allowed region. Those renders become
// alpha-only virtual views: a fraction of the training steps draw one instead of a real view and push the alpha
// outside the allowed region to 0 (weights 255 there, 0 inside; colour, SSIM, normals off). The target comes from the
// splat's own surface, so clothing, hair and anything else outside the body define it and are never trimmed; what gets
// removed is what the body interior throws outside the figure when the pose shears it (the shoulder-top spikes of an
// arms-up capture with the arms lowered).
//
// Which poses, which cameras and which splats count as interior are decisions about the rig: b2crig/rig/contain.py.
#include <vector>

#include "gpu/loss.h"
#include "gpu/util.cuh"
#include "dataset/camera.h"

namespace b2c {

struct PoseContain {
  bool on = false;
  float fraction = 0.5f;               // share of training steps that draw a containment view
  DevBuf<uint32_t> rgba;               // [views][px] alpha = allowed * 255, rgb 0 (premultiplied)
  DevBuf<uint8_t> weights;             // [views][px] 255 outside the allowed region, 0 inside
  DevBuf<uint8_t> deep;                // [n] at build time: hidden while the targets render (--pose-contain-exclude)
  std::vector<ViewGPU> views;
  std::vector<Camera> cams;
  std::vector<int> frame;              // cage frame of each view
  int n_deep = 0;
};

// Posed opacity logits of the deep splats -> -30 (a render-only override of the cage's pos_view).
void pose_contain_hide(float4* pos_view, const uint8_t* deep, int n, cudaStream_t stream);
// Allowed region = alpha > thr dilated by a disc of radius r px; writes one view's rgba and weights. Returns nothing.
void pose_contain_target(const float4* out_rgba, int W, int H, int r, float thr, uint32_t* rgba, uint8_t* weights,
                         cudaStream_t stream);

}  // namespace b2c
