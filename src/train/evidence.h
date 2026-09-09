#pragma once
#include "gpu/render.h"
#include "gpu/loss.h"
#include "dataset/camera.h"
#include "cli.h"
#include <vector>

namespace b2c {
// Per-splat multi-view evidence: [n][7] = w_in, w_all, err, views, dir xyz. Kept in a module-level buffer.
struct BodyRig;
// `rig` + `rig_views` (per view, the rig's view index or -1): render each view from its posed means (gpu/deform.h).
void compute_evidence(RenderCtx& ctx, const Model& m, const std::vector<ViewGPU>& views, const std::vector<Camera>& cams, const Config& cfg, cudaStream_t stream,
                      BodyRig* rig = nullptr, const std::vector<int>* rig_views = nullptr);
std::vector<float> download_evidence(const Model& m, cudaStream_t stream);
}
