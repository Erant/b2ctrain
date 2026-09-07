#pragma once
#include "gpu/render.h"
#include "gpu/loss.h"
#include "dataset/camera.h"
#include "cli.h"
#include <vector>

namespace b2c {
// Per-splat multi-view evidence: [n][7] = w_in, w_all, err, views, dir xyz. Kept in a module-level buffer.
void compute_evidence(RenderCtx& ctx, const Model& m, const std::vector<ViewGPU>& views, const std::vector<Camera>& cams, const Config& cfg, cudaStream_t stream);
std::vector<float> download_evidence(const Model& m, cudaStream_t stream);
}
