#pragma once
#include "gpu/render.h"
#include "gpu/loss.h"
#include "dataset/camera.h"
#include "cli.h"
#include <vector>

namespace b2c {
// Per-splat multi-view evidence: [n][7] = w_in, w_all, err, views, dir xyz. Kept in a module-level buffer.
struct BodyRig;
struct CageRig;
// `rig` + `rig_views` (per view, the rig's view index or -1): render each view from its posed means (gpu/deform.h).
void compute_evidence(RenderCtx& ctx, const Model& m, const std::vector<ViewGPU>& views, const std::vector<Camera>& cams, const Config& cfg, cudaStream_t stream,
                      BodyRig* rig = nullptr, const std::vector<int>* rig_views = nullptr,
                      CageRig* cage = nullptr, const std::vector<int>* cage_views = nullptr);   // cage: per view its frame or -1
std::vector<float> download_evidence(const Model& m, cudaStream_t stream);
// The device accumulator of the last compute_evidence: [n][*stride], w_all at offset 1 (GaussianSpa's importance).
const float* evidence_device(int* stride);
// Per-splat class vote from the views' `labels` planes (see LABEL_BINS in evidence.cu). Accumulated by
// compute_evidence when `cfg.export_labels` is set; [n][2] = (winning class id, its share of the splat's
// voting weight), both 0 for a splat no labelled view rendered.
std::vector<float> download_labels(const Model& m, cudaStream_t stream);
}
