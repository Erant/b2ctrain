#pragma once
#include "train/gpu_views.h"
#include <string>

namespace b2c {
// lod-distill (lod/merge.h): replace every view's target with the teacher splat rendered at that view, canonical, with
// its alpha (premultiplied RGBA8, a transparent view: the loss composites the step's background behind it). The
// teacher is fixed, so rendering each view once up front is the same target as rendering it at every step it is drawn.
void render_teacher_targets(const std::string& teacher_ply, GpuViews& gv, cudaStream_t stream);
}  // namespace b2c
