#pragma once
#include "gpu/render.h"

namespace b2c {

// Per-pixel depth-distribution probe of the composited splats (run after project + bin_and_sort in ctx):
//   out[pix] = (A, z_first, deep, behind)
//   A       = accumulated alpha
//   z_first = depth of the fragment at which the accumulated alpha first reaches `tau` (the first real surface)
//   deep    = weight arriving from more than `delta` behind z_first (false transparency: a later surface shows through)
//   behind  = weight arriving from more than `margin` behind the reference depth `ref_z` (+inf = none), if given
void probe_depth(RenderCtx& ctx, float tau, float delta, const float* ref_z, float margin, float4* out, cudaStream_t stream);

}  // namespace b2c
