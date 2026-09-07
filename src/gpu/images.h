#pragma once
#include "gpu/loss.h"

namespace b2c {
// 2x2 box downsample of a resident view (packed RGBA averaged per channel, normals nearest, weights averaged).
// `dst` buffers must be preallocated for (W/2) x (H/2); normal_count is recomputed on the GPU.
struct DownsampleOut { uint32_t* rgba; uint32_t* normals; uint8_t* weights; };
void downsample_view(const ViewGPU& src, const DownsampleOut& dst, int dw, int dh, float* normal_count_accum, cudaStream_t stream);
}
