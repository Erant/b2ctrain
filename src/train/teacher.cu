#include "train/teacher.h"
#include "gpu/render.h"
#include "model.h"
#include "ply.h"
#include "util/log.h"

namespace b2c {
namespace {
// Rendered on a black background the colour is already premultiplied; quantised like the dataset loader's frames.
__global__ void pack_kernel(int n, const float4* __restrict__ img, uint32_t* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 v = img[i];
  auto q = [](float x) { return (uint32_t)fminf(fmaxf(x * 255.f + 0.5f, 0.f), 255.f); };
  uint32_t a = q(v.w), r = min(q(v.x), a), g = min(q(v.y), a), b = min(q(v.z), a);
  out[i] = r | (g << 8) | (b << 16) | (a << 24);
}
}  // namespace

void render_teacher_targets(const std::string& teacher_ply, GpuViews& gv, cudaStream_t stream) {
  double t0 = now_seconds();
  SplatCloud c = read_ply(teacher_ply);
  Model teacher; teacher.upload(c, stream);
  RenderCtx ctx;
  for (size_t v = 0; v < gv.views.size(); v++) {
    ViewGPU& view = gv.views[v];
    if (view.W != ctx.W || view.H != ctx.H) ctx.setup(view.W, view.H, teacher.cap, stream);
    RenderParams rp; rp.cam = CameraGPU::from(gv.cams[v], view.W, view.H); rp.sh_degree = teacher.degree; rp.bwd_info = false;
    render_forward(ctx, teacher, rp, stream);
    const int n = view.W * view.H;
    pack_kernel<<<div_up(n, 256), 256, 0, stream>>>(n, ctx.out_rgba.ptr, const_cast<uint32_t*>(view.rgba));
    CUDA_KERNEL_CHECK();
    view.has_alpha = true; view.masked = false; view.alpha_coverage = 1.f;
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  log_info("Teacher %s: %zu splats (SH degree %d) rendered as the target of %zu views in %.1fs", teacher_ply.c_str(), c.n, c.sh_degree, gv.views.size(), now_seconds() - t0);
}

}  // namespace b2c
