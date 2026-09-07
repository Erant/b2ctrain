#include "train/evidence.h"
namespace b2c {
static DevBuf<float> g_evidence;
void compute_evidence(RenderCtx& ctx, const Model& m, const std::vector<ViewGPU>& views, const std::vector<Camera>& cams, const Config& cfg, cudaStream_t stream) {
  g_evidence.reserve((size_t)m.n * 7); g_evidence.zero(stream);
  (void)ctx; (void)views; (void)cams; (void)cfg;
}
std::vector<float> download_evidence(const Model& m, cudaStream_t stream) {
  if (g_evidence.count < (size_t)m.n * 7) return std::vector<float>((size_t)m.n * 7, 0.f);
  return g_evidence.download((size_t)m.n * 7, stream);
}
}
