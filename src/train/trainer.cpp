#include "train/trainer.h"
#include "train/init.h"
#include "dataset/views.h"
#include "gpu/render.h"
#include "gpu/loss.h"
#include "gpu/optim.h"
#include "gpu/refine.h"
#include "gpu/align.h"
#include "train/evidence.h"
#include "train/gpu_views.h"
#include "util/log.h"
#include "json.hpp"
#include "stb_image_write.h"
#include <random>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <map>
#include <thread>

namespace b2c {
namespace fs = std::filesystem;

namespace {

std::string resolve_export_dir(const Config& cfg, const Dataset& ds, uint64_t start_ts) {
  std::string p = cfg.export_path;
  auto replace_all = [](std::string& s, const std::string& a, const std::string& b) { for (size_t i = 0; (i = s.find(a, i)) != std::string::npos; i += b.size()) s.replace(i, a.size(), b); };
  replace_all(p, "{dataset}", ds.name);
  replace_all(p, "{timestamp}", std::to_string(start_ts));
  fs::path path(p);
  if (path.is_relative()) path = fs::path(ds.root).parent_path() / path;
  return path.string();
}
std::string resolve_export_name(const Config& cfg, uint32_t iter, uint32_t total) {
  std::string n = cfg.export_name;
  int digits = (int)std::floor(std::log10((double)std::max(1u, total))) + 1;
  char buf[32]; snprintf(buf, sizeof buf, "%0*u", digits, iter);
  for (size_t i = 0; (i = n.find("{iter}", i)) != std::string::npos;) n.replace(i, 6, buf);
  return n;
}

struct StageTimer {
  bool enabled = false;
  std::vector<cudaEvent_t> pool;
  std::vector<std::pair<std::string, cudaEvent_t>> marks;
  std::map<std::string, double> totals;
  cudaEvent_t start = nullptr;
  size_t used = 0;
  void init() { if (!enabled) return; CUDA_CHECK(cudaEventCreate(&start)); pool.resize(16); for (auto& e : pool) CUDA_CHECK(cudaEventCreate(&e)); }
  void begin(cudaStream_t s) { if (!enabled) return; CUDA_CHECK(cudaEventRecord(start, s)); marks.clear(); used = 0; }
  void mark(const char* name, cudaStream_t s) { if (!enabled || used >= pool.size()) return; cudaEvent_t e = pool[used++]; CUDA_CHECK(cudaEventRecord(e, s)); marks.emplace_back(name, e); }
  void end() {
    if (!enabled || marks.empty()) return;
    CUDA_CHECK(cudaEventSynchronize(marks.back().second));
    cudaEvent_t prev = start;
    for (auto& [n, e] : marks) { float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, prev, e)); totals[n] += ms; prev = e; }
  }
};

// One training phase: the cold run (all schedules, refinement, exports) or an alignment refit (fixed model size,
// full resolution, fresh optimizer state and LR schedule, no normal loss).
struct Phase {
  uint32_t total = 0, start = 0;
  const std::vector<ViewGPU>* views = nullptr;   // full-resolution views to train on
  bool refine = false, normals = false, res_schedule = false, exports = false, evals = false;
  uint32_t growth_stop = 0;
  std::string label;                              // prefix for the progress line ("" for the main run)
};

float schedule_value(const std::vector<float>& v, uint32_t iter) { return v.empty() ? 6.f : v[std::min<size_t>(iter - 1, v.size() - 1)]; }

}  // namespace

int train_main(const Config& cfg) {
  uint64_t start_ts = (uint64_t)std::time(nullptr);
  CUDA_CHECK(cudaSetDevice(cfg.device));
  cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
  double t_load = now_seconds();
  Dataset ds = load_dataset(cfg);
  SplatCloud init = initial_splats(ds, cfg);
  GpuViews gv; gv.upload(ds.train);
  // The progressive resolution schedule is a cold-start device: a warm start (init.ply — b2crunner's polish and
  // alignment refits) trains at full resolution from its first step, as the brush runs it replaces did.
  const bool warm_start = !ds.init_ply.empty();
  const bool res_schedule = (cfg.res_schedule || cfg.recipe == Recipe::Fast) && !warm_start;
  if ((cfg.res_schedule || cfg.recipe == Recipe::Fast) && warm_start) log_info("Warm start from init.ply: resolution schedule off, training at full resolution");
  if (res_schedule) gv.build_pyramid(3, stream);
  GpuViews gv_eval; if (!ds.eval.empty()) gv_eval.upload(ds.eval);
  Model model; model.sh_fp16 = cfg.sh_fp16 && !cfg.sh_fp32; model.upload(init, stream);
  RenderCtx ctx; ctx.setup(std::max(gv.max_w, gv_eval.max_w), std::max(gv.max_h, gv_eval.max_h), model.cap, stream);
  log_info("Loaded %zu initial splats, %zu views on GPU in %.1fs", init.n, ds.train.size(), now_seconds() - t_load);

  // Scene bounds / median scale for the mean LR and noise clamp.
  Bounds bounds = bounds_from_pos(init.pos.data(), init.n, 0.8f);
  float median_scale = bounds.median_size();

  const uint32_t total = cfg.total_train_iters;
  const uint32_t growth_stop = std::min(cfg.growth_stop_iter, total);
  std::mt19937_64 rng(cfg.seed);
  std::vector<int> order; size_t order_pos = 0;
  auto next_view = [&]() {
    if (order_pos >= order.size()) { order.resize(gv.views.size()); for (size_t i = 0; i < order.size(); i++) order[i] = (int)i; std::shuffle(order.begin(), order.end(), rng); order_pos = 0; }
    return order[order_pos++];
  };
  std::uniform_real_distribution<float> uni(-1.f, 1.f);
  std::string export_dir = resolve_export_dir(cfg, ds, start_ts);
  fs::create_directories(export_dir);

  RefineState refine; refine.init(model, stream);
  refine.accumulate_min_scale = cfg.accumulate_min_scale || cfg.recipe == Recipe::Brush;
  // Camera centres / focals for the 3D-filter floor.
  std::vector<float> cam_pos; std::vector<float> cam_focal;
  for (auto& c : gv.cams) { cam_pos.insert(cam_pos.end(), {c.pos[0], c.pos[1], c.pos[2]}); cam_focal.push_back(c.focal_px()); }
  refine.set_cameras(cam_pos, cam_focal);
  refine.update_min_scale(model, stream);  // brush attaches the floor only at refine; attach it at start too for consistency

  StageTimer timer; timer.enabled = cfg.bench; timer.init();
  PinnedBuf<float> h_loss; h_loss.reserve(16);
  double isect_sum = 0; uint32_t isect_max = 0; uint32_t steps_timed = 0;
  const std::vector<ViewGPU>* evidence_views = &gv.views;  // the frames evidence is measured against (warped ones after alignment)

  auto do_export = [&](uint32_t iter, bool final_export) {
    std::string name = resolve_export_name(cfg, iter, total);
    fs::path out = fs::path(export_dir) / name;
    bool want_evidence = final_export && (cfg.export_evidence || cfg.evidence_prune_inmask.has_value());
    SplatCloud c = model.download(stream);
    bake_min_scale_cpu(c, model, stream);
    if (want_evidence) {
      double te = now_seconds();
      compute_evidence(ctx, model, *evidence_views, gv.cams, cfg, stream);
      c.has_evidence = true; c.evidence = download_evidence(model, stream);
      log_info("Computed evidence for %zu splats over %zu views in %.1fs", c.n, evidence_views->size(), now_seconds() - te);
      if (cfg.evidence_prune_inmask) {
        float f = *cfg.evidence_prune_inmask;
        SplatCloud kept; kept.has_evidence = true; kept.has_scales = true;
        std::vector<size_t> idx;
        for (size_t i = 0; i < c.n; i++) { const float* e = &c.evidence[i * 7]; if (e[3] > 0.f && e[1] > 0.f && e[0] / e[1] >= f) idx.push_back(i); }
        int K = c.K();
        kept.resize(idx.size(), c.sh_degree);
        for (size_t j = 0; j < idx.size(); j++) {
          size_t i = idx[j];
          for (int k = 0; k < 3; k++) { kept.pos[j * 3 + k] = c.pos[i * 3 + k]; kept.log_scale[j * 3 + k] = c.log_scale[i * 3 + k]; }
          for (int k = 0; k < 4; k++) kept.quat[j * 4 + k] = c.quat[i * 4 + k];
          kept.opacity[j] = c.opacity[i];
          for (int k = 0; k < K * 3; k++) kept.sh[j * K * 3 + k] = c.sh[i * K * 3 + k];
          for (int k = 0; k < 7; k++) kept.evidence[j * 7 + k] = c.evidence[i * 7 + k];
        }
        log_info("Evidence prune (inmask < %g): %zu -> %zu splats", f, c.n, kept.n);
        c = std::move(kept);
      }
    }
    std::vector<std::string> comments = {"Exported from Brush", "Vertical axis: y", format("SH degree: %d", c.sh_degree), "SplatRenderMode: default"};
    write_ply(out.string(), c, comments);
    log_info("Exported %zu splats to %s", c.n, out.string().c_str());
  };

  auto train_phase = [&](const Phase& ph) {
    const uint32_t ph_total = ph.total;
    const double lr_decay = std::pow((double)cfg.lr_mean_end / cfg.lr_mean, 1.0 / std::max(1u, ph_total));
    double t0 = now_seconds(), t_report = t0;
    float last_loss = 0.f; uint32_t last_report_iter = ph.start; double last_report_time = t0;
    uint32_t step = ph.start;
    const bool use_tc = cfg.backward == "tc";
    while (step < ph_total) {
      step++;
      timer.begin(stream);
      int vi = next_view();
      int level = 0;
      if (ph.res_schedule) { float pr = (float)step / (float)std::max(1u, ph_total); level = pr < cfg.res_quarter_until ? 2 : (pr < cfg.res_half_until ? 1 : 0); }
      const ViewGPU& view = (level > 0 && !gv.lvl_views[level].empty()) ? gv.lvl_views[level][vi] : (*ph.views)[vi];
      const Camera& cam = gv.cams[vi];
      bool normals_active = ph.normals && cfg.normal_loss_weight > 0.f && step >= cfg.normal_loss_start_iter && ((step - cfg.normal_loss_start_iter) % cfg.normal_loss_every == 0) && view.normals != nullptr;
      float bg[3];
      for (int k = 0; k < 3; k++) bg[k] = std::min(std::max(cfg.background_color[k] + uni(rng) * cfg.background_noise_strength, 0.f), 1.f);

      RenderParams rp; rp.cam = CameraGPU::from(cam, view.W, view.H);
      for (int k = 0; k < 3; k++) rp.bg[k] = bg[k];
      rp.sh_degree = cfg.sh_warmup_every > 0 ? std::min<int>(model.degree, (int)(step / cfg.sh_warmup_every)) : model.degree;
      rp.feat = normals_active ? FeatureMode::Normals : FeatureMode::None;
      rp.bwd_info = true;
      if (view.W != ctx.W || view.H != ctx.H) ctx.setup(view.W, view.H, model.cap, stream);
      render_forward(ctx, model, rp, stream);
      isect_sum += ctx.num_isect; isect_max = std::max(isect_max, ctx.num_isect); steps_timed++;
      timer.mark("forward", stream);

      ctx.loss_accum.zero(stream, 4);
      LossParams lp;
      lp.l1_w = cfg.ssim_weight > 0 ? 1.f - cfg.ssim_weight : 1.f; lp.ssim_w = cfg.ssim_weight > 0 ? -cfg.ssim_weight : 0.f;
      for (int k = 0; k < 3; k++) lp.bg[k] = bg[k];
      lp.composite = view.has_alpha && !view.masked;
      lp.mask = view.masked;
      lp.alpha_lane = view.has_alpha && !view.masked && cfg.match_alpha_weight > 0.f;
      lp.match_alpha_weight = cfg.match_alpha_weight;
      lp.scale = (view.masked && cfg.normalize_masked_loss) ? 1.f / std::max(view.alpha_coverage, 0.01f) : 1.f;
      lp.grad_scale = use_tc ? 3.f * (float)view.W * (float)view.H : 1.f;
      photometric_loss(ctx, view, lp, stream);
      if (normals_active) { lp.normal_scale = cfg.normal_loss_weight * (float)cfg.normal_loss_every / std::max(view.normal_count, 1.f); normal_loss(ctx, view, lp, stream); }
      accumulate_loss(ctx, stream);
      timer.mark("loss", stream);
      if (use_tc) rasterize_backward_tc(ctx, model, rp, lp.grad_scale, stream); else rasterize_backward(ctx, model, rp, stream);
      timer.mark("backward", stream);

      OptimParams op; op.cam = rp.cam; op.active_sh_degree = rp.sh_degree; op.t = ++model.adam_t;
      op.lr_mean = (float)(cfg.lr_mean * std::pow(lr_decay, (double)step - 1.0) * median_scale);
      op.lr_rot = cfg.lr_rotation; op.lr_scale = cfg.lr_scale; op.lr_opac = cfg.lr_opac; op.lr_dc = cfg.lr_coeffs_dc; op.lr_sh_rest = cfg.lr_coeffs_dc / cfg.lr_coeffs_sh_scale;
      op.noise_weight = op.lr_mean * cfg.mean_noise_weight; op.noise_clamp = median_scale;
      op.seed = (uint32_t)cfg.seed; op.step = step; op.sparse = cfg.sparse_adam || cfg.recipe == Recipe::Fast;
      optimizer_step(ctx, model, op, stream);
      timer.mark("optim", stream);

      // Refine.
      float progress = (float)step / (float)std::max(1u, ph_total);
      if (ph.refine && step % cfg.refine_every == 0 && progress <= 0.95f) {
        RefineParams rpar;
        rpar.iter = step; rpar.total = ph_total; rpar.growth_allowed = step < ph.growth_stop;
        rpar.max_splats = cfg.max_splats; rpar.growth_grad_threshold = cfg.growth_grad_threshold; rpar.growth_select_fraction = cfg.growth_select_fraction;
        rpar.split_at_screen_size = cfg.split_at_screen_size; rpar.opac_decay = cfg.opac_decay; rpar.seed = (uint32_t)cfg.seed;
        RefineStats rs = refine.run(model, ctx, rpar, stream);
        if (model.cap > (int)ctx.tile_count.count) ctx.setup(ctx.W, ctx.H, model.cap, stream);
        bounds = refine.bounds; median_scale = bounds.median_size();
        log_info("Refine iter %u, %d splats (pruned %d, split %d, grown %d).", step, model.n, rs.pruned, rs.split_oversized, rs.grown);
        timer.mark("refine", stream);
      }
      timer.end();

      double now = now_seconds();
      if (now - t_report >= 10.0 || step == ph_total) {
        CUDA_CHECK(cudaMemcpyAsync(h_loss.ptr, ctx.loss_accum.ptr, 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        last_loss = h_loss.ptr[3] > 0 ? h_loss.ptr[2] / h_loss.ptr[3] : 0.f;
        CUDA_CHECK(cudaMemsetAsync(ctx.loss_accum.ptr + 2, 0, 2 * sizeof(float), stream));
        double rate = (step - last_report_iter) / (now - last_report_time);
        double avg = (step - ph.start) / (now - t0);
        double left = (ph_total - step) / std::max(rate, 1e-9);
        std::string line = format("📊 %siter %u/%u · %.1f it/s (avg %.1f) · loss %.5f · %s splats · %s elapsed", ph.label.c_str(), step, ph_total, rate, avg, last_loss, format_count(model.n).c_str(), format_duration(now - t0).c_str());
        if (step < ph_total) line += format(" · ~%s left", format_duration(left).c_str());
        log_info("%s", line.c_str());
        t_report = now; last_report_iter = step; last_report_time = now;
      }
      if (ph.evals && !ds.eval.empty() && cfg.eval_every > 0 && step % cfg.eval_every == 0) {
        double mse_sum = 0, ssim_sum = 0; size_t cnt = 0;
        for (size_t e = 0; e < gv_eval.views.size(); e++) {
          const ViewGPU& ev = gv_eval.views[e];
          RenderParams ep; ep.cam = CameraGPU::from(gv_eval.cams[e], ev.W, ev.H); ep.sh_degree = model.degree; ep.bwd_info = false;
          if (ev.W != ctx.W || ev.H != ctx.H) ctx.setup(ev.W, ev.H, model.cap, stream);
          render_forward(ctx, model, ep, stream);
          ctx.loss_accum.zero(stream, 8);
          bool mw = cfg.normalize_masked_loss && ev.masked && ev.has_alpha;
          eval_metrics(ctx, ev, mw, stream);
          CUDA_CHECK(cudaMemcpyAsync(h_loss.ptr, ctx.loss_accum.ptr, 8 * sizeof(float), cudaMemcpyDeviceToHost, stream));
          CUDA_CHECK(cudaStreamSynchronize(stream));
          double npx = (double)ev.W * ev.H * 3.0;
          double div = mw ? std::max(ev.alpha_coverage, 0.01f) : 1.0;
          mse_sum += h_loss.ptr[4] / npx / div; ssim_sum += h_loss.ptr[5] / npx / div; cnt++;
        }
        double mse = mse_sum / cnt, ssim = ssim_sum / cnt;
        log_info("Eval iter %u: PSNR %.3f, ssim %.4f", step, 10.0 * std::log10(1.0 / std::max(mse, 1e-12)), ssim);
      }
      if (ph.exports && cfg.export_every > 0 && step % cfg.export_every == 0 && step < ph_total) do_export(step, false);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return now_seconds() - t0;
  };

  {
    Phase main;
    main.total = total; main.start = cfg.start_iter; main.views = &gv.views;
    main.refine = true; main.normals = true; main.res_schedule = res_schedule; main.exports = true; main.evals = true; main.growth_stop = growth_stop;
    double train_time = train_phase(main);
    log_info("Training took %s (%.1f it/s)", format_duration(train_time).c_str(), (total - cfg.start_iter) / std::max(train_time, 1e-9));
  }

  // The alignment loop (b2crunner's steps/brush.py, in-process): render every transparent training view with the
  // model as it stands, flow the PRISTINE frame onto the render, warp it, and refit on the warped set with growth off.
  // The pristine frames stay resident; each iteration warps them afresh (a warp of a warp would drift).
  std::vector<ViewGPU> aligned_views;
  DevBuf<uint32_t> warped;
  std::vector<std::thread> debug_writers;
  if (cfg.align_iters > 0) {
    size_t n_px = 0; int n_align = 0;
    for (auto& v : gv.views) if (!v.masked) { n_px += (size_t)v.W * v.H; n_align++; }
    warped.reserve(n_px);
    aligned_views = gv.views;
    { size_t o = 0; for (auto& v : aligned_views) if (!v.masked) { v.rgba = warped.ptr + o; o += (size_t)v.W * v.H; } }
    AlignScratch scratch;
    nlohmann::json history = nlohmann::json::array();
    std::vector<std::string> names; for (auto& v : ds.train) names.push_back(v.name);
    int sample = -1; for (size_t i = 0; i < gv.views.size(); i++) if (!gv.views[i].masked) { sample = (int)i; break; }
    std::vector<float> means;
    for (uint32_t it = 1; it <= cfg.align_iters; it++) {
      const float sigma = schedule_value(cfg.align_flow_sigma, it), cap = schedule_value(cfg.align_flow_cap, it);
      double ta = now_seconds();
      double mean_sum = 0, p90_sum = 0;
      nlohmann::json per_view = nlohmann::json::array();
      std::vector<uint32_t> dbg_warped; std::vector<unsigned char> dbg_render; int dbg_w = 0, dbg_h = 0;
      for (size_t vi = 0; vi < gv.views.size(); vi++) {
        const ViewGPU& pv = gv.views[vi];
        if (pv.masked) continue;
        RenderParams rp; rp.cam = CameraGPU::from(gv.cams[vi], pv.W, pv.H);
        rp.bg[0] = rp.bg[1] = rp.bg[2] = 0.5f;  // align.py's BACKGROUND: both sides flattened onto the same grey
        rp.sh_degree = model.degree; rp.feat = FeatureMode::None; rp.bwd_info = false;
        if (pv.W != ctx.W || pv.H != ctx.H) ctx.setup(pv.W, pv.H, model.cap, stream);
        render_forward(ctx, model, rp, stream);
        AlignStats st = align_view_gpu(scratch, pv.rgba, ctx.out_rgba, pv.W, pv.H, sigma, cap, const_cast<uint32_t*>(aligned_views[vi].rgba), stream);
        mean_sum += st.mean; p90_sum += st.p90;
        per_view.push_back({{"name", names[vi]}, {"mean", st.mean}, {"p90", st.p90}});
        if (!cfg.align_debug_dir.empty() && (int)vi == sample) {
          // Only the device-to-host copies happen here; encoding and writing run on a thread while the refit trains.
          dbg_w = pv.W; dbg_h = pv.H;
          dbg_warped.resize((size_t)pv.W * pv.H);
          CUDA_CHECK(cudaMemcpyAsync(dbg_warped.data(), aligned_views[vi].rgba, dbg_warped.size() * 4, cudaMemcpyDeviceToHost, stream));
          std::vector<float4> rgba = ctx.out_rgba.download((size_t)pv.W * pv.H, stream);
          dbg_render.resize((size_t)pv.W * pv.H * 3);
          for (size_t p = 0; p < rgba.size(); p++) for (int c = 0; c < 3; c++) { float v = c == 0 ? rgba[p].x : (c == 1 ? rgba[p].y : rgba[p].z); dbg_render[p * 3 + c] = (unsigned char)std::min(std::max(v * 255.f + 0.5f, 0.f), 255.f); }
          CUDA_CHECK(cudaStreamSynchronize(stream));
        }
      }
      const float mean = (float)(mean_sum / std::max(n_align, 1)), p90 = (float)(p90_sum / std::max(n_align, 1));
      means.push_back(mean);
      log_info("Alignment %u/%u: the training views disagreed with their own renders by %.2f px mean, %.2f px p90 (smoothed at sigma %.1f, capped at %.1f); %d views warped in %.1fs; refitting for %u steps, growth off",
               it, cfg.align_iters, mean, p90, sigma, cap, n_align, now_seconds() - ta, cfg.align_steps);
      if (p90 >= cap)
        log_warn("Alignment %u/%u is CAP-BOUND: the measured disagreement (p90 %.2f px) is at or past the %.1f px cap, so the warp is limited by the clamp and not by the data", it, cfg.align_iters, p90, cap);
      history.push_back({{"iteration", it}, {"sigma", sigma}, {"cap", cap}, {"align_steps", cfg.align_steps}, {"mean", mean}, {"p90", p90}, {"per_view", per_view}});
      if (!cfg.align_debug_dir.empty()) {
        fs::path dir = cfg.align_debug_dir; fs::create_directories(dir);
        nlohmann::json payload = {{"views", names}, {"sample_view", sample >= 0 ? names[sample] : ""}, {"iterations", history}, {"backend", "b2ctrain"}};
        std::string json_text = payload.dump(1);
        std::string stem = sample >= 0 ? fs::path(names[sample]).stem().string() : "view";
        debug_writers.emplace_back([dir, json_text, it, stem, dbg_warped = std::move(dbg_warped), dbg_render = std::move(dbg_render), dbg_w, dbg_h]() {
          { FILE* f = fopen((dir / "alignment.json").string().c_str(), "wb"); if (f) { fwrite(json_text.data(), 1, json_text.size(), f); fclose(f); } }
          if (!dbg_warped.empty()) stbi_write_png((dir / format("iter%u_%s_warped.png", it, stem.c_str())).string().c_str(), dbg_w, dbg_h, 4, dbg_warped.data(), dbg_w * 4);
          if (!dbg_render.empty()) stbi_write_png((dir / format("iter%u_%s_render.png", it, stem.c_str())).string().c_str(), dbg_w, dbg_h, 3, dbg_render.data(), dbg_w * 3);
        });
      }
      // Refit: what a fresh warm-started invocation would do — fresh optimizer state and LR schedule, floor re-attached,
      // bounds from the model as it stands, growth and refinement off, normal loss off, full resolution.
      model.zero_optimizer(stream); model.zero_stats(stream);
      refine.update_min_scale(model, stream);
      refine.update_bounds(model, stream); bounds = refine.bounds; median_scale = bounds.median_size();
      Phase refit;
      refit.total = cfg.align_steps; refit.views = &aligned_views; refit.label = format("align %u/%u · ", it, cfg.align_iters);
      double t = train_phase(refit);
      log_info("Alignment %u/%u refit took %s (%.1f it/s)", it, cfg.align_iters, format_duration(t).c_str(), cfg.align_steps / std::max(t, 1e-9));
    }
    std::string traj; for (size_t i = 0; i < means.size(); i++) traj += (i ? " -> " : "") + format("%.2f", means[i]);
    log_info("Alignment finished: measured disagreement %s px across %u iteration(s) (reference loop: 1.02 -> 1.18 -> 1.26 -> 1.31, rising and decelerating)", traj.c_str(), cfg.align_iters);
    evidence_views = &aligned_views;
  }

  do_export(total, true);
  for (auto& t : debug_writers) t.join();
  if (cfg.bench) {
    log_info("  intersections/step: avg %.0f, max %u", isect_sum / std::max(1u, steps_timed), isect_max);
    double sum = 0; for (auto& [n, ms] : timer.totals) sum += ms;
    for (auto& [n, ms] : timer.totals) log_info("  %-10s %8.1f ms total  %6.3f ms/step  %5.1f%%", n.c_str(), ms, ms / std::max(1u, steps_timed), 100.0 * ms / sum);
  }
  return 0;
}

}  // namespace b2c
