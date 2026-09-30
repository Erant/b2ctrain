#include "cli.h"
#ifndef B2CTRAIN_GIT_SHA
#define B2CTRAIN_GIT_SHA "unknown"
#endif
#include "util/log.h"
#include <cstring>
#include <cstdlib>
#include <functional>
#include <map>

namespace b2c {

namespace {
struct Opt {
  const char* name;
  const char* value_name;  // nullptr for boolean flags
  const char* help;
  std::function<void(Config&, const char*)> set;
};

float parse_f(const char* s, const char* name) {
  char* end = nullptr; float v = strtof(s, &end);
  if (end == s || *end) fail("invalid value '%s' for '--%s': expected a number", s, name);
  return v;
}
uint64_t parse_u(const char* s, const char* name) {
  char* end = nullptr; long long v = strtoll(s, &end, 10);
  if (end == s || *end || v < 0) fail("invalid value '%s' for '--%s': expected a non-negative integer", s, name);
  return (uint64_t)v;
}

std::vector<float> parse_float_list(const char* s, const char* name) {
  std::vector<float> out; std::string cur;
  for (const char* p = s;; p++) {
    if (*p == ',' || *p == '\0') { if (!cur.empty()) out.push_back(parse_f(cur.c_str(), name)); cur.clear(); if (!*p) break; }
    else cur += *p;
  }
  if (out.empty()) fail("invalid value '%s' for '--%s': expected a comma-separated list of numbers", s, name);
  return out;
}

#define F(field) [](Config& c, const char* v) { c.field = parse_f(v, #field); }
#define U(field) [](Config& c, const char* v) { c.field = (uint32_t)parse_u(v, #field); }
#define UO(field) [](Config& c, const char* v) { c.field = (uint32_t)parse_u(v, #field); }
#define FO(field) [](Config& c, const char* v) { c.field = parse_f(v, #field); }
#define B(field) [](Config& c, const char*) { c.field = true; }
#define S(field) [](Config& c, const char* v) { c.field = v; }
#define FL(field) [](Config& c, const char* v) { c.field = parse_float_list(v, #field); }

const std::vector<std::pair<const char*, std::vector<Opt>>>& groups() {
  static const std::vector<std::pair<const char*, std::vector<Opt>>> g = {
    {"Options", {
      {"with-viewer", nullptr, "Spawn a viewer to visualize the training (unsupported in b2ctrain)", B(with_viewer)},
      {"help", nullptr, "Print help", B(help)},
    }},
    {"Training options", {
      {"total-train-iters", "TOTAL_TRAIN_ITERS", "Total number of steps to train for [default: 30000]", U(total_train_iters)},
      {"render-mode", "RENDER_MODE", "[possible values: default, mip]", S(render_mode)},
      {"lr-mean", "LR_MEAN", "Start learning rate for the mean parameters [default: 2e-5]", F(lr_mean)},
      {"lr-mean-end", "LR_MEAN_END", "End learning rate for the mean parameters [default: 2e-7]", F(lr_mean_end)},
      {"mean-noise-weight", "MEAN_NOISE_WEIGHT", "How much noise to add to the mean parameters of low opacity gaussians [default: 50.0]", F(mean_noise_weight)},
      {"lr-coeffs-dc", "LR_COEFFS_DC", "Learning rate for the base SH (RGB) coefficients [default: 2e-3]", F(lr_coeffs_dc)},
      {"lr-coeffs-sh-scale", "LR_COEFFS_SH_SCALE", "How much to divide the learning rate by for higher SH orders [default: 10.0]", F(lr_coeffs_sh_scale)},
      {"lr-opac", "LR_OPAC", "Learning rate for the opacity parameter [default: 0.012]", F(lr_opac)},
      {"lr-scale", "LR_SCALE", "Learning rate for the scale parameters [default: 5e-3]", F(lr_scale)},
      {"lr-rotation", "LR_ROTATION", "Learning rate for the rotation parameters [default: 2e-3]", F(lr_rotation)},
      {"ssim-weight", "SSIM_WEIGHT", "Weight of SSIM loss (compared to l1 loss) [default: 0.2]", F(ssim_weight)},
      {"opac-decay", "OPAC_DECAY", "Factor of the opacity decay [default: 0.004]", F(opac_decay)},
      {"normalize-masked-loss", nullptr, "Divide a masked view's loss (and its eval PSNR/SSIM) by the fraction of the frame its mask covers, so masked and transparent views contribute comparable gradient magnitude", B(normalize_masked_loss)},
      {"normal-loss-weight", "NORMAL_LOSS_WEIGHT", "Weight of the monocular normal-map supervision loss (L1 + 1-cosine against a per-frame normal prior under `normals/`). 0 disables it [default: 0.0]", F(normal_loss_weight)},
      {"normal-loss-start-iter", "NORMAL_LOSS_START_ITER", "Training step at which normal-map supervision starts [default: 5000]", U(normal_loss_start_iter)},
      {"normal-loss-every", "NORMAL_LOSS_EVERY", "Evaluate normal supervision once every N training steps (loss multiplied by N) [default: 1]", U(normal_loss_every)},
      {"background-color", "R,G,B", "Base background color (R,G,B) used during training [default: 0,0,0]",
        [](Config& c, const char* v) { if (sscanf(v, "%f,%f,%f", &c.background_color[0], &c.background_color[1], &c.background_color[2]) != 3) fail("invalid --background-color '%s'", v); }},
      {"background-noise-strength", "BACKGROUND_NOISE_STRENGTH", "Strength of random noise added to the background color each step [default: 0.1]", F(background_noise_strength)},
      {"random-init-scene-scale", "RANDOM_INIT_SCENE_SCALE", "Scene scale used for random splat initialization", FO(random_init_scene_scale)},
    }},
    {"Refine options", {
      {"max-splats", "MAX_SPLATS", "Max nr. of splats. Upper bound only [default: 10000000]", U(max_splats)},
      {"refine-every", "REFINE_EVERY", "Frequency of 'refinement' where gaussians are replaced and densified [default: 200]", U(refine_every)},
      {"growth-grad-threshold", "GROWTH_GRAD_THRESHOLD", "Threshold to control splat growth. Lower means faster growth [default: 0.0025]", F(growth_grad_threshold)},
      {"growth-select-fraction", "GROWTH_SELECT_FRACTION", "What fraction of splats that are deemed as needing to grow do actually grow [default: 0.25]", F(growth_select_fraction)},
      {"growth-stop-iter", "GROWTH_STOP_ITER", "Period after which splat growth stops [default: 15000]", U(growth_stop_iter)},
      {"split-at-screen-size", "SPLIT_AT_SCREEN_SIZE", "Split any splat whose max screen-space extent exceeds this fraction of the image dimension. 0 disables [default: 0.5]", F(split_at_screen_size)},
      {"match-alpha-weight", "MATCH_ALPHA_WEIGHT", "Weight of l1 loss on alpha if input view has transparency [default: 0.1]", F(match_alpha_weight)},
      {"lpips-loss-weight", "LPIPS_LOSS_WEIGHT", "[default: 0.0] (unsupported in b2ctrain)", F(lpips_loss_weight)},
    }},
    {"LOD options", {
      {"lod-levels", "LOD_LEVELS", "Number of LOD levels to generate after initial training (0 = disabled) [default: 0]", U(lod_levels)},
      {"lod-refine-steps", "LOD_REFINE_STEPS", "[default: 5000]", U(lod_refine_steps)},
      {"lod-decimation-keep", "LOD_DECIMATION_KEEP", "[default: 50]", U(lod_decimation_keep)},
      {"lod-image-scale", "LOD_IMAGE_SCALE", "[default: 50]", U(lod_image_scale)},
    }},
    {"Model Options", {
      {"sh-degree", "SH_DEGREE", "SH degree of splats [default: 3]", U(sh_degree)},
    }},
    {"Dataset Options", {
      {"max-frames", "MAX_FRAMES", "Max nr. of frames of dataset to load", UO(max_frames)},
      {"max-resolution", "MAX_RESOLUTION", "Max resolution of images to load [default: 1920]", U(max_resolution)},
      {"eval-split-every", "EVAL_SPLIT_EVERY", "Create an eval dataset by selecting every nth image", UO(eval_split_every)},
      {"subsample-frames", "SUBSAMPLE_FRAMES", "Load only every nth frame", UO(subsample_frames)},
      {"subsample-points", "SUBSAMPLE_POINTS", "Load only every nth point from the initial sfm data", UO(subsample_points)},
      {"alpha-mode", "ALPHA_MODE", "Whether to interpret an alpha channel (or masks) as transparency or masking [possible values: masked, transparent]",
        [](Config& c, const char* v) {
          if (!strcmp(v, "masked")) c.alpha_mode = AlphaMode::Masked;
          else if (!strcmp(v, "transparent")) c.alpha_mode = AlphaMode::Transparent;
          else fail("invalid value '%s' for '--alpha-mode' [possible values: masked, transparent]", v); }},
      {"max-scene-batch-cache-size", "MAX_SCENE_BATCH_CACHE_SIZE", "Ignored (all views are GPU resident) [default: 6GiB]", S(max_scene_batch_cache_size)},
    }},
    {"Process options", {
      {"seed", "SEED", "Random seed [default: 42]", [](Config& c, const char* v) { c.seed = parse_u(v, "seed"); }},
      {"start-iter", "START_ITER", "Iteration to resume from [default: 0]", U(start_iter)},
      {"eval-every", "EVAL_EVERY", "Eval every this many steps [default: 1000]", U(eval_every)},
      {"eval-save-to-disk", nullptr, "Save the rendered eval images to disk", B(eval_save_to_disk)},
      {"export-every", "EXPORT_EVERY", "Export every this many steps [default: 5000]", U(export_every)},
      {"export-path", "EXPORT_PATH", "Location to put exported files. Supports {dataset} and {timestamp} interpolation. Relative to the dataset's parent directory [default: ./{dataset}_exports/]", S(export_path)},
      {"export-name", "EXPORT_NAME", "Filename of exported ply file [default: export_{iter}.ply]", S(export_name)},
      {"export-evidence", nullptr, "At the end of training, measure per-splat multi-view evidence against every training view and write it into the final ply as `ev_*` vertex properties", B(export_evidence)},
      {"evidence-prune-inmask", "EVIDENCE_PRUNE_INMASK", "Before the final export, drop splats whose in-mask contribution fraction (evidence `w_in / w_all`) is below this value, or that no training view supported at all. Implies computing evidence", FO(evidence_prune_inmask)},
      {"evidence-prune-wall", "EVIDENCE_PRUNE_WALL", "Before the final export, drop splats whose total rendered mass over the training views (evidence `w_all`, pixel-weights) is below this value: they cannot affect any training view. Implies computing evidence", FO(evidence_prune_wall)},
      {"cull-weight", "CULL_WEIGHT", "At every refine, prune the splats whose rendered mass (sum of alpha*T over the pixels of the views drawn since the last refine) is below this many pixel-weights per full-resolution pass over the training views; the alignment refits prune by it too. 0 disables it [default: 0]", F(cull_weight)},
      {"cull-no-relocate", nullptr, "With --cull-weight: once growth has stopped, dead and culled splats are dropped instead of relocated onto opacity-sampled survivors (the count shrinks)", B(cull_no_relocate)},
      {"cull-start-iter", "CULL_START_ITER", "First training step at which --cull-weight applies [default: 0]", U(cull_start_iter)},
      {"evidence-normal-weight", "EVIDENCE_NORMAL_WEIGHT", "Weight of the normal-map residual folded into the evidence residual when the dataset has `normals/` [default: 0.0]", F(evidence_normal_weight)},
      {"export-labels", nullptr, "At the end of training, vote each splat a class from the dataset's `labels/` sidecar (8-bit class-id PNGs, one per training frame) by its rendered weight on each class's pixels over every training view, and write the winner and its share of the splat's weight into the final ply as `seg_label` / `seg_conf`. Views without a sidecar cast no vote", B(export_labels)},
    }},
    {"Rerun options", {
      {"rerun-enabled", nullptr, "(unsupported in b2ctrain)", B(rerun_enabled)},
      {"rerun-log-train-stats-every", "N", "(ignored)", [](Config&, const char*) {}},
      {"rerun-log-splats-every", "N", "(ignored)", [](Config&, const char*) {}},
      {"rerun-log-distribution-every", "N", "(ignored)", [](Config&, const char*) {}},
      {"rerun-max-img-size", "N", "(ignored)", [](Config&, const char*) {}},
    }},
    {"b2ctrain options", {
      {"recipe", "RECIPE", "Training recipe: `brush` reproduces the brush fork's dynamics (accumulating 3D-filter floor, dense Adam, full resolution); `fast` adds sparse Adam, the progressive resolution schedule and the non-accumulating floor [default: fast]",
        [](Config& c, const char* v) {
          if (!strcmp(v, "brush")) c.recipe = Recipe::Brush; else if (!strcmp(v, "fast")) c.recipe = Recipe::Fast;
          else fail("invalid value '%s' for '--recipe' [possible values: brush, fast]", v); }},
      {"sparse-adam", nullptr, "Only apply Adam to splats visible in the current view", B(sparse_adam)},
      {"sh-fp16", nullptr, "Store SH bands >= 1 and their Adam moments as fp16 with stochastically rounded updates: ~8% faster on a large model, but the rounding random walk shows as view-dependent colour speckle on specular surfaces at novel views. Off by default", B(sh_fp16)},
      {"sh-fp32", nullptr, "Keep SH bands >= 1 in fp32 (the default; overrides --sh-fp16)", B(sh_fp32)},
      {"res-schedule", nullptr, "Progressive resolution schedule (1/4 -> 1/2 -> 1x) over the first 40% of iterations", B(res_schedule)},
      {"res-quarter-until", "FRACTION", "Progress fraction at which the resolution schedule leaves 1/4 resolution [default: 0.15]", F(res_quarter_until)},
      {"res-half-until", "FRACTION", "Progress fraction at which the resolution schedule reaches full resolution [default: 0.4]", F(res_half_until)},
      {"accumulate-min-scale", nullptr, "Bake the Mip 3D-filter floor into scales at every refine (brush's behaviour) instead of applying it on the fly", B(accumulate_min_scale)},
      {"sh-warmup-every", "N", "Unlock one SH band every N iterations (0 = all bands from the start) [default: 0]", U(sh_warmup_every)},
      {"backward", "MODE", "Rasterizer backward kernel: tc (tensor-core reduction) or warp (shuffle reduction) [default: tc]", S(backward)},
      {"bench", nullptr, "Print a per-kernel timing breakdown at the end", B(bench)},
      {"align-iters", "N", "After training, run N alignment iterations in-process: render every training view, flow the pristine frame onto its render, warp it (Lanczos), refit for --align-steps with growth off and no normal loss. Replaces b2crunner's render/warp/re-invoke loop [default: 0]", U(align_iters)},
      {"align-steps", "N", "Refit iterations per alignment pass [default: 3000]", U(align_steps)},
      {"align-flow-sigma", "PX[,PX..]", "Gaussian smoothing of the flow field in pixels; one value, or one per alignment iteration [default: 6]", FL(align_flow_sigma)},
      {"align-flow-cap", "PX[,PX..]", "Largest displacement applied in pixels; one value, or one per iteration [default: 6]", FL(align_flow_cap)},
      {"align-warp", "MODE", "What the measured flow is applied to: frames (resample the pristine frames onto the render, Lanczos) or render (displace each splat's projected mean by the field at its position while refitting, so the pristine frames are never resampled) [default: frames]", S(align_warp)},
      {"align-debug-dir", "DIR", "Write alignment.json (per-iteration, per-view flow statistics) and one view's warped frame + render per iteration here (written off the training path)", S(align_debug_dir)},
      {"align-dump-dir", "DIR", "Write every warped training frame of the LAST alignment iteration here as RGBA png (the frames the final refit trained on; --align-warp frames only)", S(align_dump_dir)},
      {"body-rig", "PATH", "Per-view articulated deformation: a rig file (b2crunner's body tooling) with the refit body's skinning and, per training view, per joint, the rigid transform from the canonical pose; every splat renders for a view at its skinned position, the model and the export stay canonical. Takes a v2 rig (B2CRIG2) or a v3 one (B2CRIG3: also per-view displacements of the rig vertices, from b2crunner's per-view head fit, so the face follows each frame's expression and head pose)", S(body_rig)},
      {"body-rig-init", "PATH", "Seed the per-view joint rotations from a JSON in the layout of the exported body_rig_omega.json ({\"views\": [{\"name\", \"omega\": {joint: [x, y, z]}}]}, world-axis axis-angle); the rotations start there and --body-rig-zero pulls towards the seed instead of towards zero, so a pose fitted outside the trainer (the head, from each frame's landmarks) is the prior the torque refines", S(body_rig_init)},
      {"body-rig-start-iter", "N", "Start learning the per-view joint rotations at this iteration of the main run; they keep learning through the alignment refits. Measured: the earlier the sharper (1000 > 5000 > 15000) [default: 1000]", U(body_rig_start_iter)},
      {"body-rig-stop-iter", "N", "Stop learning the per-view joint rotations at this iteration of the main run and keep them frozen through the alignment refits (0 = never). Coarse-to-fine: with the fast recipe's resolution schedule, a stop at the end of the 1/2-resolution phase (40% of the run) learns the pose from coarse structure only [default: 0]", U(body_rig_stop_iter)},
      {"cage", "PATH", "Cage-posed training (b2crig): a cage file (gpu/cage.h); every training view named like one of its frames (or \"<frame>@<view>\") renders the splat posed by that frame, the others render canonical. Gradients are turned back into the canonical frame, the model and the export stay canonical. Needs a warm start whose init.ply carries seg_label (the binding); new splats inherit their parent's label", S(cage)},
      {"cage-min-conf", "C", "Splats with seg_conf below this bind to the nearest triangle of any layer [default: 0.5]", F(cage_min_conf)},
      {"pose-contain-weight", "W", "Pose containment loss (needs --cage and --pose-contain-cameras): the splat posed by the cage's frames must stay inside the silhouette its own surface draws in that pose. Targets are rendered once from the warm start, per camera of --pose-contain-cameras, posed by the cage frame the camera is named after, with the splats --pose-contain-exclude lists hidden, dilated by --pose-contain-dilate px; the virtual views train only alpha, pushing it to 0 outside. Removes what the body interior throws outside the figure when animated; clothing and hair define the target, so they stay. Which poses, cameras and splats is b2crig's choice (b2crig/rig/contain.py) (0 = off) [default: 0]", F(pose_contain_weight)},
      {"pose-contain-fraction", "F", "Share of the training steps that draw a containment view instead of a real one [default: 0.5]", F(pose_contain_fraction)},
      {"pose-contain-cameras", "PATH", "Containment views (cameras.json, body2colmap terms, one image size): each camera is named after the cage frame it poses, \"<frame>\" or \"<frame>@<anything>\"", S(pose_contain_cameras)},
      {"pose-contain-exclude", "PATH", "uint8 per warm-start splat (init.ply order): 1 = hidden while the containment targets render (b2crig: the splats deep inside the body) [default: none]", S(pose_contain_exclude)},
      {"pose-contain-dilate", "PX", "Pixels the allowed silhouette grows by (at the containment cameras' resolution) [default: 5]", U(pose_contain_dilate)},
      {"cage-stretch-weight", "W", "Stretch regulariser: splats bound to a cage triangle that grows over the cage's frames (a pose library may ride along as extra frames) get W * log(growth) added to the gradient of every log-scale above --cage-stretch-tau: they stay small where the body stretches, so they do not bloom in poses (0 = off) [default: 0]", F(cage_stretch_weight)},
      {"cage-stretch-tau", "M", "Scale (metres, per axis) under which the stretch regulariser leaves a splat alone [default: 0.004]", F(cage_stretch_tau)},
      {"cage-max-growth", "G", "Bound splats follow their triangle's size change (offset and scales) only within 1/G..G (0 = unclamped) [default: 0]", F(cage_max_growth)},
      {"cage-fade-start", "S", "Stretch fade: bound splats' opacity falls linearly to 0 as their triangle grows from S to --cage-fade-end x its canonical size (0 = off) [default: 0]", F(cage_fade_start)},
      {"cage-fade-end", "E", "Stretch fade end ratio (see --cage-fade-start) [default: 0]", F(cage_fade_end)},
      {"cage-app", nullptr, "Pose-dependent appearance (gpu/cage_app.h): a small MLP per cage vertex, fed the local surface stretch of the posed frame, blends each bound splat's colour towards a learned target and shifts its opacity and scale; zero at the canonical pose. Exported next to the ply as <name>.app (render --cage-app)", B(cage_app)},
      {"cage-app-init", "PATH", "Start the --cage-app MLP from this .app file instead of a random init", S(cage_app_init)},
      {"cage-app-lr", "LR", "Adam step of the --cage-app MLP weights [default: 0.001]", F(cage_app_lr)},
      {"cage-app-lr-latent", "LR", "Adam step of the --cage-app per-vertex latents [default: 0.001]", F(cage_app_lr_latent)},
      {"cage-app-deadzone", "DZ", "--cage-app dead zone: the log stretch features enter as sign(f) max(|f| - DZ, 0), so only surface stretched or compressed past exp(DZ) changes appearance [default: 0]", F(cage_app_deadzone)},
      {"cage-app-max-dopacity", "D", "--cage-app: the opacity logit may rise by at most D (it may always fall) [default: unlimited]", F(cage_app_max_do)},
      {"cage-app-max-dscale", "D", "--cage-app: the log scales change by at most +-D (D tanh(x / D); 0 = no scale channel) [default: 1]", F(cage_app_max_ds)},
      {"cage-app-max-offset", "M", "--cage-app: also move each bound splat by up to M metres per axis of its posed triangle's frame (M tanh(x / M); 0 = off) [default: 0]", F(cage_app_max_dp)},
      {"cage-app-rise-opacity", "W", "--cage-app: added to the gradient of every visible splat's opacity-logit delta while it is positive (penalises faint splats the MLP makes more opaque: haze) [default: 0]", F(cage_app_reg_rise_do)},
      {"cage-app-rise-scale", "W", "--cage-app: the same against scale growth [default: 0]", F(cage_app_reg_rise_ds)},
      {"cage-app-reg-out", "W", "--cage-app: L2 weight on the per-vertex outputs (blend, opacity, scale, offset) [default: 0]", F(cage_app_reg_out)},
      {"cage-app-latent-decay", "W", "--cage-app: decoupled weight decay of the per-vertex latents [default: 0]", F(cage_app_latent_decay)},
      {"cage-open", nullptr, "Two-state splats (gpu/cage_open.h): every bound splat learns a colour offset and an opacity delta, blended in by the cage's opening gate (theta per vertex and frame: how far a joint next to it has turned, e.g. the torso side under a raised arm; b2crig computes it into the cage's B2COPEN1 section, gpu/cage.h) over --cage-open-start..-end degrees; exactly the plain splat at the canonical pose. Exported as ply properties open_* plus a header comment with the gate parameters", B(cage_open)},
      {"cage-open-start", "S", "--cage-open: gate angle (degrees) at which the open state starts to blend in [default: 30]", F(cage_open_start)},
      {"cage-open-end", "E", "--cage-open: gate angle (degrees) at which the open state is fully in [default: 70]", F(cage_open_end)},
      {"cage-open-max-dopacity", "D", "--cage-open: the open state's opacity-logit delta may be at most D (it may always fall) [default: 0]", F(cage_open_max_do)},
      {"cage-open-lr", "LR", "--cage-open: Adam step of the open colour logits [default: 0.01]", F(cage_open_lr)},
      {"cage-open-lr-opacity", "LR", "--cage-open: Adam step of the open opacity delta [default: 0.012]", F(cage_open_lr_opacity)},
      {"cage-open-snr", "T", "--cage-open: consistency gate of the open-state update: the Adam ratio m/sqrt(v) (1 = steady gradient, ~0.2 = noise) enters as sign max(|r| - T, 0) / (1 - T), so only splats the posed views push the same way every frame move (an opened crease), not those a few pixels of WAN misalignment push around (a knee pad) [default: 0]", F(cage_open_snr)},
      {"cage-open-beta1", "B", "--cage-open: Adam beta1 of the open state (higher = a longer memory, a sharper --cage-open-snr) [default: 0.9]", F(cage_open_beta1)},
      {"cage-open-reg", "W", "--cage-open: pull of each open colour offset towards 0, W d added to dL/dd: the open state moves only where the posed views insist (an opened crease), not for the clips' general look [default: 0]", F(cage_open_reg)},
      {"body-rig-lr", "LR", "Adam step of the per-view joint rotations, radians [default: 0.002]", F(body_rig_lr)},
      {"body-rig-smooth", "W", "Pull of each view's rotations towards the mean of its two orbit neighbours, added to the gradient [default: 0.05]", F(body_rig_smooth)},
      {"body-rig-zero", "W", "Pull of every per-view rotation towards zero, added to the gradient; keeps joints with little evidence (fingers, a hidden limb) from wandering [default: 0.02]", F(body_rig_zero)},
      {"body-rig-global-moment", nullptr, "One Adam second moment for all joints and views instead of one per joint. Measured much worse (the root chain then takes the largest steps and moves the whole body per view); kept for experiments", B(body_rig_global_moment)},
      {"mesh", "PATH", "Body proxy triangle mesh (.ply or .obj, dataset world frame) for --hollow-weight; defaults to <dataset>/mesh.ply when present", S(mesh)},
      {"hollow-weight", "W", "Weight of the hollow loss: per pixel, the compositing weight arriving from more than --hollow-margin behind the proxy mesh surface (ramping to full penalty at twice the margin), relative to the photometric loss. Pushes the visible surface opaque and empties the body interior. 0 disables [default: 0]", F(hollow_weight)},
      {"hollow-margin", "DIST", "Depth behind the proxy surface (scene units) where the hollow penalty starts; it reaches full strength at twice this [default: 0.05]", F(hollow_margin)},
      {"hollow-dilate", "PX", "The reference depth at a pixel is the farthest mesh depth within this radius (silhouettes and folds are forgiven) [default: 2]", U(hollow_dilate)},
      {"hollow-start-iter", "N", "Apply the hollow loss from this iteration on [default: 0]", U(hollow_start_iter)},
      {"hollow-proxy", "MODE", "Where the reference surface comes from: `mesh` (a mesh is required), `points` (surfels on the dataset's points3D.txt, normals estimated from the neighbours, which b2crunner samples on the body mesh) or `auto` (the mesh when there is one, else the points) [default: auto]", S(hollow_proxy)},
      {"hollow-points-radius", "DIST", "Surfel radius of the points proxy in scene units; 0 = four times the median point spacing [default: 0]", F(hollow_points_radius)},
      {"sparsify", "RATIO", "GaussianSpa simplification: remove this fraction of the splats alive at --sparsify-start-iter by the stop. Between start and stop, every --sparsify-every steps, the opacities are pulled (--sparsify-rho) towards a copy of themselves in which all but the top (1 - RATIO) are zero, and that copy and its dual are re-projected; at the stop the bottom of the ranking is removed. Relocation and oversized splits are off from the start on, so the count only falls. Needs growth stopped by the start. 0 disables [default: 0]", F(sparsify)},
      {"sparsify-start-iter", "N", "First iteration of the sparsification phase; must be at or after --growth-stop-iter [default: 15000]", U(sparsify_start_iter)},
      {"sparsify-stop-iter", "N", "Iteration at which the sparsification phase ends and the splats outside the kept set are removed [default: 25000]", U(sparsify_stop_iter)},
      {"sparsify-every", "N", "Steps between sparsifying updates; the penalty is applied on those steps only, as GaussianSpa does [default: 50]", U(sparsify_every)},
      {"sparsify-rho", "RHO", "Weight of the penalty 0.5 * rho * ||o - z + u||^2 on the opacities, relative to the photometric loss (mean over pixels) [default: 0.0005]", F(sparsify_rho)},
      {"sparsify-score", "MODE", "What ranks the splats: `opacity` (o + u, GaussianSpa's vanilla-3DGS variant) or `importance` (each splat's blending weight summed over every training view, the Mini-Splatting variant; costs one render of every view per --sparsify-score-every) [default: opacity]", S(sparsify_score)},
      {"sparsify-score-every", "N", "Steps between recomputations of the importance score (--sparsify-score importance); GaussianSpa recomputes at every update [default: 500]", U(sparsify_score_every)},
      {"device", "N", "CUDA device index [default: 0]", [](Config& c, const char* v) { c.device = (int)parse_u(v, "device"); }},
      {"checkpoint-dir", "DIR", "Directory for debug dumps", S(checkpoint_dir)},
    }},
  };
  return g;
}
}  // namespace

std::string help_text() {
  std::string s = "b2ctrain - CUDA gaussian splat trainer (brush-compatible CLI)\n\n"
                  "Usage: b2ctrain [OPTIONS] [PATH]\n\nArguments:\n  [PATH]\n          Dataset directory (COLMAP text model with images/, masks/, normals/, weights/, init.ply)\n\n";
  for (auto& [gname, opts] : groups()) {
    s += std::string(gname) + ":\n";
    for (auto& o : opts) {
      s += "      --" + std::string(o.name);
      if (o.value_name) s += std::string(" <") + o.value_name + ">";
      s += "\n          " + std::string(o.help) + "\n\n";
    }
  }
  return s;
}

Config parse_args(int argc, char** argv) {
  Config c;
  std::map<std::string, const Opt*> lookup;
  for (auto& [g, opts] : groups()) for (auto& o : opts) lookup[o.name] = &o;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") { c.help = true; continue; }
    if (a == "-V" || a == "--version") { printf("b2ctrain 0.1.0 (%s)\n", B2CTRAIN_GIT_SHA); exit(0); }
    if (a.rfind("--", 0) != 0) {
      if (!c.source.empty()) fail("unexpected argument '%s' found", a.c_str());
      c.source = a; continue;
    }
    std::string name = a.substr(2), value;
    bool has_value = false;
    auto eq = name.find('=');
    if (eq != std::string::npos) { value = name.substr(eq + 1); name = name.substr(0, eq); has_value = true; }
    auto it = lookup.find(name);
    if (it == lookup.end()) fail("unexpected argument '--%s' found\n\nFor more information, try '--help'.", name.c_str());
    const Opt* o = it->second;
    if (o->value_name) {
      if (!has_value) {
        if (i + 1 >= argc) fail("a value is required for '--%s <%s>' but none was supplied", o->name, o->value_name);
        value = argv[++i]; 
      }
      o->set(c, value.c_str());
    } else {
      if (has_value) fail("unexpected value for flag '--%s'", o->name);
      o->set(c, nullptr);
    }
  }
  return c;
}

}  // namespace b2c
