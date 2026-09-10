#pragma once
#include <string>
#include <vector>
#include <optional>
#include <cstdint>

namespace b2c {

enum class AlphaMode { Masked, Transparent };
enum class Recipe { Brush, Fast };

struct Config {
  // positional
  std::string source;
  bool with_viewer = false;
  // training
  uint32_t total_train_iters = 30000;
  std::string render_mode = "default";
  float lr_mean = 2e-5f, lr_mean_end = 2e-7f, mean_noise_weight = 50.0f;
  float lr_coeffs_dc = 2e-3f, lr_coeffs_sh_scale = 10.0f;
  float lr_opac = 0.012f, lr_scale = 5e-3f, lr_rotation = 2e-3f;
  uint32_t max_splats = 10000000, refine_every = 200;
  float growth_grad_threshold = 0.0025f, growth_select_fraction = 0.25f;
  uint32_t growth_stop_iter = 15000;
  float split_at_screen_size = 0.5f;
  float ssim_weight = 0.2f, opac_decay = 0.004f, match_alpha_weight = 0.1f;
  bool normalize_masked_loss = false;
  float lpips_loss_weight = 0.0f;
  float normal_loss_weight = 0.0f;
  uint32_t normal_loss_start_iter = 5000, normal_loss_every = 1;
  float background_color[3] = {0, 0, 0};
  float background_noise_strength = 0.1f;
  uint32_t lod_levels = 0, lod_refine_steps = 5000, lod_decimation_keep = 50, lod_image_scale = 50;
  std::optional<float> random_init_scene_scale;
  // model
  uint32_t sh_degree = 3;
  // dataset
  std::optional<uint32_t> max_frames;
  uint32_t max_resolution = 1920;
  std::optional<uint32_t> eval_split_every, subsample_frames, subsample_points;
  std::optional<AlphaMode> alpha_mode;
  std::string max_scene_batch_cache_size = "6GiB";
  // process
  uint64_t seed = 42;
  uint32_t start_iter = 0, eval_every = 1000;
  bool eval_save_to_disk = false;
  uint32_t export_every = 5000;
  std::string export_path = "./{dataset}_exports/";
  std::string export_name = "export_{iter}.ply";
  bool export_evidence = false;
  std::optional<float> evidence_prune_inmask;
  float evidence_normal_weight = 0.0f;
  bool rerun_enabled = false;
  // b2ctrain additions
  Recipe recipe = Recipe::Fast;
  bool sparse_adam = false;
  bool sh_fp16 = false, sh_fp32 = false;  // storage of SH bands >= 1: fp32 unless --sh-fp16 (see cli.cpp for why)
  bool res_schedule = false;
  float res_quarter_until = 0.15f, res_half_until = 0.40f;  // progress fractions of the 1/4 and 1/2 resolution phases
  bool accumulate_min_scale = false;  // brush quirk: bake floor at every refine
  uint32_t sh_warmup_every = 0;       // 0 = all bands from step 1 (brush)
  bool bench = false;
  // In-trainer alignment loop (b2crunner's render -> flow -> warp -> refit, without leaving the process).
  uint32_t align_iters = 0, align_steps = 3000;
  std::vector<float> align_flow_sigma{6.f}, align_flow_cap{6.f};  // one entry, or one per iteration
  std::string align_debug_dir;
  std::string align_warp = "frames";  // frames: Lanczos-warp the pristine frames onto the render; render: move the projected splats onto the frames
  std::string backward = "tc";       // tc | warp
  // Hollow loss: a body proxy mesh (dataset's mesh.ply, or --mesh) gives every training pixel a reference surface
  // depth; splat weight arriving from behind it is penalised, and the gradient of that weight through the
  // fragments in front is what pushes the front surface opaque.
  std::string mesh;                   // explicit mesh path (default: <dataset>/mesh.ply if present)
  std::string body_rig;               // per-view articulated deformation rig (gpu/deform.h), empty = off
  uint32_t body_rig_start_iter = 1000;   // learn the per-view rotations from this iteration of the main run on
  float body_rig_lr = 2e-3f, body_rig_smooth = 0.05f, body_rig_zero = 0.02f;
  bool body_rig_global_moment = false;
  float hollow_weight = 0.f;          // 0 = off
  float hollow_margin = 0.05f;        // scene units behind the surface where the penalty starts (full at 2x)
  uint32_t hollow_dilate = 2;         // px: reference depth is the farthest surface within this radius
  uint32_t hollow_start_iter = 0;     // first iteration the loss is applied
  std::string hollow_proxy = "auto"; // auto: mesh if present, else discs on points3D; mesh; points
  float hollow_points_radius = 0.f;   // surfel radius for the points proxy, 0 = 4x the median point spacing
  float hollow_push_tau = 0.5f;       // only fragments at or behind the pixel's depth at this accumulated alpha get the opacity push; 0 = all
  float hollow_front_alpha = 0.f;     // scale the opacity push on fragments in front of penalised weight by min(alpha/this, 1); 0 = exact gradient
  float hollow_tau = 0.1f;            // adaptive reference: deeper of mesh and the splat's own first surface (alpha >= tau); 0 = mesh only
  int device = 0;
  std::string checkpoint_dir;         // debug dumps
  bool help = false;

  uint32_t total_iters() const { return total_train_iters + lod_levels * lod_refine_steps; }
};

// Parses brush's argv. Throws std::runtime_error on bad args (message is clap-like).
Config parse_args(int argc, char** argv);
std::string help_text();

}  // namespace b2c
