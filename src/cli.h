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
  bool sh_fp16 = false, sh_fp32 = false;  // storage of SH bands >= 1; fast recipe defaults to fp16 unless --sh-fp32
  bool res_schedule = false;
  float res_quarter_until = 0.15f, res_half_until = 0.40f;  // progress fractions of the 1/4 and 1/2 resolution phases
  bool accumulate_min_scale = false;  // brush quirk: bake floor at every refine
  uint32_t sh_warmup_every = 0;       // 0 = all bands from step 1 (brush)
  bool bench = false;
  std::string backward = "tc";       // tc | warp
  int device = 0;
  std::string checkpoint_dir;         // debug dumps
  bool help = false;

  uint32_t total_iters() const { return total_train_iters + lod_levels * lod_refine_steps; }
};

// Parses brush's argv. Throws std::runtime_error on bad args (message is clap-like).
Config parse_args(int argc, char** argv);
std::string help_text();

}  // namespace b2c
