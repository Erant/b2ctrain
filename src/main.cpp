#include "cli.h"
#include "train/trainer.h"
#include "render/renderer.h"
#include "render/probe.h"
#include "util/log.h"
#include <cstring>
#include <cstdio>

int main(int argc, char** argv) {
  try {
    if (argc >= 2 && !strcmp(argv[1], "render")) return b2c::render_main(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "probe")) return b2c::probe_main(argc, argv);
    b2c::Config cfg = b2c::parse_args(argc, argv);
    if (cfg.help) { fputs(b2c::help_text().c_str(), stdout); return 0; }
    if (cfg.source.empty()) b2c::fail("a dataset path is required (b2ctrain [OPTIONS] <PATH>)");
    if (cfg.with_viewer) b2c::fail("--with-viewer is not supported by b2ctrain");
    if (cfg.lod_levels > 0) b2c::fail("--lod-levels > 0 is not supported by b2ctrain");
    if (cfg.lpips_loss_weight > 0) b2c::fail("--lpips-loss-weight > 0 is not supported by b2ctrain");
    if (cfg.rerun_enabled) b2c::fail("--rerun-enabled is not supported by b2ctrain");
    if (cfg.render_mode != "default") b2c::fail("--render-mode %s is not supported by b2ctrain", cfg.render_mode.c_str());
    if (cfg.sh_degree > 4) b2c::fail("--sh-degree must be in 0..=4");
    return b2c::train_main(cfg);
  } catch (const std::exception& e) {
    fprintf(stderr, "error: %s\n", e.what());
    fflush(stderr);
    return 1;
  }
}
