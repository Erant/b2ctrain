// `b2ctrain lod-merge` and `b2ctrain lod-distill` (b2cgltf_lod_proposal.md, "b2ctrain: two subcommands").
#include "lod/merge.h"
#include "cli.h"
#include "gpu/cage.h"
#include "train/trainer.h"
#include "util/log.h"
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace b2c {
namespace fs = std::filesystem;

namespace {

const char* MERGE_HELP =
"Merge a splat down to an exact count: a LOD level (b2cgltf_lod_proposal.md 4.2)\n\n"
"A port of splat-transform 3.10.1's decimate (PlayCanvas, MIT): 16 nearest neighbours, an edge cost to each,\n"
"greedy disjoint grouping of up to 4 in cost order, moment-matched merging; generations halve the count until it\n"
"is reached exactly. Only splats of the same seg_label merge (the input's labels, as given); a merged splat's\n"
"seg_conf is its group's minimum. Evidence and b2crig per-splat state (open_*, cage_fill, cage_gate_*) are dropped.\n\n"
"Usage: b2ctrain lod-merge --splat PLY --count N --out PLY [OPTIONS]\n\n"
"Options:\n"
"      --splat PLY              The splat to merge (a trainer ply, with seg_label / seg_conf when it has them)\n"
"      --count N                Splats in the output, exactly\n"
"      --out PLY                The merged level (a trainer ply)\n"
"      --cage CAGE              Distance cap (b2crig's cage file, gpu/cage.h): a merge whose extent (3 sigma along its\n"
"                               major axis) exceeds K x the mean edge length of the cage face nearest its mean (the\n"
"                               face the binding would pick: the layer that owns the class, else any) is not made\n"
"      --max-extent-faces K     The cap's K; 0 = no cap [default: 3]\n"
"      --cage-min-conf C        A merge whose group-minimum seg_conf is under C measures against the nearest face of\n"
"                               any layer, as the binding does (pass the binding's value) [default: 0.5]\n";

const char* DISTILL_HELP =
"Fine-tune a merged LOD level against the full splat's renders\n\n"
"A warm start from --splat with a fresh optimiser; every training view's target is the teacher rendered at that\n"
"view (canonical, no rig, with its alpha), composited on a 0.5 grey background with 0.5 noise. Growth and\n"
"oversized splits are off; relocation stays on, so the count holds. The dataset's weights/ sidecar applies as usual.\n"
"seg_label / seg_conf of --splat are carried to the output (relocated splats take their parent's).\n\n"
"Usage: b2ctrain lod-distill --splat PLY --teacher PLY --dataset DIR --out PLY [OPTIONS]\n\n"
"Options:\n"
"      --splat PLY              The level to fine-tune (lod-merge's output)\n"
"      --teacher PLY            The full splat the level was merged from\n"
"      --dataset DIR            COLMAP text dataset whose cameras are the views (its images are not used)\n"
"      --out PLY                The fine-tuned level (may be --splat itself)\n"
"      --steps N                [default: 6000]\n"
"      --max-resolution N       [default: 1920]\n"
"      --seed N  --device N\n";

}  // namespace

int lod_merge_main(int argc, char** argv) {
  std::string splat, out, cage_path;
  float min_conf = 0.5f;
  long long count = -1; float k = 3.f;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    auto val = [&]() -> std::string { if (i + 1 >= argc) fail("a value is required for '%s'", a.c_str()); return argv[++i]; };
    if (a == "-h" || a == "--help") { fputs(MERGE_HELP, stdout); return 0; }
    else if (a == "--splat") splat = val();
    else if (a == "--count") count = atoll(val().c_str());
    else if (a == "--out") out = val();
    else if (a == "--cage") cage_path = val();
    else if (a == "--max-extent-faces") k = (float)atof(val().c_str());
    else if (a == "--cage-min-conf") min_conf = (float)atof(val().c_str());
    else fail("lod-merge: unexpected argument '%s'", a.c_str());
  }
  if (splat.empty() || out.empty() || count < 1) fail("lod-merge needs --splat, --count (>= 1) and --out (see --help)");
  double t0 = now_seconds();
  SplatCloud in = read_ply(splat);
  log_info("Loaded %zu splats (SH degree %d) from %s%s", in.n, in.sh_degree, splat.c_str(), in.has_labels ? "" : "; it has no seg_label, so every splat is one class");
  if ((size_t)count >= in.n) fail("lod-merge: --count %lld is not below the splat's %zu", count, in.n);
  if (in.has_evidence || in.has_fill || in.has_gate || in.has_open)
    log_info("Dropped per-splat state: %s%s%s%s", in.has_evidence ? "evidence " : "", in.has_fill ? "cage_fill " : "", in.has_gate ? "cage_gate_* " : "", in.has_open ? "open_* " : "");

  MergeParams p; p.count = (size_t)count; p.max_extent_faces = k; p.min_conf = min_conf;
  CageFaces cf;
  if (!cage_path.empty() && k > 0.f) {
    CageRig cage;
    if (!cage.load(cage_path)) fail("--cage %s: cannot open", cage_path.c_str());
    auto v = cage.verts0.download(cage.nv); auto f = cage.faces.download(cage.nf);
    for (auto& x : v) cf.verts.insert(cf.verts.end(), {x.x, x.y, x.z});
    for (auto& x : f) cf.faces.insert(cf.faces.end(), {x.x, x.y, x.z});
    cf.face_layer = cage.face_layer_h;
    for (auto& L : cage.layers) { cf.layer_classes.push_back(L.class_bits); cf.layer_v_off.push_back(L.v_off); cf.layer_v_count.push_back(L.v_count); }
    p.cage = &cf;
    log_info("Cage cap: %s, %zu layers, %d triangles; a merge may span at most %g x the mean edge of the face it would bind to (min conf %g)", cage_path.c_str(), cage.layers.size(), cage.nf, k, min_conf);
  } else if (!cage_path.empty()) {
    log_info("--max-extent-faces 0: the cage cap is off");
  }
  MergeStats st;
  SplatCloud lvl = lod_merge(in, p, &st);
  std::vector<std::string> comments;
  for (auto& c : read_ply_comments(splat)) if (c.rfind("b2c.cage_open ", 0) != 0) comments.push_back(c);
  comments.push_back(format("lod-merge: %zu -> %zu splats%s (b2ctrain; splat-transform 3.10.1 decimate port)", in.n, lvl.n,
                            p.cage ? format(", cage cap K=%g", k).c_str() : ""));
  write_ply(out, lvl, comments);
  log_info("lod-merge: %zu -> %zu splats in %d generation(s), %zu class(es); %s; wrote %s in %.2fs", in.n, lvl.n, st.generations, st.classes,
           p.cage ? format("the cage cap ruled out %zu merges, and kept %zu splats whole in the last generation", st.cap_blocked, st.cap_held).c_str() : "no cage cap", out.c_str(), now_seconds() - t0);
  return 0;
}

int lod_distill_main(int argc, char** argv) {
  Config cfg;
  std::string out;
  cfg.total_train_iters = 6000;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    auto val = [&]() -> std::string { if (i + 1 >= argc) fail("a value is required for '%s'", a.c_str()); return argv[++i]; };
    if (a == "-h" || a == "--help") { fputs(DISTILL_HELP, stdout); return 0; }
    else if (a == "--splat") cfg.init_ply = val();
    else if (a == "--teacher") cfg.teacher = val();
    else if (a == "--dataset") cfg.source = val();
    else if (a == "--out") out = val();
    else if (a == "--steps") cfg.total_train_iters = (uint32_t)atoi(val().c_str());
    else if (a == "--max-resolution") cfg.max_resolution = (uint32_t)atoi(val().c_str());
    else if (a == "--seed") cfg.seed = (uint64_t)atoll(val().c_str());
    else if (a == "--device") cfg.device = atoi(val().c_str());
    else fail("lod-distill: unexpected argument '%s'", a.c_str());
  }
  if (cfg.init_ply.empty() || cfg.teacher.empty() || cfg.source.empty() || out.empty()) fail("lod-distill needs --splat, --teacher, --dataset and --out (see --help)");
  if (cfg.total_train_iters == 0) fail("--steps must be at least 1");
  // The settings of the LOD experiment (b2crunner experiments/lod/lod.py, b2crig cage_train's distill flags).
  cfg.sh_degree = (uint32_t)read_ply(cfg.init_ply).sh_degree;
  cfg.growth_stop_iter = 0; cfg.split_at_screen_size = 0.f;
  for (float& b : cfg.background_color) b = 0.5f;
  cfg.background_noise_strength = 0.5f;
  const fs::path o = fs::absolute(out);
  cfg.export_path = o.parent_path().string(); cfg.export_name = o.filename().string();
  cfg.export_every = 0; cfg.eval_every = 0;
  return train_main(cfg);
}

}  // namespace b2c
