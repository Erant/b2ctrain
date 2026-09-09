# b2ctrain: project state

Last updated: 2026-09-09 (hollow loss). Written for whoever (human or Claude) picks this up next.

## One-line summary

The trainer and the `render` subcommand work end to end, are validated against the brush fork (`~/Projects/brush`) on
both example datasets, run at **~3.9x brush's wall time** at equal or better quality on an RTX 4070 Ti, and now also
carry b2crunner's stage-5 alignment loop in-process (which takes the stage-5 step from 5m30s to ~2m50s). b2crunner is wired to it in the working tree (Dockerfile stage,
workflow params, doctor, step backend) but those changes are **not committed there** — see "What's left".

## Plan and contract

Full design plan: `~/.claude/plans/we-re-going-to-write-happy-locket.md`. The contract to satisfy is brush's own
CLI/dataset/PLY surface, documented in that plan's "The contract to satisfy" section and cross-checked against
`~/Projects/b2crunner/pipeline/steps/brush.py` and `pipeline/doctor.py`. Architecture and kernel-level design notes
live in `docs/design.md`; this file tracks status, not design. Source of truth on GitHub: `git@github.com:Erant/b2ctrain.git`
(branch `main`; b2crunner's Dockerfile pins it by commit, `B2CTRAIN_REF`).

## Phase status (plan's "Implementation phases")

| # | Phase | Status |
|---|---|---|
| 1 | Scaffold + IO | Done |
| 2 | Forward renderer | Done — parity with `brush-splat-render`: MAE ≈ 0.00000, worst pixel 0.027/255 |
| 3 | Backward + loss + optimizer | Done — validated by `tests/b2c_tests` against a double-precision CPU reference, 0/1156 mismatches |
| 4 | Refinement + schedules, full 30k runs | Done — see measured results below |
| 5 | Evidence + integration | Done. Evidence export matches brush column-by-column; `b2ctrain render --confidence` matches `brush-splat-render` to 1/255. **Both b2crunner training steps ran end to end through b2crunner's real `brush` step class** (`bench/b2crunner_step.py`) with b2ctrain as trainer and the shim as rasteriser: stage 2 cold + 9000-step polish (3m44s), stage 5 cold + 4-iteration alignment loop (5m30s through the pipeline loop, 2m38s with the loop in the trainer). `pipeline.cli doctor` passes |
| 6 | Performance | ~3.9x brush at the default recipe (target was 5x). Kernel-level work is in single digits now; see "Performance journey" |
| 7 | Docker / b2crunner | Written in b2crunner's working tree, uncommitted: `b2ctrain-builder` stage + runtime copies in `docker/Dockerfile`, `brush_path: b2ctrain` on both trainings, doctor check, `align_backend` on the step. Not yet built as an image or run on a pod |

## Measured results (RTX 4070 Ti, argv as b2crunner's steps pass it)

| dataset | brush fork | b2ctrain default (fp32 SH) | b2ctrain `--sh-fp16` (not default, see below) |
|---|---|---|---|
| stage 2 (135 views, 720p, normals, support views, alpha weight 0.5) | 9m41s / 837k / 35.83 dB | **2m29s / 871k / 36.96 dB** | 2m18s / 887k / 37.13 dB |
| stage 5 (81 views, 1080p, alpha weight 0.1, no normals) | 6m47s / 356k / 32.88 dB (older argv) | **1m40s / 364k / 34.23 dB** | 1m39s / 389k / 34.47 dB |

PSNR/SSIM are mask-weighted against the training frames (`bench/eval_ply.py`). The stage-5 brush row was measured
with a different argv on 2026-09-06 and is only indicative; the two b2ctrain columns are a clean A/B from today.
Training-view PSNR does not see everything: the fp16 column is higher there and yet its splats carry iridescent
colour speckle on the metallic top at novel views (below).

Full b2crunner step shapes (through `bench/b2crunner_step.py`, `brush_path` = b2ctrain):

| step | what runs | wall |
|---|---|---|
| `train_splat` (stage 2) | cold 30k + polish 9k from `init.ply` (growth off, refine never, full res) | 3m44s |
| `train_final_splat` (stage 5), pipeline loop | cold 30k + 4 x (render process, Python DIS flow + warp, re-invoke 3k) | 5m30s |
| `train_final_splat` (stage 5), in-trainer loop (`--align-iters 4`) | cold 30k + 4 x (GPU render + flow + warp 0.7 s, refit 3k 14 s) | **2m38s** (fp16 SH run; ~2m50s with the fp32 default) |

Alignment trajectories (mean measured disagreement, px): pipeline loop through b2ctrain 1.06 → 1.26 → 1.42 → 1.52;
in-trainer loop 1.47 → 1.64 → 1.75 → 1.83 (both rising and decelerating like the reference 1.02 → 1.31); PSNR of
the final splat against the pristine frames 32.24 vs 32.29 dB (fp32 default). On the same model and renders the in-trainer flow's
batch mean equals DIS's (1.48 vs 1.47 px) and its p90 is 11% higher (per-view correlation ~0.6: the two algorithms
disagree per view, agree in aggregate).

Per-step kernel breakdown at HEAD, warm 866k-splat stage-2 model (nsys, ms/step): optimizer 1.94, tensor-core
backward 1.70, forward raster 0.78, projection 0.56, radix sorts 0.38, photometric loss 0.27, intersection emit 0.27;
6.0 ms total against 6.07 ms wall (GPU idle ~1%).

### Polish and alignment, measured with b2crunner's own sharpness tool (2026-09-07)

`~/Downloads/refinesplat/tools/eval_splat.py`: band-limited face sharpness `s1` (Laplacian variance after a sigma-1
blur of the Sapiens2 face crop) at the training views and at interpolated novel views. Stage 2 crops use a fresh
Sapiens2 segmentation of its 81 frames.

| splat | face s1 train / novel | hand6 / hand15 s1 | PSNR to originals |
|---|---|---|---|
| stage 5 cold (fp32) | 22.0 / 21.1 (guide's brush cold reference: 21.1 / 20.0) | 12.8 / 13.8 | 34.23 |
| stage 5 + in-trainer alignment x4 | **23.8 / 23.0** (guide's brush reference after 4 iterations: 23.8) | 13.6 / 15.1 | 32.29 |
| stage 5 + alignment + 9000-step polish | 23.6 / 22.8 | 13.4 / 14.7 | 34.90 |
| stage 5 + alignment through the pipeline loop, pre-fix build | 21.3 / 20.7 | 12.2 / 13.9 | 32.24 |
| stage 2 cold | 26.9 / 26.5 | 22.0 / 18.8 | 37.00 |
| stage 2 + 3000-step polish | 26.8 / 26.5 | 22.3 / 18.9 | 36.85 |
| stage 2 + 9000-step polish (the workflow's) | 27.2 / 26.8 | 22.6 / 19.4 | 37.23 |

- The in-trainer alignment reproduces the guide's measured gain exactly (+1.8 face s1, the same at novel views).
- A polish after the alignment adds nothing (within noise on every part); it only pulls fidelity back toward the
  unwarped originals. Not worth adding to `train_final_splat`.
- The stage-2 polish is worth +1% face s1, +3% hands, +0.2 dB for ~1 minute — a small, real, cheap gain; brush's
  measured +12% raw face sharpness does not transfer because b2ctrain's cold run already lands above brush's polished
  result (raw face Laplacian variance 162 cold vs brush's 161 polished). 3000 steps does nothing; 9000 is the minimum.
- The pipeline-loop row is the splat trained before the warm-start fix: its refits ran through the 1/4 → 1/2
  resolution schedule and lost sharpness below the cold start. Warm starts now train at full resolution.

## Performance journey (what worked, in order)

1. 8x8 rasterizer tiles instead of 16x16 — cut backward fragment work ~3x for this scene's small splats.
2. Two-pass binning: depth-sort splats once, scan tile counts in that order, emit intersections keyed by tile id only.
3. GPU-side percentile bounds for the mean-LR/noise-clamp schedule.
4. Register/local-memory cleanup in the optimizer and projection kernels.
5. Fused per-fragment tensor-core backward (WMMA fp16 reduction against a per-pixel basis).
6. Lazy sparse Adam with closed-form catch-up for skipped steps.
7. Optimizer skips gradient-buffer traffic entirely for invisible splats.
8. Compact per-splat hit info (bitmask + packed bbox) for the emit pass.
9. Warp-cooperative coalesced intersection emit.
10. fp16 storage for SH bands >= 1 and their first moments (`ShBuf`, stochastically rounded parameter stores):
    optimizer 2.28 → 1.93 ms/step, projection-side forward 2.28 → 2.14; stage 2 2m29s → 2m18s at equal
    training-view PSNR. **Kept behind `--sh-fp16`, off by default** — see the next section.

### Tried and reverted (recorded so they aren't retried)
- **fp16 SH as the default**: the stochastic rounding is a ~0.5 ulp random walk per step on every high-band
  coefficient; over 40k steps that is 0.02-0.03 absolute near 0.3, invisible at the training views and visible as
  iridescent speckle on the metallic top at novel views (the user spotted it in a render; confirmed on an elevated
  synthetic view: high-frequency chroma RMS 1.57-1.62 for both fp16 splats vs 1.32-1.42 for fp32 ones, and a direct
  fp16/fp32 pair of the in-trainer alignment run). Performance must not cost visible quality, so fp32 is the default;
  the storage path stays for an error-compensated variant if one is ever worth it.
- Merging the loss kernel's 3 colour-channel blocks into one: worse occupancy, net slower.
- Splitting the SH Adam update into a lane-parallel kernel: extra global traffic outweighed the register relief.
- Doubling the tensor-core backward's MMA group size (16→32): more shared memory, lower occupancy, 1.77 → 2.24 ms.
- Replacing the backward's vectorised shared-memory zeroing pass with unconditional per-fragment fp16 stores:
  1.70 → 1.78 ms (64 scattered 2-byte stores per thread per group cost more than 8 uint4 stores).
- CUDA graphs: not tried because not worth it — nsys shows the GPU idle ~1% per step, so launch gaps are not a lever.
- Morton reordering of the splat arrays: not tried — the compaction is already in depth order and the optimizer
  streams the arrays linearly, so there is no gather it would help.

### The remaining lever
The full-resolution phase on the warm ~880k model is ~70% of stage-2 wall time. `--res-quarter-until 0.25
--res-half-until 0.6` (default 0.15 / 0.4) gives 1m59s at 37.00 dB — 4.9x brush — but 747k splats, 11% under
brush's 837k, and the sharpness metric b2crunner judges by was not measured. Left at the default; the knobs exist.

### Confidence gate: culled speckles along the face cap's rim (2026-09-07, evening)

A real run's re-render showed black (culled) speckles along the hairline, nose and jaw — the rim of the face cap, where
`face_priority_weights` ramps the denoised frames' weight and the cap's own mask fades. Reproduced on
`splat/colmap_intermediate` with the stage-2 export: 10% of the pixels in the weight-ramp band culled at the anchor
view, 1% of the face. Not a compositing problem: with every confidence term disabled the band stayed culled at
accumulated alpha 1.0. The cause is brush's support count (ported as-is): a view counted as supporting a splat only
when the splat drew >= 1 pixel-weight of in-mask mass in it, so 66% of the splats had zero supporting views and zero
confidence, and any pixel whose visibility is carried by such fine splats is culled. The ramp is where the fit is made
of fine splats (both sources attenuated, disagreement fitted in detail). `ev_views` is now the participation ratio of
the per-view mass, `(sum w)^2 / sum w^2` — an effective view count that is scale-free (1 for one view whatever the
mass, N for N equal views, a faint tail over many views barely moves it). Same 7-column `ev_*` block, so nothing in
b2crunner changes; the pipeline's `--conf-min-views 4` keeps its meaning. Measured (rationale and numbers in
`src/train/evidence.cu`): rim 10% -> 0%, face 1% -> 0%, subject 4.2% -> 3.1%; at a view 25 degrees above the orbit
the culled fraction of opaque pixels 3.4% -> 0.5% (the black patches on the specular top go too); far-outside
floater pixels kept 18 -> 36 per 720x1280 frame, i.e. unchanged in practice. Gradient tests pass (0/1156).

### False transparency: the hollow loss (2026-09-09)

The user reported that the trained splats are partly transparent: the front of a shirt is a half-opaque layer and
the back of the shirt shows through it, moving with parallax as the view tilts (static renders look fine; the
training orbit is reproduced either way, so nothing in the photometric loss opposes it). Two things were built:

- **`b2ctrain probe`** (`src/render/probe.cpp`, `src/gpu/probe.cu`): per pixel, the compositing weight arriving from
  more than `--delta` (3 cm) behind the first surface ("deep", first surface = accumulated alpha 0.1), and with a body
  mesh the weight arriving from more than `--margin` behind it ("behind"). Reported over covered pixels (alpha > 0.5)
  and over interior pixels (4 px from any silhouette) as probe.json plus heat maps. "deep" also counts legitimate
  layering (hair in front of the face, folds), so compare it between runs rather than reading it as an absolute.
- **The hollow loss** (`--hollow-weight`, `--hollow-margin`, `--hollow-dilate`, `--hollow-start-iter`, `--mesh`):
  the body proxy mesh (`src/dataset/mesh.cpp` reads .ply/.obj) is depth-rasterised for the training view every step
  (`src/gpu/meshdepth.cu`, one thread per triangle, atomicMin on float bits, then a farthest-in-window dilation so the
  reference at a silhouette or fold is the deeper surface), and each pixel is charged
  `sum_i vis_i * clamp((z_i - z_ref - margin) / margin, 0, 1)`. The gradient runs through the compositing weights only
  (the depth is a constant): a fragment behind the surface is cheapest to remove by making everything in front of it
  opaque, which is the intended fix, and nothing pulls the far side of the body forward through it. The growth
  statistic stays photometric so the regulariser does not spawn splats. Tested in `tests/b2c_tests` (three extra
  configurations against the double-precision reference; finite differences along the means are skipped there
  because of the constant-depth choice).

Measured on the example run's final-stage dataset (81 views, 1080p, `--match-alpha-weight 0.1`, no alignment, 4070
Ti) against a Poisson proxy of the SAM-3D-Body mesh rebuilt from the dataset's 10k surface samples (the pipeline will
pass the real mesh). Probe over every 8th training view; PSNR from `bench/eval_ply.py`:

| run | deep (mean) | px with deep > 0.2 | behind body (mean) | PSNR | splats | time |
|---|---|---|---|---|---|---|
| baseline | 0.146 | 27.1% | 0.038 | 35.46 | 305k | 1m32s |
| hollow 0.5, margin 0.05 | 0.102 | 17.8% | 0.0025 | 35.39 | 312k | 1m47s |
| hollow 2.0, margin 0.05 | 0.092 | 16.3% | 0.0005 | 35.25 | 320k | 1m39s |
| hollow 0.5, margin 0.03 | 0.088 | 15.7% | 0.0008 | 35.31 | 318k | 1m42s |

(The 0.5 rows were re-measured after the growth statistic was made photometric-only; the 2.0 row predates that,
which only changes which splats the growth samples.) Run to run the step cost is 5-15%.

With the production argv (four alignment refits, hollow on through them): behind 0.0021, deep 0.094, PSNR against
the pristine frames 33.26 vs 33.30 for the delivered aligned splat (alignment lowers that number by design),
step 2m39s vs 2m38s. `out/hollow/before_after_pan-10.png` is the side-by-side.

Stage 2 (117 views at 720p with masks, normals, support views, `--match-alpha-weight 0.5`) leaks less to begin
with and moves the same way: behind 0.0066 -> 0.0020, deep 0.068 -> 0.063, PSNR 37.18 -> 37.23, 369k -> 385k splats,
1m33s -> 1m44s.

Off-orbit (elevated ±25°) views move the same way: deep 0.149 -> 0.113, behind 0.033 -> 0.003 (0.5 / 0.05). The
remaining "deep" weight sits on hair in front of the face and inside the margin band. The delivered splat from the
run (`ply/scene.ply`, aligned) measured deep 0.135 / behind 0.035, the same as the baseline retrain.

b2crunner side (working tree, uncommitted): `render.py` publishes the oriented mesh as `mesh` (`scene.mesh_world` in
the workflow, the same frame as `points_3d`), `steps/brush.py` writes it as `mesh.ply` beside the COLMAP model and
passes the flags (`hollow_weight` / `hollow_margin` / `hollow_dilate` params; `tests/test_brush_hollow.py`), the
doctor requires `--hollow-weight`, and both trainings in `fast_helical_native.yaml` set `hollow_weight: 0.5`.
Splats to look at: `out/hollow/` (baseline, the three settings, the proxy mesh, camera sets).

## What's left

- **Commit the b2crunner side.** Its working tree (`~/Projects/b2crunner`) has uncommitted changes from this work in
  `docker/Dockerfile`, `pipeline/workflows/fast_helical_native.yaml`, `pipeline/doctor.py`, `pipeline/steps/brush.py`,
  `tests/test_brush_alignment.py`, `docs/docker.md`, `README.md` — alongside unrelated uncommitted face-priority edits
  that were already there. Its test suite passes (the brush/workflow/docker/doctor files: 168 tests). The Dockerfile's
  `B2CTRAIN_REF` pin must point at the b2ctrain commit that is pushed.
- **Build the image and run a pod.** The b2ctrain stage has not been built inside Docker yet (the binary needs CUDA 13
  at build time and driver >= 580 at run time), and no b2crunner pipeline has run end to end on a pod with the swap.
- **Sharpness check of the in-trainer alignment.** PSNR, the trajectory shape and novel-view chroma noise (1.42 vs
  1.32 on an elevated view, fp32 both) match the pipeline loop; b2crunner's band-limited face-sharpness metric
  (docs/final-splat-alignment-guide.md there) is what the loop was tuned on and has not been run on the in-trainer result.
- **fp16 SH without the noise**, if the 8% is ever wanted back: error-compensated rounding (carry the rounding residual)
  rather than stochastic rounding, or fp16 moments only.
- Everything the plan marked out of scope for milestone 1 is still out of scope: viewer/GUI, LOD baking, LPIPS,
  rerun logging, nerfstudio/RealityCapture loaders, non-pinhole cameras, `--render-mode mip`, random frustum init,
  multi-GPU.

## Repo map

```
src/cli.*              brush-compatible argument parser (+ --sh-fp16 (opt-in), --res-*-until, --align-*)
src/dataset/            COLMAP + sidecar loader (masks/normals/weights/init.ply)
src/ply.*               PLY reader/writer (brush field order + ev_* evidence block)
src/model.*             GPU splat parameter/moment buffers (SH bands >= 1 optionally fp16 via ShBuf in gpu/util.cuh)
src/gpu/                CUDA kernels: project, binning, raster fwd/bwd (+ tensor-core variant), loss, optim
                        (fused projection-backward + Adam + noise), refine, images (resolution pyramid),
                        align (LK flow, Gaussian blur, Lanczos warp for the alignment loop)
src/train/              trainer (phases: main run + alignment refits), GPU-resident views, splat init, evidence
src/render/             `b2ctrain render` subcommand + confidence model
tests/                  finite-difference gradient test against a double-precision CPU reference; flow/warp tests
bench/                  eval_ply.py (mask-weighted PSNR/SSIM), colmap_to_cameras.py, compare_images.py,
                        polish_test.sh, b2crunner_step.py (drive b2crunner's real brush step with b2ctrain)
docker/                 Dockerfile stage (reference copy; the live one is in b2crunner) + brush-splat-render shim
docs/design.md          kernel-level architecture notes and the numbers table
docs/STATUS.md          this file
```

## How to reproduce the numbers

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/b2c_tests   # gradient check ("0 / N mismatches") and the flow/warp tests

# stage 2, as train_splat passes it:
./build/b2ctrain splat/colmap_intermediate --total-train-iters 30000 --sh-degree 3 \
  --export-path out --export-name export.ply --export-every 30000 --max-resolution 1920 \
  --max-splats 10000000 --refine-every 200 --match-alpha-weight 0.5 --normalize-masked-loss \
  --normal-loss-weight 0.05 --normal-loss-start-iter 5000 --normal-loss-every 1 --export-evidence
python3 bench/eval_ply.py out/export.ply splat/colmap_intermediate --every 9

# stage 5 with the alignment loop in the trainer, as train_final_splat passes it:
./build/b2ctrain splat/colmap --total-train-iters 30000 --sh-degree 3 --export-path out5 --export-name scene.ply \
  --export-every 30000 --max-resolution 1920 --max-splats 10000000 --refine-every 200 --match-alpha-weight 0.1 \
  --export-evidence --align-iters 4 --align-steps 3000 --align-debug-dir out5/alignment

# splats from these runs (not in git): out/2026-09-07/*.ply, with the alignment debug files beside them

# the same two through b2crunner's own step class (needs b2crunner's venv):
~/Projects/b2crunner/.venv/bin/python bench/b2crunner_step.py splat/colmap_intermediate out2 --polish-steps 9000 --align-iters 0 --match-alpha-weight 0.5
~/Projects/b2crunner/.venv/bin/python bench/b2crunner_step.py splat/colmap out5 --align-iters 4 --no-normals --normal-loss-strength 0
```

brush fork build for comparison: `~/Projects/brush`, release binary at `target/release/brush`
(`cargo build --release -p brush-app -p brush-splat-render`).
