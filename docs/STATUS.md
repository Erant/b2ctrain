# b2ctrain: project state

Last updated: 2026-09-07 (second session). Written for whoever (human or Claude) picks this up next.

## One-line summary

The trainer and the `render` subcommand work end to end, are validated against the brush fork (`~/Projects/brush`) on
both example datasets, run at **4.2x brush's wall time** at equal or better quality on an RTX 4070 Ti, and now also
carry b2crunner's stage-5 alignment loop in-process. b2crunner is wired to it in the working tree (Dockerfile stage,
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
| 6 | Performance | ~4.2x brush at the default recipe (target was 5x). Kernel-level work is in single digits now; see "Performance journey" |
| 7 | Docker / b2crunner | Written in b2crunner's working tree, uncommitted: `b2ctrain-builder` stage + runtime copies in `docker/Dockerfile`, `brush_path: b2ctrain` on both trainings, doctor check, `align_backend` on the step. Not yet built as an image or run on a pod |

## Measured results (RTX 4070 Ti, argv as b2crunner's steps pass it)

| dataset | brush fork | b2ctrain `--sh-fp32` | b2ctrain default (fp16 SH) |
|---|---|---|---|
| stage 2 (135 views, 720p, normals, support views, alpha weight 0.5) | 9m41s / 837k / 35.83 dB | 2m29s / 871k / 36.96 dB | **2m18s / 887k / 37.13 dB** |
| stage 5 (81 views, 1080p, alpha weight 0.1, no normals) | 6m47s / 356k / 32.88 dB (older argv) | 1m40s / 364k / 34.23 dB | **1m39s / 389k / 34.47 dB** |

PSNR/SSIM are mask-weighted against the training frames (`bench/eval_ply.py`). The stage-5 brush row was measured
with a different argv on 2026-09-06 and is only indicative; the two b2ctrain columns are a clean A/B from today.

Full b2crunner step shapes (through `bench/b2crunner_step.py`, `brush_path` = b2ctrain):

| step | what runs | wall |
|---|---|---|
| `train_splat` (stage 2) | cold 30k + polish 9k from `init.ply` (growth off, refine never, full res) | 3m44s |
| `train_final_splat` (stage 5), pipeline loop | cold 30k + 4 x (render process, Python DIS flow + warp, re-invoke 3k) | 5m30s |
| `train_final_splat` (stage 5), in-trainer loop (`--align-iters 4`) | cold 30k + 4 x (GPU render + flow + warp 0.7 s, refit 3k 14 s) | **2m38s** |

Alignment trajectories (mean measured disagreement, px): pipeline loop through b2ctrain 1.06 → 1.26 → 1.42 → 1.52;
in-trainer loop 1.43 → 1.62 → 1.74 → 1.83 (both rising and decelerating like the reference 1.02 → 1.31); PSNR of
the final splat against the pristine frames 32.24 vs 32.39 dB. On the same model and renders the in-trainer flow's
batch mean equals DIS's (1.48 vs 1.47 px) and its p90 is 11% higher (per-view correlation ~0.6: the two algorithms
disagree per view, agree in aggregate).

Per-step kernel breakdown at HEAD, warm 866k-splat stage-2 model (nsys, ms/step): optimizer 1.94, tensor-core
backward 1.70, forward raster 0.78, projection 0.56, radix sorts 0.38, photometric loss 0.27, intersection emit 0.27;
6.0 ms total against 6.07 ms wall (GPU idle ~1%).

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
10. **fp16 storage for SH bands >= 1 and their first moments** (`ShBuf`, stochastically rounded parameter stores):
    optimizer 2.28 → 1.93 ms/step, projection-side forward 2.28 → 2.14; stage 2 2m29s → 2m18s at equal quality.

### Tried and reverted (recorded so they aren't retried)
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

## What's left

- **Commit the b2crunner side.** Its working tree (`~/Projects/b2crunner`) has uncommitted changes from this work in
  `docker/Dockerfile`, `pipeline/workflows/fast_helical_native.yaml`, `pipeline/doctor.py`, `pipeline/steps/brush.py`,
  `tests/test_brush_alignment.py`, `docs/docker.md`, `README.md` — alongside unrelated uncommitted face-priority edits
  that were already there. Its test suite passes (the brush/workflow/docker/doctor files: 168 tests). The Dockerfile's
  `B2CTRAIN_REF` pin must point at the b2ctrain commit that is pushed.
- **Build the image and run a pod.** The b2ctrain stage has not been built inside Docker yet (the binary needs CUDA 13
  at build time and driver >= 580 at run time), and no b2crunner pipeline has run end to end on a pod with the swap.
- **Sharpness check of the in-trainer alignment.** PSNR and the trajectory shape match the pipeline loop; b2crunner's
  band-limited face-sharpness metric (docs/final-splat-alignment-guide.md there) is what the loop was tuned on and has
  not been run on the in-trainer result.
- Everything the plan marked out of scope for milestone 1 is still out of scope: viewer/GUI, LOD baking, LPIPS,
  rerun logging, nerfstudio/RealityCapture loaders, non-pinhole cameras, `--render-mode mip`, random frustum init,
  multi-GPU.

## Repo map

```
src/cli.*              brush-compatible argument parser (+ --sh-fp16/--sh-fp32, --res-*-until, --align-*)
src/dataset/            COLMAP + sidecar loader (masks/normals/weights/init.ply)
src/ply.*               PLY reader/writer (brush field order + ev_* evidence block)
src/model.*             GPU splat parameter/moment buffers (SH bands >= 1 fp16 via ShBuf in gpu/util.cuh)
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

# the same two through b2crunner's own step class (needs b2crunner's venv):
~/Projects/b2crunner/.venv/bin/python bench/b2crunner_step.py splat/colmap_intermediate out2 --polish-steps 9000 --align-iters 0 --match-alpha-weight 0.5
~/Projects/b2crunner/.venv/bin/python bench/b2crunner_step.py splat/colmap out5 --align-iters 4 --no-normals --normal-loss-strength 0
```

brush fork build for comparison: `~/Projects/brush`, release binary at `target/release/brush`
(`cargo build --release -p brush-app -p brush-splat-render`).
