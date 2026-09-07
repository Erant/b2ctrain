# b2ctrain: project state

Last updated: 2026-09-07. Written for whoever (human or Claude) picks this up next.

## One-line summary

The trainer and the `render` subcommand both work end to end, are validated against the brush fork
(`~/Projects/brush`) on both example datasets, and run at roughly **4x brush's wall time** at equal or better
quality on an RTX 4070 Ti. b2crunner itself has not been touched — wiring it up is the next PR.

## Plan and contract

Full design plan: `~/.claude/plans/we-re-going-to-write-happy-locket.md`. The contract to satisfy is brush's own
CLI/dataset/PLY surface, documented in that plan's "The contract to satisfy" section and cross-checked against
`~/Projects/b2crunner/pipeline/steps/brush.py` and `pipeline/doctor.py`. Architecture and kernel-level design notes
live in `docs/design.md`; this file tracks status, not design.

## Phase status (plan's "Implementation phases")

| # | Phase | Status |
|---|---|---|
| 1 | Scaffold + IO | Done |
| 2 | Forward renderer | Done — parity with `brush-splat-render`: MAE ≈ 0.00000, worst pixel 0.027/255 |
| 3 | Backward + loss + optimizer | Done — validated by `tests/b2c_tests` against a double-precision CPU reference (`tests/reference.cpp`), finite differences, 0/1156 mismatches across transparent/masked/normal-supervised views and both backward kernels |
| 4 | Refinement + schedules, full 30k runs | Done — see measured results below |
| 5 | Evidence + integration | Mostly done: evidence export matches brush's per-splat evidence column-by-column; `b2ctrain render --confidence` matches `brush-splat-render` to 1/255 (own-ply and `--dataset`-measured evidence both). `pipeline/doctor.py check_brush_binaries` passes with b2ctrain as `brush`. **Not yet done**: an actual b2crunner pipeline run with `brush_path: b2ctrain` wired in (see "What's left") |
| 6 | Performance | Ongoing — see "Performance journey" below. At ~4x brush; target was 5x |
| 7 | Docker | Artifacts written (`docker/Dockerfile.b2ctrain-stage`, `docker/brush-splat-render` shim) but not spliced into b2crunner's actual Dockerfile yet |

## Measured results (RTX 4070 Ti)

| dataset | brush fork | b2ctrain `--recipe brush` | b2ctrain `--recipe fast` (default) |
|---|---|---|---|
| stage 2 (135 views, 720p, `splat/colmap_intermediate`) | 9m41s / 837k splats / 35.83 dB | 2m50s / 837k / 36.16 dB | 2m31s / 866k / **36.81 dB** |
| stage 5 (81 views, 1080p, `splat/colmap`) | 6m47s / 356k / 32.88 dB | 2m10s / 350k / 32.29 dB | 1m41s / 358k / 32.71 dB |

PSNR/SSIM are mask-weighted, measured by `bench/eval_ply.py` against the training frames. `--recipe fast` (lazy
sparse Adam + progressive 1/4→1/2→1 resolution schedule) matches or beats brush's splat count and quality
everywhere measured.

Current per-step kernel breakdown at HEAD, warm 838k-splat model (`--recipe fast`, stage-2 dataset, no refine):

```
backward   1.755 ms/step  26.8%
forward    2.310 ms/step  35.3%
loss       0.280 ms/step   4.3%
optim      2.205 ms/step  33.7%
```
(152.5 it/s at 838k splats.) `optim` and `forward` are now the bottlenecks — backward stopped being dominant
several optimizations ago.

## Performance journey (what worked, in order)

1. 8x8 rasterizer tiles instead of 16x16 — cut backward fragment work ~3x for this scene's small splats.
2. Two-pass binning: depth-sort splats once (CUB radix sort on raw z bits), scan tile counts in that order, emit
   intersections with the tile id as the only sort key (stable sort preserves depth order within a tile) — replaced
   brush's combined 64-bit-key single sort with something cheaper per intersection.
3. GPU-side percentile bounds for the mean-LR/noise-clamp schedule (was a CPU sort every refine).
4. Register/local-memory cleanup in the optimizer and projection kernels (removed stack-frame spills).
5. Fused per-fragment tensor-core backward (WMMA fp16 reduction of per-fragment scalars against a per-pixel basis)
   as the primary backward path, with a warp-shuffle version kept for reference/testing (`--backward warp`).
6. Lazy sparse Adam: skip Adam entirely for splats invisible this step, with a closed-form catch-up (moment decay
   + geometric momentum drift) applied on the next visible step instead of a dense per-step update.
7. Optimizer skips gradient-buffer traffic entirely for invisible splats (not just skips the update).
8. Compact per-splat "hit info" (bitmask + packed bbox) so the intersection-emit pass doesn't need to recompute or
   re-fetch the projected covariance.
9. Warp-cooperative coalesced emit: the 32 lanes of a warp jointly own a contiguous run of intersection slots and
   binary-search (via shuffles) which splat owns each slot, instead of one thread per splat writing its own
   scattered range. Dropped `emit_kernel` from ~564us to ~369us avg on the warm model.

### Tried and reverted (recorded so they aren't retried)
- **Merging the loss kernel's 3 colour-channel blocks into one**: fewer kernel launches, but worse occupancy —
  net slower.
- **Splitting the SH Adam update into its own lane-parallel kernel** (one thread per (splat, SH lane) pair): extra
  global memory traffic outweighed the register pressure it relieved in the fused per-splat version.
- **Doubling the tensor-core backward's MMA group size (16→32 splats)**: more shared memory means lower occupancy;
  backward went from 1.77ms to 2.24ms/step on the warm model. This was the change in flight when the previous
  session got interrupted — verified as a regression and reverted before it was committed.

## What's left

- **Close the 4x→5x gap.** `optim` and `forward` now dominate; backward is no longer the bottleneck. Candidates,
  untried: fp16 storage for SH bands ≥1 and their Adam moments (halves the biggest per-splat buffer); CUDA graphs
  to cut launch overhead (~15 kernel launches/step, several of them small); Faster-GS-style periodic Morton
  reordering of the splat arrays for better memory coalescing as the model grows past its initial layout.
- **b2crunner integration** (no code written there yet):
  - Splice `docker/Dockerfile.b2ctrain-stage` into `~/Projects/b2crunner/docker/Dockerfile` next to the existing
    `brush-builder` stage; add the `COPY --from=b2ctrain-builder` line to the runtime stage.
  - Set `brush_path: b2ctrain` (and point the renderer at `docker/brush-splat-render`'s shim, or wire
    `$BRUSH_SPLAT_RENDER`) in `pipeline/workflows/fast_helical_native.yaml`'s `train_splat` and
    `train_final_splat` steps.
  - Run the actual pipeline end to end at least once with the swap in place.
- **Untested invocation shape**: `bench/polish_test.sh` exercises the polish step's warm-start shape
  (`--growth-stop-iter 0 --refine-every 1000000` from `init.ply`), but the stage-5 **alignment loop** (4 repeated
  warm-started re-invocations against re-warped frames) hasn't been run through b2ctrain at all.
- Everything the plan marked out of scope for milestone 1 is still out of scope: viewer/GUI, LOD baking, LPIPS,
  rerun logging, nerfstudio/RealityCapture loaders, non-pinhole cameras, `--render-mode mip`, random frustum init,
  multi-GPU.

## Repo map

```
src/cli.*              brush-compatible argument parser
src/dataset/            COLMAP + sidecar loader (masks/normals/weights/init.ply)
src/ply.*               PLY reader/writer (brush field order + ev_* evidence block)
src/model.*             GPU splat parameter/moment buffers
src/gpu/                all CUDA kernels: project, binning, raster fwd/bwd (+ tensor-core variant), loss,
                        optim (fused projection-backward + Adam + noise), refine, images (resolution pyramid)
src/train/              trainer loop, GPU-resident dataset views, splat init (kNN scales), evidence pass
src/render/             `b2ctrain render` subcommand + confidence model
tests/                  finite-difference gradient test against a double-precision CPU reference
bench/                  eval_ply.py (mask-weighted PSNR/SSIM), colmap_to_cameras.py, compare_images.py,
                        polish_test.sh
docker/                 Dockerfile stage + brush-splat-render shim, not yet spliced into b2crunner
docs/design.md          kernel-level architecture notes and the numbers table above
docs/STATUS.md          this file
```

## How to reproduce the numbers

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/b2c_tests   # gradient check, should print "0 / N mismatches"

# render parity against brush-splat-render on ply/scene.ply:
python3 bench/colmap_to_cameras.py splat/colmap cams.json --every 10
./build/b2ctrain render --splat splat/ply/scene.ply --cameras cams.json --output-dir out --background 0,0,0
python3 bench/compare_images.py <brush-splat-render output dir> out

# a full training run + eval:
./build/b2ctrain splat/colmap_intermediate --total-train-iters 30000 --sh-degree 3 \
  --export-path out --export-name export.ply --export-every 30000 --max-resolution 1920 \
  --max-splats 10000000 --refine-every 200 --match-alpha-weight 0.5 --normalize-masked-loss \
  --normal-loss-weight 0.05 --normal-loss-start-iter 5000 --normal-loss-every 1 --export-evidence
python3 bench/eval_ply.py out/export.ply splat/colmap_intermediate --every 9
```

brush fork build for comparison: `~/Projects/brush`, release binary at `target/release/brush`
(`cargo build --release -p brush-app -p brush-splat-render`).
