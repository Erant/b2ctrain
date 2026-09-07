# b2ctrain design notes

## Data layout
- Splat parameters live in structure-of-float4 arrays (`pos_op`, `quat`, `lscale`) plus a planar SH buffer
  `[K*3][cap]` so the projection and optimizer kernels read coefficients coalesced across threads. `lscale.w` holds the
  Mip-Splatting 3D-filter floor `f` (a frozen constant; `--recipe brush` bakes it into the scales at every refine, as the
  fork does; `--recipe fast` applies it on the fly and bakes only at export).
- Adam moments mirror the parameter layout; the SH second moment is one scalar per splat (brush's Adam-mini style).
- All training views are GPU resident as packed RGBA8 (premultiplied for transparent views), RGBA8 normals and u8 weights,
  plus 1/2 and 1/4 box-filtered levels when the resolution schedule is on.

## One training step (`src/train/trainer.cpp`)
1. `project_kernel` (1 thread/splat): view transform, EWA covariance with brush's Jacobian clamp and 0.3 px blur,
   opacity-adaptive extent `sqrt(2 ln(255a) sigma)`, exact ellipse-tile count over 8x8 tiles, SH colour, optional
   pseudo-normal feature. Stashes `(xy, conic, opacity, depth, power)` and colour/feature per splat.
2. Binning: splats are depth-sorted once (32-bit z bits, CUB), tile counts scanned in that order, intersections emitted
   with the tile id as key and sorted with a stable 2-pass radix sort, so each tile's list is in exact depth order.
   One 4-byte host readback (the intersection total) per step.
3. `raster_fwd_kernel` (block = 8x8 tile): batched shared loads, brush's compositing rules (alpha clamp 0.999, cutoff 1/255,
   T cutoff 1e-4), per-pixel last-contributor index for the backward; tiles with no intersections write the background.
4. `photometric_kernel` (block = 16x16 tile x channel): recompute-based fused L1+SSIM with the SSIM gradient expressed
   as five blurred partial maps, alpha lane, mask / weight / coverage scaling, and an exact early exit for tiles whose
   36x36 footprint has prediction == ground truth (zero gradient). `normal_loss_kernel` adds L1 + (1 - cos).
5. `raster_bwd_tc_kernel`: per-pixel reverse replay recovers T and the composited remainder; for each group of 16 splats
   the per-fragment scalars (vis, v_alpha*gauss, v_sigma, refine) are written as fp16 and reduced over the 64 pixels with
   WMMA against a per-pixel basis `[v_rgb, v_feat, 1, u, v, u^2, uv, v^2]` in tile-centred half-pixel coordinates (exact
   in fp16). Conic / mean gradients are recovered from the moments. Gradients are pre-scaled by `3WH` in the loss kernels
   to keep fp16 in range. `--backward warp` keeps the shuffle-reduction reference.
6. `optim_kernel` (1 thread/splat): projection backward (conic -> covariance -> J, R, S; SH basis and view-direction term;
   opacity through sigmoid, floor compensation and scale fold; quaternion normalisation VJP), Adam with brush's constants
   and schedules, MCMC noise, refine-statistics bookkeeping. `--sparse-adam` (fast recipe) skips invisible splats.
7. Every `refine_every` steps `RefineState::run` prunes, relocates dead slots and grows by Gumbel-top-k weighted sampling
   (equivalent in distribution to brush's multinomial without replacement), splits with brush's covariance-aware rule,
   decays opacity, recomputes the 80th-percentile bounds and the floor on the GPU.

## Measured on the RTX 4070 Ti (2026-09-06)
| dataset | brush fork | b2ctrain `--recipe brush` | b2ctrain `--recipe fast` |
|---|---|---|---|
| stage 2 (135 views 720p, normals, mixed alpha) | 9m41s, 837k, 35.83 dB | 2m50s, 837k, 36.16 dB | 2m28s, 854k, 36.75 dB |
| stage 5 (81 views 1080p) | 6m47s, 356k, 32.88 dB | 2m10s, 350k, 32.29 dB | 1m39s, 358k, 32.71 dB |

Lessons: 8x8 raster tiles cut fragment work ~3x for tiny-splat scenes; the optimizer is DRAM-bound (~1.2 KB per visible
splat per step); a lane-parallel SH update was slower than the fused per-splat kernel; merging the loss channels into one
block was slower than three blocks (occupancy).

## Validation
- `tests/b2c_tests`: gradients of every parameter group against central differences of a double-precision CPU
  re-implementation of the forward + loss (`tests/reference.cpp`), for transparent, masked and normal-supervised views,
  with both backward kernels. Non-smooth points (alpha cutoff, normal axis flips) are detected and skipped.
- Render parity against `brush-splat-render` (plain and `--confidence`, evidence from the ply or measured from `--dataset`)
  is at the 1/255 level.
- `pipeline/doctor.py`'s brush check passes with `b2ctrain` on PATH as `brush`.
