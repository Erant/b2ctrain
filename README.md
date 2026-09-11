# b2ctrain

A from-scratch CUDA Gaussian-splat trainer for the b2crunner pipeline, written as a drop-in replacement for the
[brush](https://github.com/Erant/brush) fork's `brush` binary (same command line, dataset layout, PLY output and
`ev_*` evidence block) and for `brush-splat-render` (`b2ctrain render`).

Targets NVIDIA Ada / Blackwell (sm_89, sm_120). Builds against CUDA 13 (12.8+ works) with CMake; no Torch, no Vulkan.

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/b2c_tests                       # finite-difference gradient check against a double-precision CPU reference
./build/b2ctrain <dataset> --total-train-iters 30000 --export-path out --export-name export.ply --export-every 30000 --export-evidence
./build/b2ctrain render --splat out/export.ply --cameras cameras.json --output-dir renders --confidence --cull-color 0,0,0
./build/b2ctrain render --splat out/export.ply --cameras cameras.json --output-dir renders --sh-degree 0   # DC colour only; bands above N are left out of the sum [default: 3]
```

## What it implements

- COLMAP text datasets with brush's sidecar conventions: embedded alpha = transparent view, `masks/` = masked view,
  `normals/` (Sapiens camera-space normals, `[+X,-Y,-Z]` decode), `weights/` per-pixel loss weights, `init.ply` warm start.
- Rendering identical to brush (EWA projection with the 0.3 px dilation, opacity-adaptive extents, exact ellipse-tile
  binning, front-to-back compositing with the same cutoffs). 8x8 rasterizer tiles, depth-sorted two-pass binning.
- Loss: fused L1 + SSIM (11-tap, sigma 1.5) with alpha-match lane, mask weighting, `--normalize-masked-loss`,
  `weights/`, background compositing; normal supervision (L1 + 1-cos) on the composited pseudo-normal feature.
- Backward: per-pixel reverse replay with per-fragment scalars reduced on tensor cores (WMMA fp16, fp32 accumulate);
  a warp-shuffle variant is kept (`--backward warp`).
- Optimizer: fused projection backward + Adam (brush's constants, Adam-mini second moment for SH) + MCMC noise.
- Refinement: brush's recipe on the GPU (prune, relocation and growth by Gumbel-top-k weighted sampling, covariance-aware
  split, opacity decay, Mip-Splatting 3D filter floor).
- Evidence export (`--export-evidence`, `--evidence-prune-inmask`, `--evidence-normal-weight`) and confidence-gated
  rendering (`b2ctrain render --confidence ...`) matching brush-splat-render's output contract. `ev_views` is an
  effective (participation-ratio) view count rather than brush's thresholded one; see `src/train/evidence.cu`.

## Recipes

`--recipe brush` reproduces the brush fork's training dynamics (dense Adam, full resolution from step one, the floor
baked into the scales at every refine). `--recipe fast` (default) adds sparse Adam, a progressive resolution schedule
(1/4 -> 1/2 -> 1x over the first 40% of iterations) and a non-accumulating floor. Both produce the same splat counts.
`--sh-fp16` stores the SH bands above DC as fp16 for ~8% more speed on large models, but its stochastic rounding
leaves view-dependent colour speckle on specular surfaces at novel views, so it is off by default.

## Hollow loss (false transparency)

A splat that reproduces its training orbit can still be half-transparent: a front surface at partial opacity with
the far side of the body showing through, which only shows as the view tilts. `--hollow-weight W` adds a geometric
prior against it, given a body proxy mesh (`--mesh mesh.ply`, or `<dataset>/mesh.ply`; b2crunner writes the
SAM-3D-Body mesh there). Every step the mesh's depth is rasterised for the current view (`src/gpu/meshdepth.cu`,
taking the farthest surface within `--hollow-dilate` pixels so silhouettes are forgiven), and each pixel is charged
the compositing weight that arrives from more than `--hollow-margin` behind that surface (ramping to full penalty at
twice the margin). The gradient of that weight runs through the fragments in front of it, which is what pushes the
visible surface opaque; the fragment's own depth is treated as a constant (no pull towards the camera, which would
drag the far side of the body forward). Costs ~5% of the step. Without a mesh the dataset's `points3D.txt` stands in (`--hollow-proxy points`, or `auto`: surfels
with PCA normals, radius `--hollow-points-radius` or 4x the median spacing), which for b2crunner's mesh-sampled points
is the body at ~1 cm resolution. `b2ctrain probe` measures the effect: per pixel,
the weight arriving from more than `--delta` behind the first surface ("deep") and, with `--mesh`, from behind
the body ("behind"), as probe.json plus heat maps.

## Alignment loop

`--align-iters N` runs b2crunner's alignment loop (`pipeline/steps/brush.py`, `pipeline/align.py`) inside the trainer:
after the main run, every transparent training view is rendered with the current model on grey, the *pristine* frame
is flowed onto its render (coarse-to-fine Lucas-Kanade on the GPU), the flow is smoothed (`--align-flow-sigma`),
zeroed outside the subject and capped (`--align-flow-cap`), the frame is Lanczos-warped by it, and the model is refit
for `--align-steps` iterations on the warped set with a fresh optimizer and learning-rate schedule, growth and
refinement off, normal loss off. Nothing leaves the GPU between iterations: no .ply export/reload, no frames written,
no renderer process. `--align-debug-dir` writes `alignment.json` (per-iteration, per-view mean/p90 displacement in
pixels) and one view's warped frame + render per iteration, encoded on a background thread off the training path.
The refit's warm start also holds for `init.ply` runs: a warm start trains at full resolution (no progressive schedule).

## Bench

`bench/eval_ply.py <ply> <colmap_dir>` reports mask-weighted PSNR/SSIM on training views;
`bench/colmap_to_cameras.py` writes a `cameras.json`; `bench/novel_cameras.py` derives off-orbit (elevated / panned) cameras from one; `bench/polish_test.sh` reproduces b2crunner's warm-start invocation.
`bench/b2crunner_step.py` (run with b2crunner's venv) feeds a dataset through b2crunner's real `brush` step class with
b2ctrain as the trainer and `docker/brush-splat-render` as the rasteriser, so the cold run, the polish and the alignment
loop are exercised exactly as the pipeline invokes them.
