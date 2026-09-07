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
  rendering (`b2ctrain render --confidence ...`) matching brush-splat-render's output contract.

## Recipes

`--recipe brush` reproduces the brush fork's training dynamics (dense Adam, full resolution from step one, the floor
baked into the scales at every refine). `--recipe fast` (default) adds sparse Adam, a progressive resolution schedule
(1/4 -> 1/2 -> 1x over the first 40% of iterations), fp16 storage for the SH bands above DC (`--sh-fp32` to keep
them in fp32) and a non-accumulating floor. Both produce the same splat counts.

## Bench

`bench/eval_ply.py <ply> <colmap_dir>` reports mask-weighted PSNR/SSIM on training views;
`bench/colmap_to_cameras.py` writes a `cameras.json`; `bench/polish_test.sh` reproduces b2crunner's warm-start invocation.
`bench/b2crunner_step.py` (run with b2crunner's venv) feeds a dataset through b2crunner's real `brush` step class with
b2ctrain as the trainer and `docker/brush-splat-render` as the rasteriser, so the cold run, the polish and the alignment
loop are exercised exactly as the pipeline invokes them.
