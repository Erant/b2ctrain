# b2ctrain

A from-scratch CUDA Gaussian-splat trainer, renderer and mesher for the b2crunner pipeline, and the posing / training
backend of b2crig. One binary: training (`b2ctrain <dataset>`), rendering (`render`), depth / weight probes (`probe`),
cage fitting (`fit-cage`) and meshing (`mesh-*`).

Targets NVIDIA Ada / Blackwell (sm_89, sm_120). Builds against CUDA 13 (12.8+ works) with CMake; no Torch, no Vulkan.

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/b2c_tests                       # finite-difference gradient check against a double-precision CPU reference
./build/b2ctrain <dataset> --total-train-iters 30000 --export-path out --export-name export.ply --export-every 30000 --export-evidence
./build/b2ctrain render --splat out/export.ply --cameras cameras.json --output-dir renders --confidence --cull-color 0,0,0
./build/b2ctrain render --splat out/export.ply --cameras cameras.json --output-dir renders --sh-degree 0   # DC colour only; bands above N are left out of the sum
./build/b2ctrain --help                 # every training option; each subcommand takes --help too
```

Design notes and measurements are in `docs/design.md`, current state in `docs/STATUS.md`.

## Training

- **Data**: COLMAP text datasets with sidecars: embedded alpha = transparent view, `masks/` = masked view, `normals/`
  (Sapiens camera-space normals, `[+X,-Y,-Z]` decode), `weights/` per-pixel loss weights, `labels/` 8-bit class ids,
  `init.ply` warm start, `mesh.ply` body proxy. All views stay GPU resident.
- **Rendering**: 3D Gaussian splatting [1] with EWA projection [2] (0.3 px dilation), opacity-adaptive extents, exact
  ellipse-tile binning on 8x8 tiles, depth-sorted two-pass binning, front-to-back compositing; Mip-Splatting's 3D
  filter as a scale floor [3]; view-dependent colour as spherical harmonics up to degree 4.
- **Loss**: fused L1 + SSIM [4] (11-tap, sigma 1.5) with an alpha-match lane, mask weighting,
  `--normalize-masked-loss`, `weights/`, background compositing with per-step noise (`--background-noise-strength`;
  a random background keeps background-coloured splats off the subject). Monocular normal supervision (L1 + 1-cos)
  on each splat's shortest axis, composited as a pseudo-normal [5].
- **Backward**: per-pixel reverse replay with per-fragment scalars reduced on tensor cores (WMMA fp16, fp32
  accumulate); a warp-shuffle variant is kept (`--backward warp`).
- **Optimizer**: fused projection backward + Adam [6], one second moment per splat for the SH bands (Adam-mini [7]),
  sparse Adam on visible splats only [8], and the MCMC position noise of 3DGS-MCMC [9].
- **Refinement** on the GPU: prune, relocation of dead splats and growth by weighted sampling without replacement
  (Gumbel-top-k [10]), covariance-aware split [9], opacity decay.
- **Visibility cull** (`--cull-weight`, `--cull-no-relocate`, `--evidence-prune-wall`): splats whose rendered mass over
  a refine window is below a threshold are pruned at every refine, the alignment refits included, and the final export
  drops what no training view renders. Measured in `docs/unsupported-splats-2026-09-28.md`: never-seen opaque splats
  47k -> 0 at equal PSNR, 46% fewer splats.
- **Sparsification** (`--sparsify RATIO`, opt-in): GaussianSpa [11] removes a fixed fraction of the splats through an
  ADMM-style opacity penalty; `--sparsify-score importance` ranks by blending weight as in Mini-Splatting [12].
  Halves the splats for about -6% off-orbit sharpness (`docs/design.md`).
- **Evidence and labels**: `--export-evidence` writes per-splat multi-view evidence into the ply as `ev_*`
  properties (`ev_views` is an effective, participation-ratio view count; see `src/train/evidence.cu`), which drives
  `--evidence-prune-inmask` and confidence-gated rendering (`render --confidence`). `--export-labels` votes each splat
  a class from `labels/` (b2crunner writes Sapiens2 Goliath ids [13]) by its rendered weight on each class's pixels
  over every training view; the winner and its share land in the ply as `seg_label` and `seg_conf`.

### Recipes

`--recipe fast` (default) uses sparse Adam, a progressive resolution schedule (1/4 -> 1/2 -> 1x over the first 40% of
iterations) and a 3D-filter floor applied on the fly. `--recipe brush` keeps the original dynamics: dense Adam, full
resolution from step one, the floor baked into the scales at every refine. Both produce the same splat counts.
`--sh-fp16` stores the SH bands above DC as fp16 for ~8% more speed on large models, but its stochastic rounding
leaves view-dependent colour speckle on specular surfaces at novel views, so it is off by default.

### Hollow loss (false transparency)

A splat that reproduces its training orbit can still be half-transparent: a front surface at partial opacity with
the far side of the body showing through, which only shows as the view tilts. `--hollow-weight W` adds a geometric
prior against it, given a body proxy mesh (`--mesh mesh.ply`, or `<dataset>/mesh.ply`; b2crunner writes the
SAM-3D-Body mesh there). Every step the mesh's depth is rasterised for the current view (`src/gpu/meshdepth.cu`,
taking the farthest surface within `--hollow-dilate` pixels so silhouettes are forgiven), and each pixel is charged
the compositing weight that arrives from more than `--hollow-margin` behind that surface (ramping to full penalty at
twice the margin). The gradient of that weight runs through the fragments in front of it, which is what pushes the
visible surface opaque; the fragment's own depth is treated as a constant (no pull towards the camera, which would
drag the far side of the body forward). Costs ~5% of the step. Without a mesh the dataset's `points3D.txt` stands in
(`--hollow-proxy points`, or `auto`: surfels with PCA normals, radius `--hollow-points-radius` or 4x the median
spacing), which for b2crunner's mesh-sampled points is the body at ~1 cm resolution. `b2ctrain probe` measures the
effect: per pixel, the weight arriving from more than `--delta` behind the first surface ("deep") and, with `--mesh`,
from behind the body ("behind"), as probe.json plus heat maps.

### Alignment loop

`--align-iters N` runs b2crunner's alignment loop inside the trainer: after the main run, every transparent training
view is rendered with the current model on grey, the *pristine* frame is flowed onto its render (coarse-to-fine
Lucas-Kanade [14, 15] on the GPU), the flow is smoothed (`--align-flow-sigma`), zeroed outside the subject and capped
(`--align-flow-cap`), and the model is refit for `--align-steps` iterations with a fresh optimizer and learning-rate
schedule, growth and refinement off, normal loss off. `--align-warp frames` (default) Lanczos-resamples [16] the frames
by the flow; `--align-warp render` instead displaces each splat's projected mean by the field, so the frames are never
resampled. Nothing leaves the GPU between iterations. `--align-debug-dir` writes `alignment.json` (per-iteration,
per-view mean/p90 displacement in pixels) and one view's warped frame + render per iteration; `--align-dump-dir`
writes the last iteration's warped frames. A warm start (`init.ply`) trains at full resolution.

### Per-view body rig

`--body-rig RIG` learns, per training view, small rotations of the arm (and other active) joints of the refit MHR
body, composed by forward kinematics and applied to the splats by linear blend skinning [17], after HUGS's per-frame
body-pose refinement [24]: the generated frames move the limbs by centimetres between orbit segments, and one
canonical body cannot explain them all. The model and the export
stay canonical. A v3 rig adds per-view vertex displacements (b2crunner's per-view head fit: expression and neck/head
pose) before the blend. `--body-rig-smooth` pulls each view towards its orbit neighbours, `--body-rig-zero` towards the
rest pose; `--body-rig-start-iter` / `--body-rig-stop-iter` / `--body-rig-init` control when and from where it learns.

## Cage posing (b2crig)

b2crig animates a trained subject and makes every decision about the rig; this repo poses, renders and trains. See
`docs/b2crig-boundary.md` for what lives where, and `src/gpu/cage.h` for the cage file.

- **Binding**: every splat binds to the nearest triangle of the cage layer (body, garment, hair) that owns its
  `seg_label`, keeps its barycentric foot point and its offset in the triangle's frame scaled by the triangle's size,
  and turns and grows with it, as in GaussianAvatars [18]. Dual bindings blend two candidate triangles where the label
  cannot say which limb a splat belongs to. `--cage-max-growth` and `--cage-fade-start/-end` bound or fade splats on
  stretched triangles.
- **Training** (`--cage CAGE` on a warm start): views named after a cage frame render the posed splat, the others render
  canonical; gradients return to the canonical frame. `--cage-stretch-weight` keeps splats small where the body
  stretches; `--pose-contain-weight` keeps the posed splat inside the silhouette its own surface draws in that pose
  (`docs/pose-contain-2026-09-29.md`).
- **Pose-dependent appearance**: `--cage-app` (a small per-vertex MLP fed the local surface stretch and occlusion
  change, zero at the canonical pose) or `--cage-open` (two-state splats: a learned colour offset and opacity delta
  blended in by b2crig's geometric opening gate).
- **`fit-cage`** fits per-frame cage displacements, dual-binding weights and optionally appearance to a video;
  **`render --cage`** renders any cage frame.

## Meshing: `b2ctrain mesh-*`

The textured mesh from the stage-2 views (`docs/mesh.md` for usage and verification, `docs/mesh-plan.md` for the plan):

- **`mesh-fuse`**: TSDF fusion [19] of `probe` depth with silhouette carving [20], the body mesh's signed distance as
  the prior for what no view saw, face protection, and marching cubes [21].
- **`mesh-refine`**: vertex positions optimised towards the Sapiens normal maps with Laplacian and positional terms.
- **`mesh-bake`**: per-vertex colour as the trimmed, facing-weighted median of the views, plus the face cap projection.
- **`mesh-unwrap`**: decimation (meshoptimizer, quadric error [22]), xatlas charts, atlas maps and jump-flood edge
  padding [23].
- **`mesh-render` / `mesh-backproject`**: the raster halves of b2crunner's texture-refinement loop.

`tools/view_atlas.py` is the research version of b2crunner's character-sheet texture layout (`docs/view-atlas.md`).

## Bench

`bench/eval_ply.py <ply> <colmap_dir>` reports mask-weighted PSNR/SSIM on training views;
`bench/colmap_to_cameras.py` writes a `cameras.json`; `bench/novel_cameras.py` derives off-orbit (elevated / panned)
cameras from one; `bench/compare_images.py` diffs render sets; `bench/polish_test.sh` reproduces b2crunner's
warm-start invocation. `bench/b2crunner_step.py` (run with b2crunner's venv) feeds a dataset through b2crunner's real
training step with b2ctrain as the trainer, so the cold run, the polish and the alignment loop are exercised exactly as
the pipeline invokes them. `tools/bundle_study/` holds the scripts behind `docs/random-background.md`.

## Third-party code

Vendored in `third_party/` (versions in `VERSIONS.txt`): [xatlas](https://github.com/jpcy/xatlas),
[meshoptimizer](https://github.com/zeux/meshoptimizer), [stb](https://github.com/nothings/stb) image I/O and resize,
and [nlohmann/json](https://github.com/nlohmann/json).

## References

1. B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis. *3D Gaussian Splatting for Real-Time Radiance Field Rendering.*
   ACM TOG 42(4), SIGGRAPH 2023. [arXiv:2308.04079](https://arxiv.org/abs/2308.04079)
2. M. Zwicker, H. Pfister, J. van Baar, M. Gross. *EWA Splatting.* IEEE TVCG 8(3), 2002.
3. Z. Yu, A. Chen, B. Huang, T. Sattler, A. Geiger. *Mip-Splatting: Alias-free 3D Gaussian Splatting.* CVPR 2024.
   [arXiv:2311.16493](https://arxiv.org/abs/2311.16493)
4. Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli. *Image Quality Assessment: From Error Visibility to
   Structural Similarity.* IEEE TIP 13(4), 2004.
5. M. Turkulainen, X. Ren, I. Melekhov, O. Seiskari, E. Rahtu, J. Kannala. *DN-Splatter: Depth and Normal Priors for
   Gaussian Splatting and Meshing.* WACV 2025. [arXiv:2403.17822](https://arxiv.org/abs/2403.17822)
6. D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015.
   [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
7. Y. Zhang et al. *Adam-mini: Use Fewer Learning Rates To Gain More.* ICLR 2025.
   [arXiv:2406.16793](https://arxiv.org/abs/2406.16793)
8. S. S. Mallick, R. Goel, B. Kerbl, F. Vicente Carrasco, M. Steinberger, F. De La Torre. *Taming 3DGS: High-Quality
   Radiance Fields with Limited Resources.* SIGGRAPH Asia 2024. [arXiv:2406.15643](https://arxiv.org/abs/2406.15643)
9. S. Kheradmand, D. Rebain, G. Sharma, W. Sun, Y.-C. Tseng, H. Isack, A. Kar, A. Tagliasacchi, K. M. Yi. *3D Gaussian
   Splatting as Markov Chain Monte Carlo.* NeurIPS 2024. [arXiv:2404.09591](https://arxiv.org/abs/2404.09591)
10. W. Kool, H. van Hoof, M. Welling. *Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling
    Sequences Without Replacement.* ICML 2019. [arXiv:1903.06059](https://arxiv.org/abs/1903.06059)
11. Y. Zhang, W. Jia, W. Niu, M. Yin. *GaussianSpa: An "Optimizing-Sparsifying" Simplification Framework for Compact
    and High-Quality 3D Gaussian Splatting.* CVPR 2025. [arXiv:2411.06019](https://arxiv.org/abs/2411.06019)
12. G. Fang, B. Wang. *Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians.* ECCV 2024.
    [arXiv:2403.14166](https://arxiv.org/abs/2403.14166)
13. R. Khirodkar et al. *Sapiens: Foundation for Human Vision Models.* ECCV 2024.
    [arXiv:2408.12569](https://arxiv.org/abs/2408.12569)
14. B. D. Lucas, T. Kanade. *An Iterative Image Registration Technique with an Application to Stereo Vision.*
    IJCAI 1981.
15. J.-Y. Bouguet. *Pyramidal Implementation of the Lucas Kanade Feature Tracker: Description of the Algorithm.*
    Intel Corporation, Microprocessor Research Labs, 2000.
16. C. E. Duchon. *Lanczos Filtering in One and Two Dimensions.* Journal of Applied Meteorology 18(8):1016-1022,
    1979.
17. N. Magnenat-Thalmann, R. Laperrière, D. Thalmann. *Joint-Dependent Local Deformations for Hand Animation and Object
    Grasping.* Graphics Interface 1988.
18. S. Qian, T. Kirschstein, L. Schoneveld, D. Davoli, S. Giebenhain, M. Nießner. *GaussianAvatars: Photorealistic Head
    Avatars with Rigged 3D Gaussians.* CVPR 2024. [arXiv:2312.02069](https://arxiv.org/abs/2312.02069)
19. B. Curless, M. Levoy. *A Volumetric Method for Building Complex Models from Range Images.* SIGGRAPH 1996.
20. A. Laurentini. *The Visual Hull Concept for Silhouette-Based Image Understanding.* IEEE TPAMI 16(2), 1994.
21. W. E. Lorensen, H. E. Cline. *Marching Cubes: A High Resolution 3D Surface Construction Algorithm.* SIGGRAPH 1987.
22. M. Garland, P. S. Heckbert. *Surface Simplification Using Quadric Error Metrics.* SIGGRAPH 1997.
23. G. Rong, T.-S. Tan. *Jump Flooding in GPU with Applications to Voronoi Diagram and Distance Transform.* I3D 2006.
24. M. Kocabas, J.-H. R. Chang, J. Gabriel, O. Tuzel, A. Ranjan. *HUGS: Human Gaussian Splats.* CVPR 2024.
    [arXiv:2311.17910](https://arxiv.org/abs/2311.17910)
