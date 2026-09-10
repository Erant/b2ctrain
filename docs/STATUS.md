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

**A/B against the real SAM-3D-Body mesh (2026-09-09, later).** The pipeline's mesh was reproduced locally
(`~/Projects/sam-3d-body` on the run's anchor front view, `--no-detector`, then a similarity ICP onto the dataset's
`points3D.txt`: 0.4 cm median / 1.35 cm p90 residual, so it is the pipeline's fitted body to within the margin).
Every arm probed against that mesh, training views and elevated ±25° views, hollow 0.5 / margin 0.05 throughout:

| reference used in training | deep (train) | behind (train) | deep (elev) | behind (elev) | PSNR |
|---|---|---|---|---|---|
| none (baseline) | 0.145 | 0.043 | 0.149 | 0.035 | 35.46 |
| Poisson proxy of the points | 0.098 | 0.008 | 0.110 | 0.007 | 35.39 |
| points as surfels (`--hollow-proxy points`, no mesh) | 0.096 | 0.005 | 0.108 | 0.004 | 35.31 |
| real mesh | 0.102 | 0.004 | 0.114 | 0.004 | 35.36 |
| real mesh, hands dropped | 0.106 | 0.008 | 0.117 | 0.007 | 35.37 |
| real mesh, hands dropped, shrunk 1 cm | 0.114 | 0.013 | 0.125 | 0.011 | 35.36 |

All hollow arms are within noise of each other; the reference's exact shape does not matter at a 5 cm margin. The
mesh's hands are posed differently from the real hands (fingers curled, sticking out below them), yet the hand
region renders identically across arms at the training view and at +25°, and its error against the frame is equal
(5.0-5.1 / 255 in the hand box for every arm): the dilation and margin absorb it. Dropping the hands is a harmless
safety measure, shrinking is not needed. (`out/hollow/hands_cmp.png`, `realmesh_*.ply`, `pointsproxy_*.ply`.)

**Fallback without a mesh** (`--hollow-proxy auto|mesh|points`, `--hollow-points-radius`): the dataset's
`points3D.txt` is splatted as surfels (PCA normals over 12 neighbours, disc of 4x the median spacing in the tangent
plane, ray-plane intersection per pixel; camera-facing discs were tried first and sit in front of the surface at
grazing angles, inflating "behind" 3-6x with the radius). `b2ctrain probe --points <colmap_dir>` measures against the
same proxy. In the pipeline the points are samples of the body mesh, so the fallback is the mesh at ~1 cm resolution.

b2crunner side (working tree, uncommitted): `render.py` publishes the oriented mesh as `mesh` (`scene.mesh_world` in
the workflow, the same frame as `points_3d`), `steps/brush.py` writes it as `mesh.ply` beside the COLMAP model and
passes the flags (`hollow_weight` / `hollow_margin` / `hollow_dilate` params; `tests/test_brush_hollow.py`), the
doctor requires `--hollow-weight`, and both trainings in `fast_helical_native.yaml` set `hollow_weight: 0.5`.
Splats to look at: `out/hollow/` (baseline, the three settings, the proxy mesh, camera sets).

### Body refit: the hollow loss against a mesh that is where the splat is (2026-09-09, later)

The SAM-3D-Body mesh the pipeline hands the trainer is where the body was *before* two camera refinements and the
diffusion drift; on the example run the trained splat sits 3 cm higher, 1.3 cm sideways and 3 deg yawed from it,
with the arms visibly off. b2crunner now has a refit (`pipeline/steps/body_refit.py`, docs/body-refit.md there):
`splat_surface` runs `b2ctrain probe --depth --tau 0.5` over the training cameras and unprojects the splat's median
surface into 300k oriented points; `refit_body_to_splat` re-runs the MHR body model with root, pose, scales and shape
free against a one-sided point-to-plane loss (1 cm clothing allowance, hands excluded, priors to SAM's fit).
Surface-to-body median 1.33 -> 0.73 cm, points > 5 mm inside the body 24.7% -> 12.9%, 87 s on the 4070 Ti; the
updated pose parameters replay the refit mesh to 0.001 mm. `out/refit/overlay.png` is the before/after silhouette.

Trained with each mesh (final-stage argv as in the hollow table, no alignment, hollow 0.5, the current
`--hollow-tau 0.1` default; probe every 8th view, "behind" measured against the refit mesh at the run's own margin):

| mesh | margin | deep | behind | PSNR |
|---|---|---|---|---|
| none (baseline) | - | 0.146 | 0.043 | 35.46 |
| SAM-3D-Body | 0.05 | 0.111 | 0.0016 | 35.40 |
| refit | 0.05 | 0.119 | 0.0007 | 35.42 |
| SAM-3D-Body | 0.03 | 0.089 | 0.0060 | 35.37 |
| refit | 0.03 | 0.094 | 0.0015 | 35.40 |
| refit | 0.02 | **0.081** | 0.0027 | 35.42 |

With the refit mesh PSNR is equal or a hair higher at every margin, four times less weight ends up behind the body
at the same margin, and a 2 cm margin becomes usable (the lowest "deep" of any hollow run so far, at no PSNR cost)
where 3 cm was the risky setting with the SAM mesh. The committed trainer (a267b55, mesh-only reference, what the
b2crunner image pins) tells the same story with the refit mesh: margin 0.03 deep 0.093 / behind 0.0008 / 35.34 dB,
margin 0.05 0.111 / 0.0005 / 35.38, against its 0.102 / 0.004 / 35.36 with the SAM mesh at 0.05 (the A/B above).
b2crunner's final training now takes the refit mesh at margin 0.03. The delivered (no-hollow) splat probed against the two meshes
tells the same story: behind-weight at 2 cm against the refit mesh (0.057) is what 5 cm gave against the SAM mesh
(0.040 at 5 cm, 0.127 at 2 cm). Wired into b2crunner's workflow 2026-09-09 (refit after stage 2, its mesh into stage 5, docs/body-refit.md there);
`probe --depth` is what the pipeline's `splat_surface` step calls.

### Render-side alignment warp (2026-09-09)

`--align-warp render` applies the alignment loop's flow to the model instead of to the frames: each view keeps a
displacement grid (the same sigma-6, 6 px-capped field, box-averaged to 1/4 resolution, `ALIGN_WARP_DOWN`) and the
projection kernel moves every splat's projected mean by minus the field at its position while the refit trains
against the PRISTINE frame. The frames are never Lanczos-resampled (b2crunner's alignment guide measured a single
0.5 px Lanczos resample costing raw Laplacian sharpness 738 -> 480). The field is measured from the pristine frame to
the model's undisplaced render every iteration, as before, so nothing accumulates; the gradient of the mean is
unchanged (a locally constant shift), the evidence pass renders with the field. Motivation and the flow decomposition
by body part that led here (per-bone rigid motion explains 16-39% of the field's energy; the rest is sub-limb
texture jitter, so a skeletal per-view deformation would duplicate the warp with less capacity) are in the session
notes of 2026-09-09.

A/B on the 2026-09-09 result bundle's final-stage dataset (81 views 1080x1920), production stage-5 argv (hollow 0.5
at 3 cm against the refit mesh, 4 alignment iterations of 3000 steps), two seeds each; PSNR against the pristine
frames every 4th view (`bench/eval_ply.py`); s1 = Laplacian variance after a sigma-1 blur of the grey render, per
region (regions from the refit mesh's projected skin weights), at the training cameras and at cameras midway between
them (novel):

| run | PSNR | s1 train head / hands / body | s1 novel head / hands / body | trajectory |
|---|---|---|---|---|
| no alignment (`out/refit/hollow_refit_m0.03.ply`) | 35.40 (fits the disagreement) | 23.4 / 40.8 / 27.0 | 22.9 / 38.6 / 25.9 | - |
| frames (default), seed 42 | 33.23 | 26.6 / 42.5 / 29.7 | 26.2 / 40.1 / 28.6 | 1.22 -> 1.63 |
| frames, seed 1 | 33.26 | 26.5 / 42.2 / 29.7 | 26.0 / 40.3 / 28.5 | 1.22 -> 1.64 |
| render, seed 42 | 33.20 | 27.0 / 44.5 / 30.4 | 26.3 / 42.3 / 29.3 | 1.21 -> 1.57 |
| render, seed 1 | 33.24 | 27.2 / 44.5 / 30.4 | 26.5 / 42.0 / 29.1 | 1.22 -> 1.59 |

Seed-to-seed noise is ~0.2 s1 per region and 0.03 dB. The render-side warp keeps the whole alignment gain and adds
+2% on the body and +5% on the hands at both training and novel cameras, reproducibly; the face gains +0.1 to +0.5
(at the noise floor); PSNR is 0.03 dB lower in both seeds. Same wall time (the field replaces the Lanczos warp,
one bilinear fetch per splat per step), 84 MB of grids for 81 views. Novel views gain as much as training views, the
guide's test for real geometry. A face crop at a novel view is indistinguishable between the two. Not yet the
default or exposed on b2crunner's step; a `align_warp` param there is a two-line change if it should be.
Runs in `out/catch/{frames,render}[_s1]/` (not in git).

### Per-view arm rotations: the double limb (2026-09-09, evening)

The delivered splats show a **double limb**: a ghost contour beside the forearms and hands, in the frame-warp and
render-warp variants alike (crisper in the latter, which fits sharper targets). Measured between ADJACENT pristine
frames, minus the parallax the static refit mesh predicts for a 4.5-degree step (scratch `limb_motion2.py`, copies in
`out/catch/`): torso, legs and feet move 2-3 px per pair (the floor), forearms and hands 7-8 px (about 9 mm), and
the motion is episodic and smooth (lag-1 correlation +0.6): the left forearm swings 20-24 px per view over 3-4
consecutive views at three places in the orbit (6-9 cm in total each time, verified against the mesh projection).
The generated video moves the arms by centimetres between orbit segments; a canonical body averages them into two
copies. Neither 2D warp can fix it (6 px cap, and the flow locks onto whichever copy is nearer; the frame-to-render
decomposition of the previous section was blind to it for the same reason).

Per-frame SAM-3D-Body fits on all 81 frames were tried first as the source of per-view arm poses and are useless
for this: they do not see the swings and add ~5 px of noise (a rig built from them made the residual WORSE: forearm
7.6 -> 9.1 px).

What works: **learn the arm pose per view inside the trainer** (`--body-rig rig.bin`, `src/gpu/deform.*`). The rig
file carries a subsample of the refit mesh with its MHR skinning, the joint tree and canonical joint positions, and
the active joints (per arm: shoulder, elbow, wrist, two finger roots). Per view and active joint one axis-angle
rotation about the joint's (moved) pivot; forward kinematics on the GPU; every splat bound to its nearest rig vertex
(re-bound after each refine) and rendered at the skinned position; the gradient is the torque of the splats'
positional gradients about each active ancestor's pivot; Adam per (view, joint) with the second moment shared per
joint across views (a view in which the arm is hidden behind the body has a tiny torque and takes a small step)
and a pull towards the two orbit neighbours (`--body-rig-smooth 0.05`); the model and export stay canonical, the
rotations go to `<export>/body_rig_omega.json`. Learning starts at `--body-rig-start-iter` (the user's suggestion:
once the splat is roughly in place) and continues through the alignment refits. Cost: none measurable.

Start-iteration sweep, production stage-5 argv (frames warp, hollow 0.5 at 3 cm), `out/catch/rig_s*` (per-parameter
Adam, no smoothing) and `out/catch/rigv_s*` (shared second moment, smooth 0.05):

| run | rotations mean / max (deg) | adjacent-frame residual, L forearm / L hand (px; static 7.6 / 7.0) | s1 novel hands / body / head (frames: 40.1 / 28.6 / 26.2) | final refit loss (frames -0.19545) | canonical PSNR |
|---|---|---|---|---|---|
| rig, start 1k | 6.5 / 37 | 6.6 / 6.1 | 50.3 / 29.0 / 25.8 | -0.19559 | 31.19 |
| rig, start 5k | 5.9 / 44 | 6.8 / 4.8 | 48.6 / 28.9 / 26.0 | | 31.65 |
| rig, start 10k | 5.5 / 39 | 6.0 / 4.8 | 46.6 / 29.1 / 26.3 | -0.19559 | 32.06 |
| rig, start 15k | 5.3 / 35 | 6.2 / 5.6 | 44.4 / 29.1 / 25.8 | -0.19537 | 32.32 |
| rig shared-v + smooth, start 1k | 1.4 / 3.8 | 7.1 / 7.2 | 50.3 / 29.2 / 25.9 | -0.19564 | 31.32 |
| rig shared-v + smooth, start 10k | 1.3 / 4.6 | 6.3 / 6.3 | 46.7 / 29.0 / 25.7 | -0.19557 | 32.17 |

- **The double limb is gone at novel views in every rig run** (`out/catch/arm_panel_front_novel33.png`,
  `arm_panel_back_novel57.png`): the far hand is single with defined fingers, the near forearm has one edge. Hand
  sharpness rises 10-25%, more with an earlier start; body +1.5%; the face is unchanged (noise 0.2).
- The training loss, which renders each view with its rotations, is equal or slightly better than the frames run.
  The canonical model's PSNR against the frames drops 1-2 dB by construction: its arms sit at the mean pose and
  each frame's arm is elsewhere. That number is not the metric for this problem.
- Without smoothing the largest rotations (35-44 degrees) sit in the views where that arm is hidden behind the body
  (left: views 8-14, 43-50, 77-81; right: 25-31, 59-65): per-parameter Adam turns a tiny noisy torque into full
  steps. Sharing the second moment and pulling towards the neighbours holds every rotation under 5 degrees with the
  same visual result and the same hand sharpness; the right arm's residual never moves (its swings coincide with its
  occluded views, where the flow measurement itself is unreliable).
- Earlier start is better for the hands (50 at 1k vs 47 at 10k vs 44 at 15k) and the crops agree; the concern that
  the rotations need a settled splat did not materialise, the quarter- and half-resolution phases help the basin.

**How many bones** (same argv, start 1k, shared second moment, smooth 0.05; `out/catch/rigall_*`, `rig_nr*`):

| active joints | s1 novel hands / body / head | adjacent residual head / torso / leg (static 4.0 / 2.5 / 2.0) | final refit loss |
|---|---|---|---|
| arms only (12) | 50.3 / 29.2 / 25.9 | - / - / - | -0.19564 |
| all with >= 30 skinned vertices (110), per-joint Adam | 43.6 / 28.2 / **20.2** | 5.7 / 2.4 / 2.1 | - |
| all (110), ONE second moment for every joint (`--body-rig-global-moment`) | 21.4 / 16.0 / 6.9 | 12.5 / 7.5 / 5.4 | - |
| all but the root chain (104: subtree <= half the body) | 49.2 / **30.3** / **28.4** | 3.7 / 2.4 / 2.0 | -0.19590 |
| all but the root chain + `--body-rig-zero 0.02` | 49.6 / 30.2 / 28.4 | 3.8 / 2.5 / 1.9 | -0.19590 |

- Every bone active with per-joint Adam softens the face (head rotations 1.5 deg mean, fingers 29 deg) and the
  hands. One global second moment is far worse: the ROOT and SPINE joints carry the largest torques, and a rotation
  there (0.7 deg mean, 5 max) moves the whole body per view — every region's residual doubled, sharpness collapsed.
- Excluding joints whose subtree is more than half the body (root, pelvis, spine) and keeping everything else —
  hips, legs, feet, clavicles, arms, hands, neck, head — is the best configuration measured: hands as the arms-only
  rig, body +6% and face +9% over the frames baseline (the face was flat with arms only), legs and torso residuals
  at the floor, the head residual 4.0 -> 3.7, the training loss the best of all runs; visually the face, arms and
  shoes are cleaner (`out/catch/face_panel_novel33.png`, `body_panel_novel33.png`, `arm_panel_allbones_novel33.png`).
  With `--body-rig-zero 0.02` the rotations stay at 0.35 deg mean (max 3) for the same result; without it the
  fingers wander to 30 deg. The recommended setting is therefore: all joints but the root chain, start 1k, shared
  second moment, smooth 0.05, zero 0.02 (`rig_noroot.bin` from `build_rig2.py` with RIG_ALL=1 RIG_MAX_FRAC=0.5).
- The canonical model's PSNR against the frames falls further with more bones (33.2 -> 31.3 -> 28.9): more of each
  frame's per-view pose lives in the rotations; it is still not the metric.

Not in b2crunner yet: the rig builder (`out/catch/build_rig2.py`, needs only the refit outputs: pose joints,
`rig_binding`, `mesh_world`) should become a step feeding `train_final_splat`, and `--body-rig` a param. Open:
whether the learned arm trajectory should ALSO drive the frames (re-warping the frames onto the posed render), and
whether the render-side warp stacks with the rig.

### The rig at the INTERMEDIATE stage (2026-09-10)

The rig was built for `train_final_splat`. Tested at stage 2 (`train_splat`) on the same bundle
(`~/Downloads/fast_helical_native-F3-pass2-shift5-20260909-020603-792cf0-result`, `colmap_intermediate`: 81 orbit
frames at 720p + 36 face-support renders). `colmap_intermediate` and `colmap` are **different trajectories in the
same world frame** (a planar 81-frame orbit, 4.5 deg/frame, vs the helical re-render, 10.4 deg/frame): the
intermediate splat, the delivered splat and `out/refit/mesh_refit.ply` agree to ~1 cm, so the refit body drops
straight in with no re-registration. Do NOT match the two datasets by frame name — the same name is a different
camera.

The problem is present at stage 2 and relatively worse than at stage 5. Adjacent-frame limb motion beyond the static
body mesh (mm at the subject, so the two stages compare): intermediate torso/legs 1.4-1.9, upper arm 8.0-9.2,
forearm 10.9-13.2, hand 12.0-12.3; final 2.3-3.2 / 9.5-9.9 / 9.0-9.2 / 7.4-8.4. Arms move ~6x the torso baseline at
stage 2, ~3x at stage 5.

Runs: production stage-2 argv, 30k iters, rig params as b2crunner ships them (start 1000, smooth 0.05, zero 0.02,
lr 0.002), rig over the 81 frames only. `out/inter/`, scripts `run_base.sh` / `run_rig{,2,3}.sh`,
`build_inter_rig.py` (the production `pipeline.body_rig.build_body_rig`, 104/127 joints active), `eval_inter.sh`.

| run | s1 body | s1 hands | s1 head | s1 subject | canonical PSNR | time |
|-----|---------|----------|---------|------------|----------------|------|
| base            | 51.26 | 84.38  | 43.06 | 51.64 | 38.07 | 1m32s |
| base2 (seed 43) | 51.23 | 84.45  | 43.53 | 51.65 | 38.07 | 1m31s |
| align           | 55.83 | 84.91  | 48.20 | 56.12 | 37.10 | 1m30s |
| rig (SAM body)  | 54.99 | 107.13 | 47.10 | 55.81 | 33.34 | 1m32s |
| rig, seed 43    | 55.03 | 106.82 | 46.42 | 55.76 | 33.43 | 1m31s |
| rig (refit body)| 55.29 | 105.58 | 45.76 | 55.88 | 33.40 | 1m32s |
| rig + align     | 57.52 | 100.48 | 49.37 | 58.00 | 33.43 | 1m32s |
| rig over all 117 views | 55.19 | 107.02 | 43.10 | 55.59 | 31.46 | 1m33s |

s1 at NOVEL cameras (train differs by <1%); seed noise <1%. Findings:

- **The rig and the 2D alignment loop are nearly orthogonal.** Alignment fixes body (+9%) and head (+12%) and does
  nothing for the hands (+0.6%); the rig is the only thing that fixes the hands (+27%). Together: body +12%, head
  +15%, hands +19%, subject +12% — the best of the eight. Visually the baseline arms are ghosted and translucent
  with a smeared hand; `align` sharpens the top and torso but leaves the arm ghost; the rig makes the arm solid and
  the fingers defined (`out/inter/arm_novel_0000{1,41}__4way.png`).
- **The SAM-3D-Body body is as good as the refit body here** (hands 107.1 vs 105.6, head 47.1 vs 45.8). Stage 2 does
  not need a splat refit — the body from `reconstruct_body` is enough, which is what makes this wirable at all
  (at stage-2 time there is no splat to refit to).
- **The rig must cover only the real frames.** Extending it over the 36 face-support renders keeps the body and hand
  gains but loses the whole head gain (47.1 -> 43.1, baseline). b2crunner's brush step writes the rig for
  `list(image_names)`, so wiring `body_rig` at stage 2 needs the support views filtered out.
- Canonical PSNR falls 38.07 -> 33.34, and the split says why: the 81 frames go 33.86 -> 27.20 while the 36
  undeformed support views are untouched (47.91 -> 47.85). Training loss is unchanged (-0.19651 vs -0.19649). Not
  the metric, as at stage 5.
- The learned rotations are dominated by torso 1.8 deg, head 1.4, legs 1.3 — arms only 0.3-0.8 (the opposite of
  stage 5, where the alignment loop has already absorbed the body-scale wobble). Per-view |rotation| mean 0.74 deg,
  max 5.7; |mean over views| per joint 0.23 deg, so the canonical pose stays the average pose. Mean over views of
  the max hand displacement from canonical: 40 mm.
- Cost is nil: 1m32s either way, 365k splats vs 369k.

Not wired: `body_rig` (and `align_iters`) on `train_splat` in b2crunner, with the support views excluded from the
rig's view list. The rig would come from `reconstruct_body`'s mesh + `rig_binding` rather than from the refit.

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
