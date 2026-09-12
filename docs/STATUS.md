# b2ctrain: project state

Last updated: 2026-09-11 (per-splat labels). Written for whoever (human or Claude) picks this up next.

**2026-09-11 — `--export-labels`.** A `labels/` sidecar (8-bit class-id PNGs, one per training frame; point-sampled
if the frame is capped) is voted onto the splats at the final export inside the evidence replay: `label_vote_kernel`
in `src/train/evidence.cu` accumulates each splat's `vis * k` (the w_all weight) into a 32-bin histogram per class,
the winner and its share of the splat's total go into the ply as float `seg_label` / `seg_conf` after the `ev_*`
block (only with `--export-evidence` does the ev_* block itself get written). Verified against an independent
gsplat-gradient vote on a b2crunner deliverable (81 views, 332k splats, Sapiens2 labels): 99.8% of splats get the
same class, confidences within 0.001 median; `b2c_tests` has a two-class seam test and a ply round trip. The pass
adds ~0.2 s to the 0.4 s evidence pass. Labels are not warped in `--align-warp frames` mode (the ~1 px flow is noise
against an 81-view vote); the per-view rig is applied, as for the evidence.

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

### The navel and the uncommitted hollow-tau/push diff (2026-09-10)

The working tree carries an unfinished diff (`hollow_tau` 0.1 and `hollow_push_tau` 0.5 on by default,
`hollow_front_alpha` 0 off) written for a report that the hollow loss pushed a soft skin surface ~2 cm towards the
camera and buried the navel. None of it is in HEAD, so the pinned trainer and every shipped image lack it. Two facts
worth recording before deciding its fate:

- **The hollow loss has never run in a production bundle.** All 32 result bundles in ~/Downloads, up to and including
  20260909-151011, have no `--mesh` and no `--hollow-weight` in either training's argv. An intact navel in those
  splats says nothing about the loss. b2crunner main now turns it on for both trainings (0.5, margin 0.03 against the
  refit body), so the next pod run is the first that will exercise it.
- **Every rig run in out/catch was trained WITH the diff active** — `run_rig.sh` used ./build/b2ctrain and
  b2ctrain_{all,gm,sharedv} all carry the three flags (they were built from the working tree before ee71363 was split
  out of it). The clean HEAD binary is the /tmp/b2c_wt worktree. So the arm-rig table above and the navel in those
  runs were measured with the navel diff on.

Measured (F3 bundle, production stage-5 argv, 30k, anchor camera frame_00038_ = the re-injected pristine photo;
navel ROI x 449-649, y 705-865; runs in out/inter's sibling out/navel/, scripts run.sh / run2.sh, navel_eval.py):

| run | trainer | hollow | mean ROI depth | relief | PSNR | navel |
|-----|---------|--------|----------------|--------|------|-------|
| none          | HEAD | off               | 2.0401 m | 4.56 mm | 33.235 | present |
| refit_m03     | HEAD | 0.5 @ 3 cm, refit | 2.0377   | 4.71    | 33.231 | present |
| sam_m03       | HEAD | 0.5 @ 3 cm, SAM   | 2.0364   | 4.39    | 33.032 | present |
| refit_tau     | diff | 0.5 @ 3 cm, refit | 2.0397   | 4.19    | 33.211 | present |
| sam_m001_head | HEAD | 0.5 @ 1 cm, SAM   | 2.0338   | 4.02    | 32.695 | present |
| sam_m001_tau  | diff | 0.5 @ 1 cm, SAM   | 2.0365   | 4.38    | 32.405 | present |
| sam_w2_head   | HEAD | 2.0 @ 1 cm, SAM   | 2.0320   | 3.34    | 32.459 | present |

`refit_m03` is what main will now run; it is indistinguishable from the no-hollow baseline at the navel and 0.004 dB
from it. The navel survived every configuration tried, including a deliberately abusive one. On the user's own read of
the panel `refit_m03` is the best of the seven and every SAM-mesh run is subpar — which the PSNR agrees with (33.231
against 33.032 / 32.695 / 32.459) and is a second argument for the refit body carrying the hollow loss.

The mechanism the diff was written for is real but an order of magnitude smaller here than reported. The surface does
migrate towards the camera under the loss, and the diff does halve it: at 1 cm / weight 0.5 the ROI's first surface
moves 6.3 mm forward on HEAD against 3.6 mm with the diff; at weight 2.0 on HEAD it is 8.1 mm, with the relief
flattening 4.56 -> 3.34 mm. It never reaches the ~2 cm that would fill a navel in.

The premise does not hold for this subject either: out/refit/stats.json puts SAM-3D-Body's body a median 1.33 cm from
the splat surface (mean signed +1.02 cm, 24.7% inside by >5 mm), not the 2-5 cm the diff's comments describe. The
likeliest explanation for the original report is the mesh it was seen with — the earlier real-SAM mesh registered onto
points3D.txt by similarity ICP, before the refit existed — rather than the loss itself. That is a hypothesis, not a
measurement: the burial was not reproduced here.

Also noted while reading it: the diff is internally inconsistent (`render.h` defaults `hollow_push_tau` to 0.f while
`cli.h` defaults it to 0.5f, so anything building RenderParams directly gets the push ungated), and its two halves are
coupled — `--hollow-tau 0` with the rest of it measured 33.8 dB / deep 0.156, worse than either end.

### A b2h3cli run end to end: refine, refit, rig (2026-09-10)

b2ctrain was run on a **b2h3cli** run rather than a b2crunner bundle: `~/Projects/b2h3cli/runs/helical720-20260826-134012`
(243 generated frames at 768x1344, a 720-degree helix of 2 loops at +-30 degrees elevation, 2.98 deg/frame, rmbg mattes,
`points3D.txt` sampled off the SAM-3D-Body mesh). Its sibling `helical720-depth-20260826-134037` (the same path with
`reference_mode: depth`) was tried first and is the dirtier of the two; both are recorded below. Artefacts, splats and
the driver scripts: `out/b2h3/` (not in git).

**The dataset loads as-is.** `07_colmap/` is already a b2ctrain dataset (cameras.txt/images.txt/points3D.txt + images/ +
masks/); no conversion needed.

**Driving b2crunner's steps outside its runner** (`out/b2h3/scripts/`): `colmap_io.py` converts a COLMAP text model to and
from `body2colmap.Camera` (the inverse of `coordinates.world_to_colmap_camera`), and the steps are instantiated directly with
`Step.resolve_params({...})`. `refine_cameras` runs in b2crunner's venv; the body steps need torch and `sam_3d_body`, so they
run in `~/Projects/sam-3d-body/.venv` with b2crunner on PYTHONPATH. Two host details: the local COLMAP needs
`LD_LIBRARY_PATH=/usr/local/lib64:<a cu12 nvidia/cudnn/lib>` or ALIKED aborts on a missing `libcudnn.so.9`, and
`tools/export.py` (what b2h3cli's fit stage shells out to) **drops `hand_pose_params`, `scale_params` and `expr_params`**,
which `refit_body_to_splat`'s MHR replay needs. Re-running the estimator on the same photograph with the STORED `bbox` and
`focal_length` (so neither the detector nor the FOV estimator runs, and the crop is identical) reproduces the fit to
**0.0006 mm** and yields the full parameter set — `refit_params2.py`.

**Camera refinement is worth as much here as it is in b2crunner, and its centre-shift gate is dataset-shaped.**

| | mesh run | depth run |
|---|---|---|
| reprojection error, given -> refined | 1.717 -> 1.700 px | 1.709 -> 1.682 px |
| BA scale inflation removed (trap 1) | +29.2% | +3.2% |
| common-mode rotation removed (trap 4) | 0.63 deg | 1.67 deg |
| mean centre shift | 2.31% of radius — **accepted** | 4.87% — **refused** (gate 3%) |
| PSNR, given -> refined | 33.688 -> **34.086** | 32.605 -> **33.125** |
| worst view | 26.58 -> 29.15 | 25.87 -> 26.75 |

On the depth run the step refused its own solve and published the given poses. The correction it refused was smooth, not
noise: per-frame displacement varied from 4 to 29 cm along the trajectory with a lag-1 correlation of the displacement
vector of **0.996**. Reconstructing what the step would have published (its own `_align_to` + `_remove_common_mode`,
`publish_refined.py`) and training on it gained +0.52 dB — the same size of gain that justified the step. So
`max_centre_shift: 0.03`, calibrated on an 81-frame planar orbit, is too tight for a 243-frame 720-degree helix whose
generated video drifts more; the honest fix is to raise it for helical runs, not to distrust the solve. Exhaustive
ALIKED/LightGlue matching of 243 frames is 16091 verified pairs and ~25 min on the 4070 Ti; the whole step is ~30 min.

**The refit moves much further than it does in b2crunner** — the generated video starts the subject at its own
orientation, so the root yaw is a per-run constant:

| | mesh run | depth run | b2crunner reference |
|---|---|---|---|
| root yaw delta | **+23.9 deg** | **-25.1 deg** | ~3 deg |
| surface -> body median | 4.75 -> **2.52 cm** | 4.11 -> 3.02 | 1.33 -> 0.73 |
| p90 | 11.6 -> 7.05 | 11.6 -> 8.8 | - |
| body -> surface median | 0.99 -> 0.69 cm | 0.74 -> 0.72 | - |
| inside by > 5 mm | 20% -> 15% | 17% -> 16% | 24.7% -> 12.9% |
| mesh vertices inside the frame matte | - | 80.6% -> **88.2%** | - |

The two runs' yaws have OPPOSITE signs from the same `body.npz`, which is the point: each H3 generation places the subject
where it likes and the refit absorbs it. The residual is larger than b2crunner's because the subject wears baggy cargo
trousers and a loose bomber jacket — 40% of surface points sit more than 3 cm outside the body whatever the fit does.
Because the per-view rig excludes the root chain, a 24-degree canonical yaw never becomes a per-view rotation.

**Results, mesh run** (30k iters, `--match-alpha-weight 0.5 --normalize-masked-loss --export-evidence`, hollow 0.5 at
3 cm against the refit mesh, rig = production settings over all 243 views; 4070 Ti):

| run | wall | PSNR | SSIM | splats | mass > 1.5x body radius | behind (probe, 3 cm) | s1 novel body / head / hands |
|---|---|---|---|---|---|---|---|
| base_given | 8m58s* | 33.688 | 0.9478 | 1.20M | 29.2% | - | - |
| base_refined | 3m54s | **34.086** | **0.9521** | 1.13M | 17.4% | 0.0403 | 26.11 / 27.81 / 41.42 |
| hollow | 3m56s | 33.224 | 0.9477 | 1.26M | **1.2%** | 0.0019 | 25.67 / 27.74 / 40.26 |
| hollow_rig | 3m55s | 24.875+ | 0.7984 | 1.16M | 8.0% | 0.0014 | 25.90 / **28.21** / 41.50 |
| hollow_rig_align | 3m54s | 24.186+ | 0.7843 | 1.16M | 5.2% | **0.0012** | **26.31** / **28.21** / **41.70** |

\* contended with the COLMAP job. + canonical export against per-view poses — not the metric for a rig run (STATUS,
"The rig at the INTERMEDIATE stage"): here every one of the 243 views is rigged, so the drop is the full effect.
The learned rotations are textbook: mean 0.66 deg, max 5.65, |mean over views| 0.205 — against the 0.74 / 5.7 / 0.23 measured
at stage 2. `hollow_rig_align` is the best deliverable: equal or better sharpness than the unregularised baseline in every
region, 21x less weight behind the body, a third of its floaters.

**Floaters, and why they are not the refinement.** A viewer render of the depth run showed the subject buried in spiky
sheets. Measured over the model (count and `alpha * scale^2` beyond a radius about the body centre, `floaters.py`), the
un-refined baseline is the WORST and every later step improves it:

| depth run | splats > 1.5R | > 3R | mass > 1.5R |
|---|---|---|---|
| base_given | 17,614 | 1,070 | 94.0% |
| base_refined | 11,520 | 11 | 64.1% |
| hollow | 2,318 | 0 | 7.5% |

The mesh run has **nothing at all beyond 3R** in any variant. The worst offenders are a handful of enormous splats: on the
depth run 39 splats beyond 2.5 m carried 27.8% of the visible mass (median scale 14 cm, alpha 0.76) and 1,215 more sat at or
outside the 1.87 m camera ring. Their in-mask evidence fraction is **1.000** — they are not background junk but
depth-ambiguous splats projecting inside the silhouette, supported by ~4 effective views out of 243 (model median 55).
Cutting at the camera-ring radius costs **0.01 dB** (33.125 -> 33.115) and removes 47% of the visible mass; cutting at 1.2 m
costs 0.21 dB and is too aggressive (`prune.py`). What survives that cut is a "ceiling" above the head and a "floor" below the
feet, inside the ring: those DO draw at the training cameras (a bounding-box cut costs 0.5-0.95 dB, `prune_box.py`) because they
sit behind the subject, occluded where they were trained and revealed by parallax in between. `render --confidence` is what
b2crunner ships against them and it clears most of them here while leaving the subject intact (`out/b2h3/panels/`). Note that a
sharpness metric taken over a crop box is inflated by them — measure inside the projected body region instead (`sharp2.py`),
or the dirtiest run wins.

**Against brush, on this dataset, b2ctrain floats LESS.** The b2h3cli run ships `07_colmap_exports/export_05000.ply`, a
brush export with no floaters at all, which looked like a trainer bug here. It is not: that export is 116k splats from a
different (lighter) recipe stopped at 5k, and at 5k nothing has floaters yet. Matched, on `07_colmap` with b2crunner's argv:

| | 5k | 30k |
|---|---|---|
| brush | 0 beyond 1.5R | **85,645 beyond 1.5R, 11,874 beyond 3R, 98.6% of mass**; PSNR 33.309 / SSIM 0.9420 |
| b2ctrain | 1 beyond 1.5R (3 on a reseed) | 5,436 beyond 1.5R, **0 beyond 3R**, 29.2%; PSNR 33.688 / SSIM 0.9478 |

At 4.2 m the brush model renders PURE BLACK — the camera sits behind an opaque shell of its own junk — while b2ctrain's
shows the subject. On-ring novel views are comparable between the two (`out/b2h3/panels/`). brush at 4.8k under this argv
already has 288k splats against the shipped export's 116k, which is how one can tell the recipes differ.

**When the floaters appear: after growth stops.** Splat count saturates by 15k (`growth_stop_iter` 15000) and the junk keeps
accumulating for the remaining 15k as unsupervised splats drift outward and inflate (b2ctrain, `07_colmap`, exports every 5k):

| iter | splats | > 1.5R | > 3R | mass > 1.5R | PSNR | SSIM |
|---|---|---|---|---|---|---|
| 5000 | 320k | 3 | 0 | 1.2% | 26.547 | 0.8090 |
| 10000 | 777k | 186 | 0 | 12.3% | 28.648 | 0.8848 |
| 15000 | 1.19M | 1,487 | 0 | 25.5% | 31.843 | 0.9292 |
| 20000 | 1.19M | 3,466 | 0 | 36.0% | 32.856 | 0.9397 |
| 25000 | 1.20M | 5,940 | 2 | 46.5% | 33.407 | 0.9452 |
| 30000 | 1.20M | 8,430 | 699 | 91.3% | 33.720 | 0.9480 |

Stopping early is NOT free — PSNR is still climbing at 30k (25k costs 0.31 dB for half the floater mass). The cheap levers are
the ones already measured: the camera-ring prune (0.01 dB), the hollow loss (1.2% of mass left at 30k) and
`render --confidence`.

### The face per view, and the anchor's eyes injected (2026-09-11)

The body rig poses the arms per view; the face got nothing, and the eyes of the helical frames are the diffusion's
invention per frame (three consecutive frames of F3 show three eye states: half closed, open looking sideways, open
with makeup; pass 2 re-denoises even the reinjected anchor frame). Nothing multi-view recovers that, so the plan was
(a) fit the MHR head per view — neck/head rotations plus the 72 expression blendshapes SAM-3D-Body leaves at zero —
and hand the trainer the per-view head geometry the way the rig hands it the arms, and (b) build eyeballs from the
MHR eye joints, texture them from the anchor photograph, and paste them into every frame through that frame's fitted
lids. All on the MHR skeleton (user's call: keep it unified; uniface was surveyed and adds nothing the MHR route needs).

What was learned building it (scratch tooling in `out/face/`, details in the session memory):

* The pipeline's face landmarker (BlazeFace short-range on the whole 1080x1920 frame) misses the ~80 px face in 63 of
  81 helical frames and hallucinates faces on the trousers in 4 of them. Cropping from the **projected MHR head**
  (roll from the projected head-up axis) needs no detector and landmarks 35 of 81 — every view facing the camera to
  ~85 deg; the rest are back views or hair-covered profiles.
* Per-view fit, all views in one batched Adam run: landmark rms 6.4 -> 4.6 (pose) -> **2.1 px** (pose + expression).
  The six neck/head rotations counter-rotate into a lateral head shift when unregularised (frame 67: 61 mm); an L2 of
  50 on them removes that at zero residual cost. The fitted lids follow the frames' half-closed eyes (frame 73).
* MHR has eye joints (122/124, gaze children 123/125) but no eyeballs; the lid ring sits 15.4-20.6 mm from the joint.
  The eyeball is a 15.5 mm sphere at the joint, clipped **in 3D** to the fitted lid contour's cylinder (a 2D polygon
  leaks at grazing angles), depth-tested at 1 mm against the head mesh with the eye-disk faces removed and back-face
  culling off, and skipped when less than half its lid polygon is visible (the far eye peeking past the model's
  narrower nose). Texture: the anchor through a PnP camera refined with expression (1.4 px); the ~30% of the iris under
  the anchor's upper lid is filled by a radial colour profile around the gaze axis; gaze fixed relative to the head.
* Trainer: **rig v3** = the v2 rig plus a per-view displacement of every rig vertex (magic `B2CRIG3`), added to a bound
  point before the LBS blend so it rides the learned rotations; `tests/tests.cpp` covers it. Frames without a fit get
  the deltas interpolated across gaps of <= 4 frames and (first runs) faded to canonical over 3 — later held, see below.

Stage-5 argv of the earlier rig runs, F3 bundle, `rig_noroot` (all joints but the root chain), HEAD binary:

| run | frames | rig | PSNR (21 train views) | head s1 novel / train | eye MAE vs the eye model, 11 novel views |
|---|---|---|---|---|---|
| base | original | v2 | 28.757 | 27.98 / 28.18 | 66.6 |
| delta | original | v3, pose+expr | 28.615 | 27.59 / 27.82 | 59.4 |
| eyes | eyes pasted (32 views) | v3, pose+expr | 28.631 | 27.61 / 27.81 | **52.5** |
| expr_eyes | eyes pasted | v3, expression only | 28.679 | 27.92 / 28.40 | 56.8 |
| v2_eyes | eyes pasted | v2 | 28.791 | **28.28 / 28.67** | 61.2 |

(eye MAE: mean |render - eye model| inside the eye model's own mask, rendered at the novel cameras from the canonical
head; a coarse number, the render is soft against a crisp model, but it orders the runs the way the panels do.)

What the panels (`out/face/panel2.jpg`) show: base has smeared dark slits for eyes at every novel view; with the eyes
pasted and the full v3 deltas there is an eyeball with sclera and a dark iris at every novel view, including the one
between frames 73/74 where the frames' lids were half closed — the deltas explain those lids per view, so the canonical
opening converges to the anchor's; without deltas (`v2_eyes`) that view stays a dark slit. The deltas cost head
sharpness (-1.4% s1) and the cost sits in the rigid part: expression-only deltas are s1-neutral and keep most of the
eye gain. Face deltas alone (`delta`) do nothing for sharpness — the alignment loop and the learned rig already absorb
the per-frame head wobble in 2D — they are the enabler for the eyes, not a lever on their own.

**Second subject (`~/Projects/facerefine`, run 18, body replayed from the ply's MHR record) and the lever sweep
(2026-09-11, later).** On this subject full deltas stabilised the face across views but softened every single view,
expression-only deltas were the crisp ones but swam like the baseline — a toss-up, so a metric for the swim was built
(`swim.py`: landmark the render at 52 dense novel cameras, measure the rendered landmarks against the projected
canonical head; the std of the per-view shift is the swim) and the rigid part was taken apart. All with the eyes
pasted; s1 split into face (expression-moved vertices, `mov > 0.5`) and the rest of the head:

| deltas | face s1 novel | hair+head s1 | swim px | eye MAE |
|---|---|---|---|---|
| none (base, v2 rig) | 30.44 | 40.29 | 5.02 | 51.3 |
| expression only | 30.87 | 40.75 | 4.99 | 43.1 |
| full (neck+head 6 DOF + expr) | 29.48 | 38.39 | 3.44 | 41.8 |
| head-only rotation (3 DOF) | 29.81 | 38.81 | 3.09 | 41.7 |
| 6 DOF, temporal smoothing | 29.51 | 39.27 | 3.39 | 40.2 |
| 6 DOF, visibility-weighted landmarks | 29.64 | 38.95 | 3.53 | 38.9 |
| 6 DOF, unfitted views hold the neighbour instead of fading | 29.10 | 38.51 | 2.39 | 43.6 |
| 6 DOF, rigid part only within 45 / 60 deg of frontal | — | 39.60 / 38.98 | 4.22 / 3.58 | 44.5 / 46.4 |
| **6 DOF + hold, deltas on the FACE only (mov > 0.5, 3 cm fade)** | **30.48** | **40.13** | **1.58** | **35.5** |
| head-only + hold, face only | 30.57 | 39.69 | 1.87 | 37.9 |

The same trio without the alignment loop keeps the ordering (rigid deltas face 26.7 vs 27.7), so the blur is not an
alignment interaction. Every variant of the rigid fit blurred the same amount — the blur is not fit noise either. It
is the hair: the deltas moved every head vertex, and the frames' hair does not follow the face fit (nor do the back
views, which pin the hair canonical), so the hair was supervised in two places and blurred, and the face with it.
Restricting the deltas to the face core (`FACE_ONLY`, `FACE_MOV=0.5`, fade over 3 cm) keeps the hair with the learned
rig: sharpness at the expression-only level, the swim 3x lower than any other variant, the best eyes. The first
"face-only" attempt used `mov > 0.01`, which is the whole head (the expression basis touches every head vertex by a
hair) — an error that hid this for two runs. Panels `out/facerefine/panel_head2.jpg`.

Also seen on this subject: frames 67-70 have the hand in front of the face and the head mesh cannot know, so one
frame got an eye pasted on the hand; a pixel gate (something darker than the skin inside the lid polygon) catches
two of the four. Occlusion from the per-view posed body is the proper fix.

**Four subjects (2026-09-11, evening).** F3 and facerefine above, plus bundles 22 (a problematic face: the frames
deviate 12 px rms from the canonical before the fit, the baseline's eyes are blue smears) and 19 (a good face, the
control that must not degrade). All from the ply's MHR record with `run_subject.sh`. base = v2 rig, original frames;
hold = full 6-DOF+expression deltas, unfitted views hold their neighbour; face05_hold = the same restricted to the face.

| subject | run | face s1 | hair s1 | swim px | eye MAE |
|---|---|---|---|---|---|
| F3 | base / hold* / face05_hold | 28.95 / 28.67 / 28.22 | 22.90 / 22.41 / 22.42 | 2.47 / 2.01 / 2.02 | 66.6 / 52.5 / 55.7 |
| facerefine | base / hold / face05_hold | 30.44 / 29.10 / 30.48 | 40.29 / 38.51 / 40.13 | 5.02 / 2.39 / 1.58 | 51.3 / 43.6 / 35.5 |
| 22 | base / hold / face05_hold | 21.48 / 21.52 / 21.94 | 19.29 / 18.39 / 19.22 | 3.03 / 2.90 / 3.09 | 65.6 / 27.8 / 28.3 |
| 19 | base / hold / face05_hold | 35.32 / 34.96 / 34.96 | 25.76 / 25.00 / 26.42 | 1.79 / 1.93 / 2.13 | 81.5 / 41.8 / 51.5 |

(*F3's "hold" column is the `eyes` run, full deltas with fade.) The eyes are the headline on every subject: base eye
error 51-82, with deltas 28-56, and the panels (`out/s22/panel_head.jpg`, `out/s19/panel_head.jpg`) show why — 22's
blue smears and 19's closed slits both become open eyes with an iris. Face sharpness stays within +-2.5% of base on
every subject in every config (the metric's resolution); the one systematic effect is the hair, which full deltas
cost 2-5% on three of four subjects and face-only deltas leave alone (19: +2.6%). Swim: face-only is a large win
where the subject swims (facerefine 5.0 -> 1.6), neutral where it does not (19, F3). Full deltas give somewhat
better eyes on 19 (42 vs 52). **Default: face-only + hold** (`FACE_ONLY=3 FACE_MOV=0.5 HOLD=1`, 6-DOF fit with the
rotation prior) — it never costs hair or face sharpness; full deltas remain the option when a subject's eyes need it.

**Integrated (2026-09-11, later).** Trainer: rig v3 is commit e8f43ac (`B2CRIG3` named in `--body-rig`'s help, which
is how b2crunner's brush step and doctor tell a v3-capable binary). b2crunner dd899c8: four steps in stage 6 of
`fast_helical_native.yaml` on the `face_refine` setting (default on, needs `refit_body`) — `detect_face_views`,
`fit_head_per_view` (sam3dbody env), `paste_eyes`, `build_face_rig` (face-only + hold, `pose_prior` 50) — see its
docs/face-refine.md; `pipeline/body_rig.py` writes/reads v3; its Dockerfile pins b2ctrain at the main HEAD that follows this commit (the builder requires branch head == pin). Driven standalone on the
facerefine subject the steps reproduce the scratch run above: 37 landmarked (the scratch's roll normalisation had its
sign inverted and doubled the roll instead of removing it — MediaPipe coped; fixed), 34 fitted at 8.4 -> 4.6 -> 2.6 px,
32 frames pasted, and trained with the same argv (`out/facerefine/integrated`): face s1 30.48 / hair 40.04 / swim
1.65 px / eye MAE 38.2 against face05_hold's 30.48 / 40.13 / 1.58 / 35.5 (base 30.44 / 40.29 / 5.02 / 51.3); panel
`out/facerefine/panel_integrated.jpg`. No image built yet.

Open: the eye region gets no loss-weight boost yet (`weights/` sidecars exist for that); gaze is fixed to the anchor's;
closed-eye detection is not attempted; an eye pasted onto a dark hand in front of the face survives the darkness gate
(occlusion from the posed body is the proper fix); the texture comes from a 90 px-wide face (iris ~9 px), the same
scale as the frames' eyes, so it is not the resolution bottleneck.

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
src/dataset/            COLMAP + sidecar loader (masks/normals/weights/labels/init.ply)
src/ply.*               PLY reader/writer (brush field order + ev_* evidence block + seg_label/seg_conf)
src/model.*             GPU splat parameter/moment buffers (SH bands >= 1 optionally fp16 via ShBuf in gpu/util.cuh)
src/gpu/                CUDA kernels: project, binning, raster fwd/bwd (+ tensor-core variant), loss, optim
                        (fused projection-backward + Adam + noise), refine, images (resolution pyramid),
                        align (LK flow, Gaussian blur, Lanczos warp for the alignment loop)
src/train/              trainer (phases: main run + alignment refits), GPU-resident views, splat init, evidence
                        (+ the --export-labels class vote, same replay)
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
