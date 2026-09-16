# Meshification in b2ctrain: implementation plan

Written 2026-09-16 from the local experiments in `out/mesh/` (README.md, REPORT.md, `tools/`). The Python tools there
are the reference implementation this plan ports, the way brush was the reference for the trainer.

## Start here (for the session that implements this)

Everything below refers to `~/Projects/b2ctrain/out/mesh/` (local experiment area, NOT part of the build). Nothing in
this plan has been committed to either repo; the Python tools are the reference, the C++/CUDA does not exist yet.

- **Reference tools:** `out/mesh/tools/*.py`. The chain scripts that ran the two subjects end to end are `tools/sg4m.sh`
  (+ `run_sg8m.sh`, 00307) and `tools/f3_c.sh` (F3); the klein loop is `tools/uv_refine_loop.py` (driver) calling
  `uv_render.py`, `klein_refine.py`, `uv_backproject.py`; masks from `uv_protect.py` / `uv_protect_cap.py`.
- **Venvs:** the geometry tools re-exec into `~/Projects/masktest/.venv` (torch, no open3d); the open3d tools use
  `~/Projects/tsdf/.venv`; the UV tools need xatlas + open3d + scipy + plyfile — a scratch venv was used
  (`uv venv --python 3.12 && uv pip install xatlas trimesh open3d opencv-python-headless scipy plyfile`); klein runs in
  `~/Projects/datasetgen/envs/ml` with `PYTHONPATH=~/Projects/datasetgen` (the fp8 loader `datasetgen/flux2_fp8.py`).
- **Subjects and assets:** 00307 = bundle `~/Downloads/all-results/helical-splat_00307-20260914-014336-4a6bed`
  (symlink `out/mesh/bundle`), splat `sg8/scene.ply`, meshes `sg8m/mesh_hybrid.ply` / `mesh_cap.ply`, SAM body
  `ds_int/mesh.ply`, cap `samcap/cap_sam.ply`, cameras `cams/sg81.json` (train, photo at index 0), `cams/helix720.json`,
  `cams/orbit4.json`, photo sheet `~/datasets/testing/splat_00307.png` (front left half, back right half), SAM head
  centroid (-0.0032, 0.663, -1.9542). F3 = bundle `~/Downloads/fast_helical_native-F3-pass2-shift5-20260909-020603-792cf0-result`,
  everything under `out/mesh/f3/` and `f3d/mesh/`, cap `f3/cap_sam.ply`, SAM body `f3/mesh_world.ply`, photo
  `~/datasets/isolated/00.png` (front left half); F3's bundle has no body rig.
- **UV/klein results to compare against:** `out/mesh/tex/s307/` and `tex/f3/` (`mesh_uv.*`, `texture.png`,
  `protect_cap.npy`, `cams_*.json`, `loopN/texture_cur.png` with `loop4` = cap protected, `loop3` = unprotected,
  `loop2` = head band protected), `tex/s307cap/` = the splat-only seed; refs split in `tex/refs/`. Panels in
  `out/mesh/samcap/panel_tex_*.jpg`. Exact prompts and flags are in the `loopN.log` files (first line of each).
- **Numbers to reproduce:** unwrap 300 k tris / 13.9 k charts / 4 min; klein 17 s per view at 6.9 GB peak, strength 0.8
  / 12 steps; seed test Laplacian sharpness (before → after klein) hybrid 379 → 552, splat 250 → 472 on body_a000.
- **Verified so far:** the full local chain on both subjects; the klein loop on both; UV-space klein = failure (control);
  strength sweep; face policies; seam-blend attempts (ineffective); splat vs frames seed. NOT run: `mesh-fuse --protect`
  on F3; anything on a pod.

## Decisions this plan rests on

1. **Splat-based.** Geometry and the seed texture come from the trained splat's own renders (`b2ctrain probe`), the SAM
   body mesh and the face cap. The pass-1 frames are not touched after training: the frames bake (body rig at bake time +
   one flow iteration) bought ~10 % on a sharpness metric after the diffusion pass on 00307 and nothing on the face
   (`out/mesh/samcap/panel_tex_seed_body.jpg`, `panel_tex_seed_head.jpg`). `align.cu` stays a training-time tool.
2. **b2ctrain owns every raster/bake/fusion step**, as `b2ctrain mesh-*` subcommands in C++/CUDA. It already has the
   pieces the Python tools re-implement in torch: splat depth (`probe --depth`), mesh depth on the GPU (`gpu/meshdepth.cu`),
   PLY/OBJ mesh IO (`dataset/mesh.cpp`), cameras JSON, PNG IO (stb), the rig (`gpu/deform.cu`, not needed here).
3. **b2crunner owns orchestration and the ML models**: SAM-3D-Body, Sapiens normals, the cap, and the FLUX.2 klein
   texture refinement (diffusers). The klein loop's two raster halves are b2ctrain calls.
4. **The face policy is a flag, decided on the pod A/B, not here**: protect the projected-photo cap (identity kept, a
   sharpness step at the cap's edge) or let klein repaint the face (seamless, mild identity drift). Both are one mask.

## The chain

```
b2crunner: train_splat ─► scene.ply
           sam3d_body  ─► mesh.ply (SAM body, world frame)      Sapiens ─► normals/ (camera space, [+X,-Y,-Z])
           face cap    ─► cap.ply (one Gaussian per photo pixel, on the SAM head)        photo panels front/back

b2ctrain probe   scene.ply  x {train cams, helix720, orbit4}  --depth --images     ─► probes/<set>/<name>.png + .zfirst.png
b2ctrain mesh-fuse    probes + carve + SAM prior                                   ─► mesh_raw.ply
b2ctrain mesh-refine  mesh_raw + normals/ (+ cap mask: keep raw)                   ─► mesh_geom.ply
b2ctrain mesh-bake    mesh_geom + orbit4/helix720 probes (+ --project photo on the cap) ─► mesh_col.ply (vertex colours)
b2ctrain mesh-unwrap  mesh_col                                                     ─► atlas/ (mesh_uv.obj, texture.png, position/normal/mask, protect masks)
b2crunner refine_texture: for each view  b2ctrain mesh-render ─► klein img2img ─► b2ctrain mesh-backproject   ─► texture_final.png
```

Runtime target on a 4070 Ti: probes ~1 min, fuse < 1 min (the torch version is 138 s and GPU-bound; a fused kernel
should be well under), refine ~10 s, bake ~10 s, unwrap ~1 min (xatlas on 300 k triangles is the floor), klein loop ~6 min
for 11 views (17 s/view klein + ~10 s model load per process today; one resident process brings it to ~4 min).

## Subcommands

Every subcommand: `--cameras CAMS.json` in the `out/mesh` cams format (OpenGL c2w `rotation`, `position`, fx fy cx cy,
width/height at the top), PLY in/out through `dataset/mesh.cpp` extended with vertex colours and normals, PNG through stb.

### `mesh-fuse` (reference: `tools/tsdf_torch.py`, `tools/body_sdf.py`)

TSDF fusion of the probe depths with silhouette carving and the SAM body as a signed-distance prior.

- Inputs: `--views CAMS DIR` (repeatable; `<name>.zfirst.png` 16-bit mm + RGBA `<name>.png` whose alpha is the mask),
  `--carve CAMS DIR` (masks for carving), `--prior mesh.ply`, `--bbox` or `--bbox-from mesh.ply` (+0.12 m margin),
  `--voxel 0.002`, `--trunc` (default 4 voxels), `--edge-tol 0.01`, `--erode 2`, `--carve-min 3`, `--carve-dilate 2`,
  `--prior-mode fill|union|protect`, `--prior-offset -0.005`, `--min-comp 0.05`,
  `--protect cap.ply RADIUS --protect-band 0.02 --protect-outside 0.005` (the sgc3 recipe: the prior is the authority
  inside the cap's region, blended over the band — the fix for F3's flattened face, see REPORT.md).
- Kernels: (1) per view, one thread per voxel: project, depth test with edge rejection (depth discontinuity within
  `edge-tol` → skip) and eroded mask, TSDF update (weighted running mean); (2) carve: count views where the voxel projects
  outside the dilated mask, carve where ≥ `carve-min`; (3) prior SDF of the SAM mesh on the same grid: closest triangle
  through a uniform grid of triangle lists (the mesh is 20 k triangles; brute force per voxel over the local cells),
  sign by the pseudo-normal; (4) fill / union / protect combine; (5) marching cubes (the standard 256-entry table, one
  thread per cell, two passes: count then emit, vertices welded on the shared edge index); (6) connected components on
  the output, drop those under `min-comp` of the total area.
- Memory: 2 mm voxels over ~0.65 × 1.7 × 0.75 m = ~100 M voxels; sdf f32 + weight f16 + prior f32 ≈ 1 GB. Fits; if a
  pod card is shared with the trainer, halve with f16 sdf.
- Verification: same inputs as `sg8m` (00307) and `f3d` (F3): vertex count within 5 %, Hausdorff distance to the torch
  mesh < 2 voxels on 99 % of vertices; `b2c_tests`: a synthetic sphere rendered from 24 views fuses to a sphere with
  radius error < 0.5 voxel; marching cubes on an analytic SDF.

### `mesh-refine` (reference: `tools/normal_refine.py`, `tools/blend_face.py`)

Move vertices so face normals match the Sapiens normal maps, keep the cap region raw.

- Inputs: mesh, `--views CAMS NORMALS_DIR`, `--iters 400 --lam-pos 10 --lam-lap 2`, `--keep cap.ply RADIUS` (vertices
  within the cap's footprint are pinned to their fused positions — blend_face's job).
- Per iteration: depth-rasterise the mesh at each view (`meshdepth.cu`), sample the normal map where the vertex is
  visible (depth within tolerance), decode `[+X,-Y,-Z]`, rotate to world, facing-weighted mean = target normal per
  vertex (computed once; the reference does it once too); then Adam on vertex positions with the loss
  Σ_f ‖n_f(V) − n_t,f‖² + λ_lap ‖L V‖² + λ_pos ‖V − V₀‖². Gradients: face-normal gradient w.r.t. its three vertices
  (closed form), uniform Laplacian by CSR adjacency, both as scatter-adds; the trainer's Adam kernel applies.
- Verification: per-vertex position difference to the torch result < 0.5 mm median on 00307/F3.

### `mesh-bake` (reference: `tools/texture.py`, `tools/smooth_colours.py`, `tools/cap_texture.py --project`)

Per-vertex colour from the splat's RGBA probes, then the photo projected onto the cap region.

- Inputs: mesh, `--views CAMS DIR` (orbit4 + helix720 probes), `--mode trimmed --trim 0.12 --power 6 --alpha-min 128`,
  `--smooth 1` (one Laplacian pass on colours), `--project PHOTO.png CAMS INDEX --cap cap.ply` (the photo's pixels through
  its camera onto the vertices the cap covers, seam-levelled against the surrounding bake as `cap_texture.py` does,
  `--lowpass 300`).
- Per view: depth-rasterise at 2× supersampling (`meshdepth.cu`), each vertex visible if its depth agrees with the
  buffer at its sub-pixel; sample the RGBA bilinearly; weight = max(0, n·v)^power; samples kept as fp16 [views × V × 4]
  (369 views × 1.4 M × 4 × 2 B = 4 GB — chunk vertices at 256 k as `gpu_raster.py` does); trimmed median per vertex on
  the GPU; unseen vertices filled by diffusion from seen neighbours (`index_add_` in the reference: Jacobi iterations).
- Verification: seen-vertex colour difference < 1/255 mean vs the torch bake (the GPU port of texture.py was verified at
  0.6/255 against the open3d original; the same bar).

### `mesh-unwrap` (reference: `tools/uv_bake.py`, `tools/uv_protect.py`, `tools/uv_protect_cap.py`)

Decimate, unwrap with xatlas, rasterise the atlas maps, transfer the vertex colours, write the protection masks.

- Vendor `xatlas.h/.cpp` into `third_party/` (single TU, MIT). Quadric decimation to `--tris 300000` needs an
  implementation: a standard edge-collapse with quadrics (~300 lines) — or vendor `meshoptimizer`'s `simplify` (single
  file, MIT), which is the pragmatic choice.
- xatlas options that matter on a TSDF surface: `max_cost 8`, `normal_deviation_weight 0.5` (2 / 2 give 24 k charts and
  12 min), pack `resolution 4096`, `padding 6`, `bilinear`; **UVs come back normalised [0,1]**; assert `atlas_count == 1`.
- Atlas maps at `--res 4096`: rasterise the flat UV mesh (the trainer's own tile raster is overkill; a plain per-triangle
  scanline kernel does it), per texel: position, normal, triangle id; colour from the ORIGINAL (undecimated) mesh at the
  closest surface point (uniform-grid closest-triangle, same kernel as `mesh-fuse`'s prior SDF); edge padding by
  nearest-covered texel (jump flooding).
- Outputs: `mesh_uv.obj` + `.mtl` (v = texture row 0 at the top, i.e. standard), `texture.png`, `position.npy`-equivalent
  (write as EXR-free raw `.f32` + a small JSON header, or 16-bit PNG per channel — b2crunner reads either),
  `mask.png`, `mask_dilated.png`, and `protect_cap.png`: texels within 3 mm of a cap Gaussian, kept where the cap fills
  ≥ 75 % of a 6 mm disc (a geometric closing of the strand holes and an erosion of the border) — `--protect-head`
  alternatively writes the wider face band (radius 0.12 m about the SAM head centroid, front half, facing the photo).
- Verification: render the atlas back at the probe cameras and compare with the vertex-coloured mesh render (< 2/255
  mean on the subject, the 4096 atlas resolves ~0.7 mm vs 1 mm vertex spacing).

### `mesh-render` (reference: `tools/uv_render.py`)

Render the UV-textured mesh at cameras: RGBA on 0.5 grey, `<stem>.depth` (z), `<stem>.cos` (|n·v|), `<stem>.dens`
((fx/z)² × 1e-6), and `--aux MAP` sampled per pixel (nearest) — the loop's "best so far" map. 2× supersampled, box-filtered.
`--make-cams` generates the orbit / head camera sets (radius 2.2 m / 0.9 m, fov 1.15 × the height / 0.36 m, `--centre`
for the head; the SAM head centroid, never the bbox centre — a ponytail or a walking pose moves the bbox centre by 0.2 m).
Same raster as the bake's depth pass with a texture lookup added.

### `mesh-backproject` (reference: `tools/uv_backproject.py`)

For every covered texel: project into the view, visible if |z − depth(pixel)| < 8 mm and inside the eroded mask,
weight = cos^4 × density, take the view's colour where the weight beats `--best MAP` (updated in place), restricted to
`--mask <stem>.mask.png` (the repainted pixels). Pure gather, one thread per texel.

## b2crunner side

- `meshify` step (after the intermediate or final splat): the probe + fuse + refine + bake + unwrap calls, parameters from
  the workflow YAML, the SAM mesh and cap from the existing steps. Outputs land in the result zip as `mesh/` (obj, mtl,
  textures, protect masks).
- `refine_texture` step: the sequential loop (`tools/uv_refine_loop.py` is the reference): views in the order head
  0/45/315 (1024²) then body 0,180,90,270,45,135,225,315 (704 × 1408: the inpaint pipeline caps init images at 1 MP);
  per view `mesh-render` → repaint mask = subject ∧ (cos⁴·dens > 1.5 × best ∨ never refined) ∧ ¬protected, dilated 8 px →
  klein (`Flux2KleinInpaintPipeline`, strength 0.8, 12 steps, guidance 1.0, seed fixed, the front and back photo panels as
  SEPARATE reference streams via `prepare_image_latents`, refs ≤ 0.3 MP each, mask feathered 12 px in image space) →
  `mesh-backproject`. One resident klein process for the whole loop (the model loads once). Prompts: the caption's
  clothing description + the fixed "keep the pose, framing and silhouette" clause; a portrait variant for head views.
  No rear head close-ups (klein paints a figure into a featureless hair mask).
- klein assets: `black-forest-labs/FLUX.2-klein-4B` (text encoder, VAE, ~8 GB) + `FLUX.2-klein-4b-fp8` (3.8 GB) on the
  volume; the fp8 loader is `datasetgen/flux2_fp8.py` (move it into b2crunner's `pipeline/`). VRAM 6.9 GB peak at
  704 × 1408 with two 0.3 MP refs; the text encoder can run on the CPU (one prompt, ~5 s).
- Workflow: `meshify` and `refine_texture` off by default until the pod A/B; `face_policy: protect_cap | protect_head |
  none`.

## Verification plan

- Unit (`b2c_tests`): marching cubes on an analytic sphere SDF; TSDF of a synthetic sphere from 24 rendered depth maps;
  closest-triangle SDF against brute force on a small mesh; a textured cube through unwrap → render → backproject
  round-trips its texture (< 1/255).
- Parity with the Python reference on the two local subjects (`out/mesh/sg8m`, `out/mesh/f3d`), each stage against its
  tool with the bars above; the panels `out/mesh/samcap/panel_tex_*` are the visual reference for the klein loop.
- End to end: the A/B packaged in `out/mesh/ab2` gains a fourth arm (the b2ctrain mesh), and the pod pass-2 run decides
  the face policy.

## Order and estimates

| phase | what | size | why first |
|---|---|---|---|
| 1 | `mesh-fuse` (+ `--protect`), `mesh-refine`, `mesh-bake` | ~1.2 k lines C++/CUDA | settled recipes; re-fusing F3 with `--protect` answers the profile question at the same time |
| 2 | `mesh-unwrap` (xatlas + meshoptimizer vendored), `mesh-render`, `mesh-backproject` | ~900 lines + vendored | the klein loop's raster halves |
| 3 | b2crunner `meshify` + `refine_texture` steps, klein assets on the volume, doctor checks | ~500 lines Python | orchestration, once the CLI surface exists |
| 4 | pod A/B (ab2 + the new arm), face policy, then the YAML defaults | — | the only decision that needs a pod |

Roughly a week of focused work for phases 1–2 with verification, a day or two for phase 3.

## Open items and risks

- F3's fused face is flat because the fill-mode prior leaves the face to the splat's depth; `--protect` is the intended
  fix but has only been run in the sgc3 chain (00307). Run it on F3 first thing in phase 1.
- Klein's per-view inventions differ between seeds of the *texture* (the sleeve patch on 00307 came out as two designs);
  fixed noise seed per view keeps a run reproducible but not the content — acceptable for now, noted for the A/B.
- The cap's sharpness step: the projected photo is softer than klein's ring (cap Gaussians ~1.1 mm, klein at ~0.5 mm/px
  in the head view). A tone blend does nothing (measured); only a sharper cap or the unprotected policy closes it.
- xatlas time scales with chart count; if a mesh comes out noisier than these two, `max_cost` up or a Laplacian smoothing
  pass on the decimated copy used for charting (the colours are transferred from the original anyway).
- Memory on shared pod cards: fuse's 1 GB grid and the bake's fp16 sample buffer are the two big allocations; both chunk.
