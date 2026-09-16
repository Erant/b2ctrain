# Meshification: `b2ctrain mesh-*`

The C++/CUDA port of the local `out/mesh/tools` chain (docs/mesh-plan.md is the plan; this is the usage). Every
subcommand takes camera lists in the `out/mesh` cams format (`{width, height, cameras: [{name, fx, fy, cx, cy,
position, rotation (OpenGL c2w)}]}`, what `probe`/`render` read), meshes as binary PLY (float xyz, optional
`nx ny nz`, optional `red green blue`) or OBJ, images as PNG, and float maps as raw little-endian float32 (`.f32`,
no header; the size is the consumer's: W x H for a view map, R x R (x 3) for an atlas map). `--help` on each.

    b2ctrain probe --splat scene.ply --cameras CAMS.json --output-dir probes/NAME --tau 0.5 --depth --images
    b2ctrain mesh-fuse --output mesh_raw.ply --views CAMS DIR [--views ...] --carve CAMS DIR [...] \
        --bbox-from body.ply [--bbox-margin 0.12] --voxel 0.002 --prior body.ply --prior-mode fill --prior-offset -0.005 \
        --carve-min 3 --carve-dilate 2 --min-comp 0.05 [--protect cap.ply 0.03 --protect-groups none --protect-band 0.02]
    b2ctrain mesh-refine --input mesh_raw.ply --output mesh_geom.ply --views CAMS NORMALS_DIR --iters 400 --lam-pos 10 --lam-lap 2 [--keep cap.ply 0.03]
    b2ctrain mesh-bake --input mesh_geom.ply --output mesh_col.ply --views CAMS DIR [--views ...] --mode trimmed --trim 0.12 --power 6 \
        --smooth 1 [--project CAMS INDEX --cap cap.ply [--photo IMG] --lowpass 300]
    b2ctrain mesh-unwrap --input mesh_col.ply --output atlas/ [--tris 300000 --res 4096 --pad 6 --max-cost 8 --chart-smooth 2] \
        [--cap cap.ply] [--protect-head --centre x,y,z --facing x,y,z]
    b2ctrain mesh-render --atlas atlas/ --cameras CAMS.json --output DIR [--texture T.png] [--depth] [--aux best.f32]
    b2ctrain mesh-render --make-cams --atlas atlas/ --output CAMS.json [--head --centre x,y,z] [--azims ..] [--elevs ..]
    b2ctrain mesh-backproject --atlas atlas/ --cameras CAMS.json --images DIR --renders DIR --output T.png --texture T.png \
        [--mask-dir DIR] [--best best.f32]

## What each one does

* **mesh-fuse**: TSDF of the probes' `<stem>.zfirst.png` depth (mm) with per-view rim erosion and depth-gradient edge
  rejection, silhouette carving (a voxel outside `--carve-min` dilated masks is free space), the body mesh's signed
  distance as the prior for what no view saw (`fill`), face protection (`--protect`: no view is fused within RADIUS of
  the cap's points, the prior shapes the face, a band blends the two), enclosed free-space regions solidified
  (`--keep-cavities` to skip), marching cubes, and the small components dropped. The prior's sign is a winding count
  per grid row (an arm pushed into the torso stays inside); the grid's z = 0 face must be outside the body.
* **mesh-refine**: target normal per vertex = the facing-weighted mean of the Sapiens maps where the vertex is
  visible (its depth agrees with the mesh's own raster); Adam on the positions with face-normal, uniform-Laplacian and
  positional terms (the reference's loss and scales). `--keep` pins the cap's region to the fused positions.
* **mesh-bake**: per-vertex colour = the trimmed weighted median (facing^power) of the RGBA views that see the vertex
  (depth-tested at 2x supersampling, eroded alpha), unseen vertices filled from their neighbours, `--smooth` 1-ring
  median passes, and `--project`: the cap's Gaussians rasterised at the photo camera are the photo's face; the
  vertices that camera sees inside the cap's coverage take that colour, seam-levelled (the cap's low band swapped for
  the bake's). Sampling is at pixel centres (the Python reference's bilinear sampler sits half a pixel off).
* **mesh-unwrap**: meshoptimizer decimation, xatlas on a Laplacian-smoothed COPY (`--chart-smooth`, 4x fewer charts
  and 8x faster on a TSDF surface; positions stay), atlas maps (`position.f32`, `normal.f32`, `mask.png`,
  `mask_dilated.png`, row 0 = v 1), colour from the ORIGINAL mesh at the closest point, jump-flood edge padding,
  `protect_cap.png` (the cap's footprint, closed and eroded) and `protect_head.png` (a face band about `--centre`).
  xatlas may pack larger than `--res` (5-6k on these subjects); the maps are rasterised at `--res` regardless.
* **mesh-render / mesh-backproject**: the texture-refinement loop's raster halves (b2crunner's `refine_texture`):
  RGBA on grey + `<stem>.depth.f32` / `.cos.f32` / `.dens.f32` / `.aux.f32`, and the gather of a repainted view back
  into the texture where its weight (cos^4 x texel density) beats `--best`.

## Verified (2026-09-16, 4070 Ti)

Against the Python reference on the two local subjects (out/mesh/sg8m = 00307, out/mesh/f3d = F3), same inputs:

| stage | bar (plan) | measured |
|---|---|---|
| fuse 00307 / F3 / sgc3-protect | V within 5 %, 99 % of vertices < 2 voxels | V -0.10 / -0.22 / -0.12 %; p99 surface distance 0.10 / 0.07 / 0.15 mm both ways |
| refine 00307 | median vertex difference < 0.5 mm | 0.012 mm (mean 0.20; the reference's angle-weighted vertex normals vs area-weighted) |
| bake 00307 (369 views) | seen-vertex colour < 1/255 mean | 0.000/255 against the reference with its half-pixel sampling offset corrected; 4/255 against it as is |
| smooth | — | identical |
| cap projection | — | 4.9/255 mean on the cap's vertices (the same half-pixel offset in the reference's remap; visually identical) |
| unwrap + render | atlas render vs vertex-colour render < 2/255 on the subject | 1.4/255 (helix views) |
| unit tests | sphere MC, sphere TSDF, closest point, cube round trip | radius error 0.005 / 0.06 voxels, exact, 0.000/255 |

Times (00307, 1.42 M vertices): fuse 2.3 s (450 views + 450 masks, 129 M voxels), refine 4.3 s, bake 2.0 s,
unwrap 56 s (xatlas 50 s), render 0.1 s/view, backproject ~0.2 s/view.

Two deliberate deviations from the reference: the cavity fill (the reference leaves interior sheets for `min-comp` to
drop, which works only when they are disconnected), and the winding-count sign of the prior (open3d's one-ray parity
calls the overlap of the arm and the torso outside; that put a sheet 3 cm inside the chest of 00307).
