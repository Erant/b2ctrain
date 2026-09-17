# View-shaped texture atlas experiment

`tools/view_atlas.py` builds a UV layout with large upright front/back panels and
smaller left/right, crown and underside panels. It transfers the existing texture;
it does not generate a new appearance. The output is real mesh UVs, so edits to the
sheet can be used directly as a material without a camera backprojection pass.

Run from the repository root with numpy, OpenCV and Open3D available:

```bash
/home/tristan/Projects/tsdf/.venv/bin/python tools/view_atlas.py \
  --atlas out/mesh/photo/atlas \
  --texture out/mesh/coverage_consensus/selected/texture_final.png \
  --output out/mesh/view_atlas --res 4096
```

The mesh is Y-up; `--front` rotates the front direction about Y in degrees, with
zero placing the front camera on +Z. Geometry and triangle count are preserved.

## Files to use

- `mesh_diffusion.obj` / `.mtl`: recommended asset, using two materials.
- `diffusion_texture.png`: 4096-square editable sheet, with upright front/back
  people across the upper portion. Lower panels supply profiles and horizontal
  views. The lower-right square is unused in this material.
- `reserve_texture.png`: unchanged source texture for surfaces that do not fit
  safely in the six view panels. Keep this image alongside the mesh.
- `edit_mask.png`: white where sheet texels belong to the mesh. Other visible
  pixels are context to help an image model understand complete body views.
- `protect*.png`: source protection masks sampled into the new coordinates.
- `preview.jpg`, `ownership_preview.jpg`: sheet overview and the same sheet with
  context-only pixels darkened.
- `face_panel.npy`, `face_uv.npy`, `source_uv.npy`: triangle ownership, per-corner
  new UVs, and original UV coordinates per covered new texel.
- `mesh_uv.obj`, `texture.png`, `atlas.json`, `position.f32`, `normal.f32`,
  `mask*.png`: a compact single-material variant compatible with `mesh-render`.
  Its reserve panel is downsampled; prefer the two-material asset for editing.
  Position/normal maps are zero outside owned texels; consumers must use the mask.

The script tests seven interior locations per triangle for visibility and assigns
front/back priority above a facing cosine of 0.25. Remaining candidates use the
best of the other directions. A continuous 2D triangle intersection test removes
positive-area overlaps; touching triangle edges are allowed. The farther member
of each overlapping pair goes to the reserve. These deliberately conservative
choices leave some potentially usable surface in the reserve.

The context pixels are complete orthographic texture views. They do not represent
additional UV ownership. A triangle has only one material location. Editing a
context-only pixel therefore does not change that surface on the mesh.

## Bringing edits to existing consumers

An OBJ viewer can use the two-material mesh directly: replace
`diffusion_texture.png` while keeping its dimensions and layout. For the existing
single-atlas pipeline, use the direct barycentric UV transfer:

```bash
/home/tristan/Projects/tsdf/.venv/bin/python tools/view_atlas_apply.py \
  --layout out/mesh/view_atlas \
  --edited out/mesh/view_atlas/diffusion_texture.png \
  --output out/mesh/view_atlas/roundtrip.png
```

This updates only surfaces owned by the six panels, plus their texture gutters.
Reserved surface texels remain exactly as in the source. The transfer does not
apply the protection masks automatically: enforce them when compositing generated
edits if those regions should stay unchanged. Run from the same repository root
used for generation, because the manifest records the supplied source paths.

## First subject: 00307

Output: `out/mesh/view_atlas/`, starting with the selected crown/boot completion
texture from the earlier experiments.

- All 300,000 triangles retained, with geometry verified after OBJ export.
- Front/back own 41.2% of total surface area; auxiliary panels add 16.0%.
- 42.8% remains reserved, at its original texture resolution in the recommended
  asset. Surface hidden by folds, overlapping limbs, or internal geometry cannot
  be represented faithfully by two ordinary body silhouettes.
- Zero positive-area triangle intersections in each of the six final panels,
  checked in continuous UV space rather than only at pixel centres.
- Unedited UV round trip: mean absolute color error 1.195/255 on mapped source
  texels; maximum error on reserved surface texels is zero.
- `validation.json` also records the comparison over 12 held-out body views.

This demonstrates an editable view-shaped layout, not improved denoising quality.
At this initial stage no diffusion pass had been run; see the completed tests below. Side panels have fewer pixels per metre than
front/back, and seams between independently edited panels remain a concern. The
reserve is a substantial limitation: the result is a hybrid layout, not a complete
anatomical front/back unwrap. A subsequent intrinsic flattening experiment could
expose more of the hidden surfaces, at the cost of distorting the body silhouettes.

## Diffusion tests completed (2026-09-17)

Follow-up outputs are in `out/mesh/view_atlas/diffusion_test/`; its `README.md`
records settings and limitations, and `review.html` compares 18 body/head cameras.
Whole-sheet Klein runs at strengths 0.55 and 0.8, separate higher-resolution
front/back crops at 0.8, and a fragmented-atlas control were evaluated. The
front/back crops reduce clothing mottling, especially on the back, with less
redrawing than the strong whole-sheet result. Hidden surfaces remain unchanged.

Testing exposed resampling seams. `view_atlas_apply.py` now accepts
`--reference-sheet ORIGINAL_SHEET.png` to transfer only edited-minus-original
pixel changes. This makes no-edit transfer byte-exact and reduces mapped/reserved
boundary contrast. The conservative candidate is
`diffusion_test/panels_s0.8_delta/mesh_uv.obj`; the stronger alternative is
`diffusion_test/sheet_s0.8_delta/mesh_uv.obj`. Both preserve the protected face and
reserved surfaces exactly. See the report before selecting a texture: generated
motifs/hair can change, and this is one subject/seed rather than a general benchmark.

## The reserve, and the extra sheet (2026-09-17)

Probing the 42.8 % reserve on 00307 against 96 directions on a sphere (visibility tested against
the whole mesh, as for the six axes) shows that **38.8 % of the surface is never visible from any
direction**: it is the TSDF's inner shell, a second wall about 1 cm inside the skin (a horizontal
slice through the chest draws it as a closed ring inside the outer ring). No camera sees it, so
it needs no diffusion. Depth peeling (removing the owned surface and casting again) exposes only
2.6 % more, because that inner wall faces the body core. The reserve a viewer can see is ~4 % of
the surface: under the skirt hem, armpits and inner arms, inside the collar, beside the ears.
In the 18 held-out cameras it is 3-4 % of the visible pixels.

On the mesh, though, most of the ugly regions (the petticoat under the hem, beside the ears)
turned out NOT to be reserve: they belong to the first sheet's left/right/top/bottom panels,
which at 620 px/m are ~300 px tall on the 2048 klein input and barely change. So the second
sheet re-homes those too.

A second problem is the front/back priority itself: it hands the front/back panels every
triangle they face above cos 0.25, i.e. up to 75 degrees off-axis. In the held-out cameras
15 % of the visible pixels were front/back-owned surface facing its panel below cos 0.5
(the petticoat under the hem, the sides of the legs, every silhouette). Those texels are
foreshortened in the front/back view and klein cannot resolve them there.

`--extra N` (default 6) lays out a second sheet, `<output>/extra/`. What may move there is
set by `--extra-scope`: `reserve` = only the visible reserve (2.8 % of the surface on 00307);
`small` = plus everything the four small panels own (17.8 %); `grazing` (default) = plus
front/back triangles facing their panel below `--steal-cos` 0.5 (25.7 %). A movable
triangle moves only to a view that gives it MORE texels: effective density = (panel px/m)^2
x facing cosine, compared between its current panel and the candidate. Front and back keep
everything they face well and stay byte-identical to the previous layout; the small panels
keep under 1 % of the surface in total.

- Candidate directions: the four small panels' axes plus a 64-point Fibonacci sphere
  (`--extra-candidates`). Each movable triangle's facing score to every candidate is tested
  against the FULL mesh, so the panels are real exterior views and their context renders a
  coherent person (klein sees a photo, not fragments). Image up is world up until the view is
  nearly vertical, so steep views stay upright.
- Panels are picked greedily by the movable area they improve; each triangle then goes to
  the picked direction it faces best among those that improve it. The same overlap test
  runs per panel. On 00307 (grazing) the picks are left-below, right-above, straight below,
  left-above, back-right-below and front-above.
- Layout: a 3 x 2 grid of full-body views on the same `--res` sheet, so each view is about
  1 400 px/m — over twice the density of the first sheet's side panels (620 px/m).
- The extra sheet has the same file set (`diffusion_texture.png` = `texture.png`, no fallback
  square, `context.png`, `edit_mask.png`, `mask.png`, `source_uv.npy`, `face_uv.npy`,
  `face_panel.npy`, `protect*.png`, `position/normal.f32`, `atlas.json`). Its `atlas.json`
  records the picked directions (`direction`, panel names `e<k>_az<yaw>_el<pitch>`), the
  reserve area fraction, the reserve area any candidate sees, and the interior (never-visible)
  area. The parent's `atlas.json` carries a summary under `extra`.
- `mesh_diffusion.obj` now has three materials: `diffusion`, `extra`, `reserve`.

Editing chains: apply the first sheet, then the extra sheet on the result:

```bash
python tools/view_atlas_apply.py --layout out/mesh/view_atlas --edited SHEET1_EDITED.png \
  --reference-sheet out/mesh/view_atlas/diffusion_texture.png --output step1.png
python tools/view_atlas_apply.py --layout out/mesh/view_atlas/extra --edited EXTRA_EDITED.png \
  --reference-sheet out/mesh/view_atlas/extra/diffusion_texture.png --source-texture step1.png \
  --output texture_final.png
```

`--source-texture` is the new override; the two sheets own disjoint triangles, so the order
only matters for gutters. On 00307: `out/mesh/view_atlas_extra/` (scope reserve: 9 763
triangles, 224 297 texels, unedited delta round trip byte-exact), `out/mesh/view_atlas_small/`
(scope small) and `out/mesh/view_atlas_graze/` (scope grazing, the default).

### Results on 00307 (`out/mesh/view_atlas_graze/README.md`)

In the 18 held-out cameras the front/back-owned surface facing its panel below cos 0.5 drops
from 15 % of the visible pixels to 1.7 %; the second sheet owns 41 % of the visible pixels at
mean cos 0.80; the small panels 0.8 %, the still-reserve 1.1 %. Klein at the same settings as
the first sheet (2048 whole sheet, 88 s; or six 1408x2048 crops, 52 s each) handles all six
views including the steep one from below, and the chained transfer changes only what the
second sheet owns (face byte-exact). What still looks blotchy under the hem is klein reading
the mottled petticoat as a print, no longer a texel without a pass. F3 not run.

### Head panels and the single-sheet trial (2026-09-17, late)

The tool now takes `--layout multi|single`, `--head M` (close-up head panels framing the top
`--head-height` metres of the mesh, chosen by the same density rule, allowed to take anything
in that band including the protected face, which then serves as context) and non-square
sheets (`--single-size`, `--single-split`). Multi writes `extra/` and `head/` next to the main
sheet; single packs front, back, obliques and heads onto one wide sheet for one denoise.

On 00307 the head sheet (four views at 4500-6200 px/m) is what finally cleans the ear and jaw
band that the front panel could never resolve. The single sheet, capped at 4.2 MP of klein
input on a 12 GB card, gives the obliques ~300 px of body height; klein erased part of the
figure in one of them and the damage reached the mesh. Multi wins here; single needs a bigger
GPU. See `out/mesh/view_atlas_m3/README.md`.
