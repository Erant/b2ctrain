# b2ctrain ↔ b2crig: what lives where

b2crig (`~/Projects/b2crig`) animates a trained subject: it builds the cage, skins it, poses it and fits it to generated
video. b2ctrain trains and renders splats. Its counterpart note is `b2crig/BOUNDARY.md`; keep the two in step.

## The rule

**b2ctrain owns the differentiable loop and the rasteriser.** It holds what has to run per step or per pixel, or needs
gradients:
- posing splats by a cage and sending gradients back to the canonical splat;
- losses on posed renders;
- per-splat state that refine, prune and compaction must carry;
- the file formats it reads.

**b2crig owns every decision about the rig**, including:
- what the body is, and which cage layer or Sapiens2 class means what;
- which poses and which cameras to use;
- which splats are interior or ambiguous;
- which geometric measure says a joint has opened.

b2crig computes these as **files or flags** and hands them over. b2ctrain does not compute them. A b2ctrain feature that
needs such a decision takes it as an input and says in its help text that b2crig supplies it.

Test for new code: *would changing the rig (a new body model, other garment classes, another pose library) require
editing it?* If yes, it belongs in b2crig. If it is maths on splats, pixels or gradients, it belongs here.

## Interfaces

b2crig is the only caller: it runs this binary as a subprocess from `b2crig/b2ctrain.py` and its tools.

| From b2crig to b2ctrain | Format | Read by |
|---|---|---|
| Cage: layers, canonical and posed vertices per frame | `B2CCAGE1` (`src/gpu/cage.h`) | `--cage` (train, render, fit-cage) |
| Opening gate θ per vertex and frame (degrees) | optional `B2COPEN1` section after the posed vertices | `--cage-open` (train); render with a ply that has `open_*` |
| Second candidate triangle per ambiguous splat | `B2CALT01` | `--alt-binding` |
| Containment views: cameras named `<frame>[@…]` | body2colmap cameras.json | `--pose-contain-cameras` |
| Splats left out of the containment targets | uint8 per warm-start splat | `--pose-contain-exclude` |
| fit-cage label groups (Sapiens2 ids) | `--groups a,b;c;d` | fit-cage (required when the label term is on) |
| Filler/crease splats and their vertex pair | ply `cage_fill`, `cage_gate_a/_b` | cage posing (`pose_bound`) |
| Size clamp, stretch fade, fill ranges | `--cage-max-growth`, `--cage-fade-*`, `--cage-fill-*` | train, render |
| Hollow proxy that follows the pose | `--mesh` whose vertex count equals cage layer 0's | trainer (`mesh_is_body`) |

| From b2ctrain to b2crig | Format |
|---|---|
| Trained splat | ply: `seg_label`/`seg_conf`, `open_*` + header `b2c.cage_open add START END MAXDO`, the subject's `b2c.*` header |
| Pose-dependent appearance MLP | `<name>.app` (`B2CAPP03`, `src/gpu/cage_app.h`) |
| fit-cage results | `delta.f32`, `vis.f32`, `fit.json`, learned `B2CALT01` |
| A posed splat | `render --export-posed` + `.frame.f32` |

## Moved to b2crig on 2026-09-29

- **Pose containment's poses, cameras and interior mask.** The trainer used to spread the cage's frames over N views,
  re-aim the training cameras at the posed layer-0 centroid and mark splats more than 2 cm inside layer 0. That is now
  `b2crig/rig/contain.py`.
  - Removed flags: `--pose-contain-views`, `--pose-contain-depth`, `--pose-contain-res`.
  - Replaced by `--pose-contain-cameras` and `--pose-contain-exclude`.
  - Verified on b24be4 (poseset_16, 6k iterations): the same 3488 excluded splats and the same view size. Silhouette
    leak on theater / walk_wave: 0.5 / 0.5 before, 0.5 / 0.3 after, coverage equal.
- **The cage-open gate geometry** (partner selection, relative rotation of the incident triangle frame). That is now
  `b2crig/rig/open_gate.py`, shipped in the cage as `B2COPEN1`.
  - `cage_open` only maps θ to the blend weight with the start/end the splat was trained with.
  - Removed flags: `--cage-open-radius`, `--cage-open-min-dist`.
  - Older ply headers with five fields still parse; the last two are ignored.
  - Verified with `--cage-open-debug` renders against the previous binary: at most 1/255 difference (fp16 θ).
  - A render with an `open_*` ply and a cage without the section renders the plain splat with a warning.
    `b2crig.b2ctrain.render` adds the section automatically.
- **fit-cage's built-in label groups** (`4;23,1;13`). b2crig passes `--groups`
  (`b2crig.b2ctrain.FIT_GROUPS`).

## Kept here, and why

- **Cage posing** (`cage.cu`), dual binding, stretch fade/fill, `cage_app`, the `cage_open` two-state splats, the
  pose-containment loss and the stretch regulariser. All of them run inside training with gradients.
- **fit-cage**. It fits the rig, but it needs the rasteriser's backward pass, which b2crig (Python) cannot reach. It
  stays here as long as there is no Python binding to the rasteriser. Its inputs (groups, bindings, masks) come from
  b2crig.
- **`render --class-mask`, `--label-maps`, `--export-posed`**. These are generic render outputs.

## Grey areas and known issues

- **Two copies of the posing maths.** `b2cviewer/web/rig.js` is a JS port of `cage.cu`'s posing and of `cage_app`.
  A change to how a splat follows its triangle has to land in both. The viewer does not implement stretch fade/fill,
  `open_*` or dual binding.
- **Layer 0 = the body** is a convention both sides rely on (the hollow proxy's `mesh_is_body`). b2crig writes the MHR
  body as layer 0.
- **`cage_gate_s`** is parsed, carried through compaction and exported, but nothing reads it. It came from the pair
  gate in b2crig `tools/open_gates.py`, which `--cage-open` no longer uses. Remove it when nothing produces it.
- **Future work on garment-edge "hairs"** (long splats posed rigidly by one triangle): the bend-aware split/loss goes
  here, driven by pose frames b2crig selects. Attaching both ends of a long splat to their own triangles would be posing
  maths here (and in the viewer), with the second binding supplied by b2crig like `B2CALT01`.
