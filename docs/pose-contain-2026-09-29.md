# Pose containment loss (`--pose-contain-weight`, 2026-09-29)

Cage training only (`--cage`). The splat, posed by the cage's frames, must not render outside the silhouette its own
surface draws in that pose. Found on b2crig subject b24be4 (arms-up capture): with the arms lowered, splats that sit
more than 2 cm inside the body (3.5k of 182k, big and anisotropic) are sheared out of the shoulder tops as long dark
spikes. Pruning them outright also removes seen clothing (the body-depth test misfires in creases), so the loss is
the fix, not a prune.

## How

`src/gpu/pose_contain.{h,cu}`, set up in `train/trainer.cpp` after the cage binding.

- Inputs from b2crig (`b2crig/rig/contain.py`, driven by `tools/cage_train.py --contain`; moved out of the trainer on
  2026-09-29, see `docs/b2crig-boundary.md`):
  - `--pose-contain-cameras`: the views. The cage's frames are spread evenly over N; each camera is one of the capture
    cameras' offsets from the canonical body centroid, re-applied to the posed body centroid, with the long side
    `--contain-res` (960). The cameras are named `<frame>@c<k>`.
  - `--pose-contain-exclude`: a uint8 mask of the splats more than `--contain-depth` (0.02 m) behind the nearest
    vertex of cage layer 0 (the MHR body).
- At start: each view is rendered posed with the excluded splats hidden (posed opacity logit -30). The alpha > 0.3,
  dilated by `--pose-contain-dilate` px (5), is the allowed region.
- Training: a `--pose-contain-fraction` (0.5) of the steps draws a virtual view instead of a real one; the photometric
  loss runs with colour / SSIM off and the alpha lane at `--pose-contain-weight` against target 0 outside the allowed
  region (weights 255) and nothing inside (weights 0). No normals, no hollow loss on those steps.
- Clothing, hair and anything else outside the body define the target, so they are never trimmed.

## Results (b24be4, 6k iterations warm start, 13 s)

Leak = alpha mass outside the allowed region per frame, held-out clips theater / walk_wave (b2crig tools/sil_eval.py).

| poses | leak theater / walk_wave | capture PSNR |
|---|---|---|
| none (straight LBS) | 45.3 / 101.6 | 29.67 |
| 300 frames of 5 dance clips | 2.0 / 1.8 | 31.19 |
| 4 selected | 3.8 / 3.6 | 31.19 |
| 16 selected | 2.8 / 1.6 | 31.20 |
| 32 selected | 1.6 / 1.5 | 31.20 |

Selected = b2crig tools/pose_select.py: greedy max coverage of per-triangle deformation (max |log principal
stretch|) over a pool (dance clips + procedural library). 16 poses cover 85% of the pool, 32 cover 91%.

Adding `--cage-stretch-weight` (0.002 or 0.0005) lowers the leak further but speckles the texture (chest,
leotard); the posed hollow loss adds nothing here.
