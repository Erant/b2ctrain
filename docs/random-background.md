# Random training background, and normals at equal splat count

2026-09-28, two helical bundles (`helical-20260928-154347-2bccb7`, `-154358-1d747b`), the final
(`train_final_splat`) training rebuilt from the bundle and re-run locally. Tools in `tools/bundle_study/`.

## The black blotches on top of raised arms

Rendered from above, the delivered splat has dark streaks along the tops of both raised arms (and a grey smear
on the crown). They are **dark splats, not holes**: they stay dark on a white background and at SH degree 0.

Where they come from: transparent frames are composited onto the *training* background, which defaults to
`--background-color 0,0,0` with `--background-noise-strength 0.1` (`±0.1` clipped at 0, so `[0, 0.1]`, exactly
black half the time), and b2crunner trains with `--match-alpha-weight 0.1`. At a soft silhouette edge the target
is "skin blended into black", and an opaque *dark* splat reproduces that as well as a semi-transparent skin one.
The tops of raised arms are silhouette edges in every view of the orbit, so that is where they collect.

Measured on the picked splats (visible and dark in the top-down view, arm-labelled): they are front-most in ~52 of
162 training views, the same as the bright arm splats around them, but 37% of those projections fall within 3 px
of the frame's silhouette (bright: 14%). The frame colour under them is not dark (0.42 vs 0.47); it is the
background they were composited against that is.

The fix is a background the splat cannot hide in: `--background-color 0.5,0.5,0.5 --background-noise-strength
0.5`, uniform random per channel per step. Top-down the streaks are gone, a white-background render shows no
see-through gaps, sharpness at the training views is unchanged (0.252 vs 0.248, render/frame Laplacian
variance), and the in-trainer alignment disagreement drops (0.98 -> 1.25 px becomes 0.79 -> 0.95).
`--match-alpha-weight 0.5` alone only removes part of them. Black-background PSNR drops ~0.6 dB, but that eval
composites on black — the old training condition — and the renders are not worse.

## Normals, and the splat count

Sapiens2 1b normals on the 162 upscaled frames, `--normal-loss-weight 0.05 --normal-loss-start-iter 5000`:

| 2bccb7                       | splats | arm mass > 2 cm off the refit body | sharpness | PSNR (black) |
|------------------------------|-------:|-----------------------------------:|----------:|-------------:|
| delivered (black bg)         |   405k |                              20.1% |     0.248 |        31.07 |
| normals (black bg)           |   469k |                               6.8% |     0.244 |        30.36 |
| random bg                    |   389k |                              15.0% |     0.252 |        30.42 |
| random bg + normals          |   553k |                               5.2% |     0.250 |        29.91 |
| random bg + normals, g 0.0035 |  396k |                               5.9% |     0.244 |        29.92 |

(1d747b, black bg: arm mass off the body 22.2% -> 10.3% with normals.)

Normals on a black background make the blotches *worse*; with the random background they are clean.

The extra splats of random bg + normals (+147k over the delivered splat) are opaque surface splats (+114k with
opacity >= 0.5, only +3k more than 3 cm off the body): shoes +50k, legs/black socks +28k, arms +27k, hidden
(no labelled view sees them) +24k, skirt +18k. Two sources, super-additive: growth is gradient-triggered, and
the normal loss switching on at 5000 is a second gradient source (the trajectories are identical to 5000 and
diverge right after); the random background makes the edges of dark objects — free against black — need real
splats.

At equal count (`--growth-grad-threshold 0.0035`, 396k against 389k) normals keep their placement benefit
(5.9% vs 15.0%) but are ~3% softer; the full-growth run's sharpness is densification. Recommendation, and what
b2crunner does since: random background on the final training, normals on it with the growth threshold at 0.0035.

## The 81 pre-extension frames

The extension adds 41 frames before the original pass and 40 after, so the pre-extension frames are
`frame_00042`..`frame_00122`. Trained on those alone: +0.7 dB on its own frames, -3.3 dB on the 81 it never saw,
and translucent / ghosted from below the orbit (the extension frames are what cover those angles).

## tools/bundle_study

Rebuilds a bundle's `train_final_splat` inputs and runs b2crunner's own `brush` step on them (b2crunner and
body2colmap on PYTHONPATH, b2crunner's venv): the argv is token-identical to the pod's and a local control
reproduces the delivered splat (405,280 vs 405,413 splats, alignment 0.98 -> 1.25 vs 0.98 -> 1.26 px).

- `a_rig_binding.py` (sam-3d-body venv, `MHR_MODEL`): the MHR skinning as `rig_binding.npz`, subject-independent.
- `b_normals.py <bundle>...` (a torch + transformers venv with Sapiens2): `colmap/normals/` in the trainer's
  sidecar format.
- `c_train.py <bundle> <variant>...`: variants `local162`, `n162` (normals), `p81`, `p81n`, each optionally
  `+rbg` (random background), `+a05` (match-alpha 0.5), `+gNN` (growth threshold NN/10000). Rig from the
  header's world joints (checked against `debug/body_rig/body_rig.json`), `photo_priority_final` re-run
  (checked against `debug/photo_priority_final`), the header's `b2c.*` comments grafted onto the export.
  Extras reach the trainer through `trainer_wrap.sh` (`$B2C_EXTRA`); `B2C_TRAINER` picks the binary.
- `d_eval.py <bundle> [extra variants]`: PSNR and sharpness on the pass and the extension frames, placement
  against the refit body, novel elevated panels. `e_crops.py`: upper-body crops of those.
- `splat_stats.py`, `pick.py`, `occl.py`, `occl2.py`, `where.py`: the blotch diagnosis above.
