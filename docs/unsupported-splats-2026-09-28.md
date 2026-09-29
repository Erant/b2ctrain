# Never-seen and poorly supported opaque splats (flagged 2026-09-28 from b2crig)

Found while animating the helical run `helical-20260928-154347-2bccb7` (arms-up capture) in b2crig. These are
b2ctrain outputs, so the fix belongs here.

## What

Every exported ply carries per-splat evidence (`ev_w_all`, `ev_views`, ...: "Computed evidence for N splats over 174
views" at the end of training). Many **opaque** splats have none:

- **Never seen**: opacity > 0.2 and `ev_w_all` < 0.5, i.e. no training view ever blends them. Their median opacity
  is 0.61, so they are not faint leftovers. Many carry the seg label Background (0).
- **Poorly supported**: opacity > 0.2 and `ev_w_all` 0.5-5, against a median of 30.5 for opaque splats.

Neither kind matters in the capture poses: they sit behind the visible surface or inside the body. Once the splat is
animated (LBS / cage binding) the body deforms, they get exposed, and they show up as specks, dark junk and haze.
b2crig currently removes some of them after the fact (`tools/prune_interior.py` with the evidence), but only those
inside the body (> 2 cm under the fitted MHR surface, mouth excepted). The rest stay.

## Numbers (run 2bccb7, every variant in `ply/`)

"Deep" = more than 2 cm inside the capture's MHR body surface (normal side), mouth region excluded.

| variant | splats | opaque (>0.2) | never seen (ev_w_all<0.5) | poorly supported (0.5-5) | deep >2cm | deep & never seen | deep & poorly supported | Background-labelled opaque |
|---|---|---|---|---|---|---|---|---|
| scene.ply | 405413 | 239853 | 14159 | 25571 | 41766 | 8370 | 6772 | 6384 |
| scene_randbg.ply | 388762 | 234062 | 5373 | 16536 | 19950 | 3088 | 3832 | 2182 |
| scene_normals.ply | 468670 | 322681 | 42133 | 56495 | 35487 | 15738 | 5718 | 21473 |
| scene_normals_randbg.ply | 552490 | 393657 | 45941 | 72273 | 29933 | 13973 | 4914 | 19853 |
| scene_normals_randbg_equalcount.ply | 395798 | 286531 | 21956 | 41960 | 21426 | 8093 | 3470 | 9316 |
| scene_81.ply | 416994 | 233679 | 10854 | 31110 | 35500 | 5559 | 7547 | 4254 |
| scene_81_normals.ply | 460205 | 308621 | 36968 | 66059 | 32781 | 14148 | 5842 | 18294 |

Reading:

- **The normal-map loss is what breeds never-seen opaque splats**: 5.4k (randbg) -> 46k (normals + randbg), and
  14k -> 42k without random background. Background-labelled opaque splats grow the same way (2.2k -> 20k).
  Presumably the normal term keeps growing / keeping opaque splats that the colour loss never sees (behind the
  surface, inside the body); nothing culls them because they never receive a colour gradient.
- Random background alone gives the cleanest interior (fewest deep splats and fewest never-seen ones).
- `equalcount` trimming (the variant b2crig uses) halves the damage, but 22k never-seen and 42k poorly supported
  opaque splats remain, about 22% of its opaque splats.

## Suggested fix (cheap)

The evidence is already computed at export. Cull by it:

1. Drop splats with opacity > ~0.05 and `ev_w_all` below a small threshold (never seen) before export. They cannot
   affect any training view by definition, so the capture loss is unchanged.
2. Consider also culling (or at least flagging) poorly supported splats, e.g. `ev_w_all` < 5 or `ev_views` < 2-3,
   especially if they are labelled Background or sit inside the fitted body. That is the bulk (42k here) and it is
   what an animated splat exposes. Check the capture PSNR before and after; it should barely move.
3. Better still, do it periodically during refinement (like opacity pruning), so the budget (equal count) is spent
   on seen surface. Also check why the normal loss creates / keeps them: maybe it should not act on splats that are
   invisible in the colour render, or not grow them.

## Reproduce

b2crig side: `work/2bccb7` (the table above came from a short script over `ply/*.ply` with `plyfile`, the `ev_w_all`
property, and the MHR body of `work/2bccb7/mhr_capture.npz` for "deep"). Evidence-gated interior prune:
`tools/prune_interior.py work/2bccb7 IN.ply OUT.ply --evidence IN.ply` (the ply's own `ev_*` block works as the
evidence) dropped 8.1k of the equal-count variant's splats.

## Findings (2026-09-28, local reruns; scripts, patch and logs under `out/cull/`)

Reruns of the `n162+rbg` variant (normals 0.05 from 5k, random background, hollow 0.5 / 3 cm, body rig, 4 alignment
refits) on the same bundle with b2ctrain ce7c29d plus a small patch (`out/cull/cull.patch`), evidence measured at every
5k-step export. The unpatched baseline reproduces the pod ply (552k splats, 46-47k never-seen opaque, 72k poorly
supported, PSNR 29.99 at the training cameras).

### Where they come from

| step | splats | never-seen opaque | poorly supported | deep > 2 cm opaque |
|---|---|---|---|---|
| 5k (quarter/half res, no normals yet) | 148k | 418 | 1.3k | 3.6k |
| 10k | 341k | 13.5k | 18k | 11.5k |
| 15k (growth stops) | 552k | 48k | 48k | 22k |
| 20k | 552k | 66k | 64k | 27k |
| 25k | 552k | 73k | 73k | 30k |
| 30k + 4 refits (final) | 552k | 47k | 72k | 31k |

- **Not early, not alignment: densification and post-growth relocation.** Nothing at 5k; they appear with growth
  between 5k and 15k and keep coming after growth stops. Lineage by slot (no compaction in a plain run, so a slot is the
  same splat unless relocation reused it): of the final never-seen splats, 48% were the same splat at 15k (79% of those
  already never-seen then), 77% at 20k, 94% at 25k. The rest were relocated after growth stopped: every refine kills
  the ~3-5k splats whose opacity fell under 1/255 and relocates them as splits of opacity-sampled visible parents
  (`refine.cu` step 2a is not gated by `growth_allowed`), and a share of the children land behind the surface. The
  alignment refits add almost none (100-260 culled per refine once the cull is on).
- **Nothing removes them.** `prune_flags_kernel` prunes on opacity < 1/255, non-finite, oversized and out-of-bounds
  only; `vis_count` gates growth and relocation parents, never pruning. A hidden splat gets no gradient, so its opacity
  never moves except by the decay (0.004 (1 - t) per refine, about 0.3 over the run, not enough from a median 0.6). The
  refits run with refinement off entirely. The evidence is computed at export but only `--evidence-prune-inmask` uses it.
- **The hollow loss makes it worse, not better.** It was on in every variant. It penalises weight arriving behind the
  mesh only while that weight is still rendered, and its gradient on a *front* splat with penalised mass behind it is
  negative (`raster_bwd_tc.cu`: `v_alpha += lam (T_before h - Sh ra)`), so the optimiser's cheapest answer is to make
  the front opaque; once T drops under the 1e-4 cutoff the interior is neither penalised nor seen, and stays. Measured
  (same run without `--hollow-weight`): deep opaque 19.4k vs 30.8k, deep and never-seen 6.3k vs 14.8k, never-seen 34k
  vs 47k, PSNR 30.03 vs 29.99. As configured it thickens the shell it was meant to hollow.

### The cull (prototype, `out/cull/cull.patch`)

- The backward already reduces each splat's blend mass per tile (`sum_vis`); it is now accumulated into a per-splat
  `vis_weight` (sum of alpha T over pixels and steps; tc and warp backends), zeroed at every refine like `vis_count`.
- `--cull-weight W`: at every refine, prune splats whose `vis_weight` over the window is below W pixel-weights per
  full-resolution pass over the training views (scaled by `refine_every / n_views` and 4^-level of the resolution
  schedule). The alignment refits run a prune-only refine (no growth, no decay) followed by compaction, so what the
  warped views no longer see goes too. `--cull-no-relocate`: once growth has stopped, culled/dead splats are dropped
  instead of relocated. `--evidence-prune-wall T`: final-export cull on `ev_w_all` (never-seen by definition).
  `--export-evidence` now also measures at intermediate exports.

| variant | splats | never-seen | poorly supp. | deep opaque | Background-labelled opaque | PSNR train cams | sharpness ratio |
|---|---|---|---|---|---|---|---|
| base (pod config) | 552k | 47062 | 71831 | 30845 | 21073 | 29.991 | 0.2848 |
| no hollow loss | 537k | 34330 | 69873 | 19435 | 10429 | 30.028 | 0.2879 |
| `--cull-weight 1` | 402k | 2 | 22728 | 3519 | 350 | 29.965 | 0.2872 |
| `--cull-weight 5` | 298k | 0 | 452 | 1991 | 206 | 29.956 | 0.2814 |
| `--cull-weight 5 --evidence-prune-wall 5` | 298k | 0 | 0 | 1949 | 194 | 29.939 | 0.2813 |
| `--cull-weight 5 --cull-no-relocate` | 201k | 0 | 285 | 1816 | 162 | 29.989 | 0.2650 |

Reading: culling what the views do not see costs nothing in PSNR (the -0.03 dB is within run-to-run noise of these
non-deterministic runs) and nothing in sharpness at W=1; W=5 also clears the poorly supported class for about -1%
sharpness. Dropping instead of relocating after growth stop halves the model again at equal PSNR but -7% sharpness: the
relocation churn does put detail on the visible surface, it just also breeds hidden splats, which the cull now removes
each window (5-6k per refine with relocation, 600-1400 and falling without). The 1.8-2k "deep" splats that remain are
mostly the mouth/hair regions the metric here does not exclude, and are seen.

### False transparency: keep the hollow loss, add the cull

`b2ctrain probe` (weight rendered from more than 3 cm behind the refit mesh, "behind"; and from more than 3 cm behind
the first surface, "deep"), 21 training cameras / 12 elevated novel cameras (+-35 deg):

| variant | behind, train cams | behind, novel | deep, novel (>0.2: % px) |
|---|---|---|---|
| base (hollow 0.5) | 0.0022 | 0.0019 | 0.0291 (4.6%) |
| no hollow loss | 0.0071 | 0.0062 | 0.0307 (4.7%) |
| cull-weight 1 | 0.0021 | 0.0018 | 0.0268 (4.2%) |
| cull-weight 5 (+ wall) | 0.0022 | 0.0020 | 0.0265 (4.2%) |
| cull-weight 5, no relocation | 0.0022 | 0.0020 | 0.0322 (5.1%) |

Transmittance to the background inside the body (white-minus-black renders at the novel cameras) is 0.14-0.15% of the
body pixels for every variant: the random background already seals the silhouette; the false transparency that matters
is the back showing through the front, which is what "behind" measures. Without the hollow loss it triples; the cull
neither adds nor removes it (it prunes what is not rendered, and behind-weight is rendered by definition). So the two
are complementary: the hollow loss seals the front (its opaque-front gradient is the anti-transparency effect), and the
cull empties what that sealed front hides (deep opaque 31k -> 2k), which is the thin shell that animates. The earlier
line "hollow loss off" is withdrawn.

Recommendation: keep `--hollow-weight 0.5 --hollow-margin 0.03`, add `--cull-weight 5 --evidence-prune-wall 5` (W=1
when sharpness is the priority). The periodic cull is the mechanism, not a final-export filter alone: 25k of the 47k
were already hidden at 15k and the model needs the remaining steps to re-fit without them.

Integration notes: the cage-rig branch keeps host-side `cage_labels` (and `--cage-open` / `--cage-app` state) indexed
by splat, so its compaction path must gather them (the same reason `--cage` refuses `--sparsify`); b2crunner's
`brush.py` needs the three parameters; the rig is already re-bound after every refine.
