#!/usr/bin/env python3
"""Compare two directories of same-named PNGs: MAE per channel, max abs diff, and where diffs concentrate."""
import sys, os, numpy as np
from PIL import Image
a_dir, b_dir = sys.argv[1], sys.argv[2]
names = sorted(n for n in os.listdir(a_dir) if n.endswith(".png") and os.path.exists(os.path.join(b_dir, n)))
tot_rgb = tot_a = 0.0; worst = (0, "")
for n in names:
    A = np.asarray(Image.open(os.path.join(a_dir, n)).convert("RGBA"), dtype=np.float32) / 255.0
    B = np.asarray(Image.open(os.path.join(b_dir, n)).convert("RGBA"), dtype=np.float32) / 255.0
    d = np.abs(A - B)
    mae_rgb = d[..., :3].mean(); mae_a = d[..., 3].mean(); mx = d.max()
    frac_big = (d.max(axis=2) > 8/255).mean()
    print(f"{n}: MAE rgb {mae_rgb:.5f} alpha {mae_a:.5f} max {mx:.3f} frac(>8/255) {frac_big*100:.3f}%")
    tot_rgb += mae_rgb; tot_a += mae_a
    if mx > worst[0]: worst = (mx, n)
print(f"mean MAE rgb {tot_rgb/len(names):.5f} alpha {tot_a/len(names):.5f}; worst max {worst[0]:.3f} in {worst[1]}")
