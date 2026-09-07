#!/usr/bin/env python3
"""Convert a COLMAP text model to a brush-splat-render cameras.json (OpenGL camera-to-world convention)."""
import json, sys, numpy as np, argparse

def quat_to_R(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=np.float64); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

ap = argparse.ArgumentParser()
ap.add_argument("colmap_dir"); ap.add_argument("out_json")
ap.add_argument("--every", type=int, default=1); ap.add_argument("--max", type=int, default=0)
ap.add_argument("--scale", type=float, default=1.0, help="rescale image size and intrinsics")
a = ap.parse_args()
cams = {}
for line in open(f"{a.colmap_dir}/cameras.txt"):
    if line.startswith("#") or not line.strip(): continue
    t = line.split(); cams[int(t[0])] = (t[1], int(t[2]), int(t[3]), [float(v) for v in t[4:]])
images = []
lines = [l for l in open(f"{a.colmap_dir}/images.txt") if not l.startswith("#") and l.strip()]
for i in range(0, len(lines), 2):
    t = lines[i].split()
    images.append((t[9], [float(v) for v in t[1:8]], int(t[8])))
images.sort(key=lambda x: x[0])
images = images[::a.every]
if a.max: images = images[:a.max]
model, W, H, params = cams[images[0][2]]
if model == "PINHOLE": fx, fy, cx, cy = params[:4]
elif model == "SIMPLE_PINHOLE": fx = fy = params[0]; cx, cy = params[1:3]
else: raise SystemExit(f"unsupported model {model}")
s = a.scale
out = {"width": int(round(W*s)), "height": int(round(H*s)), "cameras": []}
for name, p, cid in images:
    qw, qx, qy, qz, tx, ty, tz = p
    R = quat_to_R(qw, qx, qy, qz); t = np.array([tx, ty, tz])
    c2w_R = R.T; pos = -R.T @ t
    R_gl = c2w_R @ np.diag([1, -1, -1])
    out["cameras"].append({"name": name, "fx": fx*s, "fy": fy*s, "cx": cx*s, "cy": cy*s,
                           "position": pos.tolist(), "rotation": R_gl.tolist()})
json.dump(out, open(a.out_json, "w"), indent=1)
print(f"wrote {len(out['cameras'])} cameras at {out['width']}x{out['height']} to {a.out_json}")
