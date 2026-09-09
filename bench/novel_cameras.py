#!/usr/bin/env python3
"""Off-orbit cameras for novel-view checks: every N-th camera of a cameras.json, re-aimed at a target point after an
elevation and/or azimuth offset (degrees) about it, at the same distance. Same JSON format as colmap_to_cameras.py.

  novel_cameras.py cams.json out.json --every 20 --elev 25 -25 --azim 0 12 --target 0,0,-2.1
"""
import json, argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("cams"); ap.add_argument("out")
ap.add_argument("--every", type=int, default=20)
ap.add_argument("--elev", type=float, nargs="*", default=[25.0, -25.0], help="elevation offsets in degrees")
ap.add_argument("--azim", type=float, nargs="*", default=[0.0], help="azimuth offsets in degrees")
ap.add_argument("--target", default="", help="x,y,z the cameras orbit and look at (default: mean of the camera look-at points)")
ap.add_argument("--up", default="0,1,0", help="world up axis")
a = ap.parse_args()
j = json.load(open(a.cams))
up = np.array([float(v) for v in a.up.split(",")]); up /= np.linalg.norm(up)

def c2w(cam):
    R_gl = np.array(cam["rotation"]); R = R_gl @ np.diag([1, -1, -1])  # columns: right, down, forward (OpenCV)
    return R, np.array(cam["position"])

if a.target:
    target = np.array([float(v) for v in a.target.split(",")])
else:
    # Each camera's optical axis, closest approach to the mean camera position gives the orbit centre.
    pts = []
    for cam in j["cameras"]:
        R, p = c2w(cam); f = R[:, 2]
        pts.append((p, f))
    A = np.zeros((3, 3)); b = np.zeros(3)
    for p, f in pts:
        M = np.eye(3) - np.outer(f, f); A += M; b += M @ p
    target = np.linalg.solve(A, b)

def rot(axis, deg):
    axis = axis / np.linalg.norm(axis); t = np.radians(deg); K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K

out = {"width": j["width"], "height": j["height"], "cameras": []}
for i, cam in enumerate(j["cameras"][::a.every]):
    R, p = c2w(cam)
    for e in a.elev:
        for az in a.azim:
            d = p - target
            right = np.cross(R[:, 2], up); right /= np.linalg.norm(right)
            d = rot(up, az) @ rot(right, -e) @ d
            pos = target + d
            f = target - pos; f /= np.linalg.norm(f)
            r = np.cross(f, up); r /= np.linalg.norm(r); dn = np.cross(f, r)  # OpenCV: x right, y down, z forward
            Rn = np.stack([r, dn, f], 1)
            name = cam["name"].rsplit(".", 1)[0] + f"_e{e:+.0f}_a{az:+.0f}.png"
            out["cameras"].append({"name": name, "fx": cam["fx"], "fy": cam["fy"], "cx": cam["cx"], "cy": cam["cy"],
                                   "position": pos.tolist(), "rotation": (Rn @ np.diag([1, -1, -1])).tolist()})
json.dump(out, open(a.out, "w"), indent=1)
print(f"target {target.round(3).tolist()}; wrote {len(out['cameras'])} cameras to {a.out}")
