import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d_eval import read_splats
B = Path(sys.argv[1]); s = read_splats(Path(sys.argv[2])); T = json.load(open(sys.argv[3])); W, H = T["width"], T["height"]
pk = np.load("picked.npz")
xyz = np.stack([s["x"], s["y"], s["z"]], 1).astype(np.float64); opa = 1 / (1 + np.exp(-s["opacity"]))
q = np.stack([s["rot_0"], s["rot_1"], s["rot_2"], s["rot_3"]], 1); q /= np.linalg.norm(q, axis=1, keepdims=True)
sc = np.exp(np.stack([s["scale_0"], s["scale_1"], s["scale_2"]], 1))
def rotm(q):
    w, x, y, z = q.T
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1), np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1), np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], 1)
def project(cam, P):
    R = np.array(cam["rotation"]) @ np.diag([1, -1, -1]); p = np.array(cam["position"]); qq = (P - p) @ R
    return cam["fx"] * qq[:, 0] / qq[:, 2] + cam["cx"], cam["fy"] * qq[:, 1] / qq[:, 2] + cam["cy"], qq[:, 2]
front_vis = {"dark": [], "bright": []}
op = opa > 0.2
for c in T["cameras"]:
    u, v, z = project(c, xyz)
    ok = op & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    cell = (v // 4).astype(int).clip(0, H // 4) * (W // 4 + 1) + (u // 4).astype(int).clip(0, W // 4)
    front = np.full((H // 4 + 1) * (W // 4 + 1), np.inf); np.minimum.at(front, cell[ok], z[ok])
    for k in front_vis:
        ids = pk[k]; inb = ok[ids]
        front_vis[k].append(np.where(inb, z[ids] < front[cell[ids]] + 0.01, False))
up = np.array([0, 1.0, 0])
for k in front_vis:
    fv = np.array(front_vis[k])  # views x splats
    nvis = fv.sum(0)
    ids = pk[k]; R = rotm(q[ids]); axis = R[np.arange(len(ids)), :, sc[ids].argmin(1)]  # disc normal = shortest axis
    print(f"{k:7} views where it is front-most: median {np.median(nvis):.0f} / {len(T['cameras'])}, never front-most {(nvis==0).mean():.0%}, "
          f"<=3 views {(nvis<=3).mean():.0%} | disc normal |cos| with up: median {np.median(np.abs(axis@up)):.2f}")
