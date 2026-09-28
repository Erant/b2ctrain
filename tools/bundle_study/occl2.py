import sys, json, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d_eval import read_splats
B = Path(sys.argv[1]); s = read_splats(Path(sys.argv[2])); T = json.load(open(sys.argv[3])); W, H = T["width"], T["height"]
pk = np.load("picked.npz")
xyz = np.stack([s["x"], s["y"], s["z"]], 1).astype(np.float64); opa = 1 / (1 + np.exp(-s["opacity"]))
def project(cam, P):
    R = np.array(cam["rotation"]) @ np.diag([1, -1, -1]); p = np.array(cam["position"]); qq = (P - p) @ R
    return cam["fx"] * qq[:, 0] / qq[:, 2] + cam["cx"], cam["fy"] * qq[:, 1] / qq[:, 2] + cam["cy"], qq[:, 2]
rows = {"dark": [], "bright": []}
op = opa > 0.2
for vi, c in enumerate(T["cameras"]):
    im = cv2.imread(str(B / "colmap/images" / c["name"]), cv2.IMREAD_UNCHANGED); a = im[:, :, 3]
    L = cv2.erode(im[:, :, :3].mean(2).astype(np.float32), np.ones((3, 3))) / 255  # local min
    din = cv2.distanceTransform((a > 127).astype(np.uint8), cv2.DIST_L2, 5)
    u, v, z = project(c, xyz)
    ok = op & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    cell = (v // 4).astype(int).clip(0, H // 4) * (W // 4 + 1) + (u // 4).astype(int).clip(0, W // 4)
    front = np.full((H // 4 + 1) * (W // 4 + 1), np.inf); np.minimum.at(front, cell[ok], z[ok])
    for k in rows:
        ids = pk[k]; f = ok[ids] & (z[ids] < front[cell[ids]] + 0.01); ids = ids[f]
        ui, vj = u[ids].astype(int), v[ids].astype(int)
        rows[k].append(np.stack([ids, np.full(len(ids), vi), din[vj, ui], L[vj, ui], a[vj, ui]], 1))
for k in rows:
    r = np.concatenate(rows[k])
    for lo, hi in ((0, 3), (3, 8), (8, 20), (20, 1e9)):
        m = (r[:, 2] > lo) & (r[:, 2] <= hi) if lo > 0 else (r[:, 2] <= hi)
        print(f"{k:7} front-most views, {lo:>3}-{hi if hi<1e8 else 'inf':>3} px from silhouette: share {m.mean():.2f}, frame local-min lum {r[m,3].mean():.2f}")
    np.save(f"front_{k}.npy", r)
