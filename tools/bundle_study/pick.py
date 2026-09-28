# Splats visible in a novel view that are dark and sit on the arms; then: where are they in the training frames?
import sys, json, cv2, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parent))
from c_train import read_mesh_ply
from d_eval import read_splats
B = Path(sys.argv[1]); ply = Path(sys.argv[2]); camfile, camname = sys.argv[3], sys.argv[4]
s = read_splats(ply)
xyz = np.stack([s["x"], s["y"], s["z"]], 1).astype(np.float64)
rgb = np.clip(np.stack([s["f_dc_0"], s["f_dc_1"], s["f_dc_2"]], 1) * 0.28209479 + 0.5, 0, 1); lum = rgb.mean(1)
opa = 1 / (1 + np.exp(-s["opacity"])); scale = np.exp(np.stack([s["scale_0"], s["scale_1"], s["scale_2"]], 1))
lab = s["seg_label"].astype(int); ARM = [6, 7, 11, 15, 16, 20]

def project(cam, P):
    R = np.array(cam["rotation"]) @ np.diag([1, -1, -1]); p = np.array(cam["position"])
    q = (P - p) @ R  # camera coords, OpenCV (x right, y down, z forward)
    z = q[:, 2]; u = cam["fx"] * q[:, 0] / z + cam["cx"]; v = cam["fy"] * q[:, 1] / z + cam["cy"]
    return u, v, z

J = json.load(open(camfile)); cam = next(c for c in J["cameras"] if c["name"] == camname); W, H = J["width"], J["height"]
u, v, z = project(cam, xyz)
ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0) & (opa > 0.2)
# front-most depth per 4px cell among opaque splats
cell = (v[ok] // 4).astype(int) * (W // 4 + 1) + (u[ok] // 4).astype(int)
front = np.full(cell.max() + 1, np.inf); np.minimum.at(front, cell, z[ok])
vis = np.zeros(len(xyz), bool); idx = np.nonzero(ok)[0]; vis[idx] = z[idx] < front[cell] + 0.015
armvis = vis & np.isin(lab, ARM)
dark = armvis & (lum < 0.2); bright = armvis & (lum > 0.35)
print(f"visible arm splats {armvis.sum()}, dark {dark.sum()} ({dark.sum()/armvis.sum():.1%}), bright {bright.sum()}")

V, F = read_mesh_ply(B / "debug/body_refit/mesh_refit.ply")
fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); vn = np.zeros_like(V)
for k in range(3): np.add.at(vn, F[:, k], fn)
vn /= np.linalg.norm(vn, axis=1, keepdims=True)
if (np.einsum("ij,ij->i", vn, V - V.mean(0)) > 0).mean() < 0.5: vn = -vn
d, nn = cKDTree(V).query(xyz); signed = np.einsum("ij,ij->i", xyz - V[nn], vn[nn])
for m, name in ((dark, "dark"), (bright, "bright")):
    print(f"{name:7} signed dist to body mm: median {np.median(signed[m])*1000:+.1f}  p10 {np.percentile(signed[m],10)*1000:+.1f} p90 {np.percentile(signed[m],90)*1000:+.1f} | "
          f"max scale mm {np.median(scale[m].max(1))*1000:.1f} min/max {np.median(scale[m].min(1)/scale[m].max(1)):.2f} | opacity {np.median(opa[m]):.2f} | views {np.median(s['ev_views'][m]):.1f}")

# into the training frames: for each splat, per view -> is it projected onto the subject, how far from the silhouette,
# and what colour is the frame there
T = json.load(open(sys.argv[5])); names = [c["name"] for c in T["cameras"]]
stats = {"dark": [], "bright": []}
rng = np.random.default_rng(0)
samp = {"dark": rng.choice(np.nonzero(dark)[0], min(400, dark.sum()), replace=False),
        "bright": rng.choice(np.nonzero(bright)[0], min(400, bright.sum()), replace=False)}
per_view_dark = np.zeros(len(names))
for vi, c in enumerate(T["cameras"]):
    im = cv2.imread(str(B / "colmap/images" / c["name"]), cv2.IMREAD_UNCHANGED)
    a = im[:, :, 3]; L = im[:, :, :3].mean(2) / 255
    din = cv2.distanceTransform((a > 127).astype(np.uint8), cv2.DIST_L2, 5)
    dout = cv2.distanceTransform((a <= 127).astype(np.uint8), cv2.DIST_L2, 5)
    for key, ids in samp.items():
        uu, vv, zz = project(c, xyz[ids])
        inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H) & (zz > 0)
        ui, vj = uu[inb].astype(int), vv[inb].astype(int)
        edge = np.where(a[vj, ui] > 127, din[vj, ui], -dout[vj, ui])  # >0 inside, px from silhouette
        stats[key].append(np.stack([np.full(inb.sum(), vi), ids[inb], edge, L[vj, ui]], 1))
for key in stats:
    arr = np.concatenate(stats[key]); ins = arr[:, 2] > 0
    print(f"{key:7} projections inside subject {ins.mean():.2f}; of those, within 6 px of silhouette {(arr[ins,2]<=6).mean():.2f}; "
          f"frame lum at splat: interior(>6px) {arr[ins & (arr[:,2]>6),3].mean():.2f}, rim(<=6px) {arr[ins & (arr[:,2]<=6),3].mean():.2f}")
    # per splat: fraction of its in-subject views that are rim views
    np.save(f"proj_{key}.npy", arr)
np.savez("picked.npz", dark=np.nonzero(dark)[0], bright=np.nonzero(bright)[0])
