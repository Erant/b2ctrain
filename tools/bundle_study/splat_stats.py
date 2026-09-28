# Arm splats of a ply: colour vs the angle between their supporting view direction and the refit body's normal.
import sys, json, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parent))
from c_train import read_mesh_ply
from d_eval import read_splats
bundle = Path(sys.argv[1]); ply = Path(sys.argv[2])
V, F = read_mesh_ply(bundle / "debug/body_refit/mesh_refit.ply")
fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
vn = np.zeros_like(V); [np.add.at(vn, F[:, k], fn) for k in range(3)]
vn /= np.linalg.norm(vn, axis=1, keepdims=True)
if (np.einsum("ij,ij->i", vn, V - V.mean(0)) > 0).mean() < 0.5: vn = -vn
s = read_splats(ply)
xyz = np.stack([s["x"], s["y"], s["z"]], 1).astype(np.float64)
C0 = 0.28209479
rgb = np.clip(np.stack([s["f_dc_0"], s["f_dc_1"], s["f_dc_2"]], 1) * C0 + 0.5, 0, 1)
lum = rgb.mean(1); opa = 1 / (1 + np.exp(-s["opacity"]))
lab = s["seg_label"].astype(int)
arm = np.isin(lab, [6, 7, 11, 15, 16, 20]) & (opa > 0.3)
d, idx = cKDTree(V).query(xyz)
n = vn[idx]
ev = np.stack([s["ev_dir_0"], s["ev_dir_1"], s["ev_dir_2"]], 1)
evn = ev / np.maximum(np.linalg.norm(ev, axis=1, keepdims=True), 1e-12)
# ev_dir sign: which way does it point? report both conventions via the median over all well-supported splats
cosang = np.einsum("ij,ij->i", evn, n)
conc = np.linalg.norm(ev, axis=1) / np.maximum(s["ev_w_in"], 1e-12)  # 1 = all views from one direction
up = np.array([0, 1, 0.0])  # world is y-up (feet -0.78, head +0.7)
facing_up = n @ up
print("median cos(evdir, normal) over arm splats:", np.median(cosang[arm]).round(3))
sign = 1 if np.median(cosang[arm]) > 0 else -1
cosang *= sign
dark = arm & (lum < 0.18); bright = arm & (lum > 0.35)
def summ(m, name):
    print(f"{name:14} n={m.sum():6d} lum={lum[m].mean():.2f} dist_mm={np.median(d[m])*1000:5.1f} "
          f"normal·up={np.median(facing_up[m]):+.2f} cos(view,normal)={np.median(cosang[m]):+.2f} "
          f"frac_grazing(<0.3)={(cosang[m]<0.3).mean():.2f} views={np.median(s['ev_views'][m]):.1f} "
          f"w_in/w_all={np.median(s['ev_w_in'][m]/np.maximum(s['ev_w_all'][m],1e-9)):.2f} conc={np.median(conc[m]):.2f} "
          f"err/w={np.median(s['ev_err'][m]/np.maximum(s['ev_w_all'][m],1e-9)):.3f}")
summ(arm, "arm all"); summ(dark, "arm dark"); summ(bright, "arm bright")
top = arm & (facing_up > 0.5); summ(top, "arm top-face"); summ(top & (lum < 0.18), "top-face dark")
np.savez(Path(sys.argv[3]), dark=np.nonzero(top & (lum < 0.18))[0])
# elevation of the mean supporting direction (sign as ev_dir, flipped to point splat->camera if needed)
elev = np.degrees(np.arcsin(np.clip((evn * sign) @ up, -1, 1)))
for m, name in ((arm, "arm all"), (top, "arm top-face"), (top & (lum < 0.18), "top-face dark"), (top & (lum > 0.35), "top-face bright")):
    print(f"{name:16} mean view elevation {np.median(elev[m]):+5.1f} deg, normal elevation {np.median(np.degrees(np.arcsin(np.clip(n[m] @ up, -1, 1)))):+5.1f} deg, n={m.sum()}")
