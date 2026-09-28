import sys, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d_eval import read_splats
from c_train import read_mesh_ply
# where.py <bundle> name=path.ply ...   (the first is the reference the last column is measured against)
B = Path(sys.argv[1])
plys = {a.split("=", 1)[0]: Path(a.split("=", 1)[1]) for a in sys.argv[2:]}
V, F = read_mesh_ply(B / "debug/body_refit/mesh_refit.ply"); tree = cKDTree(V)
G = {"hair": [4], "face/neck": [3, 2, 24, 25], "arms+hands": [6, 7, 11, 15, 16, 20], "legs/socks/feet": [5, 8, 10, 12, 14, 17, 19, 21], "shoes": [9, 18], "torso skin": [22], "upper cloth": [23], "lower cloth (skirt)": [13], "apparel": [1], "unlabelled": [0]}
rows = {}
for k, p in plys.items():
    s = read_splats(p); lab = s["seg_label"].astype(int); opa = 1 / (1 + np.exp(-s["opacity"]))
    sc = np.exp(np.stack([s["scale_0"], s["scale_1"], s["scale_2"]], 1)).max(1)
    xyz = np.stack([s["x"], s["y"], s["z"]], 1); d, _ = tree.query(xyz)
    r = {g: int(np.isin(lab, ids).sum()) for g, ids in G.items()}
    r["other labels"] = len(lab) - sum(r.values())
    r["TOTAL"] = len(lab)
    r["opacity<0.1"] = int((opa < 0.1).sum()); r["opacity 0.1-0.5"] = int(((opa >= 0.1) & (opa < 0.5)).sum()); r["opacity>=0.5"] = int((opa >= 0.5).sum())
    r["size<1mm"] = int((sc < 0.001).sum()); r["size 1-3mm"] = int(((sc >= 0.001) & (sc < 0.003)).sum()); r["size>=3mm"] = int((sc >= 0.003).sum())
    r[">3cm off body"] = int((d > 0.03).sum())
    rows[k] = r
keys = list(next(iter(rows.values())).keys())
first, last = list(rows)[0], list(rows)[-1]
print(f"{'':22}" + "".join(f"{k:>15}" for k in rows) + f"{'last - first':>14}")
for key in keys:
    print(f"{key:22}" + "".join(f"{rows[k][key]:>15,}" for k in rows) + f"{rows[last][key]-rows[first][key]:>+14,}")
