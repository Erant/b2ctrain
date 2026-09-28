# b2crunner venv: evaluate the study's splats of one bundle.
#   python d_eval.py <bundle>
# Per ply: PSNR / sharpness at all 162 cameras, split into the 81 pre-extension frames and the 81 extension frames
# (held out for the p81 variants); splat placement against the refit body (opacity mass > 2 / 4 cm off the surface,
# overall and on arm-labelled splats); novel elevated renders as panels.
import os
import json, subprocess, sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from c_train import header_comments, comment_array, read_mesh_ply  # noqa: E402

HERE = Path(os.environ.get("BUNDLE_STUDY_OUT", ".")).resolve()  # outputs, rig_binding.npz
TRAINER = os.environ.get("B2C_TRAINER", str(Path(__file__).resolve().parents[2] / "build" / "b2ctrain"))
BENCH = Path(__file__).resolve().parents[2] / "bench"
PY = sys.executable
ARM = {6, 7, 11, 15, 16, 20}  # Goliath: hands, lower arms, upper arms (L/R)


def variants(bundle: Path):
    ply = bundle / "ply"
    return {"pod": ply / "scene.ply", "local162": HERE / bundle.name / "local162" / "scene.ply",
            "n162": ply / "scene_normals.ply", "p81": ply / "scene_81.ply", "p81n": ply / "scene_81_normals.ply"}


def read_splats(path: Path):
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    head = data[:end].decode().splitlines()
    n = int(next(l for l in head if l.startswith("element vertex")).split()[-1])
    props = [l.split()[-1] for l in head if l.startswith("property float")]
    arr = np.frombuffer(data, np.float32, n * len(props), end).reshape(n, len(props))
    col = {p: arr[:, i] for i, p in enumerate(props)}
    return col


def placement(splats, tree, mesh_normals_at):
    xyz = np.stack([splats["x"], splats["y"], splats["z"]], 1)
    opa = 1 / (1 + np.exp(-splats["opacity"]))
    d, idx = tree.query(xyz)
    out = {"splats": int(len(xyz))}
    for name, sel in (("all", np.ones(len(xyz), bool)),
                      ("arm", np.isin(splats.get("seg_label", np.full(len(xyz), -1)).astype(int), list(ARM)))):
        w = opa[sel]; dd = d[sel]
        out[name] = {"n": int(sel.sum()), "mass": float(w.sum()),
                     "off2cm": float(w[dd > 0.02].sum() / max(w.sum(), 1e-9)),
                     "off4cm": float(w[dd > 0.04].sum() / max(w.sum(), 1e-9)),
                     "median_mm": float(np.median(dd) * 1000) if len(dd) else 0.0}
    return out


def render(ply: Path, cams: Path, out: Path, bg: str):
    if out.exists() and any(out.iterdir()):
        return
    subprocess.check_call([TRAINER, "render", "--splat", str(ply), "--cameras", str(cams), "--output-dir", str(out),
                           "--background", bg], stdout=subprocess.DEVNULL)


def sharp(gray, mask):
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(lap[mask].var()) if mask.any() else 0.0


EXTRA_KEYS = []


def main(bundle: Path):
    work = HERE / bundle.name / "eval"; work.mkdir(parents=True, exist_ok=True)
    cams = work / "cams.json"
    if not cams.exists():
        subprocess.check_call([PY, str(BENCH / "colmap_to_cameras.py"), str(bundle / "colmap"), str(cams)], stdout=subprocess.DEVNULL)
    comments = header_comments(bundle / "ply" / "scene.ply")
    extras = json.loads(next(c for c in comments if c.startswith("comment b2c.orbit.extras")).split(" ", 3)[3])
    target = ",".join(str(v) for v in extras["orbit_target"])
    novel = work / "novel.json"
    if not novel.exists():
        subprocess.check_call([PY, str(BENCH / "novel_cameras.py"), str(cams), str(novel), "--every", "27",
                               "--elev", "35", "-35", f"--target={target}"], stdout=subprocess.DEVNULL)
    before = int(comment_array(comments, "b2c.orbit.extension.before")[0])
    npass = int(comment_array(comments, "b2c.orbit.pass_frames")[0])
    names = [c["name"] for c in json.load(open(cams))["cameras"]]
    in_pass = np.array([before <= i < before + npass for i in range(len(names))])

    verts, faces = read_mesh_ply(bundle / "debug" / "body_refit" / "mesh_refit.ply")
    tri = verts[faces]
    rng = np.random.default_rng(0)
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    pick = rng.choice(len(faces), 400000, p=area / area.sum())
    u, v = rng.random((2, len(pick)))
    flip = u + v > 1; u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    surf = tri[pick, 0] + u[:, None] * (tri[pick, 1] - tri[pick, 0]) + v[:, None] * (tri[pick, 2] - tri[pick, 0])
    tree = cKDTree(surf)

    gts = {}
    for n in names:
        rgba = cv2.imread(str(bundle / "colmap" / "images" / n), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
        gts[n] = (rgba[:, :, :3] * rgba[:, :, 3:], rgba[:, :, 3])

    results = {}
    vs = variants(bundle)
    for k in EXTRA_KEYS: vs[k] = HERE / bundle.name / k / "scene.ply"
    for key, ply in vs.items():
        if not ply.exists():
            print("missing", key, ply); continue
        rdir = work / key / "train_black"
        render(ply, cams, rdir, "0,0,0")
        render(ply, novel, work / key / "novel_white", "1,1,1")
        psnr, s_r, s_g = [], [], []
        for n in names:
            pred = cv2.imread(str(rdir / n), cv2.IMREAD_UNCHANGED)[:, :, :3].astype(np.float32) / 255
            gt, a = gts[n]
            mse = float(((pred - gt) ** 2).mean())
            psnr.append(10 * np.log10(1 / max(mse, 1e-12)))
            m = cv2.erode((a > 0.99).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            s_r.append(sharp(cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY) * 255, m))
            s_g.append(sharp(cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY) * 255, m))
        psnr, s_r, s_g = map(np.array, (psnr, s_r, s_g))
        results[key] = {
            "psnr_pass": float(psnr[in_pass].mean()), "psnr_ext": float(psnr[~in_pass].mean()),
            "sharp_ratio_pass": float(s_r[in_pass].sum() / s_g[in_pass].sum()),
            "sharp_ratio_ext": float(s_r[~in_pass].sum() / s_g[~in_pass].sum()),
            "placement": placement(read_splats(ply), tree, None),
        }
        print(key, json.dumps(results[key]), flush=True)
    (work / "results.json").write_text(json.dumps(results, indent=1))

    # novel-view panels: one row per variant, columns = novel cameras
    novel_names = [c["name"] for c in json.load(open(novel))["cameras"]]
    rows = []
    for key in results:
        tiles = []
        for n in novel_names:
            im = cv2.imread(str(work / key / "novel_white" / n), cv2.IMREAD_UNCHANGED)[:, :, :3]
            im = cv2.resize(im, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
            tiles.append(im)
        row = np.hstack(tiles)
        cv2.putText(row, key, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        rows.append(row)
    cv2.imwrite(str(work / "novel_panel.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    EXTRA_KEYS.extend(sys.argv[2:])
    main(Path(sys.argv[1]).expanduser().resolve())
