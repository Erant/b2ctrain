#!/usr/bin/env python3
"""Run b2crunner's real `brush` step class against an example dataset, with b2ctrain as the trainer.

Exercises exactly the invocations pipeline/steps/brush.py makes -- the cold 30k run, the polish
(--growth-stop-iter 0 --refine-every 1000000 from init.ply) and the alignment loop (render through
`brush-splat-render`, warp, refit) -- without needing the rest of the pipeline. Run with b2crunner's venv:

  B2CRUNNER=~/Projects/b2crunner
  $B2CRUNNER/.venv/bin/python bench/b2crunner_step.py splat/colmap_intermediate out2 \
      --b2crunner $B2CRUNNER --polish-steps 9000 --align-iters 0 --match-alpha-weight 0.5   # train_splat
  $B2CRUNNER/.venv/bin/python bench/b2crunner_step.py splat/colmap out5 \
      --b2crunner $B2CRUNNER --align-iters 4 --no-normals --normal-loss-strength 0        # train_final_splat
"""
import argparse, logging, os, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dataset"); ap.add_argument("out")
ap.add_argument("--b2crunner", default=str(Path.home() / "Projects/b2crunner"))
ap.add_argument("--b2ctrain", default=str(Path(__file__).resolve().parent.parent / "build/b2ctrain"))
ap.add_argument("--render-path", default=str(Path(__file__).resolve().parent.parent / "docker/brush-splat-render"))
ap.add_argument("--total-steps", type=int, default=30000)
ap.add_argument("--polish-steps", type=int, default=0)
ap.add_argument("--align-iters", type=int, default=0)
ap.add_argument("--align-steps", type=int, default=3000)
ap.add_argument("--match-alpha-weight", type=float, default=0.1, help="brush.py default; fast_helical_native.yaml sets 0.5 on train_splat")
ap.add_argument("--normal-loss-strength", type=float, default=0.05)
ap.add_argument("--no-normals", action="store_true", help="do not wire normal_maps (train_final_splat shape)")
ap.add_argument("--export-name", default="export.ply")
ap.add_argument("--param", action="append", default=[], help="extra step param, name=value")
a = ap.parse_args()

out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("B2C_DATA_DIR", str(out / "data"))
# The shim execs `b2ctrain render`, so the binary has to be on PATH.
os.environ["PATH"] = str(Path(a.b2ctrain).resolve().parent) + os.pathsep + os.environ["PATH"]
sys.path.insert(0, a.b2crunner)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

import cv2
from body2colmap.camera import Camera
from pipeline.registry import get_step_class
import pipeline.steps  # noqa: F401

def quat_to_R(q):
    q = np.asarray(q, np.float64); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

ds = Path(a.dataset)
cams = {}
for line in open(ds / "cameras.txt"):
    if line.startswith("#") or not line.strip(): continue
    t = line.split(); cams[int(t[0])] = (t[1], int(t[2]), int(t[3]), [float(v) for v in t[4:]])
model, W, H, params = cams[1]
fx, fy, cx, cy = (params[:4] if model == "PINHOLE" else (params[0], params[0], params[1], params[2]))
entries = []
for line in open(ds / "images.txt"):
    t = line.split()
    if not t or t[0].startswith("#") or len(t) < 10: continue   # comments, blank/empty POINTS2D lines
    p = [float(v) for v in t[1:8]]
    R_w2c = quat_to_R(p[:4]); tvec = np.array(p[4:7])
    pos = -R_w2c.T @ tvec
    R_gl = R_w2c.T @ np.diag([1.0, -1.0, -1.0])   # OpenCV c2w -> body2colmap's OpenGL c2w
    entries.append((t[9], Camera(focal_length=(fx, fy), image_size=(W, H), principal_point=(cx, cy),
                                 position=pos.astype(np.float32), rotation=R_gl.astype(np.float32))))
entries.sort(key=lambda e: e[0])

pts, cols = [], []
for line in open(ds / "points3D.txt"):
    if line.startswith("#") or not line.strip(): continue
    t = line.split(); pts.append([float(v) for v in t[1:4]]); cols.append([int(v) for v in t[4:7]])
points_3d = (np.array(pts, np.float32), np.array(cols, np.uint8))

def sidecar(dirname, name):
    d = ds / dirname
    if not d.is_dir(): return None
    for cand in (d / name, d / (Path(name).stem + ".png")):
        if cand.exists(): return cand
    return None

train = {"cameras": [], "image_names": [], "images": [], "masks": [], "normal_maps": [], "weights": []}
support = {"support_cameras": [], "support_image_names": [], "support_images": [], "support_masks": []}
for name, cam in entries:
    img = cv2.imread(str(ds / "images" / name), cv2.IMREAD_UNCHANGED)
    mask_path = sidecar("masks", name)
    if mask_path is not None:                      # masked view -> supporting view
        support["support_cameras"].append(cam); support["support_image_names"].append(name)
        support["support_images"].append(np.ascontiguousarray(img[..., :3]))
        support["support_masks"].append(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0)
        continue
    assert img.shape[-1] == 4, f"{name}: transparent view without alpha"
    train["cameras"].append(cam); train["image_names"].append(name)
    train["images"].append(np.ascontiguousarray(img[..., :3]))
    train["masks"].append(img[..., 3].astype(np.float32) / 255.0)
    n = sidecar("normals", name)
    if n is not None and not a.no_normals:
        nrm = cv2.imread(str(n), cv2.IMREAD_UNCHANGED)[..., :3][..., ::-1].astype(np.float32) / 255.0 * 2.0 - 1.0
        train["normal_maps"].append(np.ascontiguousarray(nrm))
    w = sidecar("weights", name)
    if w is not None:
        train["weights"].append(cv2.imread(str(w), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0)

inputs = {"cameras": train["cameras"], "image_names": train["image_names"], "points_3d": points_3d,
          "images": train["images"], "masks": train["masks"]}
if train["normal_maps"]:
    assert len(train["normal_maps"]) == len(train["images"]); inputs["normal_maps"] = train["normal_maps"]
if train["weights"]:
    assert len(train["weights"]) == len(train["images"]); inputs["weights"] = train["weights"]
if support["support_images"]:
    inputs.update(support)
print(f"{len(train['images'])} training views, {len(support['support_images'])} supporting, "
      f"normals={'normal_maps' in inputs}, weights={'weights' in inputs}, {len(pts)} points, {W}x{H}")

overrides = {
    "total_steps": a.total_steps, "polish_steps": a.polish_steps, "align_iters": a.align_iters,
    "align_steps": a.align_steps, "match_alpha_weight": a.match_alpha_weight,
    "normal_loss_strength": a.normal_loss_strength, "export_evidence": True,
    "export_dir": str(out), "export_name": a.export_name,
    "brush_path": str(Path(a.b2ctrain).resolve()), "render_path": str(Path(a.render_path).resolve()),
}
if a.align_iters > 0:
    overrides["align_debug_dir"] = str(out / "alignment")
for kv in a.param:
    k, v = kv.split("=", 1); overrides[k] = v
step_class = get_step_class("brush")
params = step_class.resolve_params(overrides)
result = step_class().run(inputs, params)
print("result:", result)
