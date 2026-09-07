#!/usr/bin/env python3
"""Evaluate a trained PLY against the training frames of a COLMAP dataset.
Renders every N-th view (b2ctrain render or brush-splat-render) on black and reports mask-weighted PSNR/SSIM,
mirroring brush's eval (8-bit quantised prediction, transparent GT premultiplied, masked views alpha-weighted).
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("ply"); ap.add_argument("colmap_dir")
ap.add_argument("--renderer", default="b2ctrain", help="'b2ctrain' or path to brush-splat-render")
ap.add_argument("--b2ctrain", default=os.path.join(os.path.dirname(__file__), "..", "build", "b2ctrain"))
ap.add_argument("--every", type=int, default=8)
ap.add_argument("--keep", default="", help="directory to keep renders in")
a = ap.parse_args()

def gauss_blur(x):
    return cv2.GaussianBlur(x, (11, 11), 1.5, borderType=cv2.BORDER_CONSTANT)
def ssim_map(x, y):
    C1, C2 = 0.01**2, 0.03**2
    mu1, mu2 = gauss_blur(x), gauss_blur(y)
    s1 = np.maximum(gauss_blur(x*x) - mu1*mu1, 0); s2 = np.maximum(gauss_blur(y*y) - mu2*mu2, 0); s12 = gauss_blur(x*y) - mu1*mu2
    return np.clip(((2*mu1*mu2 + C1)*(2*s12 + C2)) / ((mu1*mu1 + mu2*mu2 + C1)*(s1 + s2 + C2)), -1, 1)

tmp = a.keep or tempfile.mkdtemp(prefix="evalply_")
os.makedirs(tmp, exist_ok=True)
cams_json = os.path.join(tmp, "cams.json")
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), "colmap_to_cameras.py"), a.colmap_dir, cams_json, "--every", str(a.every)], stdout=subprocess.DEVNULL)
out_dir = os.path.join(tmp, "renders")
if a.renderer == "b2ctrain":
    subprocess.check_call([a.b2ctrain, "render", "--splat", a.ply, "--cameras", cams_json, "--output-dir", out_dir, "--background", "0,0,0"], stdout=subprocess.DEVNULL)
else:
    subprocess.check_call([a.renderer, "--splat", a.ply, "--cameras", cams_json, "--output-dir", out_dir, "--background", "0,0,0"], stdout=subprocess.DEVNULL)
cams = json.load(open(cams_json))["cameras"]
psnrs, ssims = [], []
for c in cams:
    name = c["name"]; stem = os.path.splitext(name)[0]
    img_path = os.path.join(a.colmap_dir, "images", name)
    gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if gt is None: continue
    mask_path = os.path.join(a.colmap_dir, "masks", stem + ".png")
    masked = os.path.exists(mask_path)
    if gt.ndim == 2: gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGRA)
    if gt.shape[2] == 3: gt = np.dstack([gt, np.full(gt.shape[:2], 255, np.uint8)])
    if masked:
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE); gt[..., 3] = m
    gt = gt.astype(np.float32) / 255.0
    alpha = gt[..., 3:4]
    rgb = gt[..., :3]
    if not masked: rgb = rgb * alpha  # transparent: premultiplied over black
    pred = cv2.imread(os.path.join(out_dir, stem + ".png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    pr = pred[..., :3]
    w = alpha if masked else np.ones_like(alpha)
    d = np.abs(pr - rgb) * w
    mse = (d * d).mean() / max(w.mean(), 0.01)
    s = np.mean([(ssim_map(pr[..., k], rgb[..., k]) * w[..., 0]).mean() for k in range(3)]) / max(w.mean(), 0.01)
    psnrs.append(10 * np.log10(1.0 / max(mse, 1e-12))); ssims.append(s)
print(f"{os.path.basename(a.ply)}: {len(psnrs)} views, PSNR {np.mean(psnrs):.3f} (min {np.min(psnrs):.2f}), SSIM {np.mean(ssims):.4f}")
