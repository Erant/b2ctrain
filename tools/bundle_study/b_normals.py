# masktest venv: Sapiens2 normals (b2crunner's sapiens2_lite step, low_vram) for every frame of a bundle's colmap/images,
# written as brush's normals/ sidecar: BGR uint8 of (n+1)/2*255 + the frame's alpha.
import sys, time, cv2, numpy as np
from pathlib import Path
from pipeline.steps.sapiens2 import Sapiens2LiteStep
bundles = [Path(p) for p in sys.argv[1:]]
step = Sapiens2LiteStep()
params = step.resolve_params({"low_vram": True}) if hasattr(step, "resolve_params") else None
for b in bundles:
    src = b / "colmap" / "images"; dst = b / "colmap" / "normals"; dst.mkdir(exist_ok=True)
    names = sorted(p.name for p in src.glob("frame_*.png"))
    t0 = time.time()
    for k, name in enumerate(names):
        if (dst / name).exists(): continue
        rgba = cv2.imread(str(src / name), cv2.IMREAD_UNCHANGED)
        n = step.run({"image": rgba[:, :, :3]}, params)["normal_map"]
        bgr = np.clip((n[..., ::-1] + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(dst / name), np.dstack([bgr, rgba[:, :, 3]]))
        if k % 20 == 0: print(b.name, k, len(names), f"{time.time()-t0:.0f}s", flush=True)
    print(b.name, "done", len(names), f"{time.time()-t0:.0f}s", flush=True)
