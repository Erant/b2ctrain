# b2crunner venv: rebuild a bundle's train_final_splat inputs and run b2crunner's own brush step on them.
#
#   python c_train.py <bundle> <variant>...   variants: local162, n162, p81, p81n
#
# Rebuilt from the bundle: frames+alpha, refined final cameras, points3D, labels (colmap/); the refit body mesh
# (debug/body_refit/mesh_refit.ply); the rig from the header's world joints + MHR skinning (rig_binding.npz);
# photo_priority_final re-run (weights + 12 masked photograph copies). Normals from colmap/normals/ (b_normals.py).
# p81* = the pre-extension pass: frames extension.before+1 .. extension.before+pass_frames.
import os
import json, logging, shutil, sys, time
from pathlib import Path

import cv2
import numpy as np

from body2colmap.camera import Camera
from pipeline.steps.body_rig import BuildBodyRigStep
from pipeline.steps.brush import BrushStep
from pipeline.steps.photo_priority import PhotoPriorityWeightsStep
from pipeline.steps.refine_cameras import _read_images_txt, _quaternion_to_rotation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", stream=sys.stdout)
log = logging.getLogger("c_train")

HERE = Path(os.environ.get("BUNDLE_STUDY_OUT", ".")).resolve()  # outputs, rig_binding.npz
TRAINER = os.environ.get("B2C_TRAINER", str(Path(__file__).resolve().parents[2] / "build" / "b2ctrain"))
NORMAL_WEIGHT = 0.05  # brush step default and the pipeline's intermediate training (from iter 5000)


def header_comments(ply: Path):
    out = []
    with open(ply, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line == "end_header":
                break
            if line.startswith("comment b2c."):
                out.append(line)
    return out


def comment_value(comments, key):
    for c in comments:
        parts = c.split(" ", 2)
        if parts[1] == key:
            return parts[2] if len(parts) > 2 else ""
    raise KeyError(key)


def comment_array(comments, key):
    shape_and_values = comment_value(comments, key).split()
    return np.array([float(v) for v in shape_and_values[1:]])


def read_mesh_ply(path: Path):
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    head = data[:end].decode().splitlines()
    nv = int(next(l for l in head if l.startswith("element vertex")).split()[-1])
    nf = int(next(l for l in head if l.startswith("element face")).split()[-1])
    v = np.frombuffer(data, np.float32, nv * 3, end).reshape(nv, 3)
    f = np.frombuffer(data, np.dtype([("n", "u1"), ("i", "<u4", 3)]), nf, end + nv * 12)
    assert (f["n"] == 3).all()
    return v.astype(np.float64), f["i"].astype(np.int64)


def read_points3d(path: Path):
    pos, col = [], []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        t = line.split()
        pos.append([float(x) for x in t[1:4]]); col.append([int(x) for x in t[4:7]])
    return np.array(pos, np.float32), np.array(col, np.uint8)


def build(bundle: Path):
    colmap = bundle / "colmap"
    cam_line = next(l for l in (colmap / "cameras.txt").read_text().splitlines() if l.strip() and not l.startswith("#")).split()
    assert cam_line[1] == "PINHOLE", cam_line
    w, h, fx, fy, cx, cy = int(cam_line[2]), int(cam_line[3]), *map(float, cam_line[4:8])
    poses = _read_images_txt(colmap / "images.txt")
    names = sorted(poses)
    gl_from_cv = np.diag([1.0, -1.0, -1.0])
    cameras = []
    for n in names:
        q, t = poses[n]
        r = _quaternion_to_rotation(q)
        cameras.append(Camera(focal_length=(fx, fy), image_size=(w, h), principal_point=(cx, cy),
                              position=-r.T @ t, rotation=r.T @ gl_from_cv))
    images, masks, labels, normals = [], [], [], []
    for n in names:
        rgba = cv2.imread(str(colmap / "images" / n), cv2.IMREAD_UNCHANGED)
        images.append(np.ascontiguousarray(rgba[:, :, :3])); masks.append(np.ascontiguousarray(rgba[:, :, 3]))
        labels.append(cv2.imread(str(colmap / "labels" / n), cv2.IMREAD_UNCHANGED))
        npath = colmap / "normals" / n
        normals.append(npath if npath.exists() else None)
    points = read_points3d(colmap / "points3D.txt")

    comments = header_comments(bundle / "ply" / "scene.ply")
    extras = json.loads(comment_value(comments, "b2c.orbit.extras").split(" ", 1)[1])
    anchor = int(comment_array(comments, "b2c.orbit.anchor_frame_index")[0])
    assert anchor == extras["anchor_frame_index"]
    before = int(comment_array(comments, "b2c.orbit.extension.before")[0])
    pass_frames = int(comment_array(comments, "b2c.orbit.pass_frames")[0])

    mesh = read_mesh_ply(bundle / "debug" / "body_refit" / "mesh_refit.ply")
    joints_world = comment_array(comments, "b2c.mhr.joints").reshape(-1, 3)
    binding = dict(np.load(HERE / "rig_binding.npz"))
    parents = comment_array(comments, "b2c.mhr.joint_parents").astype(int)
    assert (parents == binding["joint_parents"]).all()
    lo, hi = mesh[0].min(0), mesh[0].max(0)
    assert ((joints_world > lo - 0.05) & (joints_world < hi + 0.05)).all(), "joints outside the refit mesh: frame mismatch"

    rig_step = BuildBodyRigStep()
    rig = rig_step.run({"mesh_world": mesh, "joints": joints_world,
                        "world_from_raw": {"scale": 1.0, "rotation": np.eye(3), "translation": np.zeros(3)},
                        "rig_binding": binding},
                       rig_step.resolve_params({}))
    ref = json.loads((bundle / "debug" / "body_rig" / "body_rig.json").read_text())
    got = rig["body_rig_stats"]
    assert list(map(int, rig["body_rig"]["active"])) == ref["active_joints"], "rig active joints differ from the pod's"
    assert got["rig_vertices"] == ref["rig_vertices"], (got, ref)
    log.info("rig matches the pod's body_rig.json: %d active, %d rig vertices", got["active"], got["rig_vertices"])

    debug = HERE / bundle.name / "photo_priority_final"
    pp_step = PhotoPriorityWeightsStep()
    pp = pp_step.run({"cameras": cameras, "images": images, "alphas": masks, "anchor_cameras": cameras,
                      "anchor_frame_index": anchor, "anchor_position": extras.get("anchor_position"),
                      "mesh_world": mesh},
                     pp_step.resolve_params({"strength": 0.5, "debug_dir": str(debug)}))
    diffs = []
    for i in range(len(names)):
        a = cv2.imread(str(debug / f"weight_{i:03d}.png"), cv2.IMREAD_UNCHANGED).astype(np.int16)
        b = cv2.imread(str(bundle / "debug" / "photo_priority_final" / f"weight_{i:03d}.png"), cv2.IMREAD_UNCHANGED).astype(np.int16)
        diffs.append(np.abs(a - b).mean())
    log.info("photo priority weights vs the pod's: mean |diff| %.4f /255 (max view %.4f); %d copies",
             float(np.mean(diffs)), float(np.max(diffs)), len(pp["support_images"]))

    return dict(bundle=bundle, names=names, cameras=cameras, images=images, masks=masks, labels=labels,
                normals=normals, points=points, mesh=mesh, rig=rig["body_rig"], pp=pp, comments=comments,
                subset81=list(range(before, before + pass_frames)), anchor=anchor)


def decode_normal(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[:, :, :3]
    # +0.5 so brush's (n+1)/2*255 -> uint8 truncation gives back exactly these bytes
    return ((bgr[:, :, ::-1].astype(np.float32) + 0.5) / 255.0) * 2.0 - 1.0


def graft_header(src_ply: Path, dst_ply: Path, comments):
    """Copy the b2c.mhr.* / b2c.orbit.* comments of the delivered splat into a new export's header."""
    data = src_ply.read_bytes()
    end = data.index(b"end_header\n")
    head = [l for l in data[:end].decode().split("\n") if l and not l.startswith("comment b2c.")]
    at = next(i for i, l in enumerate(head) if l.startswith("element"))
    head = head[:at] + comments + head[at:]
    dst_ply.write_bytes(("\n".join(head) + "\n").encode() + data[end:])


EXTRAS = {"rbg": "--background-color 0.5,0.5,0.5 --background-noise-strength 0.5",
          "a05": "--match-alpha-weight 0.5"}
import re as _re


def _extra(m):
    if m in EXTRAS: return EXTRAS[m]
    g = _re.fullmatch(r"g(\d+)", m)  # g35 -> --growth-grad-threshold 0.0035
    if g: return f"--growth-grad-threshold {int(g.group(1)) / 10000}"
    raise KeyError(m)


def train(d, variant: str):
    import os
    base, *mods = variant.split("+")
    os.environ["B2C_EXTRA"] = " ".join(_extra(m) for m in mods)
    use = d["subset81"] if base.startswith("p81") else list(range(len(d["names"])))
    with_normals = base in ("n162", "p81n")
    out = HERE / d["bundle"].name / variant
    out.mkdir(parents=True, exist_ok=True)
    pick = lambda xs: [xs[i] for i in use]
    inputs = {
        "cameras": pick(d["cameras"]), "image_names": pick(d["names"]), "points_3d": d["points"],
        "images": pick(d["images"]), "masks": pick(d["masks"]), "mesh": d["mesh"], "body_rig": d["rig"],
        "labels": pick(d["labels"]), "weights": pick(d["pp"]["weights"]),
        "support_images": d["pp"]["support_images"], "support_masks": d["pp"]["support_masks"],
        "support_cameras": d["pp"]["support_cameras"],
    }
    if with_normals:
        missing = [d["names"][i] for i in use if d["normals"][i] is None]
        assert not missing, f"no normal map for {missing[:3]}..."
        inputs["normal_maps"] = [decode_normal(d["normals"][i]) for i in use]
    step = BrushStep()
    params = step.resolve_params({
        "total_steps": 30000, "hollow_weight": 0.5, "hollow_margin": 0.03,
        "normal_loss_strength": NORMAL_WEIGHT if with_normals else 0.0,
        "align_iters": 4, "align_debug_dir": str(out / "alignment"),
        "export_dir": str(out), "export_name": "scene.ply", "polish_steps": 0, "brush_path": str(Path(__file__).resolve().parent / "trainer_wrap.sh"),
    })
    log.info("=== %s %s: %d training views, normals %s", d["bundle"].name, variant, len(use), with_normals)
    t0 = time.time()
    step.run(inputs, params)
    log.info("=== %s %s done in %.0fs", d["bundle"].name, variant, time.time() - t0)
    graft_header(out / "scene.ply", out / "scene.ply", d["comments"])
    target = None if mods else {"n162": "scene_normals", "p81": "scene_81", "p81n": "scene_81_normals"}.get(variant)
    if target:
        ply_dir = d["bundle"] / "ply"
        shutil.copyfile(out / "scene.ply", ply_dir / f"{target}.ply")
        if (out / "body_rig_omega.json").exists():
            shutil.copyfile(out / "body_rig_omega.json", ply_dir / f"{target}_body_rig_omega.json")


if __name__ == "__main__":
    bundle = Path(sys.argv[1]).expanduser().resolve()
    data = build(bundle)
    for v in sys.argv[2:]:
        train(data, v)
