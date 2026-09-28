# sam-3d-body venv: the MHR skeleton + skinning as npz (subject independent).
import os
import numpy as np, torch
from pipeline.steps.head_fit import rig_binding_data
mhr = torch.jit.load(os.environ.get("MHR_MODEL", "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"), map_location="cpu")
d = rig_binding_data(mhr)
np.savez("rig_binding.npz", **{k: np.asarray(v) for k, v in d.items()})
print({k: np.asarray(v).shape for k, v in d.items()})
