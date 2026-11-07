"""OPTIONAL: build SGN-T warm-start weights. The Kaggle configs do NOT use this by default; they start from MRA's own
initialisation (ImageNet ResNet-50 + default init). To use it: bash kaggle/get_weights.sh sgn, run this script, then train
with --cfg-options load_from=ckpts/warm_start.pth. It was used for the first preliminary runs (see the root README).
Usage: python kaggle/make_warm_start.py [ckpts/warm_start.pth] [projects/configs/mra/MRA-T.py]   (run from the repo root)

MRA's code is built on SGN, and SGN-T's trained SemanticKITTI weights fill 86% of MRA.
MRA's extra low-resolution branch (keys ending in '_down1') has the same structure as SGN's main branch,
so we start it from those same SGN weights instead of from random values.
"""
import os, sys, importlib, torch
sys.path.insert(0, os.getcwd())
from mmcv import Config
from mmdet3d.models import build_model
importlib.import_module("projects.mmdet3d_plugin")

out = sys.argv[1] if len(sys.argv) > 1 else "ckpts/warm_start.pth"
cfg = Config.fromfile(sys.argv[2] if len(sys.argv) > 2 else "projects/configs/mra/MRA-T.py")
# Build from MRA-T so that the result is identical for every run; RaCA heads load it with strict=False.
model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
model.init_weights()                                    # ImageNet ResNet-50 for the image backbone
msd = model.state_dict()
sgn = torch.load("ckpts/sgn-T.pth", map_location="cpu"); sgn = sgn.get("state_dict", sgn)

new, direct, copied, untouched = {}, 0, 0, []
for k, v in msd.items():
    if k.startswith("tsw."):                            # teacher: MRA copies the student into it at step 0
        new[k] = v; continue
    if k in sgn and sgn[k].shape == v.shape:
        new[k] = sgn[k]; direct += 1
    elif "_down1" in k and k.replace("_down1", "") in sgn and sgn[k.replace("_down1", "")].shape == v.shape:
        new[k] = sgn[k.replace("_down1", "")]; copied += 1
    else:
        new[k] = v; untouched.append(k)
student = [k for k in msd if not k.startswith("tsw.")]
torch.save({"state_dict": new, "meta": {"note": "SGN-T + SGN main branch copied into MRA _down1 branch"}}, out)
print(f"student tensors: {len(student)} | from SGN-T: {direct} | low-res branch copied from SGN: {copied} | "
      f"left at initial values: {len(untouched)}")
print("examples left at initial values:", untouched[:6])
print("saved", out)
