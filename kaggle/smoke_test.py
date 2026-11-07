"""A few real training steps on the GPU before spending hours on a full run.
Usage (repo root): python kaggle/smoke_test.py <config> [--steps 2] [--expect-head RaCAHead] [--expect-keys loss_align_directed,loss_scan]
                                                   [--forbid-keys cda_kl_loss]
"""
import os, sys, time, importlib, argparse, torch
sys.path.insert(0, os.getcwd())
from mmcv import Config
from mmdet.apis import set_random_seed
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
importlib.import_module("projects.mmdet3d_plugin")
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

p = argparse.ArgumentParser()
p.add_argument("config"); p.add_argument("--steps", type=int, default=2)
p.add_argument("--expect-head", default=""); p.add_argument("--expect-keys", default=""); p.add_argument("--forbid-keys", default="")
a = p.parse_args()

cfg = Config.fromfile(a.config)
set_random_seed(0, deterministic=True)        # as kaggle/train.sh (--seed 0 --deterministic)
model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
head = type(model.pts_bbox_head).__name__
print("head:", head, "| params:", sum(q.numel() for q in model.parameters()))
if a.expect_head:
    assert head == a.expect_head, f"expected head {a.expect_head}, got {head}"
if cfg.get("load_from"):
    ck = load_checkpoint(model, cfg.load_from, map_location="cpu", strict=False)
model = MMDataParallel(model.cuda(), device_ids=[0]).train()
loader = build_dataloader(build_dataset(cfg.data.train), 1, 2, num_gpus=1, dist=False, shuffle=True, seed=0)
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=1e-6)
for i, data in zip(range(a.steps), loader):
    t = time.time()
    out = model.train_step(data, opt)
    opt.zero_grad(); out["loss"].backward(); opt.step(); torch.cuda.synchronize()
    lv = out["log_vars"]
    extra = " | ".join(f"{k} {lv[k]:.3f}" for k in sorted(lv) if k.startswith(("loss_align", "loss_scan", "cda_kl", "fine_leads")))
    print(f"step {i}: total loss {lv['loss']:.3f} | {extra} | {time.time() - t:.1f}s | "
          f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
print("loss keys:", sorted(k for k in lv if k.startswith("loss")))
for k in filter(None, a.expect_keys.split(",")):
    assert k in lv, f"expected loss key {k} not found"
for k in filter(None, a.forbid_keys.split(",")):
    assert k not in lv, f"loss key {k} should not be present"
print("SMOKE TEST OK")
