"""Two real training steps and one inference step with the run's config, before the long run starts (about 1 minute).

Catches in a minute what would otherwise fail somewhere inside the first epoch: a GPU the installed libraries have
no kernels for, a missing data file, starting weights that do not fit the model. Prints the loss values and the
peak GPU memory. Usage (repo root): python molab/smoke_test.py <config> [--cfg-options key=value ...]
Runs on the GPU when there is one, otherwise on the CPU (slow; only used to test the scripts).
"""
import os, sys, time, importlib, argparse
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from mmcv import Config, DictAction
from mmcv.parallel import scatter
from mmcv.runner import load_checkpoint, build_optimizer
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
importlib.import_module("projects.mmdet3d_plugin")
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

p = argparse.ArgumentParser()
p.add_argument("config"); p.add_argument("--cfg-options", nargs="+", action=DictAction)
a = p.parse_args()
cfg = Config.fromfile(a.config)
if a.cfg_options:
    cfg.merge_from_dict(a.cfg_options)
gpu = torch.cuda.is_available()
dev = "cuda" if gpu else "cpu"
torch.manual_seed(0); np.random.seed(0)


def to_device(batch):
    """What MMDataParallel does before calling the model: unwrap the DataContainers and move tensors."""
    return scatter(batch, [0])[0] if gpu else scatter(batch, [-1])[0]


model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
model.init_weights()
if cfg.get("load_from"):
    ck = load_checkpoint(model, cfg.load_from, map_location="cpu", strict=False)
    print("starting weights:", cfg.load_from)
else:
    print("starting weights: MRA's own initialisation (ImageNet ResNet-50)")
model = model.to(dev)
optimizer = build_optimizer(model, cfg.optimizer)
print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M | device: "
      + (torch.cuda.get_device_name(0) if gpu else "cpu"))

train = build_dataset(cfg.data.train)
loader = build_dataloader(train, 1, 0, num_gpus=1, dist=False, shuffle=False)
model.train()
if gpu:
    torch.cuda.reset_peak_memory_stats()
losses = []
for i, batch in enumerate(loader):
    t = time.time()
    out = model.train_step(to_device(batch), optimizer)
    optimizer.zero_grad(); out["loss"].backward()
    grad = torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 35)
    optimizer.step()
    assert torch.isfinite(out["loss"]), "loss is not finite"
    losses.append(float(out["loss"]))
    print(f"train step {i + 1}: loss {losses[-1]:.4f} | grad norm {float(grad):.2f} | {time.time() - t:.1f} s | "
          + ", ".join(f"{k} {v:.3f}" for k, v in sorted(out["log_vars"].items()) if k != "loss"), flush=True)
    if i == 1:
        break
if gpu:
    print(f"peak GPU memory while training: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB "
          f"of {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")

val_cfg = cfg.data.val
val = build_dataset(val_cfg)
model.eval()
with torch.no_grad():
    batch = to_device(next(iter(build_dataloader(val, 1, 0, num_gpus=1, dist=False, shuffle=False))))
    res = model(return_loss=False, rescale=True, **batch)
pred = torch.argmax(res["output_voxels"], dim=1)
assert tuple(pred.shape) == (1, 256, 256, 32), pred.shape
print(f"inference step: output {tuple(res['output_voxels'].shape)} | occupied voxels predicted {int((pred > 0).sum()):,}")
print(f"SMOKE TEST PASSED: train frames {len(train)} | val frames {len(val)}")
