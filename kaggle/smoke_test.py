"""A few real training steps on the GPU before spending hours on a full run.
Usage (repo root): python kaggle/smoke_test.py <config> [--steps 2] [--expect-head RaCAHead] [--expect-keys loss_align_directed,loss_scan]
                                                   [--forbid-keys cda_kl_loss] [--val-frames 2] [--deterministic]
Also runs the single-GPU validation path on --val-frames validation frames (dataset.evaluate + EarlyStoppingHook),
because the real run validates after every epoch; --val-frames 0 skips it.
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
p.add_argument("--val-frames", type=int, default=2)
p.add_argument("--deterministic", action="store_true", help="cudnn.deterministic=True as --deterministic / DETERMINISTIC=1 in train.sh (needs >13 GB on a T4)")
a = p.parse_args()

cfg = Config.fromfile(a.config)
set_random_seed(0, deterministic=a.deterministic)          # as kaggle/train.sh (--seed 0, --deterministic only with DETERMINISTIC=1)
print(f"cudnn.deterministic={torch.backends.cudnn.deterministic} cudnn.benchmark={torch.backends.cudnn.benchmark} | "
      f"GPU {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
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
          f"peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB | peak reserved "
          f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GB | now allocated {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)
print("loss keys:", sorted(k for k in lv if k.startswith("loss")))
for k in filter(None, a.expect_keys.split(",")):
    assert k in lv, f"expected loss key {k} not found"
for k in filter(None, a.forbid_keys.split(",")):
    assert k not in lv, f"loss key {k} should not be present"

if a.val_frames > 0:
    # Validation path as mmdet_train.py uses it on one GPU: custom_single_gpu_test -> dataset.evaluate -> EarlyStoppingHook
    from projects.mmdet3d_plugin.mra.utils.ssc_metric import SSCMetrics
    from projects.mmdet3d_plugin.mra.hooks.early_stopping import EarlyStoppingHook
    val_ds = build_dataset(cfg.data.val, dict(test_mode=True))
    val_loader = build_dataloader(val_ds, 1, 2, num_gpus=1, dist=False, shuffle=False)
    model.eval(); metric = SSCMetrics(len(val_ds.class_names)).cuda()
    with torch.no_grad():
        for i, data in zip(range(a.val_frames), val_loader):
            res = model(return_loss=False, rescale=True, **data)
            metric.update(y_pred=torch.argmax(res["output_voxels"], dim=1), y_true=res["target_voxels"].clone())
    scores = val_ds.evaluate({"ssc_scores": metric.compute()})
    print(f"validation on {a.val_frames} frames:", {k: scores[k] for k in ("semkitti_SC_IoU", "semkitti_SSC_mIoU")})
    hooks = [h for h in (cfg.get("custom_hooks") or []) if h.get("type") == "EarlyStoppingHook"]
    for h in hooks:
        h = {k: v for k, v in h.items() if k not in ("type", "priority")}
        es = EarlyStoppingHook(**h)
        assert es.monitor in scores, f"EarlyStoppingHook monitors {es.monitor!r}, not in dataset.evaluate() keys {sorted(scores)}"

        class _Runner:                               # just enough of EpochBasedRunner for the hook
            epoch, _max_epochs, work_dir, rank = 0, 99, "/tmp", 0
            class log_buffer: output = scores
            import logging; logger = logging.getLogger("smoke")
        es.after_train_epoch(_Runner)
        assert es.best == float(scores[es.monitor]) and es.best_epoch == 1
        print(f"EarlyStoppingHook ok: monitor={es.monitor} rule={es.rule} patience={es.patience} min_epochs={es.min_epochs}")
    if not hooks:
        print("no EarlyStoppingHook in this config")
print("SMOKE TEST OK")
