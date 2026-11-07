"""Evaluate one checkpoint on our validation frames.
Usage (repo root): python kaggle/evaluate.py <config> <checkpoint> <out.json> --name "label"

Overall numbers use MRA's own metric code (SSCMetrics), so they are computed exactly like the paper's.
Extra numbers for the methodology (Section 5): four disjoint 12.8 m range bands along the forward axis,
per-class IoU in each band with the number of ground-truth voxels, pooled vulnerable-road-user (VRU) IoU,
and frequency-weighted IoU (wIoU).
"""
import os, sys, json, time, importlib, argparse
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
importlib.import_module("projects.mmdet3d_plugin")
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.mra.utils.ssc_metric_torch import SSCMetrics

p = argparse.ArgumentParser()
p.add_argument("config"); p.add_argument("checkpoint"); p.add_argument("out_json"); p.add_argument("--name", default="")
a = p.parse_args()

CLASSES = ["empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist",
           "road", "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain",
           "pole", "traffic-sign"]
VRU = [2, 3, 6, 7, 8]
BANDS = [(0, 64, "0-12.8 m"), (64, 128, "12.8-25.6 m"), (128, 192, "25.6-38.4 m"), (192, 256, "38.4-51.2 m")]

cfg = Config.fromfile(a.config)
dataset = build_dataset(cfg.data.val)
loader = build_dataloader(dataset, 1, 2, num_gpus=1, dist=False, shuffle=False)
model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
ck = load_checkpoint(model, a.checkpoint, map_location="cpu", strict=False)
model = MMDataParallel(model.cuda(), device_ids=[0]).eval()

overall = SSCMetrics(CLASSES).cuda()
bands = [SSCMetrics(CLASSES).cuda() for _ in BANDS]
vru = np.zeros(3)                                   # tp, fp, fn for the 5 rare classes merged into one
band_vru = np.zeros((len(BANDS), 3))
times = []
for i, data in enumerate(loader):
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        res = model(return_loss=False, rescale=True, **data)
    torch.cuda.synchronize(); times.append(time.time() - t)
    pred = torch.argmax(res["output_voxels"], dim=1)          # (1, 256, 256, 32), x = forward axis
    gt = res["target_voxels"].clone()
    for (lo, hi, _), m in zip(BANDS, bands):
        m.update(y_pred=pred[:, lo:hi].clone(), y_true=gt[:, lo:hi].clone())
    for bi, (lo, hi, _) in enumerate(BANDS):
        g, pr = gt[:, lo:hi], pred[:, lo:hi]
        valid = g != 255
        gv = sum((g == c) for c in VRU).bool() & valid            # torch 1.9 has no torch.isin
        pv = sum((pr == c) for c in VRU).bool() & valid
        band_vru[bi] += [int((gv & pv).sum()), int((~gv & pv).sum()), int((gv & ~pv).sum())]
    overall.update(y_pred=pred.clone(), y_true=gt.clone())
    if i % 25 == 0:
        print(f"  evaluated {i+1}/{len(dataset)}", flush=True)
vru = band_vru.sum(0)

def summarise(m):
    s = m.compute()
    iou = s["iou_ssc"].cpu().numpy()
    gt_count = (m.tps + m.fns).cpu().numpy()
    w = gt_count[1:] / max(gt_count[1:].sum(), 1)
    return {"IoU": round(100 * s["iou"], 2), "mIoU": round(100 * s["iou_ssc_mean"], 2),
            "wIoU": round(100 * float((iou[1:] * w).sum()), 2),
            "per_class_IoU": {c: round(100 * float(v), 2) for c, v in zip(CLASSES[1:], iou[1:])},
            "gt_voxels": {c: int(v) for c, v in zip(CLASSES[1:], gt_count[1:])}}

def vru_iou(x): return round(100 * x[0] / max(x.sum(), 1), 2)

out = {"name": a.name, "checkpoint": a.checkpoint, "frames": len(dataset),
       "missing_keys_when_loading": len(ck.get("missing_keys", [])) if isinstance(ck, dict) else None,
       "seconds_per_frame": round(float(np.mean(times[2:] or times)), 3),
       "overall": summarise(overall), "VRU_IoU": vru_iou(vru),
       "bands": {name: {**summarise(m), "VRU_IoU": vru_iou(band_vru[bi])}
                 for bi, ((_, _, name), m) in enumerate(zip(BANDS, bands))}}
os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
json.dump(out, open(a.out_json, "w"), indent=1)

o = out["overall"]
print(f"\n===== RESULTS: {a.name} ({len(dataset)} validation frames) =====")
print(f"IoU {o['IoU']:.2f} | mIoU {o['mIoU']:.2f} | wIoU {o['wIoU']:.2f} | VRU IoU {out['VRU_IoU']:.2f} | "
      f"{out['seconds_per_frame']:.2f} s/frame")
print("\nPer class (IoU %, ground-truth voxels):")
for c in CLASSES[1:]:
    print(f"  {c:14s} {o['per_class_IoU'][c]:6.2f}   {o['gt_voxels'][c]:>10,}")
print("\nBy distance (forward axis):")
print(f"  {'band':12s} {'IoU':>6s} {'mIoU':>6s} {'VRU IoU':>8s}   rare-class IoU (person/bicyclist/bicycle/motorcycle/motorcyclist)")
for name, b in out["bands"].items():
    r = "/".join(f"{b['per_class_IoU'][c]:.1f}" for c in ["person", "bicyclist", "bicycle", "motorcycle", "motorcyclist"])
    print(f"  {name:12s} {b['IoU']:6.2f} {b['mIoU']:6.2f} {b['VRU_IoU']:8.2f}   {r}")
print("\nsaved", a.out_json)
