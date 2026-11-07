"""Compare two evaluate.py result files: before (original MRA) vs after (RaCA), same data, starting weights and schedule.
Usage: python kaggle/compare.py before.json after.json"""
import json, sys
b, a = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
RARE = ["person", "bicyclist", "bicycle", "motorcycle", "motorcyclist"]
def row(name, x, y, n=None):
    extra = f"   (GT voxels: {n:,})" if n is not None else ""
    print(f"  {name:22s} {x:7.2f} {y:7.2f} {y - x:+7.2f}{extra}")
print(f"\n===== BEFORE vs AFTER ({a['frames']} validation scenes) =====")
print(f"  {'metric':22s} {'before':>7s} {'after':>7s} {'change':>7s}")
for k in ["IoU", "mIoU", "wIoU"]:
    row(k, b["overall"][k], a["overall"][k])
row("VRU IoU (5 rare pooled)", b["VRU_IoU"], a["VRU_IoU"])
print("\n  Rare classes (IoU):")
for c in RARE:
    row(c, b["overall"]["per_class_IoU"][c], a["overall"]["per_class_IoU"][c], a["overall"]["gt_voxels"][c])
print("\n  By distance (mIoU | VRU IoU):")
for band in a["bands"]:
    bb, aa = b["bands"][band], a["bands"][band]
    print(f"  {band:12s} mIoU {bb['mIoU']:6.2f} -> {aa['mIoU']:6.2f} ({aa['mIoU'] - bb['mIoU']:+.2f})   "
          f"VRU {bb['VRU_IoU']:6.2f} -> {aa['VRU_IoU']:6.2f} ({aa['VRU_IoU'] - bb['VRU_IoU']:+.2f})")
print("\n  All classes (IoU):")
for c in b["overall"]["per_class_IoU"]:
    row(c, b["overall"]["per_class_IoU"][c], a["overall"]["per_class_IoU"][c])
print("\nOne seed per run: differences of a few tenths of a point can be noise (methodology, Section 3.7).")
