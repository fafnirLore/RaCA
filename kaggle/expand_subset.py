"""Rebuild the exact files MRA reads from our compact saved subset (to fit Kaggle's 20 GB output limit).

Saved (compact)                                  ->  written here (MRA's original format)
labels/XX/{fid}_1_1.npy   uint8                  ->  float32 (values 0-19, 255: converted exactly)
sequences_msnet3d_lidar/.../{fid}.bin  x,y,z     ->  x,y,z,1 float32, as MRA's depth2lidar.py writes it
Images, voxels, calib, poses and *_1_2.npy are linked, not copied.
Usage: python expand_subset.py SRC_ROOT DST_ROOT     (both ending in /, each containing dataset/)
"""
import os, sys, glob, shutil, time
import numpy as np
from multiprocessing import Pool

SRC, DST = sys.argv[1], sys.argv[2]


def label(src):
    dst = src.replace(SRC, DST, 1)
    if src.endswith("_1_1.npy"):
        a = np.load(src)
        np.save(dst, a.astype(np.float32) if a.dtype == np.uint8 else a)
    else:
        os.symlink(src, dst)


def cloud(src):
    dst = src.replace(SRC, DST, 1)
    p = np.fromfile(src, dtype=np.float32).reshape(-1, 3)
    np.concatenate([p, np.ones((len(p), 1), np.float32)], 1).tofile(dst)


if __name__ == "__main__":
    t0 = time.time()
    for seq_dir in sorted(glob.glob(f"{SRC}dataset/sequences/*")):
        seq = os.path.basename(seq_dir); d = f"{DST}dataset/sequences/{seq}"
        os.makedirs(d, exist_ok=True)
        for f in ["calib.txt", "poses.txt"]:
            shutil.copy(f"{seq_dir}/{f}", d)
        for sub in ["image_2", "voxels"]:
            if os.path.isdir(f"{seq_dir}/{sub}") and not os.path.exists(f"{d}/{sub}"):
                os.symlink(f"{seq_dir}/{sub}", f"{d}/{sub}")
    labels = glob.glob(f"{SRC}dataset/labels/*/*.npy")
    clouds = glob.glob(f"{SRC}dataset/sequences_msnet3d_lidar/sequences/*/*.bin")
    for f in labels + clouds:
        os.makedirs(os.path.dirname(f.replace(SRC, DST, 1)), exist_ok=True)
    with Pool(os.cpu_count()) as p:
        p.map(label, labels, chunksize=8)
        p.map(cloud, clouds, chunksize=8)
    print(f"expanded {len(labels)} label files and {len(clouds)} point clouds into {DST} in {time.time()-t0:.0f}s")
