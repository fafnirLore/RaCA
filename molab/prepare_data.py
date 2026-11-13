"""Find the dataset on the molab machine, bring it into MRA's file layout at DST and check it.

Usage (repo root): python molab/prepare_data.py [SOURCE] [DST]        DST defaults to /tmp/kitti_full/
SOURCE may be
  (empty or 'auto')                 search the usual places for the dataset
  a folder                          anywhere above .../dataset/sequences (searched recursively, depth-limited)
  a .zip / .tar / .tar.gz file      extracted to /tmp/kitti_src first
  hf://datasets/<user>/<repo>[/sub] downloaded from the Hugging Face Hub to /tmp/kitti_src (HF_TOKEN for private repos)

Two layouts are recognised:
  compact  the Kaggle "scenes" subset (kitti_subset/selection.json next to dataset/): uint8 labels and x,y,z point
           clouds, rebuilt into MRA's format by kaggle/expand_subset.py (about 3 minutes, written to DST);
  full     MRA's own layout (float32 labels, x,y,z,1 point clouds): linked at DST as it is, nothing is copied.
The last line printed is `DATA_ROOT=<DST>`; the checks fail loudly if a file MRA would read is missing.
"""
import glob, json, os, subprocess, sys, tarfile, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else "auto"
DST = os.path.join(sys.argv[2] if len(sys.argv) > 2 else "/tmp/kitti_full", "")
STAGE = "/tmp/kitti_src"
TRAIN_SEQS = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
TEMPORAL = [-12, -9, -6, -3]            # MRA-T's `temporal`: earlier frames read for every sample
SKIP = {"proc", "sys", "dev", "run", "boot", "etc", "usr", "bin", "sbin", "lib", "lib64", "var", "snap",
        "node_modules", "site-packages", "__pycache__", ".git", ".cache", "kitti_full", "mra_env", "mmdetection3d"}


def find_dataset_dirs(root, max_depth=6):
    """Folders named 'dataset' that contain 'sequences', at most max_depth levels below root."""
    hits, root = [], os.path.abspath(root)
    base = root.rstrip(os.sep).count(os.sep)
    for cur, dirs, _ in os.walk(root, followlinks=True):
        depth = cur.rstrip(os.sep).count(os.sep) - base
        dirs[:] = [] if depth >= max_depth else [d for d in dirs if d not in SKIP and not d.startswith(".")]
        if os.path.basename(cur) == "dataset" and os.path.isdir(os.path.join(cur, "sequences")):
            hits.append(cur); dirs[:] = []
    return hits


def extract(archive):
    os.makedirs(STAGE, exist_ok=True)
    marker = os.path.join(STAGE, ".extracted_" + os.path.basename(archive))
    if not os.path.exists(marker):
        print(f"extracting {archive} -> {STAGE}", flush=True)
        if zipfile.is_zipfile(archive):
            zipfile.ZipFile(archive).extractall(STAGE)
        else:
            tarfile.open(archive).extractall(STAGE)
        open(marker, "w").close()
    return STAGE


def from_hub(url):
    from huggingface_hub import snapshot_download
    parts = url[len("hf://"):].strip("/").split("/")
    kind = "dataset"
    if parts[0] in ("datasets", "models", "spaces"):
        kind = {"datasets": "dataset", "models": "model", "spaces": "space"}[parts[0]]; parts = parts[1:]
    repo, sub = "/".join(parts[:2]), "/".join(parts[2:])
    print(f"downloading {kind} {repo} {('/' + sub) if sub else ''} -> {STAGE}", flush=True)
    snapshot_download(repo, repo_type=kind, local_dir=STAGE, allow_patterns=[sub + "/**"] if sub else None, max_workers=16)
    for a in sorted(glob.glob(f"{STAGE}/**/*.zip", recursive=True) + glob.glob(f"{STAGE}/**/*.tar*", recursive=True)):
        extract(a)
    return STAGE


def locate(src):
    if src.startswith("hf://"):
        roots = [from_hub(src)]
    elif os.path.isfile(src):
        roots = [extract(src)]
    elif src not in ("", "auto"):
        assert os.path.isdir(src), f"SOURCE {src!r} is neither a folder, an archive nor an hf:// address"
        roots = [src]
    else:
        roots = [os.getcwd(), os.path.expanduser("~"), "/data", "/datasets", "/mnt", "/workspace", "/storage",
                 "/marimo", "/content", "/kaggle/input", "/tmp"]
    seen, hits = set(), []
    for r in roots:
        r = os.path.realpath(r)
        if os.path.isdir(r) and r not in seen:
            seen.add(r)
            hits += [h for h in find_dataset_dirs(r) if os.path.realpath(h) != os.path.realpath(DST + "dataset")]
    hits = sorted(set(os.path.realpath(h) for h in hits), key=lambda h: (not os.path.isdir(h + "/labels"), len(h)))
    assert hits, ("no dataset found: expected a folder .../dataset/sequences/XX/{image_2,voxels,calib.txt,poses.txt} "
                  f"below one of {roots}. Pass the folder, archive or hf:// address as SOURCE.")
    if len(hits) > 1:
        print("several candidates, using the first:", *hits, sep="\n  ")
    return os.path.dirname(hits[0]) + "/"            # the folder that contains dataset/


def is_compact(root):
    """Compact = point clouds stored as x,y,z (12 bytes per point) and/or labels stored as uint8."""
    labels = sorted(glob.glob(f"{root}dataset/labels/*/*_1_1.npy"))
    assert labels, f"no labels found in {root}dataset/labels/XX/*_1_1.npy"
    lab_u8 = np.load(labels[0], mmap_mode="r").dtype == np.uint8
    if os.path.exists(root + "selection.json"):
        return True
    clouds = sorted(glob.glob(f"{root}dataset/sequences_msnet3d_lidar/sequences/*/*.bin"))[:50]
    assert clouds, f"no point clouds found in {root}dataset/sequences_msnet3d_lidar/sequences/XX/*.bin"
    xyz1 = all(os.path.getsize(c) % 16 == 0 for c in clouds)
    if xyz1:                                      # sizes fit x,y,z,1: confirm with the 4th column of one file
        p = np.fromfile(clouds[0], dtype=np.float32)
        xyz1 = p.size % 4 == 0 and p.size > 0 and bool(np.all(p.reshape(-1, 4)[:, 3] == 1))
    return lab_u8 or not xyz1


def check(root):
    """Every file MRA reads for the frames it will use must exist and have the expected format."""
    problems, counts = [], {}
    for split, seqs in (("train", TRAIN_SEQS), ("val", ["08"])):
        n = 0
        for s in seqs:
            d = f"{root}dataset/sequences/{s}"
            for f in ("calib.txt", "poses.txt"):
                if not os.path.exists(f"{d}/{f}"):
                    problems.append(f"missing {d}/{f} (MRA reads it for every sequence of the split, even without frames)")
            n_poses = sum(1 for _ in open(f"{d}/poses.txt")) if os.path.exists(f"{d}/poses.txt") else 0
            frames = sorted(os.path.basename(v)[:-4] for v in glob.glob(f"{d}/voxels/*.bin"))
            n += len(frames)
            for fid in frames:
                need = {fid} | {str(int(fid) + t).zfill(6) for t in TEMPORAL if 0 <= int(fid) + t <= n_poses - 1}
                for x in sorted(need):
                    for path in (f"{d}/image_2/{x}.png", f"{root}dataset/sequences_msnet3d_lidar/sequences/{s}/{x}.bin"):
                        if not os.path.exists(path):
                            problems.append(f"missing {path} (needed by frame {s}/{fid})")
                for tag in ("_1_1.npy", "_1_2.npy"):
                    if not os.path.exists(f"{root}dataset/labels/{s}/{fid}{tag}"):
                        problems.append(f"missing {root}dataset/labels/{s}/{fid}{tag}")
                if len(problems) > 20:
                    break
        counts[split] = n
    lab = sorted(glob.glob(f"{root}dataset/labels/*/*_1_1.npy"))
    if lab:
        a = np.load(lab[0])
        if a.dtype != np.float32 or a.size != 256 * 256 * 32:
            problems.append(f"{lab[0]}: dtype {a.dtype}, shape {a.shape}; expected float32 with 256x256x32 values")
    pc = sorted(glob.glob(f"{root}dataset/sequences_msnet3d_lidar/sequences/*/*.bin"))
    if pc and os.path.getsize(pc[0]) % 16:
        problems.append(f"{pc[0]}: size is not a multiple of 16 bytes (x,y,z,1 float32)")
    if not counts.get("train") or not counts.get("val"):
        problems.append(f"no frames with voxels/*.bin: train {counts.get('train')}, val {counts.get('val')}")
    return counts, problems


if __name__ == "__main__":
    if os.path.isdir(DST + "dataset/labels") and os.path.exists(DST + ".prepared"):
        print("already prepared:", DST)
    else:
        root = locate(SRC)
        compact = is_compact(root)
        print(f"dataset: {root} | layout: {'compact (Kaggle subset)' if compact else 'full (MRA format)'}", flush=True)
        if os.path.exists(root + "selection.json"):
            s = json.load(open(root + "selection.json")).get("summary", {})
            if s:
                print(f"selection.json: train frames {s.get('train', {}).get('frames')} | val frames {s.get('val', {}).get('frames')}")
        if compact:
            subprocess.run([sys.executable, os.path.join(HERE, "..", "kaggle", "expand_subset.py"), root, DST], check=True)
        else:
            os.makedirs(DST, exist_ok=True)
            if not os.path.exists(DST + "dataset"):
                os.symlink(root + "dataset", DST + "dataset")
        open(DST + ".prepared", "w").write(root)
    counts, problems = check(DST)
    print(f"frames with ground truth: train {counts['train']} | val {counts['val']}")
    if problems:
        print(f"PROBLEMS ({len(problems)} shown):", *problems[:25], sep="\n  ")
        sys.exit(1)
    print("checks passed: images, point clouds (incl. the earlier frames MRA reads), labels, calib and poses are in place")
    print("DATA_ROOT=" + DST)
