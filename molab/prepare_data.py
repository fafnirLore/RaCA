"""Find the dataset on the molab machine, bring it into MRA's file layout at DST and check it.

Usage (repo root): python molab/prepare_data.py [SOURCE] [DST]        DST defaults to /tmp/kitti_full/
SOURCE may be
  (empty or 'auto')                 search the usual places for the dataset
  a folder                          anywhere above .../dataset/sequences (searched recursively, depth-limited)
  a .zip / .tar / .tar.gz file      extracted to /tmp/kitti_src first
  hf://datasets/<user>/<repo>[/sub] downloaded from the Hugging Face Hub to /tmp/kitti_src (HF_TOKEN for private repos)
  kaggle://datasets/<user>/<slug>   a Kaggle dataset, downloaded and unzipped to /tmp/kitti_src
  kaggle://kernels/<user>/<slug>    the output of a Kaggle notebook, downloaded to /tmp/kitti_src
                                    (private: KAGGLE_API_TOKEN in the environment; older keys: KAGGLE_USERNAME + KAGGLE_KEY)
  src1,src2,...                     several kaggle:// or hf:// sources merged into one tree (full-dataset groups
                                    from kaggle/build_full_dataset.ipynb)

Two layouts are recognised:
  compact  the Kaggle "scenes" subset (kitti_subset/selection.json next to dataset/): uint8 labels and x,y,z point
           clouds, rebuilt into MRA's format by kaggle/expand_subset.py (about 3 minutes, written to DST);
  full     MRA's own layout (float32 labels, x,y,z,1 point clouds): linked at DST as it is, nothing is copied.
The last line printed is `DATA_ROOT=<DST>`; the checks fail loudly if a file MRA would read is missing.
"""
import glob, json, os, shlex, subprocess, sys, tarfile, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else "auto"
DST = os.path.join(sys.argv[2] if len(sys.argv) > 2 else "/tmp/kitti_full", "")
STAGE = "/tmp/kitti_src"
TRAIN_SEQS = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
TEMPORAL = [-12, -9, -6, -3]            # MRA-T's `temporal`: earlier frames read for every sample
SKIP = {"proc", "sys", "dev", "run", "boot", "etc", "usr", "bin", "sbin", "lib", "lib64", "var", "snap",
        "node_modules", "site-packages", "__pycache__", ".git", ".cache", "kitti_full", "mra_env", "mmdetection3d"}


def find_dataset_dirs(root, max_depth=6, skip=SKIP):
    """Folders named 'dataset' that contain 'sequences', at most max_depth levels below root."""
    hits, root = [], os.path.abspath(root)
    base = root.rstrip(os.sep).count(os.sep)
    for cur, dirs, _ in os.walk(root, followlinks=True):
        depth = cur.rstrip(os.sep).count(os.sep) - base
        dirs[:] = [] if depth >= max_depth else [d for d in dirs if d not in skip and not d.startswith(".")]
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


def kaggle_cli():
    """The current Kaggle command-line tool in its own small environment (/tmp/kaggle_env), so that its
    dependencies cannot disturb MRA's pinned packages. Current versions understand KAGGLE_API_TOKEN."""
    import shutil
    env = "/tmp/kaggle_env"; cli = env + "/bin/kaggle"
    if os.path.exists(cli):
        return cli
    log = open("/tmp/kaggle_env.log", "w")
    def ok(cmd):
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode == 0
    uv = shutil.which("uv")
    if uv and ok([uv, "venv", "-q", "--python", "3.12", env]) and ok([uv, "pip", "install", "-q", "--python", env + "/bin/python", "kaggle", "six"]) and os.path.exists(cli):
        return cli
    for py in ("python3.13", "python3.12", "python3.11", "python3", sys.executable):
        py = shutil.which(py) or (py if os.path.exists(py) else None)
        if not py:
            continue
        shutil.rmtree(env, ignore_errors=True)
        if ok([py, "-m", "venv", env]) and ok([env + "/bin/python", "-m", "pip", "install", "-q", "kaggle", "six"]) and os.path.exists(cli):
            return cli
    raise RuntimeError("could not install the Kaggle command-line tool, see /tmp/kaggle_env.log")


def from_kaggle(url):
    parts = url[len("kaggle://"):].strip("/").split("/")
    kind = "datasets"
    if parts[0] in ("datasets", "kernels", "notebooks", "code"):
        kind = "datasets" if parts[0] == "datasets" else "kernels"; parts = parts[1:]
    assert len(parts) == 2, f"expected kaggle://datasets/<user>/<slug> or kaggle://kernels/<user>/<slug>, got {url!r}"
    ref = "/".join(parts)
    os.makedirs(STAGE, exist_ok=True)
    marker = os.path.join(STAGE, ".kaggle_" + kind + "_" + ref.replace("/", "_"))
    if os.path.exists(marker):
        return STAGE
    token = os.environ.get("KAGGLE_API_TOKEN") or os.path.exists(os.path.expanduser("~/.kaggle/access_token"))
    legacy = (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) or os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
    assert token or legacy, ("Kaggle credentials not found: put KAGGLE_API_TOKEN=<your token> in the notebook's .env "
                             "(kaggle.com -> Settings -> API -> Generate New Token). Older keys: KAGGLE_USERNAME and KAGGLE_KEY.")
    cli = kaggle_cli()
    cmd = [cli, "datasets", "download", "-d", ref, "-p", STAGE, "--unzip"] if kind == "datasets" else [cli, "kernels", "output", ref, "-p", STAGE]
    print(f"downloading Kaggle {kind[:-1]} {ref} -> {STAGE} (14 GB takes a while)", flush=True)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL if kind == "kernels" else None)
    except subprocess.CalledProcessError:
        if kind != "datasets":
            raise
        # Kaggle sometimes refuses to serve a large dataset as one zip (404) although it is ready: fetch it file by file
        print(f"whole-dataset download refused for {ref}; downloading it file by file instead", flush=True)
        subprocess.run([os.path.join(os.path.dirname(cli), "python"), os.path.join(HERE, "kaggle_files.py"), ref, STAGE],
                       check=True)
    # numbered parts of one tar (kaggle/repack_full_dataset.ipynb: kitti_full_<g>.part000, .part001, ...): join and unpack
    groups = {}
    for p in glob.glob(f"{STAGE}/**/*.part[0-9][0-9][0-9]", recursive=True):
        groups.setdefault(p[:-len(".part000")], []).append(p)
    for prefix, parts in sorted(groups.items()):
        parts.sort()
        print(f"joining {len(parts)} parts of {os.path.basename(prefix)} and unpacking -> {STAGE}", flush=True)
        subprocess.run("set -o pipefail; cat " + " ".join(shlex.quote(p) for p in parts) + f" | tar xf - -C {shlex.quote(STAGE)}",
                       shell=True, check=True, executable="/bin/bash")
        for p in parts:
            os.remove(p)
    for a in sorted(glob.glob(f"{STAGE}/**/*.zip", recursive=True) + glob.glob(f"{STAGE}/**/*.tar*", recursive=True)):
        extract(a); os.remove(a)
    open(marker, "w").close()
    return STAGE


def locate(src):
    if "," in src:                                   # several kaggle:// / hf:// sources, merged in STAGE (full dataset groups)
        for part in [p.strip() for p in src.split(",") if p.strip()]:
            assert part.startswith(("kaggle://", "hf://")), f"comma-separated SOURCE parts must be kaggle:// or hf://, got {part!r}"
            from_kaggle(part) if part.startswith("kaggle://") else from_hub(part)
        roots = [STAGE]
    elif src.startswith("kaggle://"):
        roots = [from_kaggle(src)]
    elif src.startswith("hf://"):
        roots = [from_hub(src)]
    elif os.path.isfile(src):
        roots = [extract(src)]
    elif src not in ("", "auto"):
        assert os.path.isdir(src), f"SOURCE {src!r} is neither a folder, an archive, an hf:// nor a kaggle:// address"
        roots = [src]
    else:
        roots = [os.getcwd(), os.path.expanduser("~"), "/data", "/datasets", "/mnt", "/workspace", "/storage",
                 "/marimo", "/content", "/kaggle/input", "/tmp"]
    seen, hits = set(), []
    for r in roots:
        r = os.path.realpath(r)
        if os.path.isdir(r) and r not in seen:
            seen.add(r)
            # 'kitti_full' is skipped only when searching the machine (it is DST's name); the full-dataset groups from
            # kaggle/build_full_dataset.ipynb are packed as kitti_full/ and must be found inside an explicit SOURCE
            skip = SKIP if src in ("", "auto") else SKIP - {"kitti_full"}
            hits += [h for h in find_dataset_dirs(r, skip=skip) if os.path.realpath(h) != os.path.realpath(DST + "dataset")]
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
    compact = os.path.exists(root + ".compact")      # used as it is: uint8 labels, x,y,z clouds (the loader handles both)
    lab = sorted(glob.glob(f"{root}dataset/labels/*/*_1_1.npy"))
    if lab:
        a = np.load(lab[0])
        if a.dtype not in ((np.float32, np.uint8) if compact else (np.float32,)) or a.size != 256 * 256 * 32:
            problems.append(f"{lab[0]}: dtype {a.dtype}, shape {a.shape}; expected float32 with 256x256x32 values")
    pc = sorted(glob.glob(f"{root}dataset/sequences_msnet3d_lidar/sequences/*/*.bin"))
    if pc and os.path.getsize(pc[0]) % (12 if compact else 16):
        problems.append(f"{pc[0]}: size is not a multiple of {12 if compact else 16} bytes")
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
        full = os.path.exists(root + "selection.json") and \
            json.load(open(root + "selection.json")).get("settings", {}).get("mode") == "full"
        if compact and full:
            # full dataset (kaggle/build_full_dataset.ipynb): expanding it would roughly double ~60 GB on disk, so it is
            # linked as it is and marked '.compact'; the data loader reads uint8 labels and x,y,z clouds directly.
            os.makedirs(DST, exist_ok=True)
            if not os.path.exists(DST + "dataset"):
                os.symlink(root + "dataset", DST + "dataset")
            open(DST + ".compact", "w").close()
            print("full compact dataset: used without expanding (loader reads the compact format)")
        elif compact:
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
