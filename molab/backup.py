"""Copy a run folder to storage that outlives the molab session, and bring it back.

molab clears the machine when a notebook shuts down (12 h limit, or 90 minutes idle) and does not keep files
above 1 GB, so checkpoints have to be copied somewhere else to survive.
Usage:  python molab/backup.py push <run_dir> <target>      newest checkpoint + logs + results  ->  target
        python molab/backup.py pull <run_dir> <target>      target  ->  run_dir (only when run_dir has no checkpoint)
<target> is a folder (e.g. a mounted bucket) or hf://datasets/<user>/<repo>[/<subfolder>] (needs HF_TOKEN with
write access; the repository is created as a private dataset when missing).
Only the newest checkpoint (latest.pth) and the best one by validation mIoU (best.pth) are kept in the backup.
"""
import glob, os, shutil, sys


def files_to_push(run):
    out = {}
    ck = os.path.join(run, "latest.pth")
    if os.path.exists(ck):
        out["latest.pth"] = os.path.realpath(ck)
    best = sorted(glob.glob(os.path.join(run, "best_*.pth")), key=os.path.getmtime)
    if best:                                    # kept by the evaluation hook (validation mIoU); stored under one fixed name
        out["best.pth"] = best[-1]
    for pat in ("*.log", "*.log.json", "*.json", "*.py"):
        for f in glob.glob(os.path.join(run, pat)):
            out[os.path.basename(f)] = f
    return out


def hub(target):
    parts = target[len("hf://"):].strip("/").split("/")
    if parts[0] == "datasets":
        parts = parts[1:]
    return "/".join(parts[:2]), "/".join(parts[2:])


def push(run, target):
    files = files_to_push(run)
    if not files:
        print("nothing to back up yet"); return
    if target.startswith("hf://"):
        from huggingface_hub import HfApi
        repo, sub = hub(target); api = HfApi()
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        for name, path in files.items():
            api.upload_file(path_or_fileobj=path, path_in_repo=f"{sub}/{name}" if sub else name, repo_id=repo,
                            repo_type="dataset", commit_message=f"backup {name}")
    else:
        os.makedirs(target, exist_ok=True)
        for name, path in files.items():
            tmp = os.path.join(target, name + ".tmp")
            shutil.copyfile(path, tmp); os.replace(tmp, os.path.join(target, name))
    print(f"backed up {len(files)} files to {target}" + (" (with latest.pth)" if "latest.pth" in files else ""))


def pull(run, target):
    if glob.glob(os.path.join(run, "*.pth")):
        print("run folder already has a checkpoint - nothing restored"); return
    os.makedirs(run, exist_ok=True)
    if target.startswith("hf://"):
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError
        repo, sub = hub(target)
        try:
            snapshot_download(repo, repo_type="dataset", local_dir=run + "/.hub", allow_patterns=[sub + "/*"] if sub else None)
        except RepositoryNotFoundError:
            print("no backup yet at", target); return
        src = os.path.join(run, ".hub", sub)
    else:
        src = target
    names = [n for n in (os.listdir(src) if os.path.isdir(src) else []) if os.path.isfile(os.path.join(src, n)) and not n.endswith(".tmp")]
    for n in names:
        shutil.copyfile(os.path.join(src, n), os.path.join(run, n))
    print(f"restored {len(names)} files from {target}" if names else f"no backup yet at {target}")


if __name__ == "__main__":
    mode, run, target = sys.argv[1:4]
    {"push": push, "pull": pull}[mode](run.rstrip("/"), target.rstrip("/"))
