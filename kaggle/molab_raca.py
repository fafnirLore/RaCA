# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "kagglehub", "gdrive-fsspec"]
# ///
# molab notebook (marimo) for RaCA / MRA on one 96 GB GPU. Open with:
#   https://molab.marimo.io/github/fafnirLore/RaCA/blob/farquleet_add_module/kaggle/molab_raca.py
# Mirrors kaggle/kaggle_raca.ipynb; everything it does is in the repository's kaggle/ folder (see kaggle/README.md).
import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import glob, json, os, subprocess
    return glob, json, mo, os, subprocess


@app.cell
def _(mo):
    mo.md(
        """
        # RaCA on molab (one 96 GB GPU): train and evaluate one configuration

        **Before running**

        1. Attach the GPU from the notebook specs button in the header.
        2. Data: the **scenes** subset is downloaded from Kaggle (`KAGGLE_DATASET` below) with `kagglehub`. A public
           dataset needs nothing else. For a private one, set the environment variables `KAGGLE_USERNAME` and
           `KAGGLE_KEY` (kaggle.com → Settings → API → Create New Token) as notebook secrets before pressing Start;
           do not write the key into this file. To use a copy that is already on disk, set `SCENES` to its
           `kitti_subset/` folder.
        3. Set `RUN` (`molab_raca_full` or `molab_mra_before`; any `kaggle/configs/<RUN>.py` works) and press **Start**.

        Differences from the Kaggle notebook: starting weights are MRA's own (no SGN-T warm start, which was only needed
        to fit a 16 GB T4), `--deterministic` is on, and the full method with C1 can be trained. Batch size stays 1.

        **Google Drive (optional).** Connect Drive first: Files sidebar → *Add remote storage* → Google Drive, and
        complete the sign-in. Then set `DRIVE_DIR` below to a folder name in your Drive. After every epoch the logs,
        `early_stopping.json`, the newest `epoch_N.pth` and the best checkpoint are copied to `<DRIVE_DIR>/<RUN>/`;
        checkpoints that training has deleted locally are deleted there too, so Drive holds at most two. The final
        `results_trained.json` is copied after evaluation. A write test runs before training starts.

        Step 1b checks that the pinned PyTorch 1.9.1 / CUDA 11.1 stack can compute on the attached GPU and stops the
        notebook if it cannot. molab sessions end after 12 hours and only files saved through the file browser persist,
        so copy `runs/<RUN>/` out before the session ends.
        """
    )
    return


@app.cell
def _(mo, os):
    # ---- Settings ----
    RUN = "molab_raca_full"          # which kaggle/configs/<RUN>.py to train
    REPO_URL = "https://github.com/fafnirLore/RaCA.git"
    BRANCH = "farquleet_add_module"
    EXTRA = ""                       # extra train.py args, e.g. "--cfg-options runner.max_epochs=3"
    SCENES = ""                      # .../kitti_subset/ already on disk (contains selection.json); "" = download from Kaggle
    KAGGLE_DATASET = "daniyalimran/building-subset1"   # Kaggle dataset holding kitti_subset/ (400 train / 150 val frames)
    DRIVE_DIR = ""                   # Google Drive folder for results/logs/checkpoints after every epoch, e.g. "RaCA_runs"; "" = off
    DETERMINISTIC = "1"              # "1" adds --deterministic as tools/dist_train.sh; "0" turns it off
    WORK = os.path.expanduser("~/raca_work")
    REPO = f"{WORK}/RaCA"
    OUT_ROOT = f"{WORK}/runs"
    start = mo.ui.run_button(label="Start")
    start
    return BRANCH, DETERMINISTIC, DRIVE_DIR, EXTRA, KAGGLE_DATASET, OUT_ROOT, REPO, REPO_URL, RUN, SCENES, WORK, start


@app.cell
def _(subprocess):
    def sh(cmd, env=None, check=True, on_line=None):
        """Run a bash command, print its output live, raise when it fails. on_line(line) is called for every output line."""
        p = subprocess.Popen(["bash", "-c", "set -o pipefail; " + cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in p.stdout:
            print(line, end="", flush=True)
            if on_line:
                on_line(line)
        if p.wait() != 0 and check:
            raise RuntimeError(f"command failed with exit status {p.returncode}: {cmd[:120]}")
        return p.returncode
    return (sh,)


@app.cell
def _(DRIVE_DIR, OUT_ROOT, RUN, glob, os):
    # Google Drive copy of the run folder (off when DRIVE_DIR is ""). Drive is reached through fsspec from this notebook's
    # own Python process, because that is where molab keeps the Drive sign-in; the training subprocess cannot see it.
    _sent = {}                                               # local path -> (size, mtime) already uploaded

    def drive_fs():
        from gdrive_fsspec import GoogleDriveFileSystem
        return GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True)

    def drive_sync(checkpoints=True):
        """Copy new or changed files of <OUT_ROOT>/<RUN>/ to <DRIVE_DIR>/<RUN>/. Never raises: a Drive error must not stop training."""
        if not DRIVE_DIR:
            return
        try:
            fs, _remote = drive_fs(), f"{DRIVE_DIR}/{RUN}"
            fs.makedirs(_remote, exist_ok=True)
            _local = [f for f in glob.glob(f"{OUT_ROOT}/{RUN}/*") if os.path.isfile(f) and not os.path.islink(f)]
            for f in sorted(_local, key=os.path.getsize):     # small files (logs, json) first
                if f.endswith(".pth") and not checkpoints:
                    continue
                st = os.stat(f); key = (st.st_size, st.st_mtime)
                if _sent.get(f) != key:
                    fs.put(f, f"{_remote}/{os.path.basename(f)}"); _sent[f] = key
                    print(f"[drive] {os.path.basename(f)} ({st.st_size / 2**20:.1f} MB) -> {_remote}/", flush=True)
            _keep = {os.path.basename(f) for f in _local}
            for r in fs.ls(_remote, detail=False):            # checkpoints training has replaced: remove the Drive copy too
                if r.endswith(".pth") and os.path.basename(r) not in _keep:
                    fs.rm(r); print(f"[drive] removed old {os.path.basename(r)}", flush=True)
        except Exception as e:
            print(f"[drive] SYNC FAILED ({type(e).__name__}: {e}) - training continues; files remain in {OUT_ROOT}/{RUN}/", flush=True)

    def drive_check():
        """Write, read back and delete a small file, so a Drive problem shows before training and not after an epoch."""
        if not DRIVE_DIR:
            print("Google Drive copy is off (DRIVE_DIR is empty)"); return
        fs, _remote = drive_fs(), f"{DRIVE_DIR}/{RUN}"
        fs.makedirs(_remote, exist_ok=True)
        with fs.open(f"{_remote}/.write_test", "wb") as f:
            f.write(b"ok")
        with fs.open(f"{_remote}/.write_test", "rb") as f:
            assert f.read() == b"ok", "Drive write test: content read back differs"
        fs.rm(f"{_remote}/.write_test")
        print(f"Google Drive ok: writing to {_remote}/")
    return drive_check, drive_sync


@app.cell
def _(BRANCH, REPO, REPO_URL, WORK, mo, os, sh, start):
    # 1. Code + software (~15 min the first time; skipped when already installed in this session)
    mo.stop(not start.value, "Press Start")
    os.makedirs(WORK, exist_ok=True)
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    if os.path.isdir(f"{REPO}/.git"):
        sh(f"cd {REPO} && git fetch -q origin {BRANCH} && git checkout -q {BRANCH} && git reset -q --hard origin/{BRANCH}")
    else:
        sh(f"git clone -q -b {BRANCH} {REPO_URL} {REPO}")
    sh(f"cd {REPO} && git log --oneline -1 && bash kaggle/install.sh && bash kaggle/get_weights.sh")
    # 1b. The pinned PyTorch build must be able to compute on this GPU; stops here with an explanation if it cannot.
    sh(f"cd {REPO} && /tmp/mra_env/bin/python kaggle/check_gpu.py")
    installed = True
    return (installed,)


@app.cell
def _(KAGGLE_DATASET, REPO, SCENES, glob, installed, json, os, sh):
    # 2. Data: download the saved subset from Kaggle (unless SCENES is set) and rebuild MRA's file layout at /tmp/kitti_full/
    assert installed
    if SCENES:
        _hits = [os.path.join(SCENES, "selection.json")]
    else:
        import kagglehub                                   # cached under ~/.cache/kagglehub; a second call does not download again
        _root = kagglehub.dataset_download(KAGGLE_DATASET)
        print("Kaggle dataset", KAGGLE_DATASET, "at", _root)
        sh(f"du -sh {_root}; df -h {_root} /tmp | cat")
        _hits = glob.glob(f"{_root}/**/kitti_subset/selection.json", recursive=True)
    assert _hits and os.path.isfile(_hits[0]), "kitti_subset/selection.json not found: check KAGGLE_DATASET or set SCENES (see the top of this notebook)"
    SRC = os.path.dirname(_hits[0]) + "/"
    _s = json.load(open(_hits[0]))["summary"]
    print("subset:", SRC, f"| train frames {_s['train']['frames']} | val frames {_s['val']['frames']}")
    if not os.path.exists("/tmp/kitti_full/dataset/labels"):
        sh(f"/tmp/mra_env/bin/python {REPO}/kaggle/expand_subset.py {SRC} /tmp/kitti_full/")
    data_ready = True
    return (data_ready,)


@app.cell
def _(DETERMINISTIC, REPO, RUN, data_ready, sh):
    # 3. Smoke test: 2 real training steps + the validation path (~2 min). Stops here if the configuration does not run.
    assert data_ready
    _det = "--deterministic" if DETERMINISTIC == "1" else ""
    sh(f"cd {REPO} && PYTHONPATH={REPO} CUDA_VISIBLE_DEVICES=0 timeout 20m /tmp/mra_env/bin/python kaggle/smoke_test.py "
       f"kaggle/configs/{RUN}.py --steps 2 {_det} 2>&1 | grep -v -E "
       "'mmcv - INFO|is the same before|torch.Size|PretrainedInit|XavierInit|UserWarning|floor_divide|return torch|^ *$'")
    smoke_ok = True
    return (smoke_ok,)


@app.cell
def _(DETERMINISTIC, EXTRA, OUT_ROOT, REPO, RUN, drive_check, drive_sync, os, sh, smoke_ok):
    # 4. Train. Lines like `Epoch [1][10/400] ... loss: ...` show progress; `EarlyStoppingHook: ...` lines show the
    #    monitored value and the patience counter. Output: <OUT_ROOT>/<RUN>/
    assert smoke_ok
    drive_check()
    _env = dict(os.environ, OUT_ROOT=OUT_ROOT, DETERMINISTIC=DETERMINISTIC, GPUS="1")
    _pending = [False]

    def _on_line(line):
        # "Saving checkpoint" is logged before the file is complete, so copy at the next epoch-level line: the validation
        # summary of that epoch, or the first progress line of the next epoch when validation is off.
        if "Saving checkpoint" in line:
            _pending[0] = True
        elif _pending[0] and ("Epoch(val)" in line or "Epoch [" in line):
            _pending[0] = False
            drive_sync()

    try:
        sh(f"cd {REPO} && bash kaggle/train.sh kaggle/configs/{RUN}.py {RUN} {EXTRA}", env=_env, on_line=_on_line)
    finally:
        drive_sync()                                         # last epoch, and whatever exists if training failed
    trained = True
    return (trained,)


@app.cell
def _(OUT_ROOT, REPO, RUN, drive_sync, glob, os, sh, trained):
    # 5. Evaluate on the validation frames: the best checkpoint kept by the evaluation hook, else the last one
    assert trained
    _best = sorted(glob.glob(f"{OUT_ROOT}/{RUN}/best_*.pth"), key=os.path.getmtime)
    _ckpt = _best[-1] if _best else f"{OUT_ROOT}/{RUN}/latest.pth"
    print("evaluating", _ckpt)
    sh(f"cd {REPO} && bash kaggle/eval.sh kaggle/configs/{RUN}.py {_ckpt} {OUT_ROOT}/{RUN}/results_trained.json '{RUN}: trained'")
    drive_sync(checkpoints=False)                            # results_trained.json
    sh(f"du -sh {OUT_ROOT}/{RUN}; ls {OUT_ROOT}/{RUN}")
    return


if __name__ == "__main__":
    app.run()
