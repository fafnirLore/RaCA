"""Download a Kaggle dataset file by file (fallback when Kaggle will not serve the whole dataset as one zip: 404).

Usage (run with the Kaggle tool's own Python, /tmp/kaggle_env/bin/python): python kaggle_files.py <owner>/<slug> <dst>
Every file is written to <dst>/<its path in the dataset>. Files that already exist are skipped, so a re-run resumes.
Credentials: KAGGLE_API_TOKEN (or KAGGLE_USERNAME + KAGGLE_KEY) in the environment, as for the command-line tool.
"""
import os, sys, time, zipfile, threading
from concurrent.futures import ThreadPoolExecutor
from kaggle.api.kaggle_api_extended import KaggleApi

REF, DST = sys.argv[1], sys.argv[2]
WORKERS = int(os.environ.get("KAGGLE_WORKERS", 4))
_local = threading.local()


def api():
    if not hasattr(_local, "api"):
        a = KaggleApi(); a.authenticate(); _local.api = a
    return _local.api


def attr(obj, *names):
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def list_files():
    files, token = [], None
    while True:
        for attempt in range(8):                     # 429 Too Many Requests: wait and try again
            try:
                r = api().dataset_list_files(REF, page_token=token, page_size=200)
                break
            except Exception as e:
                if attempt == 7:
                    raise
                print(f"listing files: {e!r}; retrying in {30 * (attempt + 1)} s", flush=True)
                time.sleep(30 * (attempt + 1))
        page = attr(r, "files", "dataset_files", "datasetFiles") or []
        files += [attr(f, "name", "ref") for f in page]
        token = attr(r, "next_page_token", "nextPageToken")
        if not token or not page:
            return [f for f in files if f]


def fetch(name):
    dst = os.path.join(DST, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return 0
    d = os.path.dirname(dst); os.makedirs(d, exist_ok=True)
    for attempt in range(5):
        try:
            api().dataset_download_file(REF, name, path=d, quiet=True)
            base = os.path.join(d, os.path.basename(name))
            if not os.path.exists(base) and os.path.exists(base + ".zip"):      # large files may arrive zipped
                with zipfile.ZipFile(base + ".zip") as z:
                    z.extractall(d)
                os.remove(base + ".zip")
            if base != dst and os.path.exists(base):
                os.replace(base, dst)
            if os.path.exists(dst):
                return 1
            raise RuntimeError("file not found after download")
        except Exception as e:
            if attempt == 4:
                print(f"FAILED {name}: {e!r}", flush=True)
                return -1
            if hasattr(_local, "api"):
                del _local.api
            time.sleep(20 * (attempt + 1))


if __name__ == "__main__":
    t0 = time.time()
    names = list_files()
    print(f"{REF}: {len(names)} files, downloading one by one with {WORKERS} workers", flush=True)
    done = failed = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, r in enumerate(ex.map(fetch, names), 1):
            failed += r < 0
            if i % 1000 == 0 or i == len(names):
                print(f"  {i}/{len(names)} files ({time.time() - t0:.0f}s)", flush=True)
    if failed:
        sys.exit(f"{failed} files could not be downloaded; run again to retry the missing ones")
    print(f"{REF}: complete in {time.time() - t0:.0f}s", flush=True)
