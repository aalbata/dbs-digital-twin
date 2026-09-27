"""Download the patient recordings used for anchoring into data/.

Madrid (Zenodo record 10078352, CC BY 4.0): dataset_1.zip, dataset_2.zip and
README.txt, checked against the MD5 sums of the record; the archives are
unpacked into data/madrid/dataset_1/ and data/madrid/dataset_2/.
Dusseldorf (OpenNeuro ds004998, CC0): for each of the 40 runs listed in
data/ds004998_selection.json (one run per subject and medication state), the
first 900 MiB of the FIF file, which holds the first rest block of the run
(only its leading part for the longest blocks), fetched with an HTTP range request from the public OpenNeuro S3
bucket, plus the run's BIDS sidecar files and the subject's montage, into
data/ds004998/rest/. About 35 GiB in total.
Usage: python scripts/download_data.py"""
import hashlib
import json
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ZENODO = "https://zenodo.org/api/records/10078352/files/{}/content"
MADRID = {"dataset_1.zip": "c86a5b0b977309a2a6dfb2fae831ae7b",
          "dataset_2.zip": "a85fa8cca0e2c8a19863a8a5d1e74fa9",
          "README.txt": "69bb44ee865e7b09c72c3c366bf5c513"}
OPENNEURO = "https://s3.amazonaws.com/openneuro.org/ds004998/"
NBYTES = 900 * 2 ** 20


def fetch(url, out, nbytes=None):
    """Download url to out (the first nbytes only if given); skip if present."""
    if out.exists() and (nbytes is None or out.stat().st_size >= nbytes):
        return out
    req = urllib.request.Request(url)
    if nbytes is not None:
        req.add_header("Range", f"bytes=0-{nbytes - 1}")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(out, "wb") as fh:
                while True:
                    chunk = r.read(8 * 2 ** 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            return out
        except OSError as ex:
            print("retrying", url, ex, flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(2 ** 20), b""):
            h.update(chunk)
    return h.hexdigest()


def madrid():
    dest = DATA / "madrid"
    dest.mkdir(parents=True, exist_ok=True)
    for name, digest in MADRID.items():
        out = fetch(ZENODO.format(name), dest / name)
        if md5(out) != digest:
            raise RuntimeError(f"checksum mismatch: {out}")
        if name.endswith(".zip"):
            with zipfile.ZipFile(out) as z:
                z.extractall(dest / Path(name).stem)
    print("Madrid recordings ready", flush=True)


def dusseldorf():
    sel = json.loads((DATA / "ds004998_selection.json").read_text())["files"]
    dest = DATA / "ds004998" / "rest"
    dest.mkdir(parents=True, exist_ok=True)
    small = sorted({s for r in sel for s in r["sidecars"] + r["montage"]})
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda p: fetch(OPENNEURO + p, dest / Path(p).name), small))
    jobs = [(r["file"], min(NBYTES, r["bytes"])) for r in sel]
    with ThreadPoolExecutor(4) as ex:
        for out in ex.map(lambda a: fetch(OPENNEURO + a[0], dest / Path(a[0]).name, a[1]), jobs):
            print(out.name, f"{out.stat().st_size / 2 ** 20:.0f} MiB", flush=True)
    print("Dusseldorf recordings ready", flush=True)


if __name__ == "__main__":
    madrid()
    dusseldorf()
