"""Library of linear STN potential spectra over the anchoring grid
(corticothalamic delay scale c x severity s), stable fixed points only.
Usage: python scripts/model_library.py [workers]
Writes results/patients/model_library.npz."""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from twin.anchor import library_point, FREQS  # noqa: E402

C_GRID = np.round(np.arange(0.5, 1.801, 0.05), 3)
S_GRID = np.round(np.arange(-1.0, 1.101, 0.025), 3)


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    pts = [(c, s) for c in C_GRID for s in S_GRID]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(library_point, pts, chunksize=16))
    keep = [r for r in res if r[4] is not None]
    out = ROOT / "results" / "patients"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "model_library.npz", c=np.array([r[0] for r in keep]), s=np.array([r[1] for r in keep]),
                        growth=np.array([r[2] for r in keep]), freq=np.array([r[3] for r in keep]),
                        m=np.stack([r[4] for r in keep]), f=FREQS,
                        unstable_c=np.array([r[0] for r in res if r[4] is None]),
                        unstable_s=np.array([r[1] for r in res if r[4] is None]))
    print(f"{len(keep)} stable of {len(res)} grid points")


if __name__ == "__main__":
    main()
