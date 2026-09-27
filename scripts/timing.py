"""Computation time of one 120-s medication cycle for the 40 patients of the
development cohort under the population dual-threshold policy, on an
otherwise idle machine (the value depends on the machine).
Usage: python scripts/timing.py  (writes results/model/timing.json)"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twin import closedloop as CL  # noqa: E402


def main():
    d = ROOT / "results" / "dev"
    pt = pd.read_csv(d / "patients.csv")
    calib = dict(np.load(d / "calib.npz"))
    b_ref = json.loads((d / "b_ref.json").read_text())["b_ref"]
    prm = json.loads((d / "policy_population_dual.json").read_text())["params"]
    B = len(pt)
    ref20 = CL.healthy_ref20(pt.c.values)
    pol = CL.DualThreshold(**{k: np.full(B, v) for k, v in prm.items()})
    t0 = time.perf_counter()
    CL.run(pt, pol, "cycle", calib, 1, b_ref, lib_ref20=ref20)
    sec = time.perf_counter() - t0
    out = dict(patients=B, simulated_s=CL.SCENARIOS["cycle"]["T"], wall_s=sec, processor=platform.processor(),
               python=platform.python_version())
    (ROOT / "results" / "model" / "timing.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
