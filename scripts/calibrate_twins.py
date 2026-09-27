"""Build a cohort of virtual patients, run the calibration protocol and
identify a digital twin for every patient.
Usage: python scripts/calibrate_twins.py <dev|test|synth|stress> [twin iterations] [rest seconds]
Writes results/<cohort>/patients.csv, calibration.csv, twin_initial.csv,
twin.csv and calib.npz; for dev also results/dev/b_ref.json (median true
OFF beta envelope, the fixed reference of the beta burden)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twin import study, closedloop as CL  # noqa: E402
from twin.patients import build, SEEDS  # noqa: E402
from twin.anchor import feature_table  # noqa: E402

TWIN_ITERS = 16
"""Bayesian-optimization evaluations of the twin refinement step."""


def main():
    cohort = sys.argv[1]
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else TWIN_ITERS
    rest = float(sys.argv[3]) if len(sys.argv) > 3 else study.REST_T
    tag = cohort if rest == study.REST_T else f"{cohort}_rest{int(rest)}"
    study.REST_T = rest
    out = ROOT / "results" / tag
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    chi = json.loads((ROOT / "results" / "patients" / "anchor_background.json").read_text())["chi"]
    pt = build(ROOT / "results" / "patients" / "anchor.csv", cohort)
    pt.to_csv(out / "patients.csv", index=False)
    ref20 = CL.healthy_ref20(pt.c.values)
    seed = SEEDS[cohort] % 100000
    cal, calib = study.calibrate(pt, chi, ref20, seed)
    cal.to_csv(out / "calibration.csv", index=False)
    np.savez_compressed(out / "calib.npz", **calib)
    log(f"calibration done ({len(pt)} patients)")
    if cohort == "dev" and rest == study.REST_T:
        (out / "b_ref.json").write_text(json.dumps(dict(b_ref=float(np.median(cal.b_true_off))), indent=1))
    lib = dict(np.load(ROOT / "results" / "patients" / "model_library.npz"))
    init = study.twin_initial(cal, lib, feature_table(lib, chi))
    init.insert(0, "pid", pt.pid.values)
    init.to_csv(out / "twin_initial.csv", index=False)
    twin = study.twin_refine(cal, init, chi, None, n_iter, seed + 7, log=log)
    twin.to_csv(out / "twin.csv", index=False)
    log("twin identification done")


if __name__ == "__main__":
    main()
