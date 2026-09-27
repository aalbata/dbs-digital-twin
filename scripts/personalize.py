"""Policy parameters for one cohort, policy family and method.

Methods
  population  one parameter set minimizing the mean cost over the development
              cohort (Bayesian optimization of the cohort mean)
  twin        per-patient Bayesian optimization on the patient's digital twin
  direct      per-patient Bayesian optimization on the patient itself, one
              medication-cycle trial per evaluation; recommendations are kept
              after each trial budget in DIRECT, and the recommendation after
              DIRECT_MAX trials is the reference upper bound
Usage: python scripts/personalize.py <cohort> <method> <family> [lambda]
Writes results/<cohort>/policy_<method>_<family>[_lam].json."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twin import study, closedloop as CL  # noqa: E402
from twin.bo import BatchBO  # noqa: E402
from twin.patients import SEEDS  # noqa: E402

ITERS = {"continuous": 12, "single": 30, "dual": 40}
DIRECT = {"continuous": (5, 10, 20), "single": (10, 20, 40), "dual": (10, 20, 40)}
DIRECT_MAX = {"continuous": 20, "single": 40, "dual": 60}


def load(cohort):
    d = ROOT / "results" / cohort
    pt = pd.read_csv(d / "patients.csv")
    cal = pd.read_csv(d / "calibration.csv")
    calib = dict(np.load(d / "calib.npz"))
    return pt, cal, calib


def main():
    cohort, method, family = sys.argv[1], sys.argv[2], sys.argv[3]
    lam = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    b_ref = json.loads((ROOT / "results" / "dev" / "b_ref.json").read_text())["b_ref"]
    pt, cal, calib = load(cohort)
    seed = SEEDS[cohort.split("_")[0]] % 100000 + {"population": 11, "twin": 23, "direct": 37}[method]
    box = study.POLICY_BOX[family]
    out = dict(cohort=cohort, method=method, family=family, lam=lam)
    if method == "population":
        bo = BatchBO(box, 1, study.N_INIT[box.d], seed)
        hist = []
        for it in range(ITERS[family]):
            Z = bo.ask()
            prm = {k: np.repeat(v, len(pt)) for k, v in box.to_params(Z).items()}
            res = CL.run(pt, CL.POLICIES[family](**prm), "cycle", calib, seed + 1000 + it, b_ref)
            J = float(np.mean(CL.cost(res, lam)))
            bo.tell(Z, np.array([J]))
            hist.append(J)
            log(f"population {family} {it + 1}/{ITERS[family]}: mean J {J:.4f}")
        out["params"] = {k: float(v[0]) for k, v in box.to_params(bo.recommend()).items()}
        out["history"] = hist
    elif method == "twin":
        tw = pd.read_csv(ROOT / "results" / cohort / "twin.csv")
        tcal = dict(calib, phi0=CL.initial_state(tw, tw.s_off.values))
        recs, hist = study.optimize_policy(tw, family, tcal, None, None, b_ref, ITERS[family], seed, lam=lam, log=log)
        out["params"] = {k: v.tolist() for k, v in recs[ITERS[family]].items()}
        out["history_median"] = np.median(hist, axis=1).tolist()
    else:
        recs, hist = study.optimize_policy(pt, family, calib, None, None, b_ref, DIRECT_MAX[family], seed, lam=lam,
                                           record_at=DIRECT[family], log=log)
        out["params_by_budget"] = {str(n): {k: v.tolist() for k, v in r.items()} for n, r in recs.items()}
        out["history"] = hist.tolist()
    suffix = "" if lam == 0.5 else f"_lam{lam:g}"
    (ROOT / "results" / cohort / f"policy_{method}_{family}{suffix}.json").write_text(json.dumps(out))
    log("done")


if __name__ == "__main__":
    main()
