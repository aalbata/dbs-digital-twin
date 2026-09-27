"""Accuracy of digital-twin identification: Spearman correlation and median
absolute error between the virtual patients' parameters and their twins'
estimates (delay scale c, OFF and ON severities, log stimulation efficacy g),
after step 1 (rest features only) and after step 2 (Bayesian refinement).
Usage: python scripts/twin_accuracy.py
Writes results/analysis/twin_accuracy.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
COHORTS = ("dev", "test", "synth", "stress", "test_rest30", "test_rest120")


def main():
    rows = []
    for co in COHORTS:
        d = ROOT / "results" / co
        if not (d / "twin.csv").exists():
            continue
        pt = pd.read_csv(d / "patients.csv")
        for step, fn in (("rest", "twin_initial.csv"), ("refined", "twin.csv")):
            tw = pd.read_csv(d / fn)
            assert (tw.pid.values == pt.pid.values).all()
            for k in ("c", "s_off", "s_on", "g"):
                if k not in tw:
                    continue
                x, y = pt[k].values, tw[k].values
                if k == "g":
                    x, y = np.log(x), np.log(y)
                rows.append(dict(cohort=co, step=step, param=k, n=len(x), rho=spearmanr(x, y)[0],
                                 mae=float(np.median(np.abs(x - y)))))
    out = ROOT / "results" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "twin_accuracy.csv", index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
