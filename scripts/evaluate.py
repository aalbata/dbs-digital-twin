"""Deploy every available policy on the patients of a cohort and record the
outcomes in the medication-cycle scenario with a fresh noise seed shared by
all configurations (paired comparisons).
Configurations: no stimulation; for each family, the population parameters
(development cohort), the twin-personalized parameters, the direct
per-patient parameters after each trial budget, and the checked twin policy:
one verification trial of the twin policy and one of the population policy
on the patient (noise seed CHECK_SEED, not used elsewhere), keeping for each
patient the policy with the lower trial cost.
Usage: python scripts/evaluate.py <cohort> [workers]
Writes results/<cohort>/evaluation.csv (one row per patient and configuration)
and check_trials.csv (verification-trial costs)."""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EVAL_SEED = 777
CHECK_SEED = 1777
FAMILIES = ("continuous", "single", "dual")


def configs(cohort):
    d = ROOT / "results" / cohort
    out = [("none", "none", "", None)]
    for fam in FAMILIES:
        pop = ROOT / "results" / "dev" / f"policy_population_{fam}.json"
        if pop.exists():
            out.append((fam, "population", "", json.loads(pop.read_text())["params"]))
        tw = d / f"policy_twin_{fam}.json"
        if tw.exists():
            out.append((fam, "twin", "", json.loads(tw.read_text())["params"]))
        di = d / f"policy_direct_{fam}.json"
        if di.exists():
            for n, prm in json.loads(di.read_text())["params_by_budget"].items():
                out.append((fam, "direct", n, prm))
    return out


def job(args):
    cohort, fam, method, budget, prm, seed = args
    from twin import closedloop as CL
    d = ROOT / "results" / cohort
    pt = pd.read_csv(d / "patients.csv")
    calib = dict(np.load(d / "calib.npz"))
    b_ref = json.loads((ROOT / "results" / "dev" / "b_ref.json").read_text())["b_ref"]
    B = len(pt)
    if fam == "none":
        pol = CL.Continuous(a=np.zeros(B))
    else:
        pol = CL.POLICIES[fam](**{k: np.broadcast_to(np.asarray(v, float), (B,)) for k, v in prm.items()})
    t0 = time.time()
    res = CL.run(pt, pol, "cycle", calib, seed, b_ref)
    df = pd.DataFrame(dict(pid=pt.pid, family=fam, method=method, budget=budget, B=res["B"], U=res["U"], V=res["V"]))
    df["J"] = CL.cost(res)
    return df, f"{fam} {method} {budget} seed {seed}: {time.time() - t0:.0f} s"


def checked(cohort, cfg, trials):
    """Per-patient choice between the twin and population parameters of each
    family by their verification-trial costs."""
    B = len(pd.read_csv(ROOT / "results" / cohort / "patients.csv"))
    out = []
    for fam in FAMILIES:
        prm = {m: p for f, m, b, p in cfg if f == fam and m in ("twin", "population")}
        if len(prm) < 2:
            continue
        jt = trials[(trials.family == fam) & (trials.method == "twin")].J.values
        jp = trials[(trials.family == fam) & (trials.method == "population")].J.values
        use_twin = jt <= jp
        ch = {k: np.where(use_twin, np.broadcast_to(np.asarray(prm["twin"][k], float), (B,)),
                          np.broadcast_to(np.asarray(prm["population"][k], float), (B,))).tolist()
              for k in prm["twin"]}
        out.append((fam, "checked", "", ch))
    return out


def main():
    cohort = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    cfg = configs(cohort)
    trial_jobs = [(cohort, f, m, b, p, CHECK_SEED) for f, m, b, p in cfg if m in ("twin", "population")]
    frames = []
    with ProcessPoolExecutor(workers) as ex:
        trials = []
        for df, msg in ex.map(job, trial_jobs):
            trials.append(df)
            print(msg, flush=True)
        trials = pd.concat(trials) if trials else pd.DataFrame(columns=["family", "method", "J"])
        cfg = cfg + checked(cohort, cfg, trials)
        for df, msg in ex.map(job, [(cohort,) + c + (EVAL_SEED,) for c in cfg]):
            frames.append(df)
            print(msg, flush=True)
    d = ROOT / "results" / cohort
    trials.to_csv(d / "check_trials.csv", index=False)
    pd.concat(frames).to_csv(d / "evaluation.csv", index=False)


if __name__ == "__main__":
    main()
