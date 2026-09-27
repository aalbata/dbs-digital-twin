"""Disease drift: every patient's OFF and ON severities rise by DRIFT (capped at
1.1, the upper end of the severity grid). The
dual-threshold policy personalized at baseline (stale twin) is deployed with
the baseline and with a recalibrated sensing normalization, and compared with
an updated twin, re-identified from new resting recordings only (OFF and ON,
no stimulation probes; delay scale, stimulation efficacy and the baseline
severity correction kept) and re-personalized with the same number of
evaluations, with the sensing recalibrated; the population dual-threshold
policy (recalibrated sensing) and no stimulation are included.
Usage: python scripts/drift.py <cohort>
Writes results/<cohort>/drift_twin.csv, policy_drift_dual.json and
drift_evaluation.csv."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twin import study, closedloop as CL  # noqa: E402
from twin.anchor import feature_table, fit_features  # noqa: E402
from twin.patients import SEEDS  # noqa: E402

DRIFT = 0.2
DRIFT_ITERS = 40
EVAL_SEED = 778


def main():
    cohort = sys.argv[1]
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    d = ROOT / "results" / cohort
    chi = json.loads((ROOT / "results" / "patients" / "anchor_background.json").read_text())["chi"]
    b_ref = json.loads((ROOT / "results" / "dev" / "b_ref.json").read_text())["b_ref"]
    pt = pd.read_csv(d / "patients.csv")
    calib0 = dict(np.load(d / "calib.npz"))
    tw0 = pd.read_csv(d / "twin.csv")
    init0 = pd.read_csv(d / "twin_initial.csv")
    ptd = pt.copy()
    ptd["s_off"] = np.minimum(pt.s_off + DRIFT, 1.1)
    ptd["s_on"] = np.minimum(pt.s_on + DRIFT, 1.1)
    B = len(ptd)
    seed = SEEDS[cohort] % 100000 + 51
    zero = lambda t: np.zeros(B)
    phi_off = CL.initial_state(ptd, ptd.s_off.values)
    c0 = dict(f_peak=np.full(B, 20.0), b_cal=np.ones(B), chi=np.full(B, chi), phi0=phi_off)
    rest = CL.run(ptd, None, "off", c0, seed, 1.0, T=study.REST_T, probe=zero, record_lfp=True)
    ft = study.rest_features(rest["lfp"])
    f_peak = ft.peak_hz.values
    b_cal = np.median(study.envelope(rest["lfp"], f_peak)[:, int(5 * CL.FS):], axis=1)
    on = CL.run(ptd, None, "on", dict(c0, phi0=CL.initial_state(ptd, ptd.s_on.values)), seed + 1, 1.0,
                T=study.REST_T, probe=zero, record_lfp=True)
    fo = study.rest_features(on["lfp"])
    lib = dict(np.load(ROOT / "results" / "patients" / "model_library.npz"))
    table = feature_table(lib, chi)
    ds = tw0.s_off.values - init0.s_off.values
    s_off_new, s_on_new = [], []
    for j in range(B):
        s_off_new.append(fit_features(f_peak[j], ft.peak_db.values[j], lib, table, c_fixed=tw0.c.values[j])["s"])
        s_on_new.append(fit_features(f_peak[j], fo.peak_db.values[j], lib, table, c_fixed=tw0.c.values[j])["s"])
    tw1 = study.twin_table(tw0.pid.values, tw0.c.values, np.array(s_off_new) + ds, np.array(s_on_new) + ds,
                           tw0.g.values)
    tw1.to_csv(d / "drift_twin.csv", index=False)
    log("twin updated")
    calib1 = dict(f_peak=f_peak, b_cal=b_cal, chi=np.full(B, chi), phi0=phi_off)
    tcal = dict(calib1, phi0=CL.initial_state(tw1, tw1.s_off.values))
    recs, hist = study.optimize_policy(tw1, "dual", tcal, None, None, b_ref, DRIFT_ITERS, seed + 5, log=log)
    upd = {k: v.tolist() for k, v in recs[DRIFT_ITERS].items()}
    (d / "policy_drift_dual.json").write_text(json.dumps(dict(params=upd, history_median=np.median(hist, 1).tolist())))
    stale = json.loads((d / "policy_twin_dual.json").read_text())["params"]
    pop = json.loads((ROOT / "results" / "dev" / "policy_population_dual.json").read_text())["params"]
    arr = lambda prm: {k: np.broadcast_to(np.asarray(v, float), (B,)) for k, v in prm.items()}
    cfg = [("none", CL.Continuous(a=np.zeros(B)), calib1),
           ("stale", CL.DualThreshold(**arr(stale)), dict(calib0, phi0=phi_off)),
           ("stale_recalibrated", CL.DualThreshold(**arr(stale)), calib1),
           ("updated", CL.DualThreshold(**arr(upd)), calib1),
           ("population", CL.DualThreshold(**arr(pop)), calib1)]
    frames = []
    for name, pol, cal in cfg:
        res = CL.run(ptd, pol, "cycle", cal, EVAL_SEED, b_ref)
        df = pd.DataFrame(dict(pid=ptd.pid, config=name, B=res["B"], U=res["U"], V=res["V"]))
        df["J"] = CL.cost(res)
        frames.append(df)
        log(f"{name}: median J {df.J.median():.4f}")
    pd.concat(frames).to_csv(d / "drift_evaluation.csv", index=False)


if __name__ == "__main__":
    main()
