"""Reproduction of published results of the van Albada-Robinson model:
steady-state rates of Paper I, Table 3 (columns a-o), loop gains of Paper II,
Sec. 4.1, and the limit-cycle frequencies of Paper II, Fig. 8 (a)-(d).
Writes results/model/verification.json."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twin.model import HEALTHY, PARKINSONIAN, BETA_20HZ, THETA_5HZ, HYPERDIRECT_6HZ, COMBINED, simulate  # noqa: E402
from twin.linear import fixed_point_one, loop_gains, rightmost_mode  # noqa: E402

H = HEALTHY
TABLE3 = {  # Paper I, Table 3: cortex, D1, D2, GPi/SNr, GPe, STN, relay, TRN
    "a": (H, (12, 7.4, 3.5, 69, 48, 28, 14, 28)),
    "b": (dict(H, td1=13.0, td2=13.0, vd1e=0.4, vd2e=0.1), (12, 6.0, 2.7, 69, 48, 28, 14, 28)),
    "c": (dict(H, vd1e=0.5, vd2e=1.4), (10, 1.9, 9.3, 83, 40, 29, 11, 25)),
    "d": (dict(H, vp2p2=-0.03), (16, 14, 6.4, 49, 65, 27, 20, 34)),
    "e": (dict(H, vd1e=0.5, vd2e=1.4, vp2p2=-0.03), (12, 2.4, 12, 70, 51, 27, 13, 27)),
    "f": (dict(H, vee=1.4, vei=-1.6), (22, 24, 11, 78, 48, 36, 22, 42)),
    "g": (dict(H, vd1e=0.5, vd2e=1.4, vee=1.4, vei=-1.6), (14, 2.8, 16, 100, 36, 33, 13, 29)),
    "h": (PARKINSONIAN, (12, 2.2, 12, 110, 47, 36, 10, 27)),
    "i": (dict(H, vd1d1=0.0, vd2d2=0.0), (13, 15, 5.4, 63, 46, 29, 15, 29)),
    "j": (dict(H, vd1e=0.5, vd2e=1.4, vee=1.4, vei=-1.6, vd1d1=0.0, vd2d2=0.0), (12, 2.6, 24, 110, 28, 34, 10, 27)),
    "k": (dict(H, vd1s=0.3), (13, 13, 3.9, 64, 48, 29, 15, 29)),
    "l": (dict(H, vd2s=0.3), (11, 6.7, 5.8, 72, 45, 29, 13, 27)),
    "m": (dict(H, vp1p2=0.0), (10, 5.0, 2.5, 87, 47, 27, 10, 25)),
    "n": (dict(H, vp2z=0.4), (14, 11, 4.9, 56, 58, 27, 17, 31)),
    "o": (dict(H, vze=0.2), (10, 5.1, 2.6, 85, 55, 32, 11, 25)),
}
GAINS = {"healthy": (H, dict(intracortical=0.91, direct=0.29, hyperdirect=-0.35, stn_gpe=-0.89, indirect=-0.27)),
         "parkinsonian": (PARKINSONIAN, dict(intracortical=0.59, direct=0.042, hyperdirect=-0.40, stn_gpe=-1.1,
                                             indirect=-3.0))}
CYCLE_FS, CYCLE_NPERSEG = 1000.0, 4096
"""Sampling rate (Hz) and Welch segment length of the limit-cycle spectra."""
CYCLES = {"8a_theta": (THETA_5HZ, 5.1), "8b_hyperdirect": (HYPERDIRECT_6HZ, 6.2), "8c_beta": (BETA_20HZ, 20.0),
          "8d_combined": (COMBINED, 20.0)}


def sig2(x):
    return float(f"{x:.2g}")


def main():
    out = {"table3": {}, "gains": {}, "cycles": {}, "healthy_mode": None}
    worst = 0.0
    for col, (p, ref) in TABLE3.items():
        r = fixed_point_one(p)
        ok = all(sig2(a) == b or abs(a - b) <= 0.051 * b for a, b in zip(r, ref))
        worst = max(worst, max(abs(a - b) / b for a, b in zip(r, ref)))
        out["table3"][col] = dict(model=[round(float(x), 2) for x in r], paper=list(ref), match_2sf=bool(ok))
    out["table3_max_rel_dev"] = worst
    for name, (p, ref) in GAINS.items():
        g = loop_gains(p, fixed_point_one(p))
        out["gains"][name] = {k: dict(model=round(float(g[k]), 3), paper=v) for k, v in ref.items()}
    for name, (p, f_ref) in CYCLES.items():
        t, rec = simulate(p, 12.0, dt=1e-4, B=1, noise_sd=2.0, record=("z", "e"), record_every=10, burn=4.0)
        x = rec["z"][:, 0] - rec["z"][:, 0].mean()
        f, P = welch(x, fs=CYCLE_FS, nperseg=CYCLE_NPERSEG)
        k = np.argmax(P[f > 1.0]) + np.sum(f <= 1.0)
        out["cycles"][name] = dict(peak_hz=float(f[k]), paper_hz=f_ref, stn_range=float(np.ptp(rec["z"][:, 0])))
    phi = fixed_point_one(H)
    out["healthy_mode"] = dict(zip(("growth", "freq"), rightmost_mode(H, phi)))
    phi = fixed_point_one(PARKINSONIAN)
    out["parkinsonian_mode"] = dict(zip(("growth", "freq"), rightmost_mode(PARKINSONIAN, phi)))
    dest = ROOT / "results" / "model"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "verification.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
