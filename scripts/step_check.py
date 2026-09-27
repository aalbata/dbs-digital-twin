"""Integration step check: 13-30 Hz power of the STN potential for a grid of
severities and delay scales, each with several noise seeds, simulated with
the production step (0.2 ms) and with half of it (0.1 ms) under the same noise
realization (the brainstem noise is held for 1 ms, so both steps see the same
input sequence).
Usage: python scripts/step_check.py  (writes results/model/step_check.json)"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twin.cohort import severity_params  # noqa: E402
from twin.linear import fixed_point  # noqa: E402
from twin.model import simulate  # noqa: E402

SEVERITIES = (0.0, 0.5, 0.8, 1.0)
DELAY_SCALES = (0.5, 1.0, 1.8)
SEEDS = (11, 12, 13)
T, BURN = 30.0, 5.0


def beta_power(dt, seed):
    """13-30 Hz power for every (delay scale, severity) pair, delay scale outer."""
    s = np.array([x for _ in DELAY_SCALES for x in SEVERITIES])
    p = severity_params(s)
    p["ct_scale"] = np.array([c for c in DELAY_SCALES for _ in SEVERITIES])
    B = len(s)
    t, rec = simulate(p, T, dt=dt, B=B, phi0=fixed_point(p, B), noise_sd=2.0, noise_hold=1e-3, seed=seed,
                      record_every=int(round(1e-3 / dt)), record=("Vz",), burn=BURN)
    x = rec["Vz"].T
    f, P = welch(x - x.mean(1, keepdims=True), fs=1000.0, nperseg=2000, axis=-1)
    band = (f >= 13) & (f <= 30)
    return P[:, band].sum(1) * (f[1] - f[0])


def main():
    p2 = np.array([beta_power(2e-4, seed) for seed in SEEDS])
    p1 = np.array([beta_power(1e-4, seed) for seed in SEEDS])
    rel = (p2 - p1) / p1
    shape = (len(SEEDS), len(DELAY_SCALES), len(SEVERITIES))
    out = dict(severity=list(SEVERITIES), delay_scale=list(DELAY_SCALES), seeds=list(SEEDS),
               beta_power_dt02=p2.reshape(shape).tolist(), beta_power_dt01=p1.reshape(shape).tolist(),
               rel_diff=rel.reshape(shape).tolist(), max_abs_rel_diff=float(np.max(np.abs(rel))))
    (ROOT / "results" / "model" / "step_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
