"""Supplementary figure: single-threshold stimulation with fast and slow
ramps. Three development-cohort patients (10th, 50th and 90th percentile of
off-medication severity) held off medication for 40 s, single-threshold
policy with amplitude 7 mV, thresholds 0.6 and 1.0 and ramp rates 1 to
40 mV/s, compared with no stimulation and continuous 7-mV stimulation.
(a) Command and beta envelope of the median patient at 40 and 3 mV/s
(threshold 1.0); (b) mean beta burden against ramp rate.
Usage: python scripts/fig_single_ramp.py  (writes figures/fig_single_ramp.pdf/.png
and results/model/single_ramp.csv)"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twin import closedloop as CL, plotstyle as PS  # noqa: E402

PS.apply()
import matplotlib.pyplot as plt  # noqa: E402

RAMPS = (40.0, 10.0, 3.0, 1.0)
THRESHOLDS = (0.6, 1.0)
AMP, T, SEED = 7.0, 40.0, 3


def main():
    d = ROOT / "results" / "dev"
    pt0 = pd.read_csv(d / "patients.csv")
    cal0 = dict(np.load(d / "calib.npz"))
    b_ref = json.loads((d / "b_ref.json").read_text())["b_ref"]
    order = np.argsort(pt0.s_off.values, kind="stable")
    pick = order[[int(round(q * (len(order) - 1))) for q in (0.1, 0.5, 0.9)]]
    grid = [(p, thr, r) for p in pick for thr in THRESHOLDS for r in RAMPS]
    rows = np.array([g[0] for g in grid])
    pt = pt0.iloc[rows].reset_index(drop=True)
    calib = {k: (v[..., rows] if k == "phi0" else v[rows]) for k, v in cal0.items()}
    B = len(rows)
    thr = np.array([g[1] for g in grid])
    ramp = np.array([g[2] for g in grid])
    single = CL.run(pt, CL.SingleThreshold(thr=thr, a=np.full(B, AMP), ramp=ramp), "off", calib, SEED, b_ref,
                    T=T, record=True)
    none = CL.run(pt, CL.Continuous(a=np.zeros(B)), "off", calib, SEED, b_ref, T=T)
    cont = CL.run(pt, CL.Continuous(a=np.full(B, AMP)), "off", calib, SEED, b_ref, T=T)
    tr = single["trace"]
    late = tr["t"] >= CL.BURN
    df = pd.DataFrame(dict(pid=pt.pid, s_off=pt.s_off, threshold=thr, ramp=ramp, B_single=single["B"],
                           U_single=single["U"], B_none=none["B"], B_continuous=cont["B"],
                           peak_envelope=tr["env_true"][late].max(axis=0) / b_ref))
    df.to_csv(ROOT / "results" / "model" / "single_ramp.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(PS.PAGE_W, 2.0), gridspec_kw=dict(wspace=0.3, width_ratios=[1.6, 1]))
    t = tr["t"]
    sel = (t >= 20) & (t <= 26)
    mid = pick[1]
    ax = axs[0]
    ax2 = ax.twinx()
    for r, col in ((40.0, "0.15"), (3.0, "0.6")):
        j = int(np.flatnonzero((rows == mid) & (thr == 1.0) & (ramp == r))[0])
        ax.plot(t[sel], tr["env_true"][sel, j] / b_ref, color=col, lw=0.7, label=f"{r:g} mV/s")
        ax2.plot(t[sel], tr["u"][sel, j], color=col, lw=0.6, ls=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Beta envelope (relative, solid)")
    ax2.set_ylabel("Amplitude (mV, dotted)")
    ax2.set_ylim(0, 10)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=8, borderaxespad=0.2)
    ax = axs[1]
    for p, mk in zip(pick, ("o", "s", "^")):
        for th, ls in zip(THRESHOLDS, ("-", "--")):
            x = df[(df.pid == pt0.pid.values[p]) & (df.threshold == th)].sort_values("ramp")
            ax.plot(x.ramp, x.B_single / x.B_none, marker=mk, ms=3, lw=0.7, ls=ls, color="0.2")
    ax.axhline(1.0, color=PS.COLORS["none"], lw=0.6, ls=":")
    ax.legend(handles=[plt.Line2D([], [], color="0.2", lw=0.7, ls=ls, label=f"Threshold {th:.1f}")
                       for th, ls in zip(THRESHOLDS, ("-", "--"))], loc="upper left", fontsize=8,
              borderaxespad=0.2)
    ax.set_xscale("log")
    ax.set_xticks(RAMPS)
    ax.set_xticklabels([f"{r:g}" for r in RAMPS])
    ax.minorticks_off()
    ax.set_xlabel("Ramp rate (mV/s)")
    ax.set_ylabel("Beta burden / no stimulation")
    for ax, s in zip(axs, "ab"):
        ax.text(-0.14 if s == "a" else -0.22, 1.03, f"({s})", transform=ax.transAxes, ha="left", va="bottom")
    PS.save(fig, ROOT / "figures" / "fig_single_ramp")
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
