"""Figure: example medication cycles of one test-cohort patient under
continuous stimulation (population amplitude), the population dual-threshold
policy, the twin-personalized dual-threshold policy and the twin-personalized
single-threshold policy: medication level,
beta envelope of the synaptic STN potential (relative to the outcome
reference) and stimulation amplitude.
The patient is the upper of the two middle patients of the test cohort when
ranked by the paired gain of the twin dual-threshold policy over the
population dual-threshold policy.
Usage: python scripts/fig_traces.py  (writes figures/fig_traces.pdf/.png and
results/test/traces.csv)"""
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

COHORT = "test"
EVAL_SEED = 777


def pick_patient(ev):
    d = ev[(ev.family == "dual") & ev.method.isin(["population", "twin"])].pivot(index="pid", columns="method",
                                                                                  values="J")
    gain = (d["population"] - d["twin"]).sort_values()
    return gain.index[len(gain) // 2]


def simulate(pid):
    d = ROOT / "results" / COHORT
    pt = pd.read_csv(d / "patients.csv")
    calib = dict(np.load(d / "calib.npz"))
    j = int(np.flatnonzero(pt.pid.values == pid)[0])
    b_ref = json.loads((ROOT / "results" / "dev" / "b_ref.json").read_text())["b_ref"]
    pop_c = json.loads((ROOT / "results" / "dev" / "policy_population_continuous.json").read_text())["params"]
    pop_d = json.loads((ROOT / "results" / "dev" / "policy_population_dual.json").read_text())["params"]
    tw_d = json.loads((d / "policy_twin_dual.json").read_text())["params"]
    tw_s = json.loads((d / "policy_twin_single.json").read_text())["params"]
    cfg = [("Continuous, population", "continuous", pop_c),
           ("Dual threshold, population", "dual", pop_d),
           ("Dual threshold, digital twin", "dual", tw_d),
           ("Single threshold, digital twin", "single", tw_s)]
    B = len(pt)
    out = {}
    for name, fam, prm in cfg:
        # the whole cohort with the evaluation seed, so that the traces are those of the evaluated runs
        pol = CL.POLICIES[fam](**{k: np.broadcast_to(np.asarray(v, float), (B,)) for k, v in prm.items()})
        r = CL.run(pt, pol, "cycle", calib, EVAL_SEED, b_ref, record=True)
        out[name] = dict(t=r["trace"]["t"], env=r["trace"]["env_true"][:, j] / b_ref, u=r["trace"]["u"][:, j],
                         m=r["trace"]["m"], J=float(CL.cost(r)[j]), B=float(r["B"][j]), U=float(r["U"][j]))
    return out


def main():
    ev = pd.read_csv(ROOT / "results" / COHORT / "evaluation.csv")
    pid = pick_patient(ev)
    tr = simulate(pid)
    first = next(iter(tr.values()))
    cols = {"t": first["t"], "m": first["m"]}
    for n, d in tr.items():
        cols[f"{n}|env"], cols[f"{n}|u"] = d["env"], d["u"]
    df = pd.DataFrame(cols)
    df.insert(0, "pid", pid)
    df.to_csv(ROOT / "results" / COHORT / "traces.csv", index=False, float_format="%.6g")
    fig, axs = plt.subplots(3, 1, figsize=(3.44, 3.1), sharex=True,
                            gridspec_kw=dict(height_ratios=[0.5, 1.2, 1.0], hspace=0.2))
    axs[0].plot(first["t"], first["m"], color="k", lw=0.8)
    axs[0].set_ylabel("Medication")
    axs[0].set_yticks([0, 1])
    colors = [PS.COLORS["continuous_pop"], PS.COLORS["population"], PS.COLORS["twin"], PS.COLORS["single_twin"]]
    for (name, d), col in zip(tr.items(), colors):
        axs[1].plot(d["t"], d["env"], color=col, lw=0.6, label=name)
        axs[2].plot(d["t"], d["u"], color=col, lw=0.7)
    late = first["t"] >= CL.BURN
    axs[1].set_ylim(0, 1.1 * max(d["env"][late].max() for d in tr.values()))
    axs[2].set_ylim(0, CL.U_MAX)
    axs[1].set_ylabel("Beta envelope\n(relative)")
    axs[2].set_ylabel("Amplitude (mV)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_xlim(CL.BURN, CL.SCENARIOS["cycle"]["T"])
    h, lab = axs[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.42, 0.875), handlelength=1.2,
               columnspacing=0.8)
    for ax, s in zip(axs, "abc"):
        ax.text(-0.26, 1.0, f"({s})", transform=ax.transAxes, ha="left", va="top")
    PS.save(fig, ROOT / "figures" / "fig_traces")
    print(pid, {n: (round(d["J"], 3), round(d["B"], 3), round(d["U"], 3)) for n, d in tr.items()})


if __name__ == "__main__":
    main()
