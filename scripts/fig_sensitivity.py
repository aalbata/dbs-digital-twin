"""Figure: calibration length and disease progression.
(a) Test cohort, dual threshold: median cost (bootstrap 95% CI) of the twin
and checked twin policies identified from calibrations with 30, 60 and 120 s
of rest per medication state; the population policy with the same sensing
calibration for reference.
(b) Disease progression (severities raised after personalization),
development and test cohorts: median cost of the original twin policy with the original and with recalibrated sensing, the policy of
the updated twin and the population policy (no stimulation is written to
the CSV file only).
Usage: python scripts/fig_sensitivity.py  (writes figures/fig_sensitivity.pdf/.png
and results/analysis/sensitivity.csv)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twin import plotstyle as PS  # noqa: E402
from twin.stats import bootstrap_ci  # noqa: E402

PS.apply()
import matplotlib.pyplot as plt  # noqa: E402

REST = (("test_rest30", 30), ("test", 60), ("test_rest120", 120))
DRIFT = (("stale", "Original twin", PS.COLORS["stale"]),
         ("stale_recalibrated", "Original twin,\nrecalibrated", PS.COLORS["stale_recal"]),
         ("updated", "Updated twin", PS.COLORS["twin"]),
         ("population", "Population", PS.COLORS["population"]))


def med_ci(x):
    lo, hi = bootstrap_ci(x)
    return float(np.median(x)), lo, hi


def main():
    rows = []
    for co, sec in REST:
        ev = pd.read_csv(ROOT / "results" / co / "evaluation.csv")
        for m in ("twin", "checked", "population"):
            J = ev[(ev.family == "dual") & (ev.method == m)].J.values
            rows.append(dict(panel="rest", cohort=co, rest_s=sec, config=m, n=len(J), **dict(zip(("median", "lo", "hi"),
                                                                                                   med_ci(J)))))
    for co in ("dev", "test"):
        de = pd.read_csv(ROOT / "results" / co / "drift_evaluation.csv")
        for key in ["none"] + [k for k, _, _ in DRIFT]:
            J = de[de.config == key].J.values
            rows.append(dict(panel="drift", cohort=co, rest_s=np.nan, config=key, n=len(J),
                             **dict(zip(("median", "lo", "hi"), med_ci(J)))))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "analysis" / "sensitivity.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(PS.PAGE_W, 2.0), gridspec_kw=dict(wspace=0.3, width_ratios=[1, 1.5]))
    ax = axs[0]
    for m, col, off in (("population", PS.COLORS["population"], -4), ("twin", PS.COLORS["twin"], 0),
                        ("checked", PS.COLORS["checked"], 4)):
        x = df[(df.panel == "rest") & (df.config == m)].sort_values("rest_s")
        ax.errorbar(x.rest_s + off, x["median"], yerr=[x["median"] - x.lo, x.hi - x["median"]], fmt="o-", ms=2.5,
                    lw=0.7, elinewidth=0.6, color=col, label={"twin": "Digital twin", "checked": "Checked twin",
                                                              "population": "Population"}[m])
    ax.set_xticks([30, 60, 120])
    ax.set_xlabel("Rest per medication state (s)")
    ax.set_ylabel("Cost $J$ (median, 95% CI)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, borderaxespad=0.2,
              handlelength=1.2, handletextpad=0.3, columnspacing=0.6)
    ax = axs[1]
    for ci, (co, marker) in enumerate((("dev", "s"), ("test", "o"))):
        x = df[(df.panel == "drift") & (df.cohort == co)].set_index("config")
        for k, (key, lab, col) in enumerate(DRIFT):
            r = x.loc[key]
            ax.errorbar(k + (ci - 0.5) * 0.25, r["median"], yerr=[[r["median"] - r.lo], [r.hi - r["median"]]],
                        fmt=marker, ms=3, color=col, elinewidth=0.6)
    ax.set_xticks(range(len(DRIFT)))
    ax.set_xticklabels([lab for _, lab, _ in DRIFT], fontsize=8)
    ax.set_ylabel("Cost $J$ after progression")
    ax.legend(handles=[plt.Line2D([], [], color="0.3", marker=m, ls="none", ms=3, label=PS.LABELS[c])
                       for c, m in (("dev", "s"), ("test", "o"))], loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, fontsize=8, borderaxespad=0.2)
    for ax, s in zip(axs, "ab"):
        ax.text(-0.2 if s == "a" else -0.13, 1.0, f"({s})", transform=ax.transAxes, ha="left", va="bottom")
    PS.save(fig, ROOT / "figures" / "fig_sensitivity")


if __name__ == "__main__":
    main()
