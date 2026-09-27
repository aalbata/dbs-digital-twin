"""Figure: digital-twin identification. Virtual-patient parameters against
the twin estimates for the synthetic cohort (gray) and the Madrid test
cohort (orange): delay scale c, OFF and ON severities and stimulation
efficacy g (Spearman correlations in results/analysis/twin_accuracy.csv).
Usage: python scripts/fig_twin_id.py  (writes figures/fig_twin_id.pdf/.png)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twin import plotstyle as PS  # noqa: E402

PS.apply()
import matplotlib.pyplot as plt  # noqa: E402

PARAMS = [("c", r"Delay scale $c$", (0.45, 1.85), False),
          ("s_off", r"Off-medication severity $s_\mathrm{off}$", (-1.35, 1.2), False),
          ("s_on", r"On-medication severity $s_\mathrm{on}$", (-1.35, 1.2), False),
          ("g", r"Stimulation efficacy $g$", (0.25, 4.0), True)]


def main():
    data = {}
    for co in ("synth", "test"):
        d = ROOT / "results" / co
        data[co] = pd.read_csv(d / "patients.csv").merge(pd.read_csv(d / "twin.csv"), on="pid", suffixes=("", "_tw"))
    fig, axs = plt.subplots(1, 4, figsize=(PS.PAGE_W, 1.9), gridspec_kw=dict(wspace=0.5))
    rng = np.random.default_rng(0)
    names = {"synth": "Synthetic cohort", "test": "Madrid test cohort"}
    for ax, (k, lab, lim, log), letter in zip(axs, PARAMS, "abcd"):
        for co, color, size in (("synth", PS.COLORS["synthetic"], 3), ("test", PS.COLORS["madrid"], 6)):
            x, y = data[co][k].values, data[co][k + "_tw"].values
            if k == "c":
                # estimates lie on the library grid: small vertical jitter for visibility
                y = y + rng.uniform(-0.012, 0.012, len(y))
            ax.scatter(x, y, s=size, color=color, lw=0, label=names[co])
        ax.plot(lim, lim, color="0.4", lw=0.5, zorder=0)
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([0.3, 1, 3])
            ax.set_xticklabels(["0.3", "1", "3"])
            ax.set_yticks([0.3, 1, 3])
            ax.set_yticklabels(["0.3", "1", "3"])
            ax.minorticks_off()
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Virtual patient " + lab.split(" ")[-1])
        ax.set_ylabel("Twin estimate")
        ax.text(-0.32, 1.02, f"({letter}) " + " ".join(lab.split(" ")[:-1]), transform=ax.transAxes, ha="left",
                va="bottom")
    h, lab_ = axs[0].get_legend_handles_labels()
    fig.legend(h, lab_, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.95), markerscale=1.5,
               handletextpad=0.2, columnspacing=1.5)
    PS.save(fig, ROOT / "figures" / "fig_twin_id")


if __name__ == "__main__":
    main()
