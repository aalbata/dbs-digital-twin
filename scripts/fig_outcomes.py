"""Figure: deployed cost by cohort, policy family and configuration.
For every cohort (panels) and policy family (groups): cohort median of the
medication-cycle cost J with a bootstrap 95% confidence interval for the
population policy, the twin-personalized policy, the checked twin policy and direct per-patient
optimization after each trial budget (the largest budget is the reference);
the dashed line is the median without stimulation.
Usage: python scripts/fig_outcomes.py  (writes figures/fig_outcomes.pdf/.png)"""
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

COHORTS = ("test", "synth", "stress")
FAMILIES = ("continuous", "single", "dual")
TITLES = {"test": "Madrid test cohort", "synth": "Synthetic cohort", "stress": "Stress cohort"}


def configs(ev, fam):
    """Ordered (label, method, budget, color) for one family."""
    budgets = sorted({int(b) for b in ev[(ev.family == fam) & (ev.method == "direct")].budget.dropna()})
    out = [("Population", "population", None, PS.COLORS["population"]),
           ("Digital twin", "twin", None, PS.COLORS["twin"]),
           ("Checked twin", "checked", None, PS.COLORS["checked"])]
    shades = [PS.COLORS["direct_1"], PS.COLORS["direct_2"], PS.COLORS["direct_3"]]
    for k, b in enumerate(budgets[:-1]):
        out.append((f"Direct, {b} trials", "direct", b, shades[min(k, 2)]))
    if budgets:
        out.append((f"Reference ({budgets[-1]} trials)", "direct", budgets[-1], PS.COLORS["reference"]))
    return out


def main():
    fig, axs = plt.subplots(1, len(COHORTS), figsize=(PS.PAGE_W, 2.0), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, co, letter in zip(axs, COHORTS, "abc"):
        ev = pd.read_csv(ROOT / "results" / co / "evaluation.csv")
        none = ev[ev.family == "none"].J.median()
        ax.axhline(none, color=PS.COLORS["none"], ls="--", lw=0.6)
        for gi, fam in enumerate(FAMILIES):
            cf = configs(ev, fam)
            xs = gi + np.linspace(-0.32, 0.32, len(cf))
            for x, (lab, meth, bud, col) in zip(xs, cf):
                sel = (ev.family == fam) & (ev.method == meth)
                if bud is not None:
                    sel &= ev.budget == bud
                J = ev[sel].J.values
                if J.size == 0:
                    continue
                lo, hi = bootstrap_ci(J)
                ax.errorbar(x, np.median(J), yerr=[[np.median(J) - lo], [hi - np.median(J)]], fmt="o", ms=2.5,
                            color=col, elinewidth=0.6, capsize=0)
        ax.set_xticks(range(len(FAMILIES)))
        ax.set_xticklabels([PS.LABELS[f].replace(" ", "\n") for f in FAMILIES])
        ax.set_xlim(-0.5, len(FAMILIES) - 0.5)
        ax.text(0.0, 1.03, f"({letter}) {TITLES[co]}", transform=ax.transAxes, ha="left", va="bottom")
    axs[0].set_ylabel("Cost $J$ (median, 95% CI)")
    handles = [plt.Line2D([], [], color=PS.COLORS["none"], ls="--", lw=0.6, label="No stimulation")]
    for lab, key in (("Population", "population"), ("Digital twin", "twin"), ("Checked twin", "checked"),
                     ("Direct, 1st budget", "direct_1"), ("Direct, 2nd budget", "direct_2"),
                     ("Direct, 3rd budget", "direct_3"), ("Reference", "reference")):
        handles.append(plt.Line2D([], [], color=PS.COLORS[key], marker="o", ms=2.5, ls="none", label=lab))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.97), handletextpad=0.3,
               columnspacing=1.2)
    PS.save(fig, ROOT / "figures" / "fig_outcomes")


if __name__ == "__main__":
    main()
