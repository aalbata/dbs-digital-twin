"""Figure: overview of the study (virtual patients, calibration, digital
twin, personalization and deployment, comparators).
Usage: python scripts/fig_overview.py  (writes figures/fig_overview.pdf/.png)"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twin import plotstyle as PS  # noqa: E402

PS.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

W, H = 7.05, 2.5


def box(ax, x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", lw=0.6,
                                edgecolor=color, facecolor="white"))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=8, color=color)
    ax.text(x + w / 2, y + h - 0.125, "\n".join(lines), ha="center", va="top", fontsize=8, linespacing=1.2)


def arrow(ax, x0, y0, x1, y1, color="0.2", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=color, linestyle=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=6))


def main():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    dark, red, blue, green = "0.15", PS.COLORS["twin"], PS.COLORS["population"], PS.COLORS["direct_3"]
    # top row: from recordings to virtual patients
    bw, bh, ty = 0.225, 0.36, 0.60
    box(ax, 0.01, ty, bw, bh, "Recorded STN LFPs",
        ["Rest, off and on medication", "Düsseldorf (development)", "Madrid (test)"], dark)
    box(ax, 0.29, ty, bw, bh, "Anchoring",
        ["Beta peak frequency and height", "matched to a model library give", r"delay scale $c$ and severities",
         r"$s_\mathrm{off}$ and $s_\mathrm{on}$"], dark)
    box(ax, 0.57, ty, 0.42, bh, "Virtual patient (mean-field BGTC model)",
        [r"Anchored $c$, $s_\mathrm{off}$, $s_\mathrm{on}$; medication cycle $s(t)$",
         r"Hidden traits (stimulation efficacy $g$, STN threshold,",
         "noise, recording background; in the stress cohort also",
         "STN-GPe strengths that the twin does not model)"], dark)
    arrow(ax, 0.235, ty + bh / 2, 0.29, ty + bh / 2)
    arrow(ax, 0.515, ty + bh / 2, 0.57, ty + bh / 2)
    # bottom row: calibration, twin, personalization, deployment
    by, bh2 = 0.04, 0.40
    box(ax, 0.01, by, 0.225, bh2, "Calibration protocol",
        ["60 s rest off medication", "Probe run of rest, 2.5, 5", "and 7.5 mV (20 s each)", "60 s rest on medication"],
        dark)
    box(ax, 0.26, by, 0.235, bh2, "Digital twin",
        [r"Step 1, library $\to c, s_\mathrm{off}, s_\mathrm{on}$",
         r"Step 2, GP-BO of $\Delta s$ and $g$", "against peak height and", "probe responses"], red)
    box(ax, 0.52, by, 0.225, bh2, "Personalization",
        ["GP-BO of policy parameters", "on the twin, medication-", r"cycle cost $J$", "(continuous, single,", "dual threshold)"],
        red)
    box(ax, 0.77, by, 0.22, bh2, "Deployment on patient",
        ["Medication cycle", r"$J = B + \lambda U + w D$", "Comparators are the", "population policy and",
         "direct BO on the patient"], dark)
    ymid = 0.535
    ax.plot([0.78, 0.78, 0.12], [ty, ymid, ymid], color="0.2", lw=0.6)
    arrow(ax, 0.12, ymid, 0.12, by + bh2)
    arrow(ax, 0.235, by + bh2 / 2, 0.26, by + bh2 / 2)
    arrow(ax, 0.495, by + bh2 / 2, 0.52, by + bh2 / 2)
    arrow(ax, 0.745, by + bh2 / 2, 0.77, by + bh2 / 2)
    ax.text(0.45, ymid - 0.02, "After disease progression, the twin is re-identified from new rest recordings",
            ha="center", va="top", fontsize=8, color=red, style="italic")
    PS.save(fig, ROOT / "figures" / "fig_overview")


if __name__ == "__main__":
    main()
