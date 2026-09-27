"""Figure: virtual patients against the recordings they are anchored to.
(a) Beta peak frequency and (b) peak height (off medication filled, on
medication open) of each recorded hemisphere against the same features of
the simulated calibration recording of its virtual patient (development and
test cohorts, hidden traits and recording noise included); (c) anchored off-
and on-medication severities.
Usage: python scripts/fig_anchor.py  (writes figures/fig_anchor.pdf/.png)"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twin import plotstyle as PS  # noqa: E402

PS.apply()
import matplotlib.pyplot as plt  # noqa: E402

COHORTS = (("dev", "dusseldorf"), ("test", "madrid"))


def paired():
    """Recorded features and simulated calibration features, one row per
    hemisphere (virtual patients are built in anchor.csv order)."""
    an = pd.read_csv(ROOT / "results" / "patients" / "anchor.csv")
    out = []
    for co, src in COHORTS:
        a = an[an.cohort == src].reset_index(drop=True)
        pt = pd.read_csv(ROOT / "results" / co / "patients.csv")
        cal = pd.read_csv(ROOT / "results" / co / "calibration.csv")
        assert (pt.source.values == (a.patient + "_" + a.hemisphere).values).all()
        assert (pt.pid.values == cal.pid.values).all()
        out.append(pd.DataFrame(dict(cohort=src, rec_hz=a.peak_hz_off, sim_hz=cal.f_peak, rec_db=a.peak_db_off,
                                     sim_db=cal.peak_db_off, rec_db_on=a.peak_db_on,
                                     sim_db_on=cal.peak_db_on.where(~pt.s_on_imputed), s_off=a.s_off,
                                     s_on=a.s_on)))
    return pd.concat(out, ignore_index=True)


def main():
    d = paired()
    d.to_csv(ROOT / "results" / "patients" / "virtual_vs_recorded.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(PS.PAGE_W, 2.0), gridspec_kw=dict(wspace=0.45))
    ax = axs[0]
    for co in ("dusseldorf", "madrid"):
        x = d[d.cohort == co]
        ax.scatter(x.rec_hz, x.sim_hz, s=7, color=PS.COLORS[co], lw=0, label=PS.LABELS[co])
    ax.plot([10, 37], [10, 37], color="0.6", lw=0.5, zorder=0)
    ax.set_xlim(10, 37)
    ax.set_ylim(10, 37)
    ax.set_xlabel("Recorded peak frequency (Hz)")
    ax.set_ylabel("Virtual patient (Hz)")
    ax.legend(loc="upper left", handletextpad=0.2, borderaxespad=0.2, markerscale=1.5)
    ax = axs[1]
    for co in ("dusseldorf", "madrid"):
        x = d[d.cohort == co]
        ax.scatter(x.rec_db, x.sim_db, s=7, color=PS.COLORS[co], lw=0)
        ax.scatter(x.rec_db_on, x.sim_db_on, s=7, facecolor="none", edgecolor=PS.COLORS[co], lw=0.5)
    ax.plot([-2, 20], [-2, 20], color="0.6", lw=0.5, zorder=0)
    ax.set_xlim(-2, 20)
    ax.set_ylim(-2, 20)
    ax.set_xlabel("Recorded peak height (dB)")
    ax.set_ylabel("Virtual patient (dB)")
    ax = axs[2]
    for co in ("dusseldorf", "madrid"):
        x = d[(d.cohort == co) & d.s_on.notna()]
        ax.scatter(x.s_off, x.s_on, s=7, color=PS.COLORS[co], lw=0)
    ax.plot([-1.05, 1.15], [-1.05, 1.15], color="0.6", lw=0.5, zorder=0)
    ax.set_xlim(-1.05, 1.15)
    ax.set_ylim(-1.05, 1.15)
    ax.set_xlabel(r"Off-medication severity $s_\mathrm{off}$")
    ax.set_ylabel(r"On-medication severity $s_\mathrm{on}$")
    for ax, s in zip(axs, "abc"):
        ax.text(-0.25, 1.03, f"({s})", transform=ax.transAxes, ha="left", va="bottom")
    PS.save(fig, ROOT / "figures" / "fig_anchor")


if __name__ == "__main__":
    main()
