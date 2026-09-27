"""Common figure style: vector PDF, serif fonts of 8 pt, IEEE column widths,
and one fixed color per method and cohort across all figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL_W = 3.48    # in, single column (252 pt in the TNSRE template)
PAGE_W = 7.14   # in, double column (516 pt)

COLORS = {
    "none": "#7f7f7f",
    "population": "#1f77b4",
    "twin": "#d62728",
    "checked": "#ff9896",
    "continuous_pop": "#17becf",
    "single_twin": "#8c564b",
    "stale": "#e377c2",
    "stale_recal": "#7b4173",
    "direct_1": "#a1d99b",
    "direct_2": "#41ab5d",
    "direct_3": "#006d2c",
    "reference": "#000000",
    "dusseldorf": "#9467bd",
    "madrid": "#ff7f0e",
    "synthetic": "#bdbdbd",
}
LABELS = {
    "none": "No stimulation",
    "population": "Population",
    "twin": "Digital twin",
    "reference": "Reference",
    "continuous": "Continuous",
    "single": "Single threshold",
    "dual": "Dual threshold",
    "dusseldorf": "D\u00fcsseldorf",
    "madrid": "Madrid",
    "dev": "Development",
    "test": "Test",
    "synth": "Synthetic",
    "stress": "Stress",
}


def apply():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.5,
        "lines.linewidth": 0.8,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig, path):
    """Save as PDF (vector) and PNG (inspection copy) with neutral metadata."""
    meta = {"Creator": "Matplotlib", "Author": None, "Title": None, "Subject": None, "Keywords": None,
            "CreationDate": None}
    fig.savefig(str(path) + ".pdf", metadata=meta)
    fig.savefig(str(path) + ".png", dpi=300, metadata={"Software": None})
    plt.close(fig)
