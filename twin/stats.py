"""Paired statistics for per-instance comparisons."""
import numpy as np
from scipy import stats


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation of paired differences d
    (positive when positive differences dominate); zeros are dropped."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def bootstrap_ci(x, fn=np.median, n_boot=10000, seed=7, alpha=0.05):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    idx = rng.integers(0, x.size, (n_boot, x.size))
    boot = fn(x[idx], axis=1)
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (p.size - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_compare(a, b):
    """a, b: per-instance values (lower is better). Returns the median of
    b - a, its bootstrap CI, the fraction of instances where a < b, the
    two-sided Wilcoxon signed-rank p-value and the rank-biserial effect size
    (positive when a is lower than b)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    lo, hi = bootstrap_ci(d)
    w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return dict(median_diff=float(np.median(d)), ci_lo=lo, ci_hi=hi, mean_diff=float(d.mean()),
                frac_better=float(np.mean(d > 0)), p=float(w.pvalue), r_rb=rank_biserial(d),
                shapiro_p=float(stats.shapiro(d).pvalue) if 3 <= d.size <= 5000 else np.nan, n=int(d.size))


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (float(c - h), float(c + h))
