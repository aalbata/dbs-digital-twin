"""Statistics of the deployed policies for one cohort (results/<cohort>/
evaluation.csv).

For every policy family: outcome summaries per method; paired comparisons of
the twin-personalized policy and the checked twin policy with the population
policy, with each other and with direct per-patient optimization at each
trial budget (Wilcoxon signed-rank test,
Holm correction within the family, median paired difference with bootstrap
95% CI, matched-pairs rank-biserial correlation); the share of the attainable
gain captured, (J_population - J_twin) / (J_population - J_reference), where
the reference is direct optimization with the largest budget, computed on
cohort medians.
Budget table: cohort median cost of the twin, population and direct
configurations per family, and the smallest direct budget whose median cost
is at or below the twin's.
Usage: python scripts/analyze.py <cohort>
Writes results/analysis/<cohort>_summary.csv, <cohort>_paired.csv and
<cohort>_budget.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ["twin", "checked"]
"""Configurations compared with every other configuration of the same family."""
sys.path.insert(0, str(ROOT))
from twin.stats import paired_compare, holm  # noqa: E402


def main():
    cohort = sys.argv[1]
    ev = pd.read_csv(ROOT / "results" / cohort / "evaluation.csv")
    ev["budget"] = ev["budget"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    ev["config"] = ev.method + np.where(ev.budget != "", "_" + ev.budget, "")
    out = ROOT / "results" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    summ = ev.groupby(["family", "config"])[["J", "B", "U", "V"]].agg(["median", "mean"]).reset_index()
    summ.columns = ["family", "config"] + [f"{a}_{b}" for a, b in summ.columns[2:]]
    summ.to_csv(out / f"{cohort}_summary.csv", index=False)
    rows = []
    for fam, g in ev[ev.family != "none"].groupby("family"):
        piv = g.pivot(index="pid", columns="config", values="J")
        res = []
        for primary in [c for c in PRIMARY if c in piv]:
            for other in [c for c in piv.columns if c not in PRIMARY[:PRIMARY.index(primary) + 1]]:
                r = paired_compare(piv[primary].values, piv[other].values)
                r.update(family=fam, method=primary, comparator=other)
                res.append(r)
        if not res:
            continue
        adj = holm(np.array([r["p"] for r in res]))
        for r, a in zip(res, adj):
            r["p_holm"] = a
        direct = sorted([c for c in piv.columns if c.startswith("direct_")], key=lambda c: int(c.split("_")[1]))
        if "population" in piv and direct:
            jp, jr = piv["population"].median(), piv[direct[-1]].median()
            for r in res:
                r["gain_share"] = (jp - piv[r["method"]].median()) / (jp - jr) if jp != jr else np.nan
        rows += res
    pd.DataFrame(rows).to_csv(out / f"{cohort}_paired.csv", index=False)
    budget = []
    for fam, g in ev[ev.family != "none"].groupby("family"):
        med = g.groupby("config").J.median()
        if "twin" not in med:
            continue
        direct = sorted([c for c in med.index if c.startswith("direct_")], key=lambda c: int(c.split("_")[1]))
        matched = [int(c.split("_")[1]) for c in direct if med[c] <= med["twin"]]
        budget.append(dict(family=fam, twin=med["twin"], population=med.get("population", np.nan),
                           **{c: med[c] for c in direct},
                           first_direct_budget_at_or_below_twin=matched[0] if matched else np.nan))
    pd.DataFrame(budget).to_csv(out / f"{cohort}_budget.csv", index=False)
    pd.set_option("display.width", 220)
    print(summ.round(4).to_string())
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()
