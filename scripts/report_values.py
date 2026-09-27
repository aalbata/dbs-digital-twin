"""Every value reported in the article and its supplement, computed from the
result files: model verification, cohorts, anchoring, twin identification,
closed-loop behavior, deployed outcomes (including Table II rows),
patient-trial comparisons, calibration length, disease progression and the
supplementary tables. Values are formatted as printed.
Usage: python scripts/report_values.py  (writes results/analysis/reported_values.json)"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twin import study, closedloop as CL  # noqa: E402
from twin.patients import N_SYNTH  # noqa: E402
from twin.stats import paired_compare, holm  # noqa: E402

RES = ROOT / "results"
PHRASE = {"continuous": "continuous", "single": "single threshold", "dual": "dual threshold"}
FAMILY_NAME = {"continuous": "Continuous", "single": "Single threshold", "dual": "Dual threshold"}
COHORT_NAME = {"dev": "Development", "test": "Test", "synth": "Synthetic", "stress": "Stress",
               "test_rest30": "Rest 30~s", "test_rest120": "Rest 120~s"}
CO = {"D": "dev", "T": "test", "S": "synth", "R": "stress"}
FAIL_MARGIN = 0.05
"""A policy counts as a failure for a patient when its cost exceeds that of the
population policy of the same family by more than this margin."""
LARGE_MED = 1.0
"""A medication effect s_off - s_on of at least this size counts as large."""
FA = {"CONT": "continuous", "SING": "single", "DUAL": "dual"}


def f1(x):
    return f"{x:.1f}"


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def sgn(x, d=3):
    """Signed number; negative values in math mode so that they print a minus sign."""
    s = f"{x:.{d}f}"
    return f"${s}$" if s.startswith("-") else s


def r0(p):
    """Integer percentage, halves rounded up."""
    return f"{np.floor(p + 0.5 + 1e-9):.0f}"


def upper(x, d):
    """Upper bound: x rounded up at d decimals."""
    return f"{np.ceil(x * 10 ** d - 1e-9) / 10 ** d:.{d}f}"


def pct(x):
    return r0(100 * x)


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def join(items):
    items = list(items)
    return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else items[0]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_eval(co):
    ev = pd.read_csv(RES / co / "evaluation.csv")
    ev["budget"] = ev["budget"].fillna(-1).astype(int)
    return ev


def paired(co):
    return pd.read_csv(RES / "analysis" / f"{co}_paired.csv").set_index(["family", "method", "comparator"])


def methods_values():
    v = {}
    ver = json.loads((RES / "model" / "verification.json").read_text())
    assert all(t["match_2sf"] for t in ver["table3"].values())
    v["VER_NCOLS"] = str(len(ver["table3"]))
    v["VER_MAXDEV"] = upper(100 * ver["table3_max_rel_dev"], 1)
    cyc = ver["cycles"]
    v["VER_CYCLES"] = join(f1(cyc[k]["peak_hz"]) for k in sorted(cyc))
    v["VER_CYCLES_PAPER"] = join(f"{cyc[k]['paper_hz']:g}" for k in sorted(cyc))
    v["SIM_SPEED_40"] = f"{json.loads((RES / 'model' / 'timing.json').read_text())['wall_s']:.0f}"
    sc = json.loads((RES / "model" / "step_check.json").read_text())
    words = {2: "two", 3: "three", 4: "four", 5: "five"}
    v["STEP_DIFF"] = upper(100 * sc["max_abs_rel_diff"], 1)
    v["STEP_NSEV"] = words[len(sc["severity"])]
    v["STEP_NC"], v["STEP_NSEED"] = words[len(sc["delay_scale"])], words[len(sc["seeds"])]
    mv = load_script("model_verification")
    v["VER_RES"] = f2(mv.CYCLE_FS / mv.CYCLE_NPERSEG)
    an = pd.read_csv(RES / "patients" / "anchor.csv")
    dus, mad = an[an.cohort == "dusseldorf"], an[an.cohort == "madrid"]
    v["N_DUS_HEMI"], v["N_DUS_ON"] = str(len(dus)), str(int(dus.s_on.notna().sum()))
    v["N_MAD_HEMI"], v["N_MAD_ON"] = str(len(mad)), str(int(mad.s_on.notna().sum()))
    v["N_DUS_PAT"], v["N_MAD_PAT"] = str(dus.patient.nunique()), str(mad.patient.nunique())
    v["N_MAD_PAT_ON"] = str(mad.dropna(subset=["s_on"]).patient.nunique())
    per_pat = mad.groupby("patient").size()
    assert set(per_pat) == {1, 2}
    v["N_MAD_ONEHEMI"] = str(int((per_pat == 1).sum()))
    rf = pd.read_csv(RES / "patients" / "recording_features.csv")
    rest = rf[rf.cohort == "dusseldorf"].rest_s
    v["DUS_REST_MIN"], v["DUS_REST_MAX"] = f"{np.floor(rest.min()):.0f}", f"{np.ceil(rest.max()):.0f}"
    v["CHI"] = f2(json.loads((RES / "patients" / "anchor_background.json").read_text())["chi"])
    for co, key in (("dev", "N_DEV"), ("test", "N_TEST"), ("synth", "N_SYNTH"), ("stress", "N_STRESS")):
        v[key] = str(len(pd.read_csv(RES / co / "patients.csv")))
    assert int(v["N_SYNTH"]) == N_SYNTH["synth"] and int(v["N_STRESS"]) == N_SYNTH["stress"]
    v["CAL_TOTAL_S"] = f"{2 * study.REST_T + study.PROBE_REST + study.PROBE_STEP * len(study.PROBE_U):.0f}"
    lib = dict(np.load(RES / "patients" / "model_library.npz"))
    crit = {c: lib["unstable_s"][np.isclose(lib["unstable_c"], c)].min() for c in np.unique(lib["c"])
            if np.isclose(lib["unstable_c"], c).any()}
    stable = [c for c in np.unique(lib["c"]) if c not in crit]
    assert max(stable) < min(crit) and np.isclose(lib["s"].max(), 1.1)
    v["S_CRIT_MIN"], v["S_CRIT_MAX"] = f"{min(crit.values()):g}", f"{max(crit.values()):g}"
    v["C_STABLE_MAX"] = f"{max(stable):g}"
    pers = load_script("personalize")
    fams = ("continuous", "single", "dual")
    for fam in fams:
        assert len(json.loads((RES / "dev" / f"policy_population_{fam}.json").read_text())["history"]) == pers.ITERS[fam]
    v["POP_ITERS"] = join(str(pers.ITERS[f]) for f in fams) + \
        " evaluations for the continuous, single-threshold and dual-threshold families"
    v["TWIN_POLICY_ITERS"] = "the same numbers of evaluations"
    budgets = {f: sorted(set(pers.DIRECT[f]) | {pers.DIRECT_MAX[f]}) for f in fams}
    v["DIRECT_BUDGETS"] = ", ".join(f"{join(str(b) for b in budgets[f])} ({PHRASE[f]})" for f in fams[:2]) + \
        f" and {join(str(b) for b in budgets['dual'])} ({PHRASE['dual']})"
    v["BO_NINIT_DUAL"] = str(study.N_INIT[study.POLICY_BOX["dual"].d])
    cal = load_script("calibrate_twins")
    v["TWIN_ITERS"] = str(cal.TWIN_ITERS)
    dr = load_script("drift")
    v["DRIFT"] = f"{dr.DRIFT:g}"
    v["DRIFT_ITERS"] = str(dr.DRIFT_ITERS)
    capped = {}
    for co in ("dev", "test"):
        pt = pd.read_csv(RES / co / "patients.csv")
        capped[co] = int(((pt.s_off + dr.DRIFT > 1.1) | (pt.s_on + dr.DRIFT > 1.1)).sum())
    assert capped["test"] == 0
    v["DRIFT_NCAPPED"] = {1: "one", 2: "two", 3: "three"}[capped["dev"]]
    return v


def anchor_values():
    v = {}
    an = pd.read_csv(RES / "patients" / "anchor.csv")
    ok = an.peak_db_off >= 1.5
    v["N_PEAK15"], v["N_HEMI"] = str(int(ok.sum())), str(len(an))
    v["ANC_MAXHZ"] = f"{(an.model_peak_hz_off - an.peak_hz_off).abs()[ok].max():g}"
    v["ANC_MAXDB"] = upper((an.model_peak_db_off - an.peak_db_off).abs()[ok].max(), 2)
    pr = an.dropna(subset=["s_on"])
    v["N_SON_LOWER"], v["N_PAIRED"] = str(int((pr.s_on < pr.s_off).sum())), str(len(pr))
    v["MED_DELTA"] = f2(np.median(pr.s_off - pr.s_on))
    for co, key in (("dusseldorf", "DEV"), ("madrid", "TEST")):
        x = pr[pr.cohort == co]
        v[f"MED_DELTA_{key}"] = f2(np.median(x.s_off - x.s_on))
    d = pd.read_csv(RES / "patients" / "virtual_vs_recorded.csv")
    for co, key in (("dusseldorf", "DEV"), ("madrid", "TEST")):
        x = d[d.cohort == co]
        v[f"RHO_DB_{key}"] = f2(spearmanr(x.rec_db, x.sim_db)[0])
        y = x.dropna(subset=["rec_db_on", "sim_db_on"])
        v[f"RHO_DB_ON_{key}"] = f2(spearmanr(y.rec_db_on, y.sim_db_on)[0])
        v[f"PCT_HZ_{key}"] = pct(np.mean((x.rec_hz - x.sim_hz).abs() <= 1))
    med = d.rec_db.median()
    dhz = (d.rec_hz - d.sim_hz).abs()
    big = dhz > 3
    assert (d.rec_db[big] < med).all()
    v["MED_REC_DB"] = f1(med)
    v["N_HZ3"], v["MAX_DHZ"] = str(int(big.sum())), f"{dhz.max():g}"
    return v


def twin_values():
    v = {}
    acc = pd.read_csv(RES / "analysis" / "twin_accuracy.csv")
    ref = acc[acc.step == "refined"].set_index(["cohort", "param"])
    rest = acc[acc.step == "rest"].set_index(["cohort", "param"])
    for k, name in (("c", "C"), ("s_off", "SOFF"), ("s_on", "SON"), ("g", "G")):
        v[f"TID_TEST_{name}"] = f2(ref.rho[("test", k)])
    v["TID_STRESS_C"] = f2(ref.rho[("stress", "c")])
    v["TID_STRESS_SOFF"] = f2(ref.rho[("stress", "s_off")])
    v["TID_STRESS_G"] = f2(ref.rho[("stress", "g")])
    v["TID_STRESS_GMAE"] = f2(ref.mae[("stress", "g")])
    v["TID_TEST_GMAE"] = f2(ref.mae[("test", "g")])
    cohorts = ("dev", "test", "synth", "stress", "test_rest30", "test_rest120")
    lowest = {co: ref.rho.loc[co].idxmin() for co in cohorts}
    assert all(p == "g" for co, p in lowest.items() if co != "stress") and lowest["stress"] == "c"
    assert ref.mae[("stress", "g")] == max(ref.mae[(co, "g")] for co in cohorts)
    for k in ("c", "s_off", "s_on"):
        assert ref.rho[("stress", k)] == min(ref.rho[(co, k)] for co in cohorts)
    pairs = [(co, p) for co in cohorts for p in ("s_off", "s_on")]
    v["TID_REFINE_LOWER"] = str(sum(ref.rho[i] < rest.rho[i] for i in pairs))
    v["TID_REFINE_NPAIRS"] = str(len(pairs))
    v["TID_MAXDRHO"] = upper(max(abs(ref.rho[i] - rest.rho[i]) for i in pairs), 2)
    g = pd.read_csv(RES / "test" / "twin.csv").g.values
    lo, hi = study.TWIN_BOX.spec[1][1], study.TWIN_BOX.spec[1][2]
    assert not (g <= lo * 1.01).any()
    v["TID_TEST_GBOUND"] = pct(np.mean(g >= hi * 0.99))
    for co, K in (("test_rest30", "30"), ("test", "60"), ("test_rest120", "120")):
        v[f"TID_GMAE_{K}"] = f2(ref.mae[(co, "g")])
    # rest length: the efficacy error grows with the rest recordings, its correlation peaks at 60 s
    mae = [ref.mae[(co, "g")] for co in ("test_rest30", "test", "test_rest120")]
    rho = [ref.rho[(co, "g")] for co in ("test_rest30", "test", "test_rest120")]
    assert mae[0] < mae[1] < mae[2] and rho[1] == max(rho)
    rows = []
    for co in cohorts:
        r = ref.loc[co]
        rows.append(f"{COHORT_NAME[co]} & {int(r.n.iloc[0])} & " + " & ".join(f2(r.rho[k]) for k in ("c", "s_off", "s_on", "g"))
                    + f" & {f2(r.mae['g'])} \\\\")
    v["TAB_TWINID"] = "\n".join(rows)
    return v


def config_rows(ev, fam):
    rows = [("Population", "population", -1), ("Digital twin", "twin", -1), ("Checked twin", "checked", -1)]
    budgets = sorted(ev[(ev.family == fam) & (ev.method == "direct")].budget.unique())
    for b in budgets[:-1]:
        rows.append((f"Direct, {b} trials", "direct", b))
    if budgets:
        rows.append((f"Reference, {budgets[-1]} trials", "direct", budgets[-1]))
    return rows


def cell(ev, fam, meth, bud):
    x = ev[(ev.family == fam) & (ev.method == meth) & (ev.budget == bud)].set_index("pid")
    if x.empty:
        return None
    pop = ev[(ev.family == fam) & (ev.method == "population")].set_index("pid").J
    better = np.nan if meth == "population" else 100 * np.mean(x.J.reindex(pop.index) < pop)
    return dict(J=x.J.median(), B=x.B.median(), U=x.U.median(), better=better)


def outcome_table():
    evs = {co: load_eval(co) for co in ("test", "synth", "stress")}

    def fmt(c, k):
        if c is None or (k == "better" and not np.isfinite(c[k])):
            return ""
        return r0(c[k]) if k == "better" else f"{c[k]:.3f}"
    none = {co: ev[ev.family == "none"] for co, ev in evs.items()}
    lines = ["\\multicolumn{2}{l}{No stimulation} & " + f"{none['test'].J.median():.3f} & {none['test'].B.median():.3f}"
             f" & 0.000 & & {none['synth'].J.median():.3f} & & {none['stress'].J.median():.3f} & \\\\"]
    for fam in ("continuous", "single", "dual"):
        rows = config_rows(evs["test"], fam)
        lines.append("\\midrule")
        for k, (lab, meth, bud) in enumerate(rows):
            t = cell(evs["test"], fam, meth, bud)
            sy, st = cell(evs["synth"], fam, meth, bud), cell(evs["stress"], fam, meth, bud)
            head = f"\\multirow{{{len(rows)}}}{{*}}{{{FAMILY_NAME[fam]}}}" if k == 0 else ""
            lines.append(f"{head} & {lab} & {fmt(t, 'J')} & {fmt(t, 'B')} & {fmt(t, 'U')} & {fmt(t, 'better')} & "
                         f"{fmt(sy, 'J')} & {fmt(sy, 'better')} & {fmt(st, 'J')} & {fmt(st, 'better')} \\\\")
    return "\n".join(lines)


def results_values():
    v = {}
    evs = {co: load_eval(co) for co in ("dev", "test", "synth", "stress")}

    def sel(co, fam, m, b=-1):
        e = evs[co]
        return e[(e.family == fam) & (e.method == m) & (e.budget == b)]
    med = lambda co, fam, m, b=-1, k="J": sel(co, fam, m, b)[k].median()
    mean = lambda co, fam, m, b=-1: sel(co, fam, m, b).J.mean()
    # example patient (Fig. traces): upper of the two middle patients by paired gain
    tr = pd.read_csv(RES / "test" / "traces.csv")
    pid = tr.pid.iloc[0]
    d = evs["test"][(evs["test"].family == "dual") & evs["test"].method.isin(["population", "twin"])].pivot(
        index="pid", columns="method", values="J")
    gain = (d.population - d.twin).sort_values()
    assert gain.index[len(gain) // 2] == pid
    v["EX_RANK"], v["EX_N"] = str(len(gain) - len(gain) // 2), str(len(gain))  # rank from the largest gain
    v["EX_GAIN"], v["EX_GAIN_MED"] = f3(gain[pid]), f3(gain.median())
    e = evs["test"][(evs["test"].pid == pid) & (evs["test"].budget == -1)].set_index(["family", "method"]).sort_index()
    v["EX_J_CONT"] = f3(e.J[("continuous", "population")])
    v["EX_J_POPD"] = f3(e.J[("dual", "population")])
    v["EX_J_TWD"] = f3(e.J[("dual", "twin")])
    v["EX_J_TWS"] = f3(e.J[("single", "twin")])
    late = tr.t >= CL.BURN
    v["EX_U_TWD"] = f1(tr.loc[late, "Dual threshold, digital twin|u"].median())
    v["EX_U_POPD"] = f1(tr.loc[late, "Dual threshold, population|u"].median())
    us = tr.loc[late, "Single threshold, digital twin|u"]
    v["EX_US_MIN"], v["EX_US_MAX"] = f1(us.min()), f1(us.max())
    # adaptation of the optimized policies
    adapt, adapt_d = [], []
    for co in ("dev", "test", "synth", "stress"):
        x = evs[co][evs[co].family == "dual"]
        for (m, b), g in x.groupby(["method", "budget"]):
            (adapt_d if m == "direct" else adapt).append(100 * np.mean(g.V * CL.R_MAX > 0.1))
    v["DUAL_ADAPT_MAX"] = f1(max(adapt))
    v["DUAL_ADAPT_DMAX"] = f1(max(adapt_d))
    ts = sel("test", "single", "twin")
    v["SINGLE_TV"] = f2((ts.V * CL.R_MAX).median())
    twc = json.loads((RES / "test" / "policy_twin_continuous.json").read_text())["params"]["a"]
    twd = json.loads((RES / "test" / "policy_twin_dual.json").read_text())["params"]
    v["RHO_CONT_DUAL"] = f2(spearmanr(twc, np.array(twd["floor"]) * np.array(twd["a_max"]))[0])
    d60 = json.loads((RES / "test" / "policy_direct_dual.json").read_text())["params_by_budget"]
    v["DIRECT_FLOOR"] = f2(np.median(d60[max(d60, key=int)]["floor"]))
    # patients with a large medication effect: command change under the largest direct dual-threshold budget
    v["LARGE_MED"] = f"{LARGE_MED:g}"
    for co, C in (("test", "T"), ("synth", "S"), ("stress", "R")):
        pt_ = pd.read_csv(RES / co / "patients.csv").set_index("pid")
        big = pt_.index[pt_.s_off - pt_.s_on >= LARGE_MED - 1e-9]
        e = evs[co]
        bmax = e[(e.family == "dual") & (e.method == "direct")].budget.max()
        x = e[(e.family == "dual") & (e.method == "direct") & (e.budget == bmax)].set_index("pid").reindex(big)
        assert x.V.notna().all() and len(x)
        v[f"LM_{C}"], v[f"LM_{C}_N"] = str(int((x.V * CL.R_MAX <= 0.1).sum())), str(len(x))
    # personalized versus population
    v["T_NONE"] = f3(med("test", "none", "none"))
    for key in ("T_CONT", "T_SING", "T_DUAL", "S_CONT", "S_SING", "S_DUAL", "R_CONT", "R_DUAL"):
        C, F = key.split("_")
        r = paired(CO[C]).loc[(FA[F], "twin", "population")]
        v[f"{key}_BETTER"] = pct(r.frac_better)
        if key not in ("S_CONT", "R_CONT"):
            v[f"{key}_DIFF"], v[f"{key}_P"] = sgn(r.median_diff), pfmt(r.p_holm)
        if key in ("T_CONT", "T_SING", "T_DUAL", "S_DUAL"):
            v[f"{key}_CILO"], v[f"{key}_CIHI"] = sgn(r.ci_lo), sgn(r.ci_hi)
        if key in ("T_CONT", "T_DUAL", "S_DUAL", "R_DUAL"):
            v[f"{key}_R"] = f2(r.r_rb)
    for key in ("T_SING_POP", "T_SING_TW", "T_DUAL_POP", "T_DUAL_CHK", "S_DUAL_POP", "S_DUAL_TW", "S_DUAL_CHK",
                "R_DUAL_POP", "R_DUAL_TW", "R_DUAL_CHK"):
        C, F, M = key.split("_")
        v[key] = f3(med(CO[C], FA[F], {"POP": "population", "TW": "twin", "CHK": "checked"}[M]))
    v["T_DUAL_U_POP"], v["T_DUAL_U_TW"] = f3(med("test", "dual", "population", k="U")), f3(med("test", "dual", "twin", k="U"))
    v["T_DUAL_B_POP"], v["T_DUAL_B_TW"] = f3(med("test", "dual", "population", k="B")), f3(med("test", "dual", "twin", k="B"))
    v["T_DUAL_U_CHK"] = f3(med("test", "dual", "checked", k="U"))
    v["T_DUAL_U_RED"] = f"{100 * (1 - med('test', 'dual', 'twin', k='U') / med('test', 'dual', 'population', k='U')):.0f}"
    # every twin policy of the continuous and dual-threshold families used less energy at a higher burden
    ut, up, bt, bp = [], [], [], []
    for co in ("test", "synth", "stress"):
        for fam in ("continuous", "dual"):
            u_t, u_p = med(co, fam, "twin", k="U"), med(co, fam, "population", k="U")
            b_t, b_p = med(co, fam, "twin", k="B"), med(co, fam, "population", k="B")
            assert u_t < u_p and b_t > b_p, (co, fam)
            if co != "test":
                for lst, val in ((ut, u_t), (up, u_p), (bt, b_t), (bp, b_p)):
                    lst.append(val)
    for key, x in (("U_TW", ut), ("U_POP", up), ("B_TW", bt), ("B_POP", bp)):
        v[f"SR_{key}_MIN"], v[f"SR_{key}_MAX"] = f3(min(x)), f3(max(x))
    pops = json.loads((RES / "dev" / "policy_population_single.json").read_text())["params"]
    box = dict((n, (lo, hi)) for n, lo, hi, _ in study.POLICY_BOX["single"].spec)
    assert np.isclose(pops["thr"], box["thr"][0]) and np.isclose(pops["ramp"], box["ramp"][0])
    v["POP_SING_THR"], v["POP_SING_A"], v["POP_SING_RAMP"] = f1(pops["thr"]), f1(pops["a"]), f1(pops["ramp"])
    b_s = max(evs["test"][(evs["test"].family == "single") & (evs["test"].method == "direct")].budget)
    v["T_SING_REF"] = f3(med("test", "single", "direct", b_s))
    v["R_DUAL_MEAN_POP"], v["R_DUAL_MEAN_TW"] = f3(mean("stress", "dual", "population")), f3(mean("stress", "dual", "twin"))
    # held-out and development shares of patients better than the population policy
    held = [paired(co).loc[(fam, "twin", "population")].frac_better for co in ("test", "synth", "stress")
            for fam in ("continuous", "dual")]
    v["BETTER_MIN"], v["BETTER_MAX"] = pct(min(held)), pct(max(held))
    sr = [paired(co).loc[(fam, "twin", "population")].frac_better for co in ("synth", "stress")
          for fam in ("continuous", "dual")]
    v["SR_BETTER_MIN"], v["SR_BETTER_MAX"] = pct(min(sr)), pct(max(sr))
    v["D_CONT_BETTER"] = pct(paired("dev").loc[("continuous", "twin", "population")].frac_better)
    v["D_DUAL_BETTER"] = pct(paired("dev").loc[("dual", "twin", "population")].frac_better)
    # checked twin
    kept, worse = [], []
    for co in ("dev", "test", "synth", "stress"):
        for fam in ("continuous", "single", "dual"):
            e = evs[co][evs[co].family == fam].pivot_table(index="pid", columns="method", values="J")
            kept.append(100 * np.mean(e.checked == e.twin))
            worse.append(100 * np.mean(e.checked > e.population))
    v["CHK_KEPT_MIN"], v["CHK_KEPT_MAX"] = r0(min(kept)), r0(max(kept))
    v["CHK_WORSE_PCT"] = f"{np.ceil(max(worse)):.0f}"
    # failures: cost more than FAIL_MARGIN above the population policy of the same family
    tw_fail, chk_fail = [], []
    for co in ("dev", "test", "synth", "stress"):
        for fam in ("continuous", "single", "dual"):
            e = evs[co][evs[co].family == fam].pivot_table(index="pid", columns="method", values="J")
            tw_fail.append(100 * np.mean(e.twin > e.population + FAIL_MARGIN))
            chk_fail.append(100 * np.mean(e.checked > e.population + FAIL_MARGIN))
    v["FAIL_MARGIN"] = f"{FAIL_MARGIN:g}"
    v["TW_FAIL_MIN"], v["TW_FAIL_MAX"] = r0(min(tw_fail)), r0(max(tw_fail))
    v["CHK_FAIL_MAX"] = upper(max(chk_fail), 2)
    e = evs["stress"][evs["stress"].family != "none"].pivot_table(index=["family", "pid"], columns="method", values="J")
    assert (e.twin > e.population + FAIL_MARGIN).any() and not (e.checked > e.population + FAIL_MARGIN).any()
    for co, C in (("test", "T"), ("synth", "S"), ("stress", "R")):
        v[f"{C}_DUAL_MEAN_CHK"] = f3(mean(co, "dual", "checked"))
        v[f"{C}_DUAL_MEAN_POPX"] = f3(mean(co, "dual", "population"))
    # patient trials
    bud = pd.read_csv(RES / "analysis" / "test_budget.csv").set_index("family")
    v["T_DUAL_MATCH"] = str(int(bud.first_direct_budget_at_or_below_twin["dual"]))
    v["T_DUAL_D10"] = f3(med("test", "dual", "direct", 10))
    v["S_DUAL_D10"], v["R_DUAL_D10"] = f3(med("synth", "dual", "direct", 10)), f3(med("stress", "dual", "direct", 10))
    v["S_DUAL_MEAN_D10"], v["S_DUAL_MEAN_TW"] = f3(mean("synth", "dual", "direct", 10)), f3(mean("synth", "dual", "twin"))
    pt = paired("test")
    for fam, b, K in (("dual", 10, "T_DUAL_VS10"), ("dual", 20, "T_DUAL_VS20"), ("continuous", 5, "T_CONT_VS5"),
                      ("continuous", 10, "T_CONT_VS10")):
        r = pt.loc[(fam, "twin", f"direct_{b}")]
        v[f"{K}_DIFF"], v[f"{K}_P"] = sgn(-r.median_diff), pfmt(r.p_holm)
        if K in ("T_DUAL_VS10", "T_CONT_VS5"):
            v[f"{K}_CILO"], v[f"{K}_CIHI"] = sgn(-r.ci_hi), sgn(-r.ci_lo)
    for co, C in (("test", "T"), ("synth", "S"), ("stress", "R")):
        v[f"{C}_DUAL_VS60"] = f3(-paired(co).loc[("dual", "twin", "direct_60")].median_diff)
        r = paired(co).loc[("dual", "twin", "population")]
        v[f"{C}_DUAL_SHARE"] = f"{100 * r.gain_share:.0f}"
        v[f"{C}_CHK_SHARE"] = f"{100 * paired(co).loc[('dual', 'checked', 'population')].gain_share:.0f}"
    assert float(v.pop("R_DUAL_SHARE")) < 0  # reported as "none"
    vs60 = [float(v[f"{C}_DUAL_VS60"]) for C in ("T", "S", "R")]
    v["DUAL_VS60_MIN"], v["DUAL_VS60_MAX"] = f3(min(vs60)), f3(max(vs60))
    v["T_CONT_SHARE"] = f"{100 * pt.loc[('continuous', 'twin', 'population')].gain_share:.0f}"
    # calibration length
    se = pd.read_csv(RES / "analysis" / "sensitivity.csv")
    g = lambda panel, co, cfg: se[(se.panel == panel) & (se.cohort == co) & (se.config == cfg)]["median"].iloc[0]
    for co, K in (("test_rest30", "30"), ("test", "60"), ("test_rest120", "120")):
        v[f"CL_TW_{K}"] = f3(g("rest", co, "twin"))
    v["CL_CHK_120"] = f3(g("rest", "test_rest120", "checked"))
    v["CL_POP"] = f3(g("rest", "test", "population"))
    r = paired("test_rest120").loc[("dual", "twin", "population")]
    v["CL120_BETTER"], v["CL120_DIFF"], v["CL120_P"] = pct(r.frac_better), sgn(r.median_diff), pfmt(r.p_holm)
    # disease progression: paired comparisons within each cohort, Holm over the three comparisons
    for co, C in (("dev", "D"), ("test", "T")):
        for cfg, K in (("none", "NONE"), ("stale", "STALE"), ("stale_recalibrated", "RECAL"), ("updated", "UPD"),
                       ("population", "POP")):
            v[f"DR_{C}_{K}"] = f3(g("drift", co, cfg))
        de = pd.read_csv(RES / co / "drift_evaluation.csv").pivot(index="pid", columns="config", values="J")
        comps = [("UPD_VS_STALE", "updated", "stale"), ("UPD_VS_POP", "updated", "population"),
                 ("STALE_VS_POP", "stale", "population")]
        res = [paired_compare(de[a].values, de[b].values) for _, a, b in comps]
        adj = holm(np.array([r["p"] for r in res]))
        for (K, _, _), r, p in zip(comps, res, adj):
            v[f"DR_{C}_{K}"] = pct(r["frac_better"])
            v[f"DR_{C}_{K}_P"] = pfmt(p)
            if K != "UPD_VS_STALE":
                v[f"DR_{C}_{K}_DIFF"] = sgn(r["median_diff"])
        v[f"DR_{C}_MEAN_UPD"], v[f"DR_{C}_MEAN_POP"] = f3(de.updated.mean()), f3(de.population.mean())
    upd = [float(v[f"DR_{c}_UPD_VS_STALE"]) for c in ("D", "T")]
    assert all(60 <= x <= 72 for x in upd), upd
    a, b = v.pop("DR_D_UPD_VS_STALE"), v.pop("DR_T_UPD_VS_STALE")
    v["DR_UPD_VS_STALE_PHRASE"] = (f"{a}\\% of patients in each cohort" if a == b else
                                   f"{a}\\% and {b}\\% of patients in the two cohorts")
    v["TAB_OUTCOMES"] = outcome_table()
    return v


SEVERITY_ROWS = [
    ("vee", r"$\nu_{ee}$", "Intracortical excitation"), ("vei", r"$\nu_{ei}$", "Intracortical inhibition"),
    ("vd1e", r"$\nu_{d_1e}$", "Cortex to D1 striatum"), ("vd2e", r"$\nu_{d_2e}$", "Cortex to D2 striatum"),
    ("vp2d2", r"$\nu_{p_2d_2}$", "D2 striatum to GPe"), ("vp2p2", r"$\nu_{p_2p_2}$", "Intrapallidal (GPe)"),
    ("vsp1", r"$\nu_{tp_1}$", "GPi/SNr to relay nuclei"), ("tp2", r"$\theta_{p_2}$", "GPe threshold"),
    ("tz", r"$\theta_{\zeta}$", "STN threshold"), ("phin", r"$\phi_n$", "Brainstem input"),
]


def supplement_values():
    from twin.model import HEALTHY, PARKINSONIAN, BETA_20HZ
    changed = [k for k in HEALTHY if not (HEALTHY[k] == PARKINSONIAN[k] == BETA_20HZ[k])]
    assert sorted(changed) == sorted(k for k, _, _ in SEVERITY_ROWS)
    rows = [f"{sym} & {txt} & {sgn(HEALTHY[k], 2)} & {sgn(PARKINSONIAN[k], 2)} & {sgn(BETA_20HZ[k], 2)} \\\\"
            for k, sym, txt in SEVERITY_ROWS]
    v = {"TAB_SEVERITY": "\n".join(rows)}
    ev = load_eval("dev")
    no = ev[ev.family == "none"]
    lines = [f"\\multicolumn{{2}}{{l}}{{No stimulation}} & {no.J.median():.3f} & {no.J.mean():.3f} & "
             f"{no.B.median():.3f} & 0.000 & \\\\"]
    for fam in ("continuous", "single", "dual"):
        lines.append("\\midrule")
        for k, (lab, meth) in enumerate((("Population", "population"), ("Digital twin", "twin"),
                                         ("Checked twin", "checked"))):
            x = ev[(ev.family == fam) & (ev.method == meth)]
            c = cell(ev, fam, meth, -1)
            better = "" if meth == "population" else r0(c["better"])
            head = FAMILY_NAME[fam] if k == 0 else ""
            lines.append(f"{head} & {lab} & {c['J']:.3f} & {x.J.mean():.3f} & {c['B']:.3f} & {c['U']:.3f} & {better} \\\\")
    v["TAB_DEV"] = "\n".join(lines)
    sr = pd.read_csv(RES / "model" / "single_ramp.csv")
    ratio = sr.B_single / sr.B_none
    fast, slow = sr.ramp == sr.ramp.max(), sr.ramp <= 3.0
    assert (ratio[fast] > 1).all()
    v["RAMP_R40_MIN"], v["RAMP_R40_MAX"] = f2(ratio[fast].min()), f2(ratio[fast].max())
    v["RAMP_PEAK40"] = f1(sr.peak_envelope[fast].max())
    v["RAMP_R3_MAX"] = upper(ratio[slow].max(), 2)
    return v


def compute_all():
    vals = {}
    for part in (methods_values(), anchor_values(), twin_values(), results_values(), supplement_values()):
        assert not set(part) & set(vals)
        vals.update(part)
    return vals


def main():
    vals = compute_all()
    (RES / "analysis" / "reported_values.json").write_text(json.dumps(vals, indent=1, sort_keys=True))
    print(f"{len(vals)} values")


if __name__ == "__main__":
    main()
