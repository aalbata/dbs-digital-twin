"""Calibration protocol, digital-twin identification and policy
personalization.

Calibration (per patient, as in a clinical review): 60 s of rest OFF
medication, a stimulation probe run (20 s rest, then 20 s at each of three
amplitudes) and 60 s of rest ON medication, all observed through the sensed
signal only.

Digital twin: a model instance with the delay scale c, OFF and ON severities
and stimulation efficacy g of the patient estimated from the calibration
features; the other hidden traits (STN threshold offset, noise level,
background level, STN-GPe weights) stay at their nominal values. Identification
has two steps: (1) c and severities from the rest spectra with the anchoring
library (twin.anchor), (2) joint refinement of a common severity correction
(added to the OFF and ON severities) and g by
Gaussian-process Bayesian optimization, matching the rest peak height and the
three probe responses of simulated twin calibrations to the recorded ones.

Personalization: Bayesian optimization of the policy parameters on the twin
(scenario cost), deployment on the patient."""
import numpy as np
import pandas as pd
from scipy.signal import sosfilt

from . import closedloop as CL
from .anchor import FREQS, KAPPA, fit_features
from .bo import Box, BatchBO
from .spectra import psd, features

PROBE_U = (2.5, 5.0, 7.5)
PROBE_REST, PROBE_STEP, PROBE_SETTLE = 20.0, 20.0, 5.0
REST_T = 60.0
TWIN_BOX = Box([("ds", -0.3, 0.3, "lin"), ("g", 0.3, 3.0, "log")])
POLICY_BOX = {
    "continuous": Box([("a", 0.0, CL.U_MAX, "lin")]),
    "single": Box([("thr", 0.3, 2.5, "lin"), ("a", 1.0, CL.U_MAX, "lin"), ("ramp", 0.5, 40.0, "log")]),
    "dual": Box([("lo", 0.3, 2.5, "lin"), ("gap", 0.05, 1.5, "lin"), ("floor", 0.0, 1.0, "lin"),
                 ("a_max", 1.0, CL.U_MAX, "lin"), ("ramp", 0.5, 40.0, "log")]),
}
N_INIT = {1: 3, 2: 4, 3: 6, 5: 8}
"""Initial Sobol design size by search dimension."""
NOMINAL = dict(dtz=0.0, noise=2.0, kappa=KAPPA, m_p2z=1.0, m_zp2=1.0)


def envelope(lfp, f_peak, fs=CL.FS):
    """Causal sensing envelope of recorded signals (rows), as in the device."""
    out = np.empty_like(lfp, dtype=float)
    sos = CL._bandpass(f_peak - 3.0, f_peak + 3.0)
    a = 1.0 / (fs * CL.TAU_ENV)
    for j in range(lfp.shape[0]):
        y = np.abs(sosfilt(np.ascontiguousarray(sos[..., j]), lfp[j].astype(float))) * (0.5 * np.pi)
        out[j] = sosfilt(np.array([[a, 0.0, 0.0, 1.0, a - 1.0, 0.0]]), y)
    return out


def rest_features(lfp, fs=CL.FS, skip=5.0):
    x = lfp[:, int(skip * fs):].astype(float)
    f, P = psd(x - x.mean(1, keepdims=True), fs)
    return pd.DataFrame(features(f, P))


def probe_command(B):
    edges = PROBE_REST + PROBE_STEP * np.arange(len(PROBE_U) + 1)

    def cmd(t):
        k = np.searchsorted(edges, t, side="right") - 1
        return np.full(B, 0.0 if k < 0 or k >= len(PROBE_U) else PROBE_U[k])
    return cmd, PROBE_REST + PROBE_STEP * len(PROBE_U)


def probe_ratios(env, fs=CL.FS):
    """Mean envelope in the settled part of each probe step relative to the
    settled rest segment."""
    def seg(a, b):
        return env[:, int(a * fs):int(b * fs)].mean(1)
    base = seg(PROBE_SETTLE, PROBE_REST)
    r = [seg(PROBE_REST + k * PROBE_STEP + PROBE_SETTLE, PROBE_REST + (k + 1) * PROBE_STEP) / base
         for k in range(len(PROBE_U))]
    return np.stack(r, axis=1)


def calibrate(pt, chi, ref20, seed):
    """Run the calibration protocol on patients pt. Returns a DataFrame of
    calibration features and the calib dict used by the sensing chain."""
    B = len(pt)
    phi0 = CL.initial_state(pt, pt.s_off.values)
    c0 = dict(f_peak=np.full(B, 20.0), b_cal=np.ones(B), chi=np.full(B, chi), phi0=phi0)
    zero = lambda t: np.zeros(B)
    rest = CL.run(pt, None, "off", c0, seed, 1.0, T=REST_T, lib_ref20=ref20, probe=zero, record_lfp=True)
    ft = rest_features(rest["lfp"])
    f_peak = ft.peak_hz.values
    env = envelope(rest["lfp"], f_peak)
    b_cal = np.median(env[:, int(5 * CL.FS):], axis=1)
    calib = dict(f_peak=f_peak, b_cal=b_cal, chi=np.full(B, chi), phi0=phi0)
    cmd, T = probe_command(B)
    pr = CL.run(pt, None, "off", calib, seed + 1, 1.0, T=T, lib_ref20=ref20, probe=cmd, record_lfp=True)
    ratios = probe_ratios(envelope(pr["lfp"], f_peak))
    on = CL.run(pt, None, "on", dict(calib, phi0=CL.initial_state(pt, pt.s_on.values)), seed + 2, 1.0, T=REST_T,
                lib_ref20=ref20, probe=zero, record_lfp=True)
    fo = rest_features(on["lfp"])
    out = pd.DataFrame(dict(pid=pt.pid.values, f_peak=f_peak, b_cal=b_cal, peak_db_off=ft.peak_db.values,
                            peak_db_on=fo.peak_db.values, b_true_off=rest["be_mean"]))
    for k in range(len(PROBE_U)):
        out[f"r{k}"] = ratios[:, k]
    return out, calib


def twin_initial(cal, lib, table):
    """Step 1: delay scale and severities from the rest features."""
    rows = []
    for _, r in cal.iterrows():
        fo = fit_features(r.f_peak, r.peak_db_off, lib, table)
        fn = fit_features(r.f_peak, r.peak_db_on, lib, table, c_fixed=fo["c"])
        rows.append(dict(c=fo["c"], s_off=fo["s"], s_on=fn["s"]))
    return pd.DataFrame(rows)


def twin_table(pid, c, s_off, s_on, g):
    df = pd.DataFrame(dict(pid=pid, c=c, s_off=s_off, s_on=s_on, g=g))
    for k, v in NOMINAL.items():
        df[k] = v
    return df


def twin_refine(cal, init, chi, ref20, n_iter, seed, log=None):
    """Step 2: Bayesian optimization of (common severity shift, g) against the
    rest peak height and probe responses of simulated twin calibrations."""
    B = len(cal)
    bo = BatchBO(TWIN_BOX, B, 4, seed)
    target = np.log(cal[[f"r{k}" for k in range(len(PROBE_U))]].values.clip(1e-3))
    cmd, T = probe_command(B)
    for it in range(n_iter):
        Z = bo.ask()
        prm = TWIN_BOX.to_params(Z)
        s_off = init.s_off.values + prm["ds"]
        tw = twin_table(cal.pid.values, init.c.values, s_off, init.s_on.values + prm["ds"], prm["g"])
        calib = dict(f_peak=cal.f_peak.values, b_cal=cal.b_cal.values, chi=np.full(B, chi),
                     phi0=CL.initial_state(tw, s_off))
        pr = CL.run(tw, None, "off", calib, seed + 100 + it, 1.0, T=T, lib_ref20=ref20, probe=cmd, record_lfp=True)
        ratios = np.log(probe_ratios(envelope(pr["lfp"], cal.f_peak.values)).clip(1e-3))
        ft = rest_features(pr["lfp"][:, :int(PROBE_REST * CL.FS)])
        err = ((ft.peak_db.values - cal.peak_db_off.values) / 2.0) ** 2 + ((ratios - target) ** 2).sum(1)
        bo.tell(Z, err)
        if log:
            log(f"twin iteration {it + 1}/{n_iter}: median error {np.median(err):.3f}")
    prm = TWIN_BOX.to_params(bo.recommend())
    return twin_table(cal.pid.values, init.c.values, init.s_off.values + prm["ds"], init.s_on.values + prm["ds"],
                      prm["g"])


def optimize_policy(plant, family, calib, chi, ref20, b_ref, n_iter, seed, lam=0.5, record_at=(), log=None):
    """Bayesian optimization of policy parameters on plant (twin or patient
    table), one optimization per row, each evaluation a medication-cycle run.
    Returns the recommended parameters after n_iter evaluations and, for every
    budget in record_at, the recommendation after that many evaluations."""
    box = POLICY_BOX[family]
    B = len(plant)
    n_init = N_INIT[box.d]
    bo = BatchBO(box, B, n_init, seed)
    hist = []
    for it in range(n_iter):
        Z = bo.ask()
        pol = CL.POLICIES[family](**box.to_params(Z))
        out = CL.run(plant, pol, "cycle", calib, seed + 1000 + it, b_ref, lib_ref20=ref20)
        J = CL.cost(out, lam)
        bo.tell(Z, J)
        hist.append(J)
        if log:
            log(f"{family} iteration {it + 1}/{n_iter}: median J {np.median(J):.4f}")
    recs = {n: box.to_params(bo.recommend(upto=n)) for n in record_at}
    recs[n_iter] = box.to_params(bo.recommend())
    return recs, np.array(hist)
