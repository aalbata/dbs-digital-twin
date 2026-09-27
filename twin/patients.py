"""Virtual patients.

A virtual patient is one STN hemisphere: the anchored delay scale c and the
OFF- and ON-medication severities (twin.cohort), plus hidden traits that a
resting recording does not reveal:
  g      stimulation efficacy (multiplies the stimulation input)
  dtz    offset of the STN firing threshold (mV)
  noise  standard deviation of the brainstem input (s^-1)
  kappa  recording background level (twin.anchor)
and, in the stress set only, scale factors of the STN-GPe weights that the
digital twin keeps at their nominal values.
Cohorts: 'dev' (Dusseldorf-anchored), 'test' (Madrid-anchored), 'synth'
(smoothed bootstrap of the development hemispheres) and 'stress' (as synth,
wider hidden traits and STN-GPe mismatch)."""
from pathlib import Path

import numpy as np
import pandas as pd

from .anchor import KAPPA
from .cohort import severity_params

SEEDS = {"dev": 20260926, "test": 20260927, "synth": 20260928, "stress": 20260929}
HIDDEN = {"normal": dict(g_sd=0.3, dtz=1.0, noise=(1.5, 2.5), kappa_ln=0.7, mismatch=0.0),
          "stress": dict(g_sd=0.5, dtz=2.0, noise=(1.0, 3.0), kappa_ln=1.0, mismatch=0.2)}
JITTER = dict(c=0.05, s=0.1, d=0.1)
N_SYNTH = {"synth": 400, "stress": 200}


def _hidden(n, rng, kind):
    h = HIDDEN[kind]
    return pd.DataFrame(dict(
        g=np.exp(h["g_sd"] * rng.standard_normal(n)),
        dtz=rng.uniform(-h["dtz"], h["dtz"], n),
        noise=rng.uniform(*h["noise"], n),
        kappa=KAPPA * np.exp(rng.uniform(-h["kappa_ln"], h["kappa_ln"], n)),
        m_p2z=1.0 + rng.uniform(-h["mismatch"], h["mismatch"], n),
        m_zp2=1.0 + rng.uniform(-h["mismatch"], h["mismatch"], n)))


def build(anchor_csv, cohort):
    """Patient table for a cohort (deterministic given the seed)."""
    a = pd.read_csv(anchor_csv)
    rng = np.random.default_rng(SEEDS[cohort])
    dev = a[a.cohort == "dusseldorf"].reset_index(drop=True)
    if cohort in ("dev", "test"):
        base = dev if cohort == "dev" else a[a.cohort == "madrid"].reset_index(drop=True)
        paired = base.dropna(subset=["s_on"])
        delta = float(np.median(paired.s_off - paired.s_on))
        df = pd.DataFrame(dict(pid=[f"{cohort}{k:03d}" for k in range(len(base))],
                               source=base.patient + "_" + base.hemisphere, c=base.c, s_off=base.s_off,
                               s_on=base.s_on.fillna(base.s_off - delta), s_on_imputed=base.s_on.isna()))
        hid = _hidden(len(df), rng, "normal")
    else:
        n = N_SYNTH[cohort]
        pick = rng.integers(0, len(dev), n)
        d_on = (dev.s_off - dev.s_on.fillna(dev.s_off)).values
        c = np.clip(dev.c.values[pick] + JITTER["c"] * rng.standard_normal(n), 0.5, 1.8)
        s_off = np.clip(dev.s_off.values[pick] + JITTER["s"] * rng.standard_normal(n), -1.0, 1.1)
        s_on = np.clip(s_off - (d_on[pick] + JITTER["d"] * rng.standard_normal(n)), -1.0, 1.1)
        df = pd.DataFrame(dict(pid=[f"{cohort}{k:03d}" for k in range(n)], source=dev.patient.values[pick] + "_" +
                               dev.hemisphere.values[pick], c=np.round(c, 3), s_off=s_off, s_on=s_on,
                               s_on_imputed=False))
        hid = _hidden(n, rng, "stress" if cohort == "stress" else "normal")
    return pd.concat([df.reset_index(drop=True), hid], axis=1)


def params(pt, s):
    """Model parameter dict (column arrays) for patients pt at severity s."""
    p = severity_params(s)
    p["ct_scale"] = pt.c.values
    p["tz"] = p["tz"] + pt.dtz.values
    p["vp2z"] = p["vp2z"] * pt.m_p2z.values
    p["vzp2"] = p["vzp2"] * pt.m_zp2.values
    return p
