"""Anchoring virtual patients to recorded STN spectra.

Measurement model: the recorded spectrum is the linear spectrum of the STN
mean membrane potential at corticothalamic delay scale c and severity s
(stable fixed points only) plus an aperiodic recording background,
    P(f) = m(f; c, s) + kappa * m(20 Hz; c, -1) * (f/20)^(-chi).
Every library spectrum with this background is summarized by its beta peak
frequency and height above the aperiodic fit (twin.spectra.features); a
recording is anchored to the library entry with the nearest pair of
features. The ON-medication recording of the same hemisphere keeps c and is
matched for s by peak height."""
import numpy as np

from .cohort import severity_params
from .linear import fixed_point_one, potential_spectrum, rightmost_mode

FREQS = np.round(np.arange(3.0, 45.01, 0.5), 2)


def library_point(args):
    """Linear STN potential spectrum at (c, s); None if the fixed point is
    unstable."""
    c, s = args
    q = {k: float(v[0]) for k, v in severity_params(s).items()}
    q["ct_scale"] = c
    phi = fixed_point_one(q)
    growth, freq = rightmost_mode(q, phi, fmin=2.0, fmax=60.0)
    if not np.isfinite(growth) or growth >= 0:
        return c, s, growth, freq, None
    return c, s, growth, freq, potential_spectrum(q, phi, FREQS)


KAPPA = 3.0
"""Recording background at 20 Hz relative to the healthy-state (s = -1) STN
potential power at the same delay scale; with this level the model peak
height spans the recorded range over the severity axis."""


def background(lib, c, chi, kappa=KAPPA):
    """Aperiodic recording background kappa * ref_c * (f/20)^-chi, where
    ref_c is the healthy-state model power at 20 Hz for delay scale c."""
    sel = np.isclose(lib["c"], c)
    k = np.flatnonzero(sel)[np.argmin(lib["s"][sel])]
    ref = lib["m"][k][np.argmin(np.abs(FREQS - 20.0))]
    return kappa * ref * (FREQS / 20.0) ** (-chi)


def feature_table(lib, chi, kappa=KAPPA):
    """Beta peak frequency and height (dB above the aperiodic fit) of every
    library spectrum with the recording background added."""
    from .spectra import features
    P = np.empty_like(lib["m"])
    for c in np.unique(lib["c"]):
        sel = np.isclose(lib["c"], c)
        P[sel] = lib["m"][sel] + background(lib, c, chi, kappa)
    ft = features(FREQS, P)
    return np.array([x["peak_hz"] for x in ft]), np.array([x["peak_db"] for x in ft])


def fit_features(peak_hz, peak_db, lib, table, c_fixed=None, hz_scale=1.0, db_scale=1.0):
    """Library entry closest to the recorded beta peak: squared distance in
    units of hz_scale (Hz) and db_scale (dB). With c_fixed only the peak
    height is matched along the severity axis of that delay scale."""
    pk, hdb = table
    if c_fixed is None:
        d = ((pk - peak_hz) / hz_scale) ** 2 + ((hdb - peak_db) / db_scale) ** 2
        k = int(np.argmin(d))
    else:
        idx = np.flatnonzero(np.isclose(lib["c"], c_fixed))
        k = int(idx[np.argmin(np.abs(hdb[idx] - peak_db))])
    return dict(c=float(lib["c"][k]), s=float(lib["s"][k]), k=k, model_peak_hz=float(pk[k]),
                model_peak_db=float(hdb[k]))
