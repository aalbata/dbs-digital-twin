"""Severity axis and virtual-patient parameters.

Severity s moves every parameter piecewise linearly along the published
states of the model: the healthy state of Paper I (s = -1), the parkinsonian
steady state of Paper I, Sec. 3.4 (s = 0), and the 20-Hz limit-cycle set of
Paper II, Fig. 8(c) (s = 1). Values above 1 extrapolate along the last
segment."""
import numpy as np

from .model import HEALTHY, PARKINSONIAN, BETA_20HZ


def severity_params(s):
    """Parameter dict (arrays) at severity s (scalar or array)."""
    s = np.atleast_1d(np.asarray(s, float))
    lo = np.clip(s, -1.0, 0.0) + 1.0
    hi = np.clip(s, 0.0, None)
    return {k: HEALTHY[k] + lo * (PARKINSONIAN[k] - HEALTHY[k]) + hi * (BETA_20HZ[k] - PARKINSONIAN[k])
            for k in HEALTHY}
