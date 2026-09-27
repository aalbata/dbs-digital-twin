"""Steady state, loop gains, linear stability and linear spectra of the
mean-field model (Paper I, Eqs. 7-14; Paper II, Eqs. 7-22).

Linearizing about the low stable fixed point with slopes
rho_a = phi_a (1 - phi_a / Qmax_a) / sigma' gives, for potentials v and
Laplace variable lambda,

    M(lambda) v = nu_sn e_s phi_n,
    M(lambda) = D(lambda) I - A(lambda),
    D(lambda) = (1 + lambda/alpha)(1 + lambda/beta),
    A_ab = nu_ab rho_b exp(-lambda tau_ab), with rho_e divided by
           (1 + lambda/gamma)^2 for the propagated cortical field; the
           intracortical inhibitory term uses rho_e without propagation.

Roots of det M(lambda) = 0 give the stability and frequency of each mode;
the STN rate spectrum for white brainstem input follows from M(i omega)^-1."""
import numpy as np
from scipy.optimize import brentq

from .model import POPS, IDX, sigmoid, columns, delay_s

_CONN = [("e", "e", "vee"), ("e", "s", "ves"),
         ("d1", "e", "vd1e"), ("d1", "d1", "vd1d1"), ("d1", "s", "vd1s"),
         ("d2", "e", "vd2e"), ("d2", "d2", "vd2d2"), ("d2", "s", "vd2s"),
         ("p1", "d1", "vp1d1"), ("p1", "p2", "vp1p2"), ("p1", "z", "vp1z"),
         ("p2", "d2", "vp2d2"), ("p2", "p2", "vp2p2"), ("p2", "z", "vp2z"),
         ("z", "e", "vze"), ("z", "p2", "vzp2"),
         ("s", "e", "vse"), ("s", "p1", "vsp1"), ("s", "r", "vsr"),
         ("r", "e", "vre"), ("r", "s", "vrs")]


def _tau(a, b, q):
    return float(delay_s(a + b, q))


def _rates_given_s(q, phis, stim=(0.0, 0.0, 0.0)):
    S = lambda v, a: sigmoid(v, q["Q" + a], q["t" + a], q["sigma"])
    phie = brentq(lambda x: x - S((q["vee"] + q["vei"]) * x + q["ves"] * phis, "e"), 0.0, q["Qe"])
    d1 = brentq(lambda x: x - S(q["vd1e"] * phie + q["vd1d1"] * x + q["vd1s"] * phis, "d1"), 0.0, q["Qd1"])
    d2 = brentq(lambda x: x - S(q["vd2e"] * phie + q["vd2d2"] * x + q["vd2s"] * phis, "d2"), 0.0, q["Qd2"])

    def f(x):
        z = S(q["vze"] * phie + q["vzp2"] * x + stim[0], "z")
        return x - S(q["vp2d2"] * d2 + q["vp2p2"] * x + q["vp2z"] * z + stim[2], "p2")
    p2 = brentq(f, 0.0, q["Qp2"])
    z = S(q["vze"] * phie + q["vzp2"] * p2 + stim[0], "z")
    p1 = S(q["vp1d1"] * d1 + q["vp1p2"] * p2 + q["vp1z"] * z + stim[1], "p1")
    r = S(q["vre"] * phie + q["vrs"] * phis, "r")
    return np.array([phie, d1, d2, p1, p2, z, phis, r])


def fixed_point_one(q, stim=(0.0, 0.0, 0.0)):
    """Low fixed point of one parameter set (dict of scalars)."""
    def F(phis):
        R = _rates_given_s(q, phis, stim)
        v = q["vse"] * R[0] + q["vsp1"] * R[3] + q["vsr"] * R[7] + q["vsn"] * q["phin"]
        return phis - sigmoid(v, q["Qs"], q["ts"], q["sigma"])
    xs = np.concatenate([np.linspace(1e-6, 5.0, 60), np.linspace(5.0, q["Qs"] - 1e-6, 600)[1:]])
    fs = np.array([F(x) for x in xs])
    k = np.where(np.sign(fs[:-1]) != np.sign(fs[1:]))[0][0]
    return _rates_given_s(q, brentq(F, xs[k], xs[k + 1]), stim)


def fixed_point(p, B=None, stim=(0.0, 0.0, 0.0)):
    """Low fixed point for every column; returns rates (8, B)."""
    if B is None:
        B = max(np.size(v) for v in p.values())
    q = columns(p, B)
    st = [np.broadcast_to(np.asarray(s, float), (B,)) for s in stim]
    return np.stack([fixed_point_one({k: v[j] for k, v in q.items()}, tuple(s[j] for s in st))
                     for j in range(B)], axis=1)


def slopes(q, phi):
    return np.array([phi[IDX[a]] * (1.0 - phi[IDX[a]] / q["Q" + a]) / q["sigma"] for a in POPS])


def char_matrix(q, rho, lam):
    """M(lambda) for one parameter set (scalars) and slopes rho (8,)."""
    D = (1.0 + lam / q["alpha"]) * (1.0 + lam / q["beta"])
    De = (1.0 + lam / q["gamma"]) ** 2
    M = D * np.eye(8, dtype=complex)
    for a, b, key in _CONN:
        r = rho[IDX[b]] / De if b == "e" else rho[IDX[b]]
        M[IDX[a], IDX[b]] -= q[key] * r * np.exp(-lam * _tau(a, b, q))
    M[IDX["e"], IDX["e"]] -= q["vei"] * rho[IDX["e"]]
    return M


def rightmost_mode(q, phi, fmin=2.0, fmax=60.0, stim_rho=None):
    """Least damped oscillatory root in [fmin, fmax] Hz: (growth rate s^-1,
    frequency Hz). Newton iteration on det M from a grid of starting points."""
    rho = slopes(q, phi) if stim_rho is None else stim_rho
    f = lambda lam: np.linalg.det(char_matrix(q, rho, lam))
    roots = []
    for w0 in np.arange(fmin, fmax + 1e-9, 2.0) * 2 * np.pi:
        for s0 in (-40.0, -10.0, 10.0):
            lam = complex(s0, w0)
            for _ in range(60):
                h = 1e-6 * (1.0 + abs(lam))
                fl = f(lam)
                d = (f(lam + h) - f(lam - h)) / (2 * h)
                if d == 0:
                    break
                step = fl / d
                lam -= step
                if abs(step) < 1e-9 * (1 + abs(lam)):
                    break
            if abs(f(lam)) < 1e-6 and fmin * 2 * np.pi <= lam.imag <= fmax * 2 * np.pi:
                roots.append(lam)
    if not roots:
        return np.nan, np.nan
    lam = max(roots, key=lambda z: z.real)
    return float(lam.real), float(lam.imag / (2 * np.pi))


def stn_spectrum(q, phi, freqs, pop="z"):
    """Linear power spectrum (per unit white-noise density of phi_n) of the
    firing rate of population pop."""
    rho = slopes(q, phi)
    out = np.empty(len(freqs))
    e_s = np.zeros(8)
    e_s[IDX["s"]] = q["vsn"]
    for k, fr in enumerate(freqs):
        M = char_matrix(q, rho, 2j * np.pi * fr)
        v = np.linalg.solve(M, e_s)
        resp = v[IDX[pop]] * rho[IDX[pop]]
        if pop == "e":
            resp = v[IDX["e"]] * rho[IDX["e"]] / (1 + 2j * np.pi * fr / q["gamma"]) ** 2
        out[k] = abs(resp) ** 2
    return out


def potential_spectrum(q, phi, freqs, pop="z"):
    """Linear power spectrum of the mean membrane potential of pop per unit
    white-noise density of phi_n."""
    rho = slopes(q, phi)
    e_s = np.zeros(8)
    e_s[IDX["s"]] = q["vsn"]
    out = np.empty(len(freqs))
    for k, fr in enumerate(freqs):
        v = np.linalg.solve(char_matrix(q, rho, 2j * np.pi * fr), e_s)
        out[k] = abs(v[IDX[pop]]) ** 2
    return out


def loop_gains(q, phi):
    """Loop gains of Paper II, Sec. 4.1 (zero-frequency products)."""
    rho = slopes(q, phi)
    G = lambda a, b, key: rho[IDX[a]] * q[key]
    Ges, Gsp1 = G("e", "s", "ves"), G("s", "p1", "vsp1")
    return dict(
        intracortical=rho[0] * (q["vee"] + q["vei"]),
        direct=Ges * Gsp1 * G("p1", "d1", "vp1d1") * G("d1", "e", "vd1e"),
        indirect=Ges * Gsp1 * G("p1", "z", "vp1z") * G("z", "p2", "vzp2") * G("p2", "d2", "vp2d2") * G("d2", "e", "vd2e"),
        hyperdirect=Ges * Gsp1 * G("p1", "z", "vp1z") * G("z", "e", "vze"),
        stn_gpe=G("p2", "z", "vp2z") * G("z", "p2", "vzp2"),
        corticothalamic=Ges * G("s", "e", "vse"),
    )
