"""Mean-field basal ganglia-thalamocortical model of van Albada and Robinson
(J. Theor. Biol. 257, 2009, Papers I and II), spatially uniform form.

Populations: cortical excitatory (e) and inhibitory (i), striatal D1 (d1) and
D2 (d2), GPi/SNr (p1), GPe (p2), STN (z), thalamic relay (s), thalamic
reticular nucleus (r); brainstem input (n) drives the relay nuclei.

    Q_a = Qmax_a / (1 + exp(-(V_a - theta_a) / sigma'))
    (1/(alpha beta)) V_a'' + (1/alpha + 1/beta) V_a' + V_a
        = sum_b nu_ab phi_b(t - tau_ab) + stimulation input
    phi_a = Q_a for every population except e, whose field obeys
    (1/gamma^2) phi_e'' + (2/gamma) phi_e' + phi_e = Q_e.

With random intracortical connectivity (nu_ie = nu_ee, nu_ii = nu_ei,
nu_is = nu_es) and the same delay from the relay nuclei, maximum firing
rate and threshold for both cortical populations, the inhibitory potential
equals the excitatory one, so phi_i = Q_e and the i population is not
integrated separately.
All parameters may be column arrays so that many model instances (virtual
patients) are simulated at once."""
import numpy as np

POPS = ("e", "d1", "d2", "p1", "p2", "z", "s", "r")
IDX = {a: k for k, a in enumerate(POPS)}

HEALTHY = dict(
    alpha=160.0, beta=640.0, gamma=125.0, sigma=3.8,
    Qe=300.0, Qd1=65.0, Qd2=65.0, Qp1=250.0, Qp2=300.0, Qz=500.0, Qs=300.0, Qr=500.0,
    te=14.0, td1=19.0, td2=19.0, tp1=10.0, tp2=9.0, tz=10.0, ts=13.0, tr=13.0,
    vee=1.6, vei=-1.9, ves=0.4,
    vd1e=1.0, vd1d1=-0.3, vd1s=0.1,
    vd2e=0.7, vd2d2=-0.3, vd2s=0.05,
    vp1d1=-0.1, vp1p2=-0.03, vp1z=0.3,
    vp2d2=-0.3, vp2p2=-0.1, vp2z=0.3,
    vze=0.1, vzp2=-0.04,
    vse=0.8, vsp1=-0.03, vsr=-0.4, vsn=0.5,
    vre=0.15, vrs=0.03,
    phin=10.0,
)
"""Healthy parameters of Paper I (Table 2 and Sec. 3.2), with the connection
strengths as tabulated by Quan et al. (2024, Table I) and sigma' = 3.8 mV as
in the published code of Grado et al. (2018); units s^-1, mV and mV s. Only
the product vsn * phin is constrained by the published rates."""

DELAYS_MS = dict(es=35, d1e=2, d2e=2, d1s=2, d2s=2, p1d1=1, p1p2=1, p1z=1, p2d2=1, p2z=1,
                 ze=1, zp2=1, se=50, re=50, sp1=3, sr=2, rs=2)
"""Axonal delays in ms (Paper I, Table 2); intracortical, self and brainstem
projections have no delay."""

DELAYED = [("e", "s", "ves", "es"),
           ("d1", "e", "vd1e", "d1e"), ("d1", "s", "vd1s", "d1s"),
           ("d2", "e", "vd2e", "d2e"), ("d2", "s", "vd2s", "d2s"),
           ("p1", "d1", "vp1d1", "p1d1"), ("p1", "p2", "vp1p2", "p1p2"), ("p1", "z", "vp1z", "p1z"),
           ("p2", "d2", "vp2d2", "p2d2"), ("p2", "z", "vp2z", "p2z"),
           ("z", "e", "vze", "ze"), ("z", "p2", "vzp2", "zp2"),
           ("s", "e", "vse", "se"), ("s", "p1", "vsp1", "sp1"), ("s", "r", "vsr", "sr"),
           ("r", "e", "vre", "re"), ("r", "s", "vrs", "rs")]
"""Delayed projections (target, source, strength key, delay key); the
cortical source e is the propagated field phi_e. Intracortical, self and
brainstem terms act without delay."""

CT_KEYS = ("es", "se", "re")
"""Delays of the corticothalamic loop (thalamocortical es, corticothalamic se
and re), multiplied by the parameter ct_scale (default 1), which sets the
frequency of the beta mode."""


def delay_s(key, q):
    """Delay of projection key in seconds for parameters q (scalar or array)."""
    d = DELAYS_MS.get(key, 0) * 1e-3
    return d * q.get("ct_scale", 1.0) if key in CT_KEYS else d

PARKINSONIAN = dict(HEALTHY, vd1e=0.5, vd2e=1.4, vp2p2=-0.07, vee=1.4, vei=-1.6, vp2d2=-0.5, tp2=8.0, tz=9.0)
"""Full parkinsonian steady state (Paper I, Sec. 3.4, state V)."""

BETA_20HZ = dict(HEALTHY, tp2=7.0, tz=6.0, vee=0.6, vei=-0.8, vd1e=0.5, vd2e=1.4, vp2d2=-0.4, vp2p2=0.0,
                 vsp1=-0.04, phin=20.0)
"""20-Hz limit cycle (Paper II, Fig. 8(c))."""

THETA_5HZ = dict(HEALTHY, tp2=4.0, tz=5.0, vee=1.2, vei=-1.3, vd2e=1.7, vp2d2=-0.5, vp2p2=0.0, vsp1=-0.07,
                 vse=0.6, phin=30.0)
HYPERDIRECT_6HZ = dict(HEALTHY, tp2=4.0, tz=20.0, vd2e=0.3, vze=0.7, vsp1=-0.15, phin=30.0)
COMBINED = dict(HEALTHY, tp2=5.0, tz=4.0, vee=0.5, vei=-0.7, vd1e=0.5, vd2e=1.4, vp2d2=-0.4, vp2p2=0.0,
                vsp1=-0.07, phin=30.0)
"""Other limit cycles of Paper II, Fig. 8 (a) theta, (b) hyperdirect, (d) combined."""


def columns(p, B):
    """Broadcast a parameter dict to column arrays of length B."""
    return {k: np.broadcast_to(np.asarray(v, float), (B,)).copy() for k, v in p.items()}


def sigmoid(v, qmax, theta, sigma):
    return qmax / (1.0 + np.exp(-(v - theta) / sigma))


def inv_sigmoid(phi, qmax, theta, sigma):
    return theta + sigma * np.log(phi / (qmax - phi))


class Model:
    """Batch integrator (Heun's method, fixed step, ring-buffer delays)."""

    def __init__(self, p, B=None, dt=1e-4):
        if B is None:
            B = max(np.size(v) for v in p.values())
        self.B, self.dt = B, dt
        self.q = q = columns(p, B)
        self.ab, self.apb, self.g = q["alpha"] * q["beta"], q["alpha"] + q["beta"], q["gamma"]
        self.Qm = np.stack([q["Q" + a] for a in POPS])
        self.th = np.stack([q["t" + a] for a in POPS])
        self.sg = q["sigma"]
        self.lag = {}
        for k in DELAYS_MS:
            lag = np.rint(np.broadcast_to(delay_s(k, q), (B,)) / dt).astype(int)
            self.lag[k] = int(lag[0]) if np.all(lag == lag[0]) else lag
        self.L = max(int(np.max(v)) for v in self.lag.values()) + 1
        self.cols = np.arange(B)
        # delayed connections grouped by delay: weight tensors (8, 8, B)
        self.groups, self.where = [], {}
        for a, b, key, dkey in DELAYED:
            lag = self.lag[dkey]
            for gi, (glag, _) in enumerate(self.groups):
                if (isinstance(glag, int) and isinstance(lag, int) and glag == lag) or \
                        (not isinstance(glag, int) and not isinstance(lag, int) and np.array_equal(glag, lag)):
                    break
            else:
                self.groups.append((lag, np.zeros((8, 8, B))))
                gi = len(self.groups) - 1
            self.groups[gi][1][IDX[a], IDX[b]] = q[key]
            self.where.setdefault(key, []).append((gi, IDX[a], IDX[b]))

    def set_params(self, upd):
        """Replace parameters (column arrays), e.g. when severity changes."""
        for k, v in upd.items():
            self.q[k] = np.broadcast_to(np.asarray(v, float), (self.B,)).copy()
            if k.startswith("t") and k[1:] in IDX:
                self.th[IDX[k[1:]]] = self.q[k]
            for gi, a, b in self.where.get(k, ()):
                self.groups[gi][1][a, b] = self.q[k]
        self._cache_k = None

    def reset(self, phi0):
        """Start at rates phi0 (8, B), zero derivatives, constant history."""
        self.V = inv_sigmoid(phi0, self.Qm, self.th, self.sg)
        self.dV = np.zeros_like(self.V)
        self.phie, self.dphie = phi0[0].copy(), np.zeros(self.B)
        self.hist = np.repeat(phi0[None].astype(float), self.L, axis=0)
        self.k = 0
        self._cache_k = None

    def _delayed(self, k):
        """Sum of delayed inputs at step k (depends only on the history)."""
        L = self.L
        u = np.zeros((8, self.B))
        for lag, N in self.groups:
            if isinstance(lag, int):
                H = self.hist[(k - lag) % L]
            else:
                H = self.hist[(k - lag) % L, :, self.cols].T
            u += np.einsum("abj,bj->aj", N, H)
        return u

    def _input(self, V, phie, k, phin_t, stim):
        q = self.q
        Q = sigmoid(V, self.Qm, self.th, self.sg)
        if self._cache_k != k:
            self._cache_u, self._cache_k = self._delayed(k), k
        u = self._cache_u.copy()
        u[0] += q["vee"] * phie + q["vei"] * Q[0]
        u[1] += q["vd1d1"] * Q[1]
        u[2] += q["vd2d2"] * Q[2]
        u[4] += q["vp2p2"] * Q[4]
        u[6] += q["vsn"] * phin_t
        if stim is not None:
            u[5] += stim[0]
            u[3] += stim[1]
            u[4] += stim[2]
        return u, Q

    def step(self, phin_t, stim=None):
        """Advance one step. stim = (STN, GPi/SNr, GPe) input terms in mV."""
        dt, ab, apb, g = self.dt, self.ab, self.apb, self.g
        V, dV, phie, dphie, k = self.V, self.dV, self.phie, self.dphie, self.k
        u, Q = self._input(V, phie, k, phin_t, stim)
        aV = ab * (u - V) - apb * dV
        ap = g * g * (Q[0] - phie) - 2.0 * g * dphie
        V1, dV1, p1, dp1 = V + dt * dV, dV + dt * aV, phie + dt * dphie, dphie + dt * ap
        u1, Q1 = self._input(V1, p1, k + 1, phin_t, stim)
        aV1 = ab * (u1 - V1) - apb * dV1
        ap1 = g * g * (Q1[0] - p1) - 2.0 * g * dp1
        self.V = V + 0.5 * dt * (dV + dV1)
        self.dV = dV + 0.5 * dt * (aV + aV1)
        self.phie = phie + 0.5 * dt * (dphie + dp1)
        self.dphie = dphie + 0.5 * dt * (ap + ap1)
        self.k = k + 1
        Qn = sigmoid(self.V, self.Qm, self.th, self.sg)
        Qn[0] = self.phie
        self.hist[self.k % self.L] = Qn
        return Qn


def simulate(p, T, dt=1e-4, B=None, phi0=None, noise_sd=0.0, noise_hold=1e-3, seed=0, stim=None,
             record_every=10, record=("z",), burn=0.0):
    """Run a batch for T seconds. noise_sd (s^-1) is the standard deviation of
    the brainstem input around phin, independent per column and held for
    noise_hold s. stim(t) returns the (STN, GPi/SNr, GPe) input terms.
    record lists populations (firing rates) or "V<pop>" for the mean
    membrane potential of a population (e.g. "Vz" for the STN, used as the
    local field potential proxy).
    Returns t (n,) and rec[name] (n, B) after burn."""
    from .linear import fixed_point
    m = Model(p, B, dt)
    if phi0 is None:
        phi0 = fixed_point(p, m.B)
    m.reset(phi0)
    rng = np.random.default_rng(seed)
    hold = max(1, int(round(noise_hold / dt)))
    n_steps, n_burn = int(round(T / dt)), int(round(burn / dt))
    t_rec, out = [], {a: [] for a in record}
    noise = np.zeros(m.B)
    for n in range(n_steps):
        if noise_sd > 0 and n % hold == 0:
            noise = noise_sd * rng.standard_normal(m.B)
        Qn = m.step(m.q["phin"] + noise, None if stim is None else stim(n * dt))
        if n + 1 > n_burn and (n + 1 - n_burn) % record_every == 0:
            t_rec.append((n + 1) * dt)
            for a in record:
                out[a].append(m.V[IDX[a[1:]]].copy() if a.startswith("V") else Qn[IDX[a]].copy())
    return np.array(t_rec), {a: np.array(v) for a, v in out.items()}
