"""Batched Gaussian-process Bayesian optimization: one independent
minimization per patient, all advanced together so that every iteration needs
a single batched simulation.

Search space: a box in d dimensions (linear or logarithmic per coordinate),
mapped to [0, 1]^d. Initial design: scrambled Sobol points. Surrogate: GP
with a constant times Matern (nu = 2.5, one length scale per dimension) kernel
plus a white-noise kernel, fitted by maximum marginal likelihood on
standardized observations (scikit-learn). Acquisition: expected improvement
over the best posterior mean at the evaluated points, maximized over 1024
random candidates plus 256 local perturbations of the incumbent.
Recommendation: the evaluated point with the lowest posterior mean."""
import warnings

import numpy as np
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


class Box:
    """Search box: list of (name, low, high, scale) with scale 'lin' or 'log'."""

    def __init__(self, spec):
        self.spec = spec
        self.d = len(spec)

    def to_params(self, Z):
        """Map rows of Z in [0, 1]^d to parameter dicts of arrays."""
        Z = np.atleast_2d(Z)
        out = {}
        for j, (name, lo, hi, sc) in enumerate(self.spec):
            z = Z[:, j]
            out[name] = np.exp(np.log(lo) + z * (np.log(hi) - np.log(lo))) if sc == "log" else lo + z * (hi - lo)
        return out


def _gp(d, seed):
    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.full(d, 0.3), (0.02, 5.0), nu=2.5)
              + WhiteKernel(0.05, (1e-6, 1.0)))
    return GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=1, random_state=seed)


class BatchBO:
    def __init__(self, box, n_patients, n_init, seed):
        self.box, self.P, self.seed = box, n_patients, seed
        self.X = [[] for _ in range(n_patients)]
        self.y = [[] for _ in range(n_patients)]
        sob = qmc.Sobol(box.d, scramble=True, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.init = sob.random(max(2, n_init))
        self.rng = np.random.default_rng(seed + 1)
        self.it = 0

    def _fit(self, j, n=None):
        X, y = np.array(self.X[j][:n]), np.array(self.y[j][:n])
        gp = _gp(self.box.d, self.seed + j)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(X, y)
        return gp, X

    def ask(self):
        """Next point (row in [0,1]^d) for every patient."""
        if self.it < len(self.init):
            return np.repeat(self.init[self.it][None], self.P, axis=0)
        Z = np.empty((self.P, self.box.d))
        for j in range(self.P):
            gp, X = self._fit(j)
            mu_obs = gp.predict(X)
            best = mu_obs.min()
            inc = X[np.argmin(mu_obs)]
            cand = np.vstack([self.rng.random((1024, self.box.d)),
                              np.clip(inc + 0.05 * self.rng.standard_normal((256, self.box.d)), 0, 1)])
            mu, sd = gp.predict(cand, return_std=True)
            sd = np.maximum(sd, 1e-9)
            zz = (best - mu) / sd
            ei = (best - mu) * norm.cdf(zz) + sd * norm.pdf(zz)
            Z[j] = cand[np.argmax(ei)]
        return Z

    def tell(self, Z, y):
        for j in range(self.P):
            self.X[j].append(Z[j].copy())
            self.y[j].append(float(y[j]))
        self.it += 1

    def recommend(self, upto=None):
        """Evaluated point with the lowest posterior mean, using the first
        `upto` evaluations of every patient."""
        Z = np.empty((self.P, self.box.d))
        for j in range(self.P):
            if (upto or len(self.X[j])) < 3:
                X, y = np.array(self.X[j][:upto]), np.array(self.y[j][:upto])
                Z[j] = X[np.argmin(y)]
                continue
            gp, X = self._fit(j, upto)
            Z[j] = X[np.argmin(gp.predict(X))]
        return Z
