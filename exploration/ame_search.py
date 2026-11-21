"""
Numerical search for AME(n,d) states.

An n-party pure state |psi> of qudits with local dimension d is AME if every
reduced state on floor(n/2) parties is maximally mixed.  Equivalently the
average purity over all balanced bipartitions equals 1/d^floor(n/2) (its
minimum possible value).

We minimise   E(psi) = d^m * <Tr rho_A^2>_A - 1   (m = floor(n/2))
which is >= 0 and is exactly 0 iff psi is AME.  Optimiser: L-BFGS with
random restarts on the unnormalised vector (objective is scale-invariant).
"""
import itertools
import sys
import time

import numpy as np
from scipy.optimize import minimize


def balanced_subsets(n):
    m = n // 2
    subs = list(itertools.combinations(range(n), m))
    if n % 2 == 0:  # A and its complement have the same purity
        subs = [S for S in subs if 0 in S]
    return subs


class AMEProblem:
    def __init__(self, n, d):
        self.n, self.d, self.m = n, d, n // 2
        self.subs = balanced_subsets(n)
        self.perms, self.iperms = [], []
        for S in self.subs:
            p = list(S) + [i for i in range(n) if i not in S]
            self.perms.append(p)
            self.iperms.append(list(np.argsort(p)))
        self.dA = d ** self.m
        self.dB = d ** (n - self.m)
        self.N = d ** n
        self.K = len(self.subs)

    def purities(self, psi):
        psi = psi / np.linalg.norm(psi)
        T = psi.reshape([self.d] * self.n)
        out = []
        for p in self.perms:
            M = np.transpose(T, p).reshape(self.dA, self.dB)
            rho = M @ M.conj().T
            out.append(np.vdot(rho, rho).real)
        return np.array(out)

    def excess_and_grad(self, psi):
        """E = dA*f - 1 with f = mean purity of normalised psi; returns E and dE/dpsi*."""
        d, n = self.d, self.n
        nrm2 = np.vdot(psi, psi).real
        T = psi.reshape([d] * n)
        P = 0.0
        G = np.zeros([d] * n, dtype=complex)
        for p, ip in zip(self.perms, self.iperms):
            M = np.transpose(T, p).reshape(self.dA, self.dB)
            rho = M @ M.conj().T
            P += np.vdot(rho, rho).real
            G += np.transpose((2.0 * (rho @ M)).reshape([d] * n), ip)
        P /= self.K
        G = G.reshape(-1) / self.K
        f = P / nrm2 ** 2
        g = G / nrm2 ** 2 - 2.0 * P / nrm2 ** 3 * psi
        return self.dA * f - 1.0, self.dA * g

    def fun(self, x):
        psi = x[: self.N] + 1j * x[self.N:]
        E, g = self.excess_and_grad(psi)
        return E, np.concatenate([2 * g.real, 2 * g.imag])

    def solve(self, seed, maxiter=5000, time_limit=None, x0=None, verbose=False):
        rng = np.random.default_rng(seed)
        if x0 is None:
            x0 = rng.standard_normal(2 * self.N)
        t0 = time.time()
        hist = []

        def cb(xk):
            if time_limit is not None and time.time() - t0 > time_limit:
                raise StopIteration
            if verbose:
                hist.append(self.fun(xk)[0])
                if len(hist) % 50 == 0:
                    print(f"    it {len(hist):5d}  E={hist[-1]:.3e}  t={time.time()-t0:.0f}s", flush=True)

        res = minimize(self.fun, x0, jac=True, method="L-BFGS-B", callback=cb,
                       options=dict(maxiter=maxiter, maxcor=30, ftol=1e-16, gtol=1e-13))
        psi = res.x[: self.N] + 1j * res.x[self.N:]
        psi /= np.linalg.norm(psi)
        E = self.excess_and_grad(psi)[0]
        return E, psi, res.nit, time.time() - t0


def run_case(n, d, restarts=5, maxiter=5000, time_limit=None):
    prob = AMEProblem(n, d)
    best = (np.inf, None)
    Es = []
    t0 = time.time()
    for r in range(restarts):
        E, psi, nit, dt = prob.solve(seed=1000 * n + 10 * d + r, maxiter=maxiter, time_limit=time_limit)
        Es.append(E)
        if E < best[0]:
            best = (E, psi)
    pur = prob.purities(best[1]) * prob.dA - 1  # per-bipartition relative excess
    return dict(n=n, d=d, best=best[0], all=Es, worst_cut=pur.max(), time=time.time() - t0, psi=best[1])


if __name__ == "__main__":
    # (n, d, expected) -- expected: True = AME exists, False = proven not to exist
    cases = [
        (4, 2, False), (4, 3, True), (5, 2, True), (6, 2, True), (7, 2, False),
        (8, 2, False), (6, 3, True), (6, 4, True), (7, 3, True), (8, 3, False), (4, 6, True),
    ]
    for n, d, exp in cases:
        r = run_case(n, d, restarts=6 if d != 6 else 20)
        tag = "exists" if exp else "does NOT exist"
        found = r["best"] < 1e-9
        print(f"AME({n},{d}) [{tag:>14}]  best E={r['best']:.2e}  "
              f"restarts E=[{', '.join(f'{e:.1e}' for e in r['all'])}]  "
              f"{'FOUND' if found else 'not found'}  ({r['time']:.1f}s)", flush=True)
