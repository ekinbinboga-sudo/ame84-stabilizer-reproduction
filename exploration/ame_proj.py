"""
Alternating-projection search for AME(n,d) states (multi-unitary / iterative
polar method), followed by an L-BFGS polish.

For each balanced cut A the state reshaped as a dA x dB matrix M_A must satisfy
M_A M_A^dag = I/dA.  The closest such matrix (Frobenius) is W V^dag / sqrt(dA)
from the SVD M_A = W S V^dag.  We cycle through all cuts (random order each
sweep), projecting onto each constraint in turn.
"""
import sys
import time

import numpy as np

from ame_search import AMEProblem


def project_cut(prob, psi, p, ip):
    d, n = prob.d, prob.n
    M = np.transpose(psi.reshape([d] * n), p).reshape(prob.dA, prob.dB)
    W, _, Vh = np.linalg.svd(M, full_matrices=False)
    M = (W @ Vh) / np.sqrt(prob.dA)
    return np.transpose(M.reshape([d] * n), ip).reshape(-1)


def alt_proj(prob, seed, sweeps=2000, tol=1e-12, time_limit=None, psi0=None, report=0):
    rng = np.random.default_rng(seed)
    if psi0 is None:
        psi = rng.standard_normal(prob.N) + 1j * rng.standard_normal(prob.N)
    else:
        psi = psi0.copy()
    psi /= np.linalg.norm(psi)
    t0 = time.time()
    order = np.arange(prob.K)
    E = np.inf
    for s in range(sweeps):
        rng.shuffle(order)
        for k in order:
            psi = project_cut(prob, psi, prob.perms[k], prob.iperms[k])
            psi /= np.linalg.norm(psi)
        if s % 10 == 0 or s == sweeps - 1:
            E = prob.excess_and_grad(psi)[0]
            if report and s % report == 0:
                print(f"    sweep {s:5d}  E={E:.3e}  t={time.time()-t0:.0f}s", flush=True)
            if E < tol:
                break
        if time_limit is not None and time.time() - t0 > time_limit:
            break
    E = prob.excess_and_grad(psi)[0]
    return E, psi, s + 1, time.time() - t0


def polish(prob, psi, maxiter=3000):
    from scipy.optimize import minimize
    x0 = np.concatenate([psi.real, psi.imag])
    res = minimize(prob.fun, x0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, maxcor=30, ftol=1e-16, gtol=1e-14))
    out = res.x[: prob.N] + 1j * res.x[prob.N:]
    out /= np.linalg.norm(out)
    return prob.excess_and_grad(out)[0], out


if __name__ == "__main__":
    cases = [(4, 2, False), (6, 4, True), (7, 3, True), (4, 6, True), (8, 3, False), (7, 2, False)]
    for n, d, exp in cases:
        prob = AMEProblem(n, d)
        Es = []
        t0 = time.time()
        for r in range(8 if (n, d) != (8, 3) else 3):
            E, psi, sw, dt = alt_proj(prob, seed=r, sweeps=1500, time_limit=40)
            if E > 1e-12:
                E2, psi = polish(prob, psi)
                E = min(E, E2)
            Es.append(E)
        tag = "exists" if exp else "does NOT exist"
        print(f"AME({n},{d}) [{tag:>14}]  best E={min(Es):.2e}  runs=[{', '.join(f'{e:.1e}' for e in Es)}]  "
              f"({time.time()-t0:.0f}s)", flush=True)
