"""Turn a 16-qubit graph state into an 8-ququart state vector, cross-check its AME
defect with the continuous code, then try to polish it continuously towards AME(8,4)."""
import json
import sys
import time

import numpy as np

from ame_search import AMEProblem


def graph_state_ququarts(edges, n=8, k=2):
    nq = n * k
    x = np.arange(1 << nq, dtype=np.int64)
    bits = [(x >> q) & 1 for q in range(nq)]
    phase = np.zeros(1 << nq, dtype=np.int64)
    for (i, j) in edges:
        phase ^= bits[i] & bits[j]
    amp = np.where(phase == 1, -1.0, 1.0) / np.sqrt(1 << nq)
    # party j value = 2*bit(k*j) + bit(k*j+1); flat index C-order over parties 0..n-1
    idx = np.zeros(1 << nq, dtype=np.int64)
    for j in range(n):
        v = np.zeros(1 << nq, dtype=np.int64)
        for a in range(k):
            v = (v << 1) | bits[k * j + a]
        idx = idx * (1 << k) + v
    psi = np.zeros(1 << nq, dtype=complex)
    psi[idx] = amp
    return psi


if __name__ == "__main__":
    logfile, tlimit = sys.argv[1], float(sys.argv[2])
    rows = [json.loads(l) for l in open(logfile)]
    best = min(rows, key=lambda r: r["cost"])
    print(f"starting from SA seed {best['seed']}: {best['bad_elements']} bad stabilizer elements", flush=True)
    psi = graph_state_ququarts([tuple(e) for e in best["edges"]])
    prob = AMEProblem(8, 4)
    t0 = time.time()
    E0, _ = prob.excess_and_grad(psi)
    print(f"one objective+gradient evaluation: {time.time()-t0:.2f}s", flush=True)
    pur = prob.purities(psi) * prob.dA - 1
    print(f"continuous defect E={E0:.4f}; per-cut excess (should be integers = #bad elements in that cut): "
          f"{np.round(pur, 3).tolist()}", flush=True)
    # small random kick so we are not sitting exactly on a stationary point of the symmetric manifold
    rng = np.random.default_rng(0)
    psi = psi + 0.02 * (rng.standard_normal(psi.size) + 1j * rng.standard_normal(psi.size)) / np.sqrt(psi.size)
    E, psi2, nit, dt = prob.solve(seed=0, x0=np.concatenate([psi.real, psi.imag]), maxiter=100000,
                                  time_limit=tlimit, verbose=True)
    print(f"after continuous polish: E={E:.4e}  ({nit} iterations, {dt:.0f}s)", flush=True)
    np.save("hybrid_best.npy", psi2)
