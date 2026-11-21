"""
Stabilizer (graph-state) search for AME(n, 2^k) states.

Each party is a block of k qubits, so a ququart (d=4) is 2 qubits.  Every
stabilizer state is local-Clifford equivalent to a graph state, and local
Cliffords never change which parties an operator touches, so searching graph
states is complete for Z2^k-stabilizer AME states.  Edges inside one party are
a local unitary on that party, so they are dropped.

For graph G the stabilizer group is {X^s Z^(G s) : s in GF(2)^nq}.  The state
is AME(n, 2^k) iff every non-identity element touches >= floor(n/2)+1 parties.
cost(G) = sum over s != 0 of max(0, need - parties_touched(s)); AME iff cost=0.
"""
import itertools
import sys
import time

import numpy as np


class GraphAME:
    def __init__(self, n, k):
        self.n, self.k = n, k
        self.nq = n * k
        self.need = n // 2 + 1
        self.Nst = 1 << self.nq
        self.s = np.arange(self.Nst, dtype=np.uint32)
        block = (1 << k) - 1
        pw = np.zeros(self.Nst, dtype=np.int8)
        for j in range(n):
            pw += (((self.s >> (j * k)) & block) != 0).astype(np.int8)
        self.pwtab = pw
        self.party = [q // k for q in range(self.nq)]
        self.edges = [(i, j) for i in range(self.nq) for j in range(i + 1, self.nq)
                      if self.party[i] != self.party[j]]

    def zvec(self, cols):
        z = np.zeros(self.Nst, dtype=np.uint32)
        for b in range(self.nq):
            z[1 << b: 1 << (b + 1)] = z[: 1 << b] ^ np.uint32(cols[b])
        return z

    def cost(self, cols):
        pw = self.pwtab[self.s | self.zvec(cols)].astype(np.int16)
        deficit = np.maximum(0, self.need - pw[1:])
        return int(deficit.sum()), int((deficit > 0).sum())

    def cols_from_edges(self, E):
        cols = [0] * self.nq
        for (i, j) in E:
            cols[i] ^= 1 << j
            cols[j] ^= 1 << i
        return cols

    def anneal(self, seed, steps=200000, T0=2.0, T1=0.05, init=None):
        rng = np.random.default_rng(seed)
        if init is None:
            on = rng.random(len(self.edges)) < 0.5
        else:
            on = init.copy()
        cols = self.cols_from_edges([e for e, b in zip(self.edges, on) if b])
        c, nbad = self.cost(cols)
        best = (c, nbad, on.copy())
        for t in range(steps):
            if c == 0:
                break
            T = T0 * (T1 / T0) ** (t / steps)
            e = rng.integers(len(self.edges))
            i, j = self.edges[e]
            cols[i] ^= 1 << j
            cols[j] ^= 1 << i
            c2, nb2 = self.cost(cols)
            if c2 <= c or rng.random() < np.exp((c - c2) / T):
                c, nbad = c2, nb2
                on[e] = not on[e]
                if c < best[0]:
                    best = (c, nbad, on.copy())
            else:
                cols[i] ^= 1 << j
                cols[j] ^= 1 << i
        return best


def edge_orbits_cyclic(ga, shift_party):
    """Orbits of inter-party edges under a permutation of parties (qubit index inside party kept)."""
    k = ga.k

    def mapq(q):
        return shift_party[q // k] * k + q % k

    seen, orbits = set(), []
    for (i, j) in ga.edges:
        if (i, j) in seen:
            continue
        orb, a, b = [], i, j
        while True:
            e = (min(a, b), max(a, b))
            if e in seen:
                break
            seen.add(e)
            orb.append(e)
            a, b = mapq(a), mapq(b)
        orbits.append(orb)
    return orbits


def exhaustive_symmetric(ga, orbits, report_every=None):
    """Try every union of orbits; return (#graphs, best cost, best #bad, best edge set, #solutions)."""
    best = (10 ** 9, None, None)
    sols = []
    no = len(orbits)
    t0 = time.time()
    for mask in range(1 << no):
        E = [e for o in range(no) if mask >> o & 1 for e in orbits[o]]
        c, nb = ga.cost(ga.cols_from_edges(E))
        if c < best[0]:
            best = (c, nb, E)
        if c == 0:
            sols.append(E)
        if report_every and mask % report_every == 0 and mask:
            print(f"    {mask}/{1<<no}  best cost so far {best[0]}  ({time.time()-t0:.0f}s)", flush=True)
    return 1 << no, best, sols


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if mode == "validate":
        # (n, k, expected): d = 2^k
        for n, k, exp, steps in [(5, 1, True, 20000), (6, 1, True, 20000), (8, 1, False, 20000),
                                 (4, 2, True, 20000), (5, 2, True, 50000), (6, 2, True, 100000),
                                 (7, 2, None, 150000)]:
            ga = GraphAME(n, k)
            t0 = time.time()
            res = []
            for r in range(4):
                c, nb, on = ga.anneal(seed=r, steps=steps)
                res.append(c)
                if c == 0:
                    break
            tag = {True: "exists", False: "does NOT exist", None: "exists (stab? unknown)"}[exp]
            print(f"AME({n},{2**k}) [{tag}]  SA costs per restart: {res}  "
                  f"{'FOUND' if 0 in res else 'not found'}  ({time.time()-t0:.0f}s)", flush=True)
