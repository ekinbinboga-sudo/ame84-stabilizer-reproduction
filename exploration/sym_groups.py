"""Exhaustive search of 16-qubit graph states whose graph is invariant under a group
acting on the 8 ququart parties (qubit label inside a party is kept fixed)."""
import time

from ame_graph import GraphAME, exhaustive_symmetric

ga = GraphAME(8, 2)


def orbits_from_generators(ga, gens):
    k = ga.k
    parent = {e: e for e in ga.edges}

    def find(e):
        while parent[e] != e:
            parent[e] = parent[parent[e]]
            e = parent[e]
        return e

    for g in gens:
        for (i, j) in ga.edges:
            a, b = g[i // k] * k + i % k, g[j // k] * k + j % k
            e2 = (min(a, b), max(a, b))
            r1, r2 = find((i, j)), find(e2)
            if r1 != r2:
                parent[r1] = r2
    groups = {}
    for e in ga.edges:
        groups.setdefault(find(e), []).append(e)
    return list(groups.values())


# GF(8) = GF(2)[x]/(x^3+x+1), elements as ints 0..7; alpha = 2
def gf8_mul(a, b):
    r = 0
    for i in range(3):
        if b >> i & 1:
            r ^= a << i
    for deg in (4, 3):
        if r >> deg & 1:
            r ^= 0b1011 << (deg - 3)
    return r


translations = [[x ^ t for x in range(8)] for t in (1, 2, 4)]
mult_alpha = [gf8_mul(2, x) for x in range(8)]
# PSL(2,7) on the projective line {0..6, inf=7}
inv7 = {1: 1, 2: 4, 3: 5, 4: 2, 5: 3, 6: 6}
t_shift = [(x + 1) % 7 for x in range(7)] + [7]
m_sq = [(2 * x) % 7 for x in range(7)] + [7]
s_inv = [7] + [(-inv7[x]) % 7 for x in range(1, 7)] + [0]
rot8 = [(x + 1) % 8 for x in range(8)]
refl8 = [(-x) % 8 for x in range(8)]

groups = {
    "Z2^3 (translations of GF(8))": translations,
    "AGL(1,8)": translations + [mult_alpha],
    "PSL(2,7)": [t_shift, m_sq, s_inv],
    "D8 (dihedral of 8-cycle)": [rot8, refl8],
    "Z7 x| Z3 (x->x+1, x->2x, inf fixed)": [t_shift, m_sq],
}
for name, gens in groups.items():
    orbs = orbits_from_generators(ga, gens)
    if len(orbs) > 20:
        print(f"{name}: {len(orbs)} orbits, skipped", flush=True)
        continue
    t0 = time.time()
    ng, best, sols = exhaustive_symmetric(ga, orbs)
    print(f"{name}: {len(orbs)} edge orbits -> {ng} graphs, min cost {best[0]} "
          f"(bad elements {best[1]}), solutions {len(sols)} ({time.time()-t0:.0f}s)", flush=True)
