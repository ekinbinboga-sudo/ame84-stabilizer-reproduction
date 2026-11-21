"""
Independent check of the last two levels, written in the language of subspaces.

Given the first m-2 block rows (their row spaces V_1..V_{m-2} in F2^(4m)), any completion
must split U = (V_1 + ... + V_{m-2})^perp (8-dimensional) as U = V_{m-1} (+) V_m with
V_{m-1} a non-degenerate 4-dim subspace and V_m its symplectic complement inside U.
The AME (MDS) condition is: for every set I of row spaces and set K of input coordinate
blocks W_k with |I| + |K| = m,  V_I + W_K = F2^(4m).
We enumerate ALL 4-dim subspaces of U (200,787 of them) -- no normalisation, no Schur
complements -- and count those that work.
"""
import itertools
import sys

import numpy as np
from numba import njit


def swapJ(y, m):
    mask = int("0011" * m, 2)  # low pair of every 4-bit block
    return ((y >> 2) & mask) | ((y & mask) << 2)


def sym(x, y, m):
    return bin(x & swapJ(y, m)).count("1") & 1


def kernel_basis(rows, nbits):
    """Basis of {x : parity(x & r) = 0 for all r in rows} in F2^nbits."""
    piv_rows, pivs = [], []
    for r in rows:
        for pr, pc in zip(piv_rows, pivs):
            if r >> pc & 1:
                r ^= pr
        if r:
            pc = r.bit_length() - 1
            for t in range(len(piv_rows)):
                if piv_rows[t] >> pc & 1:
                    piv_rows[t] ^= r
            piv_rows.append(r)
            pivs.append(pc)
    free = [c for c in range(nbits) if c not in pivs]
    basis = []
    for f in free:
        x = 1 << f
        for pr, pc in zip(piv_rows, pivs):
            if pr >> f & 1:
                x |= 1 << pc
        basis.append(x)
    return basis


@njit(cache=True)
def rank_vecs(v, n, nbits=64):
    a = v[:n].copy()
    r = 0
    for bit in range(nbits - 1, -1, -1):
        piv = -1
        for i in range(r, n):
            if (a[i] >> bit) & 1:
                piv = i
                break
        if piv < 0:
            continue
        a[r], a[piv] = a[piv], a[r]
        for i in range(n):
            if i != r and (a[i] >> bit) & 1:
                a[i] ^= a[r]
        r += 1
    return r


@njit(cache=True)
def parity(x):
    c = 0
    while x:
        x &= x - 1
        c ^= 1
    return c


@njit(cache=True)
def count_completions(U, Vfix, nfix, m, mask03, conds_I, conds_K, nA, ncond, max_report, report):
    """Enumerate all 4-dim subspaces of span(U) (U has 8 basis vectors)."""
    nb = 4 * m
    found = 0
    basis = np.zeros(4, dtype=np.int64)
    coef = np.zeros(4, dtype=np.int64)
    V4 = np.zeros(4, dtype=np.int64)
    buf = np.zeros(4 * m + 4, dtype=np.int64)
    for p0 in range(8):
        for p1 in range(p0 + 1, 8):
            for p2 in range(p1 + 1, 8):
                for p3 in range(p2 + 1, 8):
                    piv = (p0, p1, p2, p3)
                    # free positions of each RREF row: columns > pivot that are not pivots
                    nfree = 0
                    fpos = np.zeros(16, dtype=np.int64)
                    frow = np.zeros(16, dtype=np.int64)
                    for t in range(4):
                        for c in range(piv[t] + 1, 8):
                            if c != p0 and c != p1 and c != p2 and c != p3:
                                fpos[nfree] = c
                                frow[nfree] = t
                                nfree += 1
                    for assign in range(1 << nfree):
                        for t in range(4):
                            coef[t] = 1 << piv[t]
                        for q in range(nfree):
                            if (assign >> q) & 1:
                                coef[frow[q]] |= 1 << fpos[q]
                        for t in range(4):
                            v = 0
                            for c in range(8):
                                if (coef[t] >> c) & 1:
                                    v ^= U[c]
                            basis[t] = v
                        # non-degenerate?
                        g = np.zeros(4, dtype=np.int64)
                        for a in range(4):
                            row = 0
                            for b in range(4):
                                y = basis[b]
                                sy = ((y >> 2) & mask03) | ((y & mask03) << 2)
                                if parity(basis[a] & sy):
                                    row |= 1 << b
                            g[a] = row
                        if rank_vecs(g, 4, nb) < 4:
                            continue
                        okA = True
                        for ci in range(nA):
                            n = 0
                            for i in range(m):
                                if (conds_I[ci] >> i) & 1:
                                    for t in range(4):
                                        if i < nfix:
                                            buf[n] = Vfix[i, t]
                                        else:
                                            buf[n] = basis[t]
                                        n += 1
                            for kk in range(m):
                                if (conds_K[ci] >> kk) & 1:
                                    for t in range(4):
                                        buf[n] = 1 << (4 * kk + t)
                                        n += 1
                            if rank_vecs(buf, n, nb) < nb:
                                okA = False
                                break
                        if not okA:
                            continue
                        # V4 = basis^perp inside span(U): solve sum_c x_c <U_c, b_a> = 0
                        eq = np.zeros(4, dtype=np.int64)
                        for a in range(4):
                            y = basis[a]
                            sy = ((y >> 2) & mask03) | ((y & mask03) << 2)
                            e = 0
                            for c in range(8):
                                if parity(U[c] & sy):
                                    e |= 1 << c
                            eq[a] = e
                        # kernel of eq (4 x 8) over GF(2)
                        A = eq.copy()
                        pivc = np.full(4, -1, dtype=np.int64)
                        r = 0
                        for c in range(8):
                            pr = -1
                            for i in range(r, 4):
                                if (A[i] >> c) & 1:
                                    pr = i
                                    break
                            if pr < 0:
                                continue
                            A[r], A[pr] = A[pr], A[r]
                            for i in range(4):
                                if i != r and (A[i] >> c) & 1:
                                    A[i] ^= A[r]
                            pivc[r] = c
                            r += 1
                        nk = 0
                        for c in range(8):
                            isp = False
                            for i in range(r):
                                if pivc[i] == c:
                                    isp = True
                            if isp:
                                continue
                            x = 1 << c
                            for i in range(r):
                                if (A[i] >> c) & 1:
                                    x |= 1 << pivc[i]
                            v = 0
                            for cc in range(8):
                                if (x >> cc) & 1:
                                    v ^= U[cc]
                            V4[nk] = v
                            nk += 1
                        # all MDS conditions
                        ok = True
                        for ci in range(nA, ncond):
                            n = 0
                            for i in range(m):
                                if (conds_I[ci] >> i) & 1:
                                    for t in range(4):
                                        if i < nfix:
                                            buf[n] = Vfix[i, t]
                                        elif i == nfix:
                                            buf[n] = basis[t]
                                        else:
                                            buf[n] = V4[t]
                                        n += 1
                            for kk in range(m):
                                if (conds_K[ci] >> kk) & 1:
                                    for t in range(4):
                                        buf[n] = 1 << (4 * kk + t)
                                        n += 1
                            if rank_vecs(buf, n, nb) < nb:
                                ok = False
                                break
                        if ok:
                            if found < max_report:
                                report[found, :] = basis
                            found += 1
    return found


def check(m, fixed_rows_blocks, B):
    """fixed_rows_blocks: list of block rows (each list of m block ints) for rows 1..m-2."""
    from exact_search import assemble
    nb = 4 * m
    Vfix = np.array([assemble(B, [r], m) for r in fixed_rows_blocks], dtype=np.int64)
    allv = [int(v) for r in Vfix for v in r]
    Ub = kernel_basis([swapJ(v, m) for v in allv], nb)
    assert len(Ub) == 8, len(Ub)
    nfix = len(fixed_rows_blocks)
    A, Bc = [], []
    for r in range(1, m + 1):
        for I in itertools.combinations(range(m), r):
            for K in itertools.combinations(range(m), m - r):
                item = (sum(1 << i for i in I), sum(1 << k for k in K))
                if nfix + 1 in I:
                    Bc.append(item)          # involves the last row V_m
                elif nfix in I:
                    A.append(item)           # involves V_{m-1} only
                # conditions on the fixed rows alone are already guaranteed
    conds_I = [c[0] for c in A + Bc]
    conds_K = [c[1] for c in A + Bc]
    report = np.zeros((10, 4), dtype=np.int64)
    mask03 = int("0011" * m, 2)
    n = count_completions(np.array(Ub, dtype=np.int64), Vfix, len(fixed_rows_blocks), m, mask03,
                          np.array(conds_I, dtype=np.int64), np.array(conds_K, dtype=np.int64),
                          len(A), len(conds_I), 10, report)
    return n


if __name__ == "__main__":
    import json, os, random, time
    import fast_search as fs
    mode = sys.argv[1]
    if mode == "control":  # AME(6,4), class 4: the fast search found 2272 valid second rows
        F = fs.Fast(3)
        L1 = F.E.level1_classes()
        for cls in range(len(L1)):
            row1 = [F.B.repR[Q] for Q in L1[cls]]
            t = time.time()
            n = check(3, [row1], F.B)
            rows2, _ = fs.run_level(F, [row1], 0, -1, None)
            print(f"AME(6,4) class {cls}: subspace method {n} valid V_2, fast search {len(rows2)} rows "
                  f"-> match {n == len(rows2)}  ({time.time()-t:.1f}s)", flush=True)
    elif mode == "all":  # AME(8,4): check EVERY row-2 candidate of one class
        cls = int(sys.argv[2])
        F = fs.Fast(4)
        L1 = [tuple(t) for t in json.load(open("L1_84.json"))]
        row1 = [F.B.repR[Q] for Q in L1[cls]]
        t = time.time()
        rows2, _ = fs.run_level(F, [row1], 0, -1, None)
        print(f"class {cls}: {len(rows2)} row-2 candidates ({time.time()-t:.0f}s)", flush=True)
        tot = 0
        for idx, (r2, p2) in enumerate(rows2):
            tot += check(4, [row1, r2], F.B)
            if (idx + 1) % 1000 == 0:
                print(f"  {idx+1}/{len(rows2)} checked, completions so far {tot} ({time.time()-t:.0f}s)", flush=True)
        print(f"DONE class {cls}: {len(rows2)} row-2 candidates checked independently, "
              f"completions found: {tot} ({time.time()-t:.0f}s)", flush=True)
        json.dump(dict(cls=cls, n_row2=len(rows2), completions=tot, secs=time.time() - t),
                  open(os.path.join(os.environ.get("AME_OUT", "."), f"subspace_class{cls}.json"), "w"))
    else:  # AME(8,4): sample row-2 candidates of a class, check that NO completion exists
        cls, nsample = int(sys.argv[2]), int(sys.argv[3])
        F = fs.Fast(4)
        L1 = [tuple(t) for t in json.load(open("L1_84.json"))]
        row1 = [F.B.repR[Q] for Q in L1[cls]]
        rows2, _ = fs.run_level(F, [row1], 0, -1, None)
        random.seed(cls)
        sample = random.sample(rows2, min(nsample, len(rows2)))
        t = time.time()
        tot = 0
        for (r2, p2) in sample:
            tot += check(4, [row1, r2], F.B)
        print(f"AME(8,4) class {cls}: {len(sample)} random row-2 candidates, completions found by "
              f"subspace method: {tot}  ({time.time()-t:.0f}s)", flush=True)
