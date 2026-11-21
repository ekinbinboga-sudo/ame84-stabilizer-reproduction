"""
Compiled (numba) version of exact_search.py for the ququart case (k = 2, 4x4 blocks).
Python prepares each level (linear solve + minor tests); numba runs the inner loops.

Usage:  python fast_search.py M [class_indices...]     (M = 3 -> AME(6,4), M = 4 -> AME(8,4))
"""
import json
import os
import sys
import time

import numpy as np
from numba import njit

from exact_search import ExactAME, verify

K_QUQUART = 2


def row_table(B, K):
    """16-entry table so that (A @ K) row r = T[row r of A]."""
    rows = B.rows(K)
    T = np.zeros(16, dtype=np.int32)
    for v in range(16):
        acc = 0
        for c in range(4):
            if v >> c & 1:
                acc ^= rows[c]
        T[v] = acc
    return T


class Fast:
    def __init__(self, m):
        self.E = ExactAME(m, K_QUQUART)
        B = self.B = self.E.B
        self.m = m
        self.INV = np.array([-1 if B.inv[A] is None else B.inv[A] for A in range(B.N)], dtype=np.int32)
        self.FORM = np.array([B.form.get(A, 0) for A in range(B.N)], dtype=np.int32)
        self.GL = np.array(B.GL, dtype=np.int32)
        self.REPL = np.array([B.repL[p] for p in B.formsL], dtype=np.int32)
        self.J0 = B.J0

    def ctx_arrays(self, rows):
        free, solved, H, by_col = self.E._prep_level(rows)
        m = self.m
        nfree, nsol = len(free), len(solved)
        Htab = np.zeros((4, 4, 16), dtype=np.int32)
        for (j, a), Hm in H.items():
            Htab[j, a] = row_table(self.B, Hm)
        sol_cols = np.zeros(4, dtype=np.int32)
        sol_cols[:nsol] = solved
        tests = []
        col_start = np.zeros(5, dtype=np.int32)
        for c in range(m):
            col_start[c] = len(tests)
            tests.extend(by_col[c])
        col_start[m] = len(tests)
        T = max(1, len(tests))
        t_cstar = np.zeros(T, dtype=np.int32)
        t_n = np.zeros(T, dtype=np.int32)
        t_cols = np.zeros((T, 3), dtype=np.int32)
        t_K = np.zeros((T, 3, 16), dtype=np.int32)
        for idx, (cstar, terms) in enumerate(tests):
            t_cstar[idx] = cstar
            t_n[idx] = len(terms)
            for q, (tc, Km) in enumerate(terms):
                t_cols[idx, q] = tc
                t_K[idx, q] = row_table(self.B, Km)
        return (np.int32(nfree), np.int32(nsol), sol_cols, Htab, col_start, t_cstar, t_n, t_cols, t_K)


@njit(cache=True, inline="always")
def mulK(A, T):
    return (T[A & 15] | (T[(A >> 4) & 15] << 4) | (T[(A >> 8) & 15] << 8) | (T[(A >> 12) & 15] << 12))


@njit(cache=True)
def tests_ok(row, c, col_start, t_cstar, t_n, t_cols, t_K, INV):
    for t in range(col_start[c], col_start[c + 1]):
        s = row[t_cstar[t]]
        for q in range(t_n[t]):
            s ^= mulK(row[t_cols[t, q]], t_K[t, q])
        if INV[s] < 0:
            return False
    return True


@njit(cache=True)
def leaf(row, m, nfree, nsol, sol_cols, Htab, col_start, t_cstar, t_n, t_cols, t_K, INV, FORM, J0,
         pi, prev_p, prev_row):
    for a in range(nsol):
        x = 0
        for j in range(nfree):
            x ^= mulK(row[j], Htab[j, a])
        if INV[x] < 0:
            return False
        row[sol_cols[a]] = x
    f = 0
    for j in range(m):
        f ^= FORM[row[j]]
    if f != J0:
        return False
    for a in range(nsol):
        if not tests_ok(row, sol_cols[a], col_start, t_cstar, t_n, t_cols, t_K, INV):
            return False
    if pi == prev_p:  # canonical order of rows 2..m
        for j in range(m):
            if row[j] > prev_row[j]:
                return True
            if row[j] < prev_row[j]:
                return False
        return False
    return True


@njit(cache=True)
def level_kernel(m, nfree, nsol, sol_cols, Htab, col_start, t_cstar, t_n, t_cols, t_K,
                 INV, FORM, GL, REPL, J0, p_min, prev_p, prev_row, out, out_p):
    cnt = 0
    leaves = 0
    row = np.zeros(4, dtype=np.int32)
    ng = GL.shape[0]
    for pi in range(p_min, REPL.shape[0]):
        row[0] = REPL[pi]
        if not tests_ok(row, 0, col_start, t_cstar, t_n, t_cols, t_K, INV):
            continue
        if nfree == 1:
            leaves += 1
            if leaf(row, m, nfree, nsol, sol_cols, Htab, col_start, t_cstar, t_n, t_cols, t_K,
                    INV, FORM, J0, pi, prev_p, prev_row):
                if cnt < out.shape[0]:
                    out[cnt, :] = row
                    out_p[cnt] = pi
                cnt += 1
            continue
        for g1 in range(ng):
            row[1] = GL[g1]
            if not tests_ok(row, 1, col_start, t_cstar, t_n, t_cols, t_K, INV):
                continue
            if nfree == 2:
                leaves += 1
                if leaf(row, m, nfree, nsol, sol_cols, Htab, col_start, t_cstar, t_n, t_cols, t_K,
                        INV, FORM, J0, pi, prev_p, prev_row):
                    if cnt < out.shape[0]:
                        out[cnt, :] = row
                        out_p[cnt] = pi
                    cnt += 1
                continue
            for g2 in range(ng):
                row[2] = GL[g2]
                if not tests_ok(row, 2, col_start, t_cstar, t_n, t_cols, t_K, INV):
                    continue
                leaves += 1
                if leaf(row, m, nfree, nsol, sol_cols, Htab, col_start, t_cstar, t_n, t_cols, t_K,
                        INV, FORM, J0, pi, prev_p, prev_row):
                    if cnt < out.shape[0]:
                        out[cnt, :] = row
                        out_p[cnt] = pi
                    cnt += 1
    return cnt, leaves


def run_level(F, rows, p_min, prev_p, prev_row, cap=4_000_000):
    ctx = F.ctx_arrays(rows)
    out = np.zeros((cap, 4), dtype=np.int32)
    out_p = np.zeros(cap, dtype=np.int32)
    pr = np.zeros(4, dtype=np.int32) if prev_row is None else np.array(prev_row + [0] * (4 - len(prev_row)), dtype=np.int32)
    cnt, leaves = level_kernel(np.int32(F.m), *ctx, F.INV, F.FORM, F.GL, F.REPL, np.int32(F.J0),
                               np.int32(p_min), np.int32(prev_p), pr, out, out_p)
    if cnt > cap:
        raise RuntimeError("output buffer too small")
    return [(out[t, :F.m].tolist(), int(out_p[t])) for t in range(cnt)], leaves


def run_class(F, row1, log=None):
    m = F.m
    sols, nodes = [], [0] * (m + 1)
    t0 = time.time()

    def rec(rows, last_p, depth):
        if len(rows) == m:
            sols.append([list(r) for r in rows])
            return
        p_min = 0 if len(rows) == 1 else last_p
        prev_p = -1 if len(rows) == 1 else last_p
        prev_row = None if len(rows) == 1 else rows[-1]
        cands, _ = run_level(F, rows, p_min, prev_p, prev_row)
        nodes[len(rows) + 1] += len(cands)
        for k_, (r, p) in enumerate(cands):
            rec(rows + [r], p, depth + 1)
            if depth == 0 and log and (k_ + 1) % 200 == 0:
                print(f"      row-2 candidates done {k_+1}/{len(cands)}  solutions so far {len(sols)}  "
                      f"({time.time()-t0:.0f}s)", file=log, flush=True)

    rec([list(row1)], None, 0)
    return sols, nodes


if __name__ == "__main__":
    m = int(sys.argv[1])
    F = Fast(m)
    if m == 4:
        L1 = [tuple(t) for t in json.load(open("L1_84.json"))]
    else:
        L1 = F.E.level1_classes()
    which = [int(a) for a in sys.argv[2:]] or list(range(len(L1)))
    print(f"AME({2*m},4): {len(L1)} row-1 classes; running {which}", flush=True)
    total = 0
    for idx in which:
        t0 = time.time()
        row1 = [F.B.repR[Q] for Q in L1[idx]]
        sols, nodes = run_class(F, row1, log=sys.stdout)
        total += len(sols)
        ok = [verify(m, K_QUQUART, s)[0] for s in sols[:20]]
        print(f"  class {idx}: rows at each level {nodes[2:]}  solutions {len(sols)}  "
              f"verified(first 20) {all(ok) if ok else '-'}  ({time.time()-t0:.0f}s)", flush=True)
        with open(os.path.join(os.environ.get("AME_OUT", "."), f"result_m{m}_class{idx}.json"), "w") as fh:
            json.dump(dict(cls=idx, Q=list(L1[idx]), nodes=nodes, n_solutions=len(sols),
                           solutions=sols[:50], secs=time.time() - t0), fh)
    print(f"TOTAL normalized solutions for AME({2*m},4): {total}")
