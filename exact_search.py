"""
Exact (exhaustive) search for stabilizer AME(2m, 2^k) states.

Reformulation
-------------
Each party is k qubits, so its Pauli part is a vector in F2^b, b = 2k, with the
symplectic form w(v,w) = v^T J0 w, J0 = [[0, I_k], [I_k, 0]].
A stabilizer AME(2m, 2^k) state is an F2-linear code S of length 2m over F2^b with
|S| = 2^(bm) whose non-zero elements all touch >= m+1 parties.  That makes S an
MDS code, so any m parties form an information set.  Taking parties 1..m:

    S = {(u, M u)},   M = m x m array of b x b binary blocks, and
    (a) S is Lagrangian            <=>  M J M^T = J,  J = diag(J0, ..., J0)
    (b) every element touches >=m+1 parties
                                   <=>  every square block-submatrix of M is invertible.

Symmetries used to cut the search (all preserve (a) and (b)):
  * local Cliffords on every party: M -> D M E^-1 with D, E block-diagonal in Sp(b,2);
  * permutations of the input parties (columns) and of output parties 2..m (rows).
Row 1 is brought to a canonical form; every later row has its first block fixed to a
coset representative; rows 2..m are kept in a canonical order.
"""
import itertools
import sys
import time


# ---------------------------------------------------------------- GF(2) helpers
def gf2_inv(rows, n):
    """Inverse of an n x n GF(2) matrix given as row bit-masks (bit c = column c)."""
    A, I = list(rows), [1 << r for r in range(n)]
    for col in range(n):
        piv = next((r for r in range(col, n) if A[r] >> col & 1), None)
        if piv is None:
            return None
        A[col], A[piv] = A[piv], A[col]
        I[col], I[piv] = I[piv], I[col]
        for r in range(n):
            if r != col and A[r] >> col & 1:
                A[r] ^= A[col]
                I[r] ^= I[col]
    return I


class Blocks:
    """b x b binary matrices stored as ints: entry (r, c) is bit r*b + c."""

    def __init__(self, k):
        self.k, self.b = k, 2 * k
        b = self.b
        self.rm = (1 << b) - 1
        self.N = 1 << (b * b)
        J0 = 0
        for r in range(b):
            for c in range(b):
                if c == r + k or r == c + k:
                    J0 |= 1 << (r * b + c)
        self.J0 = J0
        self.tr = [self._transpose(A) for A in range(self.N)]
        self.inv = [None] * self.N
        for A in range(self.N):
            inv_rows = gf2_inv(self.rows(A), b)
            self.inv[A] = None if inv_rows is None else self.from_rows(inv_rows)
        self.GL = [A for A in range(self.N) if self.inv[A] is not None]
        self.form = {A: self.mul(self.mul(A, J0), self.tr[A]) for A in self.GL}      # A J0 A^T
        self.formL = {A: self.mul(self.mul(self.tr[A], J0), A) for A in self.GL}     # A^T J0 A
        self.Sp = [A for A in self.GL if self.form[A] == J0]
        self.repR, self.repL = {}, {}
        for A in self.GL:
            self.repR.setdefault(self.form[A], A)
            self.repL.setdefault(self.formL[A], A)
        self.forms = sorted(self.repR)          # all non-degenerate alternating forms
        self.formsL = sorted(self.repL)

    def rows(self, A):
        return [(A >> (r * self.b)) & self.rm for r in range(self.b)]

    def from_rows(self, rows):
        out = 0
        for r, v in enumerate(rows):
            out |= v << (r * self.b)
        return out

    def _transpose(self, A):
        b, out = self.b, 0
        for r in range(b):
            for c in range(b):
                if A >> (r * b + c) & 1:
                    out |= 1 << (c * b + r)
        return out

    def mul(self, A, B):
        b, rm = self.b, self.rm
        Brows = [(B >> (c * b)) & rm for c in range(b)]
        out = 0
        for r in range(b):
            ar, row = (A >> (r * b)) & rm, 0
            c = 0
            while ar:
                if ar & 1:
                    row ^= Brows[c]
                ar >>= 1
                c += 1
            out |= row << (r * b)
        return out


# ---------------------------------------------------- block-matrix assembly helpers
def assemble(B, blocks, ncol):
    """List of block-rows (each a list of ncol block ints) -> list of GF(2) row masks."""
    out = []
    for brow in blocks:
        for r in range(B.b):
            v = 0
            for bc in range(ncol):
                v |= ((brow[bc] >> (r * B.b)) & B.rm) << (bc * B.b)
            out.append(v)
    return out


def split(B, rows, nbr, nbc):
    """GF(2) row masks -> nbr x nbc array of blocks."""
    blk = [[0] * nbc for _ in range(nbr)]
    for br in range(nbr):
        for bc in range(nbc):
            v = 0
            for r in range(B.b):
                v |= ((rows[br * B.b + r] >> (bc * B.b)) & B.rm) << (r * B.b)
            blk[br][bc] = v
    return blk


def matmul_rows(X, Y, ncolY):
    """GF(2) product of row-mask matrices X (len p, entries over q columns) and Y (q rows)."""
    out = []
    for x in X:
        v, c = 0, 0
        while x:
            if x & 1:
                v ^= Y[c]
            x >>= 1
            c += 1
        out.append(v)
    return out


# ------------------------------------------------------------------ the search
class ExactAME:
    def __init__(self, m, k):
        self.m = m
        self.B = Blocks(k)

    # ---- row 1 ------------------------------------------------------------
    def level1_classes(self):
        B, m = self.B, self.m
        act = {g: {Q: B.mul(B.mul(g, Q), B.tr[g]) for Q in B.forms} for g in B.Sp}
        seen, reps = set(), []
        for tup in itertools.product(B.forms, repeat=m - 1):
            last = B.J0
            for Q in tup:
                last ^= Q
            if last not in B.repR:
                continue
            full = tup + (last,)
            canon = min(tuple(sorted(a[Q] for Q in full)) for a in act.values())
            if canon not in seen:
                seen.add(canon)
                reps.append(canon)
        return reps

    # ---- later rows ---------------------------------------------------------
    def _prep_level(self, rows):
        """Precompute everything the next row needs, given the rows placed so far."""
        B, m = self.B, self.m
        b, i = B.b, len(rows)
        free = list(range(m - i))
        solved = list(range(m - i, m))
        # N_{a,c} = J0 rows[c][solved[a]]^T ; equation sum_a X_a N_{a,c} = sum_j B_j G_{j,c}
        Nblk = [[B.mul(B.J0, B.tr[rows[c][solved[a]]]) for c in range(i)] for a in range(i)]
        Ninv = gf2_inv(assemble(B, Nblk, i), i * b)
        assert Ninv is not None, "N singular -- an earlier minor check failed"
        Ninv = split(B, Ninv, i, i)
        H = {}
        for j in free:
            for a in range(i):
                acc = 0
                for c in range(i):
                    G = B.mul(B.J0, B.tr[rows[c][j]])
                    acc ^= B.mul(G, Ninv[c][a])
                H[j, a] = acc
        # minor tests involving the new row: for |C| = r, R subset of old rows with |R| = r-1
        tests = []
        for r in range(1, min(i + 1, m) + 1):
            for R in itertools.combinations(range(i), r - 1):
                for C in itertools.combinations(range(m), r):
                    cstar, Cpp = C[-1], C[:-1]
                    if r == 1:
                        tests.append((cstar, ()))
                        continue
                    P = assemble(B, [[rows[rr][cc] for cc in Cpp] for rr in R], r - 1)
                    Pinv = gf2_inv(P, (r - 1) * b)
                    Q = assemble(B, [[rows[rr][cstar]] for rr in R], 1)
                    K = split(B, matmul_rows(Pinv, Q, 1), r - 1, 1)
                    tests.append((cstar, tuple(zip(Cpp, (K[t][0] for t in range(r - 1))))))
        # order tests by the largest column they use, so they can be run as soon as possible
        by_col = {c: [] for c in range(m)}
        for cstar, terms in tests:
            last = max([cstar] + [t for t, _ in terms])
            by_col[last].append((cstar, terms))
        return free, solved, H, by_col

    def _minor_ok(self, row, tests):
        B = self.B
        for cstar, terms in tests:
            s = row[cstar]
            for t, K in terms:
                s ^= B.mul(row[t], K)
            if B.inv[s] is None:
                return False
        return True

    def search(self, row1, stop_at_first=False, stats=None):
        sols = []
        self._rec([list(row1)], [None], sols, stop_at_first, stats)
        return sols

    def _rec(self, rows, pcls, sols, stop, stats):
        B, m = self.B, self.m
        i = len(rows)
        if i == m:
            sols.append([list(r) for r in rows])
            return stop
        ctx = self._prep_level(rows)
        prev_p = pcls[-1] if i >= 2 else None
        for p in B.formsL:
            if prev_p is not None and p < prev_p:
                continue
            row = [0] * m
            row[0] = B.repL[p]
            if not self._minor_ok(row, ctx[3][0]):
                continue
            if self._fill(rows, pcls, sols, stop, stats, row, 1, ctx, p, prev_p):
                return True
        return False

    def _fill(self, rows, pcls, sols, stop, stats, row, col, ctx, p, prev_p):
        B, m = self.B, self.m
        free, solved, H, by_col = ctx
        if col < len(free):
            for A in B.GL:
                row[col] = A
                if self._minor_ok(row, by_col[col]):
                    if self._fill(rows, pcls, sols, stop, stats, row, col + 1, ctx, p, prev_p):
                        return True
            return False
        # all free blocks set: the remaining blocks are forced by orthogonality to earlier rows
        if stats is not None:
            stats["leaves"] = stats.get("leaves", 0) + 1
        for a, c in enumerate(solved):
            x = 0
            for j in free:
                x ^= B.mul(row[j], H[j, a])
            if B.inv[x] is None:
                return False
            row[c] = x
        f = 0
        for blk in row:
            f ^= B.form[blk]
        if f != B.J0:
            return False
        for c in solved:
            if not self._minor_ok(row, by_col[c]):
                return False
        if prev_p is not None and p == prev_p and tuple(row) <= tuple(rows[-1]):
            return False
        return self._rec(rows + [list(row)], pcls + [p], sols, stop, stats)


# ------------------------------------------------------- independent verification
def verify(m, k, M):
    """Build S = {(u, Mu)} explicitly and check (1) isotropy, (2) every non-zero element
    touches >= m+1 of the 2m parties.  Independent of the minor logic above."""
    B = Blocks(k)
    b, n = B.b, 2 * m
    gens = []
    for j in range(m):
        for r in range(b):
            v = [0] * n
            v[j] = 1 << r
            for i in range(m):
                col = 0
                for s in range(b):  # column r of block M[i][j]
                    if M[i][j] >> (s * b + r) & 1:
                        col |= 1 << s
                v[m + i] = col
            gens.append(v)

    def omega(x, y):
        tot = 0
        for p in range(n):
            xr = x[p]
            # x^T J0 y : J0 swaps the x- and z-halves
            yz = ((y[p] >> B.k) | ((y[p] & ((1 << B.k) - 1)) << B.k))
            tot ^= bin(xr & yz).count("1") & 1
        return tot

    for g1 in gens:
        for g2 in gens:
            if omega(g1, g2):
                return False, "not isotropic"
    # enumerate all 2^(bm) elements with a Gray code
    cur = [0] * n
    min_w = n
    for t in range(1, 1 << (b * m)):
        flip = (t & -t).bit_length() - 1
        g = gens[flip]
        for p in range(n):
            cur[p] ^= g[p]
        w = sum(1 for p in range(n) if cur[p])
        min_w = min(min_w, w)
    return min_w >= m + 1, f"min party-weight {min_w} (need {m + 1})"


if __name__ == "__main__":
    m, k = int(sys.argv[1]), int(sys.argv[2])
    E = ExactAME(m, k)
    t0 = time.time()
    L1 = E.level1_classes()
    print(f"AME({2*m},{2**k}): |GL|={len(E.B.GL)} |Sp|={len(E.B.Sp)} forms={len(E.B.forms)} "
          f"row-1 classes={len(L1)}  ({time.time()-t0:.1f}s)", flush=True)
    total = 0
    for idx, Qs in enumerate(L1):
        row1 = [E.B.repR[Q] for Q in Qs]
        stats = {}
        sols = E.search(row1, stop_at_first=False, stats=stats)
        total += len(sols)
        for s in sols[:1]:
            print("   example solution verified:", verify(m, k, s), flush=True)
        print(f"  class {idx}: {len(sols)} solutions, leaves {stats.get('leaves', 0)}", flush=True)
    print(f"TOTAL normalized solutions: {total}  ({time.time()-t0:.1f}s)")
