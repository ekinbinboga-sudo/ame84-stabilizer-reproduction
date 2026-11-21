"""
Independent search for GF(16)-linear Hermitian self-dual MDS codes [2k, k, k+1]_16,
i.e. GF(4)-linear stabilizer AME(2k, 4) states (Grassl-Roetteler style).

Systematic form G = [I | A]:  Hermitian self-dual  <=>  A conj(A)^T = I   (conj(x) = x^4),
                              MDS                  <=>  every square submatrix of A is non-singular.
Symmetries used: rows and columns of A may be scaled by norm-1 scalars (x^5 = 1), so the
first row and the first column can be taken from the coset representatives {1, g, g^2}.
Rows are built one by one inside the Hermitian complement of the rows already chosen.
"""
import itertools
import sys
import time

# ---------------------------------------------------------------- GF(16), x^4 + x + 1
EXP, LOG = [0] * 30, [0] * 16
v = 1
for i in range(15):
    EXP[i] = EXP[i + 15] = v
    LOG[v] = i
    v <<= 1
    if v & 16:
        v ^= 0b10011


def mul(a, b):
    return 0 if a == 0 or b == 0 else EXP[LOG[a] + LOG[b]]


def inv(a):
    return EXP[(15 - LOG[a]) % 15]


def conj(a):
    return mul(mul(a, a), mul(a, a))  # a^4


G = EXP[1]
REPS = [1, G, mul(G, G)]  # coset representatives of GF(16)* / {x : x^5 = 1}


def herm(u, w):
    s = 0
    for a, b in zip(u, w):
        s ^= mul(a, conj(b))
    return s


def det(M):
    M = [list(r) for r in M]
    n, d = len(M), 1
    for c in range(n):
        p = next((r for r in range(c, n) if M[r][c]), None)
        if p is None:
            return 0
        M[c], M[p] = M[p], M[c]
        d = mul(d, M[c][c])
        ic = inv(M[c][c])
        for r in range(c + 1, n):
            if M[r][c]:
                f = mul(M[r][c], ic)
                M[r] = [x ^ mul(f, y) for x, y in zip(M[r], M[c])]
    return d


def nullspace(rows, k):
    """Basis of {x in GF(16)^k : sum_j x_j * r_j = 0 for every r in rows}."""
    A = [list(r) for r in rows]
    piv, r = [], 0
    for c in range(k):
        p = next((i for i in range(r, len(A)) if A[i][c]), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        ic = inv(A[r][c])
        A[r] = [mul(ic, x) for x in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c]:
                f = A[i][c]
                A[i] = [x ^ mul(f, y) for x, y in zip(A[i], A[r])]
        piv.append(c)
        r += 1
    basis = []
    for f in (c for c in range(k) if c not in piv):
        x = [0] * k
        x[f] = 1
        for i, pc in enumerate(piv):
            x[pc] = A[i][f]  # char 2: minus = plus
        basis.append(x)
    return basis


def span(basis, k):
    for coeffs in itertools.product(range(16), repeat=len(basis)):
        x = [0] * k
        for c, b in zip(coeffs, basis):
            if c:
                x = [xi ^ mul(c, bi) for xi, bi in zip(x, b)]
        yield x


def minors_ok(rows, new, k):
    allrows = rows + [new]
    i = len(rows)
    for r in range(1, i + 2):
        for R in itertools.combinations(range(i), r - 1):
            for C in itertools.combinations(range(k), r):
                sub = [[allrows[rr][c] for c in C] for rr in list(R) + [i]]
                if det(sub) == 0:
                    return False
    return True


def search(k, stop_first=False):
    sols = []

    def rec(rows):
        if len(rows) == k:
            sols.append([list(r) for r in rows])
            return stop_first
        if not rows:
            cands = (list(t) for t in itertools.product(REPS, repeat=k))
        else:
            basis = nullspace([[conj(x) for x in r] for r in rows], k)
            cands = span(basis, k)
        for x in cands:
            if 0 in x or herm(x, x) != 1:
                continue
            if rows and x[0] not in REPS:
                continue
            if not minors_ok(rows, x, k):
                continue
            if rec(rows + [x]):
                return True
        return False

    rec([])
    return sols


if __name__ == "__main__":
    for k in [int(a) for a in sys.argv[1:]]:
        t = time.time()
        sols = search(k)
        ok = all(all(herm(a, b) == (1 if i == j else 0) for j, b in enumerate(s) for i, a in enumerate(s))
                 for s in sols)
        print(f"Hermitian self-dual MDS [{2*k},{k},{k+1}]_16  (GF(4)-linear AME({2*k},4)): "
              f"{len(sols)} normalized solutions, orthonormality re-checked: {ok}  ({time.time()-t:.1f}s)",
              flush=True)
