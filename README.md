# Exhaustive search for stabilizer AME(8,4) states — an independent reproduction

**Short version.** There is no stabilizer AME(8,4) state built from qubit pairs: no F₂-additive
[[8,0,5]]₄ code, and no F₂-additive [[7,1,4]]₄ code. **This is a known result.** It was proved by
Ball, Moreno and Simoens [BMS] with a different, geometric computer-assisted method. This repository
reproduces it independently through a different reduction, with several internal cross-checks.

Whether an AME(8,4) state exists at all is still open: the remaining candidates are non-stabilizer
states (Open Quantum Problems, Problem 35 [OQP]).

## Background

- A pure state of n parties with local dimension d is *absolutely maximally entangled*, AME(n,d), if
  every reduction to ⌊n/2⌋ parties is maximally mixed. AME(8,4), eight ququarts, is the first open case
  of the existence table.
- AME(2m, d) is equivalent to a quantum MDS code [[2m−1, 1, m]]_d. For AME(8,4) this is [[7,1,4]]₄.
- Here a ququart is a pair of qubits, so the local Pauli group is that of two qubits (F₂-additive codes
  over F₂⁴). This class contains the GF(4)-linear and the GF(16)-Hermitian constructions.

## The reduction

The Pauli part of one party is a vector in F₂⁴ with symplectic form ω(v,w) = vᵀJ₀w,
J₀ = [[0, I₂], [I₂, 0]]. A stabilizer AME(8,4) state is a 16-dimensional isotropic subspace S of
(F₂⁴)⁸ whose non-zero elements all touch at least 5 parties. Such an S is an MDS code, so any 4 parties
form an information set. Taking parties 1–4 as inputs, S = {(u, M u)}, where M is a 4×4 array of 4×4
binary blocks, and

- S is Lagrangian ⇔ **M J Mᵀ = J**, J = diag(J₀, J₀, J₀, J₀);
- every non-zero element touches ≥ 5 parties ⇔ **every square block-submatrix of M is invertible**.

The first three block rows of M already describe a [[7,1,4]]₄ code (they span the normalizer of the
punctured code), so the search can stop as soon as no third row fits.

## Symmetry reduction

Local Cliffords act as Sp(4,2) on each party: M → D M E⁻¹ with D, E block diagonal in Sp(4,2). The
input parties (columns) and the output parties 2–4 (rows) can be permuted. With this:

- row 1 reduces to one of **10 canonical classes** (`L1_84.json`);
- the first block of every later row is fixed to one of the 28 coset representatives of Sp(4,2) in GL(4,2);
- rows 2–4 are kept in a canonical order.

The remaining blocks of a new row are forced by symplectic orthogonality to the rows above, through a
linear solve.

## Result

| row-1 class | valid row 2 | valid row 3 | solutions |
|---|---|---|---|
| 0 | 24,488 | 0 | 0 |
| 1 | 28,960 | 0 | 0 |
| 2 | 24,128 | 0 | 0 |
| 3 | 24,128 | 0 | 0 |
| 4 | 26,556 | 0 | 0 |
| 5 | 24,488 | 0 | 0 |
| 6 | 28,960 | 0 | 0 |
| 7 | 28,656 | 0 | 0 |
| 8 | 28,960 | 0 | 0 |
| 9 | 28,656 | 0 | 0 |
| **total** | **267,980** | **0** | **0** |

No third row can be added in any class. Hence there is no F₂-additive [[7,1,4]]₄ code and no
F₂-additive stabilizer AME(8,4) state.

## Validation

- **Known cases.** AME(4,2), AME(8,2), AME(10,2): no solutions (correct). AME(6,2), AME(4,4), AME(6,4):
  found, and every solution was re-verified by enumerating its whole stabilizer group (`verify()`).
- **[[4,0,3]]₄ uniqueness.** The exact search returns exactly one normalized AME(4,4) solution,
  consistent with the uniqueness theorem in [BMS].
- **AME(6,4), two methods.** The pure-Python and numba searches both give 1552 normalized solutions.
  An independent subspace-enumeration method reproduces the per-class counts 288 / 64 / 112 / 368 / 2272
  exactly.
- **AME(8,4) row-2 kernel.** Identical to the pure-Python search on two restricted slices (124 and 140 rows).
- **Independent completion check.** All 267,980 row-2 candidates were re-examined with the subspace
  method, which uses neither the row normalization nor Schur complements: 0 completions.
- **Independent GF(16) search** (Hermitian self-dual MDS codes, i.e. GF(4)-linear stabilizer AME states):
  [4,2,3]₁₆: 2 solutions, [6,3,4]₁₆: 60, [10,5,6]₁₆: found, [8,4,5]₁₆: 0.
- **Two machines.** A 2-core Linux machine and a Windows PC gave identical counts for every class.
- **Not independently re-derived:** the full list of row-2 candidates for AME(8,4). It was checked
  against the pure-Python code only on slices.

## Comparison with [BMS]

| | [BMS] | this repository |
|---|---|---|
| Path | [[6,2,3]]₄ → [[7,1,4]]₄ → [[8,0,5]]₄ | same: rows 1–2 = [[6,2,3]]₄, row 3 = [[7,1,4]]₄ |
| Language | solids in PG(7,2) with symplectic polarities | symplectic block matrix with invertible block minors |
| [[6,2,3]]₄ stage | 341 configurations of six solids, at most 1311 codes | 267,980 normalized candidates (weaker symmetry reduction) |
| Second implementation | GAP / FinInG | subspace method, separate GF(16) program |
| Software | Sage | Python + numba |

[BMS] also note that Z/4Z stabilizer codes with these parameters would give binary codes that do not
exist. Wójcik, Makuta, Bruzda and Augusiak [WMBA] show that AME states of 4n parties with even local
dimension cannot be canonical graph states over Z_d. A solution for AME(8,4), if one exists, therefore
has to come from outside these stabilizer constructions.

## Running it

Requirements: Python 3.9+, `numpy`, `numba` (`pip install -r requirements.txt`).

```
python fast_search.py 3            # validation: AME(6,4), expect "TOTAL ... 1552" (about 1 min)
python fast_search.py 4 0 1 2      # AME(8,4) for row-1 classes 0, 1, 2 (10-20 min per class)
python subspace_check.py control   # subspace method vs fast search on AME(6,4)
python subspace_check.py all 0     # independent completion check of class 0 (1-1.5 h)
python gf16_search.py 2 3 4 5      # GF(16)-Hermitian search for lengths 4, 6, 8, 10
```

`run_windows.bat` and `run_linux.sh` run everything, one process per class, and write to `out/`.
The environment variable `AME_OUT` sets where JSON results are written.

`L1_84.json` can be regenerated (about 1 min):

```
python -c "import json; from exact_search import ExactAME; json.dump([list(t) for t in ExactAME(4,2).level1_classes()], open('L1_84.json','w'))"
```

## Files

- `exact_search.py` — reduction, symmetry reduction, pure-Python search and the independent `verify()`
- `fast_search.py` — compiled (numba) search, used for AME(8,4)
- `subspace_check.py` — independent completion check (`control`, `sample`, `all`)
- `gf16_search.py` — independent GF(16)-Hermitian search
- `L1_84.json` — the 10 canonical first rows for AME(8,4)
- `results/cloud`, `results/pc` — logs and JSON outputs of the two runs
- `exploration/` — first, heuristic attempts (simulated annealing over graph states, continuous
  optimization). A useful negative lesson: random-start continuous optimization does not even find the
  known AME(6,4), AME(7,3) or AME(4,6) states.

## References

- **[BMS]** S. Ball, E. Moreno, R. Simoens, *Stabiliser codes over fields of even order*,
  IEEE Trans. Inf. Theory 71, 3707 (2025), doi:10.1109/TIT.2024.3454480, arXiv:2401.06618.
  Code: https://github.com/robinsimoens/stabiliser-codes-over-fields-of-even-order
- **[WMBA]** Wójcik, Makuta, Bruzda, Augusiak, arXiv:2603.18193 (2026).
- **[GR]** M. Grassl, M. Rötteler, *Quantum MDS codes over small fields*, ISIT 2015, arXiv:1502.05267.
- **[OQP]** Open Quantum Problems, Problem 35: https://oqp.iqoqi.oeaw.ac.at/?p=536.
  Lean statement: google-deepmind/formal-conjectures, `OpenQuantumProblems/35`.

## Authors and AI use

Ekin Binbuga. The code and the analysis were developed together with Claude (Anthropic), an AI model,
in November 2026.

## License

MIT — see `LICENSE`.
