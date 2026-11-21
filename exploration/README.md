# Exploration (first session)

Heuristic attempts made before the exact search. They are kept because the failures are informative.

- `ame_search.py`, `ame_proj.py` — continuous optimization of the average purity (L-BFGS) and
  alternating polar projections. They find easy cases but get stuck on the known AME(6,4), AME(7,3) and
  AME(4,6) states, so a failure on AME(8,4) says nothing.
- `ame_graph.py`, `run_84.py`, `sym_groups.py` — simulated annealing over graph states (two qubits per
  ququart) and exhaustive search over symmetric graph families. They find AME(4,4), AME(5,4), AME(6,4) and
  AME(7,4) quickly; for AME(8,4) the best candidate violates 9 of the 35 balanced cuts.
- `hybrid.py` — continuous polishing starting from the best graph-state candidate (stalls at 8 of 35).
- `sa_84_seed0.log` — annealing log with the best graphs found.
