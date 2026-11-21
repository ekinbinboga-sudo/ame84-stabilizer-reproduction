"""AME(8,4) stabilizer search: simulated annealing restarts + exhaustive symmetric families."""
import json
import sys
import time

import numpy as np

from ame_graph import GraphAME, edge_orbits_cyclic, exhaustive_symmetric

ga = GraphAME(8, 2)  # 8 ququarts = 16 qubits, need every stabilizer element to touch >= 5 parties

mode = sys.argv[1]
if mode == "sa":
    nrest, steps = int(sys.argv[2]), int(sys.argv[3])
    seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    log = open(f"sa_84_seed{seed0}.log", "a")
    for r in range(nrest):
        t0 = time.time()
        c, nb, on = ga.anneal(seed=seed0 + r, steps=steps, T0=3.0, T1=0.05)
        E = [e for e, b in zip(ga.edges, on) if b]
        msg = dict(seed=seed0 + r, cost=c, bad_elements=nb, edges=E, secs=round(time.time() - t0))
        print(json.dumps(msg), file=log, flush=True)
        print(f"restart {seed0 + r}: best cost {c}, bad stabilizer elements {nb} ({msg['secs']}s)", flush=True)
        if c == 0:
            print("!!! COST ZERO -- candidate AME(8,4) found", flush=True)
            break
elif mode == "sym":
    # Z8: rotate all 8 parties;  Z7: rotate parties 0..6, fix party 7;
    # Z4xZ2-like: two 4-cycles (0123)(4567)
    groups = {
        "Z8 (8-cycle)": [1, 2, 3, 4, 5, 6, 7, 0],
        "Z7 (7-cycle + fixed)": [1, 2, 3, 4, 5, 6, 0, 7],
        "two 4-cycles": [1, 2, 3, 0, 5, 6, 7, 4],
    }
    for name, perm in groups.items():
        orbits = edge_orbits_cyclic(ga, perm)
        t0 = time.time()
        if len(orbits) > 20:
            print(f"{name}: {len(orbits)} orbits -> too many for exhaustive, skipping", flush=True)
            continue
        ng, best, sols = exhaustive_symmetric(ga, orbits)
        print(f"{name}: {len(orbits)} edge orbits, {ng} graphs checked exhaustively, "
              f"min cost {best[0]} (bad elements {best[1]}), solutions {len(sols)}  ({time.time()-t0:.0f}s)",
              flush=True)
        if sols:
            json.dump(sols[0], open(f"solution_{name.split()[0]}.json", "w"))
