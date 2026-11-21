# Results

`cloud/` — 2-core Linux machine:
- `run84_even.log`, `run84_odd.log` — AME(8,4) search, classes 0,2,4,6,8 and 1,3,5,7,9
- `result_m4_class*.json` — per-class AME(8,4) output
- `result_m3_class*.json` — AME(6,4) validation output (1552 normalized solutions)

`pc/` — Windows PC, independent rerun:
- `pc_dogrulama_ame64.log` — AME(6,4) validation ("dogrulama" = validation)
- `pc_class*.log` — AME(8,4) search, one file per class
- `pc_subspace_class*.log`, `pc_subspace_class*.json` — independent completion check of all row-2 candidates
