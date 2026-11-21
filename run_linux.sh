#!/usr/bin/env bash
# Full reproduction: validation, then one background job per row-1 class.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out
export AME_OUT=out
python3 fast_search.py 3 > out/validation_ame64.log 2>&1
grep TOTAL out/validation_ame64.log
for c in 0 1 2 3 4 5 6 7 8 9; do
  ( python3 fast_search.py 4 "$c" > "out/class$c.log" 2>&1 &&
    python3 subspace_check.py all "$c" > "out/subspace_class$c.log" 2>&1 ) &
done
wait
grep -h "class [0-9]*:" out/class*.log
grep -h "DONE" out/subspace_class*.log
