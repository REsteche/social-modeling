"""Run every experiment and regenerate all paper figures.

Usage:  python scripts/run_all.py
"""

import time

import exp1_mobility
import exp2_static_digital
import exp3_adaptive
import exp4_engagement
import exp5_phase_diagram

# The __main__ guard is required: exp5 uses multiprocessing, whose spawned
# workers re-import this file and must not re-run the experiments.
if __name__ == "__main__":
    for mod in (exp1_mobility, exp2_static_digital, exp3_adaptive,
                exp4_engagement, exp5_phase_diagram):
        t0 = time.time()
        print(f"=== {mod.__name__} ===")
        mod.main()
        print(f"=== {mod.__name__} finished in {time.time() - t0:.0f} s ===\n")
