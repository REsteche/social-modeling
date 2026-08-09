"""Radicalization threshold in the digital attention share lambda.

Fine sweep of Var(x) versus lambda for the three Level-4 platforms at
N = 200, plus a finite-size analysis (N = 100 ... 800, constant density)
for the neutral and similarity platforms. The empirical threshold
lambda* is located by the crossing Var(x) = 0.5; the sweep data also feed
the comparison with the two-bloc reduction (scripts/theory_twobloc.py).

Outputs:
    fig7_threshold.(pdf|png)     (produced by theory_twobloc.py overlaying
                                  the reduced-model prediction)
    results/exp7_threshold.json
"""

import numpy as np

from common import save_json, run_parallel
from socialsim import ModelParams, run_simulation

PLATFORMS = ["similarity", "neutral", "controversy"]
LAM_FINE = np.round(np.linspace(0.0, 1.0, 21), 3)
SEEDS_FINE = range(12)

FSS_PLATFORMS = ["neutral", "similarity"]
FSS_NS = [100, 200, 400, 800]
LAM_COARSE = np.round(np.linspace(0.0, 1.0, 9), 3)
SEEDS_FSS = range(6)


def make_params(engagement, lam, n, seed):
    return ModelParams(n_agents=n, box_size=float(np.sqrt(n / 200)),
                       D=1e-3, seed=seed, n_steps=4000, dt=0.02,
                       epsilon=0.3, ell=0.02, sigma_x=0.02,
                       digital=True, k_digital=10, rewire_prob=5.0,
                       engagement=engagement, gamma=4.0, delta=0.8,
                       s_width=0.2,
                       repulsion=True, eps1=0.3, eps2=0.9, eta=0.4,
                       heterogeneous_influence=True, kappa=2.5,
                       attention_digital=lam)


def run_one(job):
    eng, lam, n, seed = job
    traj = run_simulation(make_params(eng, lam, n, seed))
    return {"platform": eng, "lambda": float(lam), "N": n, "seed": seed,
            "polarization": float(np.var(traj.x[-1])),
            "extremism": float(np.mean(np.abs(traj.x[-1])))}


def main():
    jobs = [(eng, lam, 200, seed) for eng in PLATFORMS
            for lam in LAM_FINE for seed in SEEDS_FINE]
    jobs += [(eng, lam, n, seed) for eng in FSS_PLATFORMS
             for n in FSS_NS if n != 200
             for lam in LAM_COARSE for seed in SEEDS_FSS]
    rows = run_parallel(run_one, jobs, chunksize=1)
    save_json(rows, "exp7_threshold")
    print("data saved; run theory_twobloc.py to produce fig7")


if __name__ == "__main__":
    main()
