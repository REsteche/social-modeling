"""Robustness controls for the Level-4 inversion.

Variants of the reference Level-4 configuration (lambda = 0.6, T = 80):
    homog    - homogeneous influence strengths (s_i = 1): the regime where
               the two-bloc reduction is exact in the mean
    kappa4   - finite-variance strength law (kappa = 4)
    reflect  - reflecting instead of clipping opinion boundaries
    k5, k20  - halved / doubled attention slots
    rho1, rho20 - slower / faster rewiring (fast-rewiring assumption check)

Also: a Level-1 noise-sensitivity check (sigma_x = 0.02 instead of 0).

Outputs:
    results/exp9_controls.json
"""

import numpy as np

from common import save_json, run_parallel
from socialsim import ModelParams, run_simulation, metrics

PLATFORMS = ["similarity", "neutral", "controversy"]
SEEDS = range(12)

VARIANTS = {
    "homog":   {"heterogeneous_influence": False},
    "kappa4":  {"kappa": 4.0},
    "reflect": {"boundary": "reflect"},
    "k5":      {"k_digital": 5},
    "k20":     {"k_digital": 20},
    "rho1":    {"rewire_prob": 1.0},
    "rho20":   {"rewire_prob": 20.0},
}


def make_params(engagement, seed, **overrides):
    base = dict(n_agents=200, D=1e-3, seed=seed, n_steps=4000, dt=0.02,
                epsilon=0.3, ell=0.02, sigma_x=0.02,
                digital=True, k_digital=10, rewire_prob=5.0,
                engagement=engagement, gamma=4.0, delta=0.8, s_width=0.2,
                repulsion=True, eps1=0.3, eps2=0.9, eta=0.4,
                heterogeneous_influence=True, kappa=2.5,
                attention_digital=0.6)
    base.update(overrides)
    return ModelParams(**base)


def run_one(job):
    kind, payload = job
    if kind == "variant":
        variant, eng, seed = payload
        traj = run_simulation(make_params(eng, seed, **VARIANTS[variant]))
        return {"kind": kind, "variant": variant, "platform": eng,
                "seed": seed,
                "polarization": float(np.var(traj.x[-1])),
                "extremism": float(np.mean(np.abs(traj.x[-1])))}
    # level-1 noise check
    D, seed = payload
    p = ModelParams(n_agents=200, D=D, seed=seed, n_steps=3000, dt=0.02,
                    epsilon=0.3, ell=0.02, sigma_x=0.02,
                    attention_digital=0.0)
    traj = run_simulation(p)
    nc, _, _ = metrics.opinion_clusters(traj.x[-1], gap=0.05)
    return {"kind": "level1_noise", "D": D, "seed": seed,
            "n_clusters": int(nc),
            "polarization": float(np.var(traj.x[-1]))}


def main():
    jobs = [("variant", (v, eng, seed)) for v in VARIANTS
            for eng in PLATFORMS for seed in SEEDS]
    jobs += [("level1", (D, seed)) for D in [1e-5, 1e-3, 1e-1]
             for seed in range(6)]
    rows = run_parallel(run_one, jobs, chunksize=1)
    save_json(rows, "exp9_controls")

    # console summary
    for v in VARIANTS:
        parts = []
        for eng in PLATFORMS:
            vals = [r["polarization"] for r in rows if r.get("variant") == v
                    and r.get("platform") == eng]
            parts.append(f"{eng[:7]}={np.mean(vals):.3f}±{np.std(vals):.3f}")
        print(f"{v:8s} " + "  ".join(parts))
    for D in [1e-5, 1e-3, 1e-1]:
        vals = [r["n_clusters"] for r in rows if r.get("kind") ==
                "level1_noise" and r.get("D") == D]
        print(f"level1 noise D={D:.0e}: nc={np.mean(vals):.2f}±{np.std(vals):.2f}")


if __name__ == "__main__":
    main()
