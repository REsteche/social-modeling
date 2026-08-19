"""Two-bloc reduction and well-mixed limit of the Level-4 dynamics.

Part A - local-rate analysis. Bounded-confidence fragmentation forms first
(blocs near +-y0 with 2*y0 > eps2), and the fragmented state is unstable to
repulsion for ANY lambda > 0. With the stationary cross-bloc slot fraction

    p0 = E(2 y0) / (E(0) + E(2 y0)),

the two-bloc equation dy/dt = 2 lambda eta p(y) y gives the frozen-p0
local-rate estimate

    t_loc(lambda) = ln(y_f / y0) / (2 lambda eta p0),

so the apparent onset in a finite-horizon simulation is the local-rate
crossover estimate

    lambda_c^(0)(T) = ln(y_f / y0) / (2 eta p0 T),

not a phase transition. The exact target-crossing time follows from the
quadrature t = (1/(2 lambda eta)) * int dy / (y p(y)); the frozen-p0 value
is exact for the neutral kernel only. (A naive linear-stability analysis
of the uniform state would predict much larger thresholds; the simulations
falsify it because fragmentation preempts it.)

Part B - well-mixed particle model. In the fast-rewiring, well-mixed
limit the model reduces to the spaceless dynamics

    dx_i = (1-lambda) <x_j - x_i>_{|dx|<eps} dt
         + lambda <F(x_i, x_j)>_{w prop E(|dx|) s_j} dt + sigma dB_i,

which we integrate as a spaceless interacting-particle system and compare
with the spatial ABM sweep of exp7_threshold.py.

Outputs:
    fig7_threshold.(pdf|png)
    results/theory_twobloc.json
"""

import json
import sys

import numpy as np
import matplotlib.pyplot as plt

from common import save_fig, save_json, run_parallel, RESULTS_DIR
from socialsim.model import ModelParams, engagement_kernel, influence_function

PLATFORMS = ["similarity", "neutral", "controversy"]
COLORS = {"similarity": "#33507a", "neutral": "#a04040",
          "controversy": "#3d7a4d"}
LABELS = {"similarity": "similarity-driven", "neutral": "neutral",
          "controversy": "controversy-driven"}

EPS1, EPS2, ETA, EPS = 0.3, 0.9, 0.4, 0.3
LAM_THEORY = np.round(np.linspace(0.0, 1.0, 21), 3)
M = 600            # well-mixed particles
REPS = 12


def base_params(engagement):
    return ModelParams(epsilon=EPS, eps1=EPS1, eps2=EPS2, eta=ETA,
                       repulsion=True, engagement=engagement,
                       gamma=4.0, delta=0.8, s_width=0.2)


Y0 = 0.6         # initial bloc position after BC fragmentation
YF = 1.0         # boundary
T_HORIZON = 80.0


def rate_analysis(engagement):
    """Cross-bloc slot fraction p0 and finite-horizon crossover lambda_c."""
    p = base_params(engagement)
    E = lambda d: float(engagement_kernel(np.asarray(d), p))
    p0 = E(2 * Y0) / (E(0.0) + E(2 * Y0))
    coeff = np.log(YF / Y0) / (2 * ETA * p0)   # t_rad = coeff / lambda
    lam_c = coeff / T_HORIZON
    return p0, coeff, lam_c


def meanfield_var(job):
    """Integrate the well-mixed particle system; return final Var(x)."""
    engagement, lam, rep = job
    p = base_params(engagement)
    rng = np.random.default_rng(1000 + rep)
    x = rng.uniform(-1, 1, M)
    s = 1.0 + rng.pareto(1.5, M)
    s /= s.mean()
    dt, n_steps, sigma = 0.02, 4000, 0.02
    for _ in range(n_steps):
        dxm = x[None, :] - x[:, None]
        adx = np.abs(dxm)
        # physical layer in the well-mixed limit: global bounded confidence
        W = (adx < EPS).astype(float)
        np.fill_diagonal(W, 0.0)
        wsum = W.sum(axis=1)
        drift = (1 - lam) * np.where(
            wsum > 0, (W * dxm).sum(axis=1) / np.maximum(wsum, 1e-12), 0.0)
        # digital layer in the fast-rewiring limit: slots are sampled AND
        # weighted proportionally to s, giving s^2 weights (the s^2 factors
        # cancel in expectation since strengths are independent of opinions)
        Ew = engagement_kernel(adx, p) * s[None, :] ** 2
        np.fill_diagonal(Ew, 0.0)
        esum = Ew.sum(axis=1)
        F = influence_function(dxm, p)
        drift += lam * (Ew * F).sum(axis=1) / np.maximum(esum, 1e-12)
        x = np.clip(x + drift * dt
                    + sigma * np.sqrt(dt) * rng.standard_normal(M), -1, 1)
    return {"platform": engagement, "lambda": float(lam), "rep": rep,
            "polarization": float(np.var(x))}


def main():
    # ---- Part A: radicalization rates and finite-horizon crossovers
    rates = {}
    for eng in PLATFORMS:
        p0, coeff, lam_c = rate_analysis(eng)
        rates[eng] = {"p0": p0, "t_rad_coeff": coeff, "lambda_c": lam_c}
        print(f"{eng:12s} p0 = {p0:.4f}  t_rad = {coeff:.2f}/lambda"
              f"  lambda_c(T={T_HORIZON:.0f}) = {lam_c:.3f}")

    # ---- Part B: mean-field curves
    data_file = RESULTS_DIR / "theory_twobloc.json"
    if "--replot" in sys.argv and data_file.exists():
        mf_rows = json.loads(data_file.read_text())["meanfield"]
    else:
        # top-up: keep existing realisations, compute only missing ones
        mf_rows = []
        if data_file.exists():
            mf_rows = json.loads(data_file.read_text())["meanfield"]
        have = {(r["platform"], r["lambda"], r["rep"]) for r in mf_rows}
        jobs = [(eng, float(lam), rep) for eng in PLATFORMS
                for lam in LAM_THEORY for rep in range(REPS)
                if (eng, float(lam), rep) not in have]
        if jobs:
            mf_rows = mf_rows + run_parallel(meanfield_var, jobs)
    save_json({"rates": rates, "meanfield": mf_rows}, "theory_twobloc")

    # ---- figure: ABM sweep + well-mixed prediction + thresholds + FSS
    abm = json.loads((RESULTS_DIR / "exp7_threshold.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7), layout="constrained")

    ax = axes[0]
    lams = sorted({r["lambda"] for r in abm if r["N"] == 200})
    for eng in PLATFORMS:
        means = [np.mean([r["polarization"] for r in abm
                          if r["platform"] == eng and r["N"] == 200
                          and r["lambda"] == lam]) for lam in lams]
        stds = [np.std([r["polarization"] for r in abm
                        if r["platform"] == eng and r["N"] == 200
                        and r["lambda"] == lam]) for lam in lams]
        ax.errorbar(lams, means, yerr=stds, fmt="o", ms=3, capsize=2,
                    color=COLORS[eng], label=LABELS[eng])
        mf = [np.mean([r["polarization"] for r in mf_rows
                       if r["platform"] == eng and r["lambda"] == lam])
              for lam in LAM_THEORY]
        ax.plot(LAM_THEORY, mf, "-", color=COLORS[eng], lw=1.2, alpha=0.8)
        lc = rates[eng]["lambda_c"]
        if lc <= 1.0:
            ax.axvline(lc, color=COLORS[eng], lw=0.8, ls=":")
    ax.set_xlabel("digital attention $\\lambda$")
    ax.set_ylabel("polarization Var$(x)$")
    ax.set_title("ABM (points) vs well-mixed model (lines)", fontsize=12)
    ax.legend(frameon=False, fontsize=8.5)

    # finite-size panels: neutral and similarity
    for k, eng in enumerate(["neutral", "similarity"]):
        ax = axes[1 + k]
        ns = sorted({r["N"] for r in abm if r["platform"] == eng})
        for n in ns:
            lams_n = sorted({r["lambda"] for r in abm
                             if r["platform"] == eng and r["N"] == n})
            means = [np.mean([r["polarization"] for r in abm
                              if r["platform"] == eng and r["N"] == n
                              and r["lambda"] == lam]) for lam in lams_n]
            ax.plot(lams_n, means, "o-", ms=2.5, lw=1,
                    label=f"$N = {n}$")
        lc = rates[eng]["lambda_c"]
        if lc <= 1.0:
            ax.axvline(lc, color="gray", lw=0.8, ls=":")
            ax.text(lc + 0.02, 0.35, "$\\lambda_c^{(0)}(T)$", fontsize=9.5,
                    color="gray")
        ax.set_xlabel("digital attention $\\lambda$")
        ax.set_title(f"{LABELS[eng]}: system-size dependence", fontsize=12)
        ax.legend(frameon=False, fontsize=8.5)

    save_fig(fig, "fig7_threshold")


if __name__ == "__main__":
    main()
