"""Core stochastic adaptive multiplex opinion-dynamics model.

State of agent i:
    r_i(t) in [0, L)^2   physical position (periodic boundary conditions)
    x_i(t) in [-1, 1]    opinion

Dynamics (Euler-Maruyama discretisation):

    dr_i = sqrt(2 D) dW_i

    dx_i = alpha_i * <K(r_ij) B_eps(x_j - x_i) (x_j - x_i)>_j dt      (physical)
         + beta_i  * <s_j A_ij F(x_i, x_j)>_j dt                      (digital)
         + sigma_x dB_i

with K(r) = exp(-r^2 / 2 ell^2), B_eps a bounded-confidence indicator,
F an attraction/neutral/repulsion influence function, s_j heterogeneous
influence strengths, and A_ij(t) a directed adjacency matrix (row i = the
set of accounts whose content agent i sees) rewired stochastically with
probability proportional to an engagement kernel E(|x_i - x_j|) * s_j.

The attention budget is implemented in two ways: (i) each agent has a fixed
number of digital attention slots k_digital (rows of A have constant sum),
and (ii) the attention split lambda in [0,1] enforces
alpha_i = alpha_total * (1 - lambda), beta_i = alpha_total * lambda so that
digital attention displaces physical attention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ModelParams:
    # population / discretisation
    n_agents: int = 200
    box_size: float = 1.0
    dt: float = 0.02
    n_steps: int = 3000
    seed: int = 0

    # physical layer
    D: float = 1e-3                 # spatial diffusion coefficient
    ell: float = 0.02               # physical interaction range
    r_cut_factor: float = 3.0       # hard cutoff of the kernel at r_cut_factor * ell
    epsilon: float = 0.30           # bounded-confidence threshold
    chi: float = 0.0                # homophilic mobility: drift toward compatible
                                    # neighbours, away from incompatible ones (Level 5)
    boundary: str = "clip"          # opinion boundary handling: "clip" | "reflect"
    alpha_total: float = 1.0        # total social attention rate
    attention_digital: float = 0.0  # lambda: fraction of attention on digital layer
    sigma_x: float = 0.0            # opinion noise amplitude

    # digital layer
    digital: bool = False
    k_digital: int = 10             # attention slots (in-degree of A)
    rewire_prob: float = 0.0        # per-agent per-unit-time slot replacement rate (0 = static graph)
    engagement: str = "similarity"  # "similarity" | "neutral" | "controversy"
    gamma: float = 4.0              # homophily strength of similarity kernel
    delta: float = 0.8              # preferred ideological distance (controversy kernel)
    s_width: float = 0.2            # width of the controversy kernel

    # influence function F (defaults = the paper's reference values)
    repulsion: bool = False
    eps1: float = 0.30              # attract below eps1 (defaults to epsilon)
    eps2: float = 0.90              # repel above eps2
    eta: float = 0.4                # repulsion strength

    # heterogeneous influence / stubborn agents
    heterogeneous_influence: bool = False
    kappa: float = 2.5              # Pareto exponent of influence strengths
    n_stubborn: int = 0
    stubborn_values: tuple = (-1.0, 1.0)

    # recording
    record_every: int = 10


def engagement_kernel(delta_x: np.ndarray, p: ModelParams) -> np.ndarray:
    """E(|x_i - x_j|): how much the platform promotes content at that distance."""
    if p.engagement == "similarity":
        return np.exp(-p.gamma * delta_x)
    if p.engagement == "neutral":
        return np.ones_like(delta_x)
    if p.engagement == "controversy":
        return np.exp(-((delta_x - p.delta) ** 2) / (2.0 * p.s_width**2))
    raise ValueError(f"unknown engagement kernel: {p.engagement}")


def influence_function(dx: np.ndarray, p: ModelParams) -> np.ndarray:
    """F(x_i, x_j) applied to the opinion difference dx = x_j - x_i.

    Assimilation below eps1, indifference in [eps1, eps2), repulsion above eps2.
    Without repulsion this reduces to plain bounded confidence with threshold eps1.
    """
    a = np.abs(dx)
    out = np.where(a < p.eps1, dx, 0.0)
    if p.repulsion:
        out = np.where(a >= p.eps2, -p.eta * dx, out)
    return out


@dataclass
class Trajectory:
    """Recorded snapshots of a simulation run."""

    times: np.ndarray
    x: np.ndarray            # (n_snapshots, N) opinions
    r: np.ndarray            # (n_snapshots, N, 2) positions
    edges: list = field(default_factory=list)   # list of (n_edges, 2) int arrays, or None
    s: Optional[np.ndarray] = None              # influence strengths
    stubborn: Optional[np.ndarray] = None       # boolean mask
    params: Optional[ModelParams] = None


class Simulation:
    def __init__(self, params: ModelParams):
        self.p = params
        self.rng = np.random.default_rng(params.seed)
        p = params
        n = p.n_agents

        self.r = self.rng.uniform(0.0, p.box_size, size=(n, 2))
        self.x = self.rng.uniform(-1.0, 1.0, size=n)

        # heterogeneous influence strengths (heavy tailed), normalised to mean 1
        if p.heterogeneous_influence:
            s = (1.0 + self.rng.pareto(p.kappa - 1.0, size=n))
            self.s = s / s.mean()
        else:
            self.s = np.ones(n)

        # stubborn agents (zealots): opinions never update; give them the
        # largest influence strengths so they act like media hubs
        self.stubborn = np.zeros(n, dtype=bool)
        if p.n_stubborn > 0:
            idx = np.argsort(self.s)[-p.n_stubborn:]
            self.stubborn[idx] = True
            vals = np.resize(np.asarray(p.stubborn_values, dtype=float), p.n_stubborn)
            self.x[idx] = vals

        # digital adjacency: A[i, j] = 1 means agent i sees content from j
        self.A = np.zeros((n, n), dtype=bool)
        if p.digital:
            self._init_digital_graph()

    # ------------------------------------------------------------------ #

    def _init_digital_graph(self):
        p, n = self.p, self.p.n_agents
        k = min(p.k_digital, n - 1)
        for i in range(n):
            choices = self.rng.choice(n - 1, size=k, replace=False)
            choices[choices >= i] += 1  # exclude self
            self.A[i, choices] = True

    def _pairwise_distance(self) -> np.ndarray:
        """Pairwise Euclidean distances with periodic (minimum-image) boundaries."""
        L = self.p.box_size
        d = self.r[:, None, :] - self.r[None, :, :]
        d -= L * np.round(d / L)
        return np.sqrt((d**2).sum(axis=-1))

    def _pairwise_displacement(self):
        """Minimum-image displacement vectors d[i, j] = r_j - r_i and distances."""
        L = self.p.box_size
        d = self.r[None, :, :] - self.r[:, None, :]
        d -= L * np.round(d / L)
        return d, np.sqrt((d**2).sum(axis=-1))

    def _rewire_digital(self):
        """Each agent independently replaces one attention slot with probability
        rewire_prob * dt, sampling the new source proportionally to
        E(|x_i - x_j|) * s_j (the platform's recommendation distribution)."""
        p, n = self.p, self.p.n_agents
        prob = p.rewire_prob * p.dt
        movers = np.flatnonzero(self.rng.random(n) < prob)
        if movers.size == 0:
            return
        for i in movers:
            current = np.flatnonzero(self.A[i])
            if current.size == 0:
                continue
            drop = self.rng.choice(current)
            w = engagement_kernel(np.abs(self.x[i] - self.x), p) * self.s
            w[i] = 0.0
            w[current] = 0.0
            total = w.sum()
            if total <= 0.0:
                continue
            new = self.rng.choice(n, p=w / total)
            self.A[i, drop] = False
            self.A[i, new] = True

    # ------------------------------------------------------------------ #

    def step(self):
        p = self.p
        n = p.n_agents

        dxm = self.x[None, :] - self.x[:, None]   # dxm[i, j] = x_j - x_i
        adx = np.abs(dxm)

        alpha = p.alpha_total * (1.0 - p.attention_digital)
        beta = p.alpha_total * p.attention_digital

        need_kernel = alpha > 0.0 or p.chi > 0.0
        if need_kernel:
            dvec, dist = self._pairwise_displacement()
            # hard cutoff: without it the normalised drift would let agents
            # with no nearby peers average with the whole box at O(1) rate
            K = np.exp(-(dist**2) / (2.0 * p.ell**2))
            K[dist > p.r_cut_factor * p.ell] = 0.0
            np.fill_diagonal(K, 0.0)

        # --- physical motion: Brownian diffusion + optional homophilic drift
        dr = np.sqrt(2.0 * p.D * p.dt) * self.rng.standard_normal((n, 2))
        if p.chi > 0.0:
            # move toward compatible neighbours, away from incompatible ones
            unit = dvec / np.maximum(dist, 1e-12)[:, :, None]
            g = np.where(adx < p.epsilon, 1.0, -1.0)
            Kg = K * g
            ksum = K.sum(axis=1)
            force = (Kg[:, :, None] * unit).sum(axis=1)
            force /= np.maximum(ksum, 1e-12)[:, None]
            dr += p.chi * force * p.dt
        self.r += dr
        self.r %= p.box_size

        drift = np.zeros(n)

        # --- physical layer: distance kernel x bounded confidence
        if alpha > 0.0:
            W = K * (adx < p.epsilon)
            wsum = W.sum(axis=1)
            drift += alpha * np.where(wsum > 0, (W * dxm).sum(axis=1) / np.maximum(wsum, 1e-12), 0.0)

        # --- digital layer: adjacency x engagement-weighted influence
        if p.digital and beta > 0.0:
            F = influence_function(dxm, p)
            M = self.A * self.s[None, :]
            msum = M.sum(axis=1)
            drift += beta * np.where(msum > 0, (M * F).sum(axis=1) / np.maximum(msum, 1e-12), 0.0)

        noise = p.sigma_x * np.sqrt(p.dt) * self.rng.standard_normal(n)
        dx = drift * p.dt + noise
        dx[self.stubborn] = 0.0
        x_new = self.x + dx
        if p.boundary == "reflect":
            # single fold suffices: per-step increments are << 1
            x_new = np.where(x_new > 1.0, 2.0 - x_new, x_new)
            x_new = np.where(x_new < -1.0, -2.0 - x_new, x_new)
        self.x = np.clip(x_new, -1.0, 1.0)

        if p.digital and p.rewire_prob > 0.0:
            self._rewire_digital()

    # ------------------------------------------------------------------ #

    def run(self, record_edges: bool = False) -> Trajectory:
        p = self.p
        times, xs, rs, edges = [], [], [], []

        def record(t):
            times.append(t * p.dt)
            xs.append(self.x.copy())
            rs.append(self.r.copy())
            if record_edges and p.digital:
                edges.append(np.argwhere(self.A))

        record(0)
        for t in range(1, p.n_steps + 1):
            self.step()
            if t % p.record_every == 0 or t == p.n_steps:
                record(t)

        return Trajectory(
            times=np.asarray(times),
            x=np.asarray(xs),
            r=np.asarray(rs),
            edges=edges,
            s=self.s.copy(),
            stubborn=self.stubborn.copy(),
            params=p,
        )


def run_simulation(params: ModelParams, record_edges: bool = False) -> Trajectory:
    return Simulation(params).run(record_edges=record_edges)
