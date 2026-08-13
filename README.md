# Mobility, Algorithms, and Attention

A stochastic adaptive multiplex model of opinion dynamics with Brownian agents.

Agents diffuse through a two-dimensional periodic world by Brownian motion and
interact through two layers at once:

- a **physical layer**: short-range, distance-kernel bounded-confidence
  influence (who you bump into);
- a **digital layer**: a directed attention graph curated by a platform
  recommendation kernel, coevolving with the opinions it exposes (who you are
  shown).

A finite attention budget forces digital interaction to displace physical
interaction. The model is built in five nested levels — Brownian motion +
bounded confidence, static long-range links, adaptive homophilic
recommendation, engagement-maximising platforms with repulsive influence and
heavy-tailed influencers, and opinion-dependent (Schelling-type) mobility —
so each mechanism's marginal effect can be isolated.

Headline result: once influence includes a repulsive response to strongly
opposed views, the ranking of platform designs inverts — the *neutral*,
uncurated platform radicalizes the population the most, and algorithmic
homophily paradoxically shields it. A two-bloc reduction and a McKean–Vlasov
mean-field limit make the inversion analytical, including a parameter-free
radicalization threshold in the digital attention share. The paper also
reports a geography null result (without opinion-dependent motion, geography
never predicts opinion) and its repair above a Péclet-number threshold.

## Repository layout

```
├── src/socialsim/          model implementation
│   ├── model.py            SDE integration, digital layer, rewiring
│   └── metrics.py          polarization, clusters, assortativity, modularity, ...
├── scripts/                one script per experiment / paper figure
│   ├── exp1_mobility.py         Level 1: mobility sweep
│   ├── exp2_static_digital.py   Level 2: static digital layer
│   ├── exp3_adaptive.py         Level 3: algorithmic homophily
│   ├── exp4_engagement.py       Level 4: platform designs + repulsion
│   ├── exp5_phase_diagram.py    (D, lambda) phase diagram
│   ├── exp6_robustness.py       inversion robustness: (eta, eps2), gamma, delta
│   ├── exp6b_nscaling.py        system-size scan at constant density (N <= 1600)
│   ├── exp7_threshold.py        radicalization threshold in lambda + finite size
│   ├── exp8_level5.py           Level 5: homophilic mobility, Peclet collapse
│   ├── theory_twobloc.py        two-bloc reduction + McKean-Vlasov mean field
│   └── run_all.py               run the original five experiments
├── visualizations/         Manim animations of the coupled dynamics
├── paper/                  LaTeX source, references, figures, compiled PDF
└── results/                raw sweep data (JSON) produced by the scripts
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .            # installs numpy/scipy/matplotlib/networkx
pip install manim           # optional, only for the videos (needs cairo+pango)
```

## Reproducing the paper

```bash
python scripts/run_all.py   # ~20 min: all experiments + figures into paper/figures/
cd paper && tectonic main.tex
```

Every figure in the paper is produced by exactly one script in `scripts/`;
raw sweep data are written to `results/*.json`.

## Videos

Manim renders of the evolution: agents in physical space coloured by opinion
(dot area tracks the heavy-tailed influence strength), digital attention
edges coloured by influence zone (assimilate / ignore / repel), live opinion
histogram, polarization Var(x) on a fixed axis comparable across scenes,
and the platform's engagement kernel E(Delta) with the repulsive zone
shaded. All labels use the paper's Computer Modern typography. Parameters
are the paper's reference values.

```bash
manim -qm visualizations/opinion_evolution.py LowMobilityScene          # Level 1, D = 1e-5 (Fig. 1)
manim -qm visualizations/opinion_evolution.py HighMobilityScene         # Level 1, D = 1e-1 (Fig. 1)
manim -qm visualizations/opinion_evolution.py SimilarityPlatformScene   # Level 4, lambda = 0.6
manim -qm visualizations/opinion_evolution.py NeutralPlatformScene      # Level 4 headline (Fig. 6)
manim -qm visualizations/opinion_evolution.py ControversyPlatformScene  # Level 4, lambda = 0.6
manim -qm visualizations/opinion_evolution.py HomophilicMobilityScene   # Level 5, Pe = 2 (Fig. 11)
```

Pre-rendered 720p versions are committed under `visualizations/renders/`.

## Minimal usage

```python
from socialsim import ModelParams, run_simulation, metrics

params = ModelParams(n_agents=200, D=1e-3, digital=True, rewire_prob=5.0,
                     engagement="similarity", gamma=4.0, attention_digital=0.5)
traj = run_simulation(params, record_edges=True)
print(metrics.summarize(traj.x[-1], traj.r[-1], params.box_size, params.ell,
                        edges=traj.edges[-1]))
```
