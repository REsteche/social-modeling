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
interaction. The model is built in four nested levels — Brownian motion +
bounded confidence, static long-range links, adaptive homophilic
recommendation, and engagement-maximising platforms with repulsive influence
and heavy-tailed influencers — so each mechanism's marginal effect can be
isolated. The accompanying paper reports the resulting phase diagram in the
mobility / digital-attention plane, and a null result: without
opinion-dependent motion, geography never becomes a predictor of opinion.

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
│   └── run_all.py               run everything
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

Manim renders of the evolution (agents in physical space coloured by opinion,
digital attention edges, live opinion histogram and polarization curve):

```bash
manim -qm visualizations/opinion_evolution.py SimilarityPlatformScene
manim -qm visualizations/opinion_evolution.py ControversyPlatformScene
manim -qm visualizations/opinion_evolution.py MobilityConsensusScene
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
