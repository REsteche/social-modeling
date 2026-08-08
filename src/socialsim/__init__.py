"""socialsim: stochastic adaptive multiplex model of opinion dynamics.

Agents diffuse in physical space (Brownian motion) while interacting through
two layers: a distance-kernel bounded-confidence physical layer and an
adaptive, algorithmically curated digital layer.
"""

from .model import ModelParams, Simulation, run_simulation
from . import metrics

__all__ = ["ModelParams", "Simulation", "run_simulation", "metrics"]
