"""Manim animation of the coupled physical/digital opinion dynamics.

Left: agents diffusing in the periodic physical box, coloured by opinion,
with a sample of the current digital attention edges drawn between them.
Right: the instantaneous opinion histogram and the polarization time series.

Render (from the repository root, with the venv active):

    manim -qm visualizations/opinion_evolution.py SimilarityPlatformScene
    manim -qm visualizations/opinion_evolution.py ControversyPlatformScene
    manim -qm visualizations/opinion_evolution.py MobilityConsensusScene
"""

import sys
from pathlib import Path

import numpy as np
from manim import (
    BLACK, WHITE, GRAY, UP, DOWN, LEFT, RIGHT, ORIGIN,
    Axes, Create, Dot, FadeIn, Line, Rectangle, Scene, Text, VGroup,
    ValueTracker, always_redraw, color_gradient, rgb_to_color,
)
import matplotlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socialsim import ModelParams, run_simulation  # noqa: E402

COOLWARM = matplotlib.colormaps["coolwarm"]


def opinion_color(x):
    r, g, b, _ = COOLWARM((float(x) + 1.0) / 2.0)
    return rgb_to_color((r, g, b))


class EvolutionScene(Scene):
    """Base scene: subclasses set PARAMS and TITLE."""

    PARAMS: ModelParams = None
    TITLE: str = ""
    MAX_EDGES = 120          # digital edges drawn per frame (sampled)
    DURATION = 24.0          # seconds of animation

    def construct(self):
        self.camera.background_color = "#101418"
        traj = run_simulation(self.PARAMS, record_edges=True)
        n_frames = len(traj.times)
        L = self.PARAMS.box_size
        variance = traj.x.var(axis=1)

        # ---------- layout
        box_side = 5.6
        box = Rectangle(width=box_side, height=box_side, stroke_color=GRAY,
                        stroke_width=1.5).shift(LEFT * 3.2)
        box_origin = box.get_corner(DOWN + LEFT)

        def to_scene(pos):
            return box_origin + np.array([pos[0] / L * box_side,
                                          pos[1] / L * box_side, 0.0])

        title = Text(self.TITLE, font_size=26, color=WHITE)
        title.to_edge(UP, buff=0.25)

        frame = ValueTracker(0.0)

        def fidx():
            return min(int(frame.get_value()), n_frames - 1)

        # ---------- agents
        n = self.PARAMS.n_agents
        dots = VGroup(*[Dot(radius=0.045) for _ in range(n)])

        def update_dots(group):
            k = fidx()
            for i, d in enumerate(group):
                d.move_to(to_scene(traj.r[k, i]))
                d.set_color(opinion_color(traj.x[k, i]))
        dots.add_updater(update_dots)

        # ---------- digital edges (thin, sampled for readability)
        rng = np.random.default_rng(0)

        def make_edges():
            k = fidx()
            group = VGroup()
            if not traj.edges:
                return group
            E = traj.edges[k]
            if len(E) > self.MAX_EDGES:
                E = E[rng.choice(len(E), self.MAX_EDGES, replace=False)]
            for i, j in E:
                a, b = traj.r[k, i], traj.r[k, j]
                # skip edges that wrap around the periodic boundary
                if np.any(np.abs(a - b) > L / 2):
                    continue
                group.add(Line(to_scene(a), to_scene(b), stroke_width=0.6,
                               stroke_opacity=0.25, color=WHITE))
            return group
        edges = always_redraw(make_edges)

        # ---------- opinion histogram
        bins = np.linspace(-1, 1, 21)
        hist_axes = Axes(x_range=[-1, 1, 0.5], y_range=[0, 0.6, 0.15],
                         x_length=4.4, y_length=2.0,
                         axis_config={"stroke_width": 1.5,
                                      "include_ticks": False},
                         tips=False).shift(RIGHT * 3.4 + UP * 1.7)
        hist_label = Text("opinion distribution", font_size=18,
                          color=WHITE).next_to(hist_axes, UP, buff=0.15)

        bar_colors = color_gradient(
            [opinion_color(-1), opinion_color(0), opinion_color(1)],
            len(bins) - 1)

        def make_hist():
            k = fidx()
            counts, _ = np.histogram(traj.x[k], bins=bins)
            frac = counts / counts.sum()
            group = VGroup()
            for b in range(len(bins) - 1):
                h = min(float(frac[b]), 0.6)
                x0 = hist_axes.c2p(bins[b], 0)
                x1 = hist_axes.c2p(bins[b + 1], h)
                bar = Rectangle(width=abs(x1[0] - x0[0]) * 0.9,
                                height=max(abs(x1[1] - x0[1]), 1e-3),
                                fill_color=bar_colors[b], fill_opacity=0.9,
                                stroke_width=0)
                bar.move_to([(x0[0] + x1[0]) / 2, (x0[1] + x1[1]) / 2, 0])
                group.add(bar)
            return group
        hist = always_redraw(make_hist)

        # ---------- polarization time series
        var_axes = Axes(x_range=[0, traj.times[-1], traj.times[-1] / 4],
                        y_range=[0, max(0.05, variance.max() * 1.1),
                                 max(0.05, variance.max() * 1.1) / 4],
                        x_length=4.4, y_length=1.8,
                        axis_config={"stroke_width": 1.5,
                                     "include_ticks": False},
                        tips=False).shift(RIGHT * 3.4 + DOWN * 1.9)
        var_label = Text("polarization  Var(x)", font_size=18,
                         color=WHITE).next_to(var_axes, UP, buff=0.15)

        def make_var_curve():
            k = fidx()
            pts = [var_axes.c2p(traj.times[m], variance[m])
                   for m in range(0, k + 1, 2)]
            curve = VGroup()
            if len(pts) > 1:
                line = Line(pts[0], pts[0])
                vm = line.copy()
                vm.set_points_as_corners(pts)
                vm.set_stroke("#e0b040", width=2.5)
                curve.add(vm)
            return curve
        var_curve = always_redraw(make_var_curve)

        clock = always_redraw(lambda: Text(
            f"t = {traj.times[fidx()]:.1f}", font_size=20, color=WHITE
        ).next_to(box, DOWN, buff=0.2))

        self.add(box, edges, dots, hist_axes, hist, var_axes, var_curve,
                 clock)
        self.play(FadeIn(title), FadeIn(hist_label), FadeIn(var_label),
                  run_time=0.8)
        self.play(frame.animate.set_value(n_frames - 1),
                  run_time=self.DURATION, rate_func=lambda t: t)
        dots.clear_updaters()
        self.wait(1.5)


def _base_params(**kw):
    defaults = dict(n_agents=150, D=1e-3, dt=0.02, n_steps=3000,
                    epsilon=0.3, ell=0.02, sigma_x=0.02,
                    digital=True, k_digital=8, rewire_prob=5.0,
                    attention_digital=0.6, record_every=10, seed=3)
    defaults.update(kw)
    return ModelParams(**defaults)


class SimilarityPlatformScene(EvolutionScene):
    PARAMS = _base_params(engagement="similarity", gamma=4.0)
    TITLE = "Similarity-driven platform: echo chambers without geography"


class ControversyPlatformScene(EvolutionScene):
    PARAMS = _base_params(engagement="controversy", delta=0.8, s_width=0.2,
                          repulsion=True, eps2=0.9, eta=0.4,
                          heterogeneous_influence=True)
    TITLE = "Controversy-driven platform: engagement-fuelled polarization"


class MobilityConsensusScene(EvolutionScene):
    PARAMS = _base_params(D=5e-2, digital=False, attention_digital=0.0)
    TITLE = "High mobility, no digital layer: mixing produces consensus"
