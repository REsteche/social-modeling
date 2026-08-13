"""Manim animations of the coupled physical/digital opinion dynamics.

Each scene shows, side by side:

  left   agents diffusing in the periodic physical box, coloured by opinion
         (dot area tracks the heavy-tailed influence strength s_i when
         heterogeneous influence is on), with the current digital attention
         edges coloured by the influence zone of the exposure they carry:
         assimilative (|x_i - x_j| < eps1), ignored, or repulsive
         (|x_i - x_j| >= eps2);
  right  the instantaneous opinion histogram, the polarization time series
         Var(x) on a fixed [0, 1] axis (comparable across scenes), and, for
         the Level-4 scenes, the platform's engagement kernel E(Delta) with
         the repulsive zone shaded.

All text is typeset with matplotlib's Computer Modern mathtext -- the same
typography as the paper's figures -- rendered to SVG and imported into
Manim, so no external LaTeX installation is required.

Scene / paper correspondence (parameters = the paper's reference values):

  LowMobilityScene          Level 1, D = 1e-5   (Fig. 1, leftmost column)
  HighMobilityScene         Level 1, D = 1e-1   (Fig. 1, rightmost column)
  SimilarityPlatformScene   Level 4, lambda = 0.6, similarity kernel
  NeutralPlatformScene      Level 4, lambda = 0.6, neutral kernel (Fig. 6)
  ControversyPlatformScene  Level 4, lambda = 0.6, controversy kernel
  HomophilicMobilityScene   Level 5, chi = 0.1, D = 1e-4 (Pe = 20, Fig. 11)

Render (from the repository root, with the venv active):

    manim -qm visualizations/opinion_evolution.py NeutralPlatformScene
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath

from manim import (
    WHITE, GRAY, UP, DOWN, LEFT, RIGHT,
    Axes, Dot, FadeIn, Line, DashedLine, Polygon, Rectangle, Scene,
    SVGMobject, VGroup, ValueTracker, always_redraw, color_gradient,
    rgb_to_color,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socialsim import ModelParams, run_simulation  # noqa: E402
from socialsim.model import engagement_kernel      # noqa: E402

COOLWARM = matplotlib.colormaps["coolwarm"]

# influence-zone palette for digital edges (assimilate / ignore / repel)
C_ASSIM = "#6b8fbf"
C_IGNORE = "#9a9a9a"
C_REPEL = "#c0504d"
C_VAR = "#e0b040"

# ---------------------------------------------------------------------- #
# Paper-typography text: matplotlib CM mathtext -> SVG -> Manim
# ---------------------------------------------------------------------- #

_TEX_DIR = ROOT / "visualizations" / ".tex_cache"
_TEX_DIR.mkdir(exist_ok=True)
# scene units per point of rendered text, at fontsize 30
_PT_TO_SCENE = 0.60 / 30.0

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "svg.fonttype": "path",
    "axes.unicode_minus": False,
})

_MOBJECT_CACHE: dict = {}
_SERIF = FontProperties(family="serif")


def _textpath_to_svg(s: str, size: float, out: Path) -> float:
    """Render `s` to a single-<path> SVG (y flipped to SVG convention).

    Returns the text height in points. Using one explicit path avoids the
    <defs>/<use> glyph structure of matplotlib's figure SVGs, which Manim's
    parser cannot ingest.
    """
    tp = TextPath((0.0, 0.0), s, size=size, prop=_SERIF, usetex=False)
    verts, codes = tp.vertices, tp.codes
    d, i = [], 0
    while i < len(codes):
        c = codes[i]
        if c == MplPath.MOVETO:
            d.append(f"M {verts[i][0]:.2f} {-verts[i][1]:.2f}")
            i += 1
        elif c == MplPath.LINETO:
            d.append(f"L {verts[i][0]:.2f} {-verts[i][1]:.2f}")
            i += 1
        elif c == MplPath.CURVE3:
            (x1, y1), (x2, y2) = verts[i], verts[i + 1]
            d.append(f"Q {x1:.2f} {-y1:.2f} {x2:.2f} {-y2:.2f}")
            i += 2
        elif c == MplPath.CURVE4:
            (x1, y1), (x2, y2), (x3, y3) = verts[i], verts[i + 1], verts[i + 2]
            d.append(f"C {x1:.2f} {-y1:.2f} {x2:.2f} {-y2:.2f} "
                     f"{x3:.2f} {-y3:.2f}")
            i += 3
        else:                      # CLOSEPOLY / STOP
            d.append("Z")
            i += 1
    bbox = tp.get_extents()
    pad = 0.5
    x0, w = bbox.x0 - pad, bbox.width + 2 * pad
    y0, h = -bbox.y1 - pad, bbox.height + 2 * pad
    out.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.2f}pt" '
        f'height="{h:.2f}pt" viewBox="{x0:.2f} {y0:.2f} {w:.2f} {h:.2f}">'
        f'<path d="{" ".join(d)}" fill="#ffffff"/></svg>')
    return h


def tex(s: str, size: float = 30.0, color=WHITE) -> SVGMobject:
    """Typeset `s` (prose and/or $math$) in the paper's CM typography.

    The mobject is scaled from the text's physical size in points, so
    different strings keep their relative proportions (like text on a
    single page would).
    """
    key = hashlib.md5(f"{s}|{size}".encode()).hexdigest()[:16]
    if key not in _MOBJECT_CACHE:
        path = _TEX_DIR / f"{key}.svg"
        height_pt = _textpath_to_svg(s, size, path)
        m = SVGMobject(str(path))
        m.set_height(height_pt * _PT_TO_SCENE)
        _MOBJECT_CACHE[key] = m
    m = _MOBJECT_CACHE[key].copy()
    m.set_color(color)
    return m


def opinion_color(x):
    r, g, b, _ = COOLWARM((float(x) + 1.0) / 2.0)
    return rgb_to_color((r, g, b))


# ---------------------------------------------------------------------- #
# Base scene
# ---------------------------------------------------------------------- #

class EvolutionScene(Scene):
    """Subclasses set PARAMS, TITLE and PARAM_LINE."""

    PARAMS: ModelParams = None
    TITLE: str = ""
    PARAM_LINE: str = ""
    MAX_EDGES = 140          # digital edges drawn per frame (sampled)
    DURATION = 26.0          # seconds of animation

    def construct(self):
        self.camera.background_color = "#101418"
        p = self.PARAMS
        traj = run_simulation(p, record_edges=True)
        n_frames = len(traj.times)
        L = p.box_size
        variance = traj.x.var(axis=1)
        digital = p.digital and p.attention_digital > 0.0

        frame = ValueTracker(0.0)

        def fidx():
            return min(int(frame.get_value()), n_frames - 1)

        # ---------------- header (clamped to the frame width)
        title = tex(self.TITLE, size=17)
        if title.width > 12.8:
            title.scale_to_fit_width(12.8)
        title.to_edge(UP, buff=0.22)
        params = tex(self.PARAM_LINE, size=13, color="#b8c2cc")
        if params.width > 12.0:
            params.scale_to_fit_width(12.0)
        params.next_to(title, DOWN, buff=0.12)

        # ---------------- physical box (left)
        box_side = 4.7
        box = Rectangle(width=box_side, height=box_side, stroke_color=GRAY,
                        stroke_width=1.5)
        box.to_edge(LEFT, buff=0.8).shift(DOWN * 0.42)
        box_origin = box.get_corner(DOWN + LEFT)

        def to_scene(pos):
            return box_origin + np.array([pos[0] / L * box_side,
                                          pos[1] / L * box_side, 0.0])

        # dot area tracks the influence strength (capped for readability)
        base_r = 0.042
        radii = base_r * np.clip(traj.s, 0.3, 8.0) ** 0.25
        dots = VGroup(*[Dot(radius=float(radii[i]))
                        for i in range(p.n_agents)])

        def update_dots(group):
            k = fidx()
            for i, d in enumerate(group):
                d.move_to(to_scene(traj.r[k, i]))
                d.set_color(opinion_color(traj.x[k, i]))
        dots.add_updater(update_dots)

        # digital edges, coloured by the influence zone of the exposure
        rng = np.random.default_rng(0)

        def edge_style(dx_abs):
            if dx_abs < p.eps1:
                return C_ASSIM, 0.9, 0.45
            if p.repulsion and dx_abs >= p.eps2:
                return C_REPEL, 1.4, 0.65
            return C_IGNORE, 0.6, 0.20

        def make_edges():
            group = VGroup()
            if not (digital and traj.edges):
                return group
            k = fidx()
            E = traj.edges[k]
            if len(E) > self.MAX_EDGES:
                E = E[rng.choice(len(E), self.MAX_EDGES, replace=False)]
            for i, j in E:
                a, b = traj.r[k, i], traj.r[k, j]
                if np.any(np.abs(a - b) > L / 2):   # periodic wrap: skip
                    continue
                color, width, opacity = edge_style(
                    abs(traj.x[k, i] - traj.x[k, j]))
                group.add(Line(to_scene(a), to_scene(b), color=color,
                               stroke_width=width, stroke_opacity=opacity))
            return group
        edges = always_redraw(make_edges)

        # clock under the box (cached per formatted value)
        def make_clock():
            m = tex(f"$t = {traj.times[fidx()]:.1f}$", size=15)
            return m.next_to(box, DOWN, buff=0.12)
        clock = always_redraw(make_clock)

        # edge legend / dot-size caption under the clock
        captions = VGroup()
        if digital:
            legend = VGroup()
            for label, color in [("assimilate", C_ASSIM),
                                 ("ignore", C_IGNORE),
                                 ("repel", C_REPEL)]:
                swatch = Line([0, 0, 0], [0.34, 0, 0], color=color,
                              stroke_width=2.4)
                text = tex(label, size=11.5, color="#b8c2cc")
                text.next_to(swatch, RIGHT, buff=0.08)
                legend.add(VGroup(swatch, text))
            legend.arrange(RIGHT, buff=0.3)
            captions.add(legend)
        if p.heterogeneous_influence:
            captions.add(tex("dot area $\\propto$ influence strength $s_i$",
                             size=11.5, color="#b8c2cc"))
        if len(captions):
            captions.arrange(DOWN, buff=0.08)
            captions.next_to(clock, DOWN, buff=0.10)

        # ---------------- right-hand panels
        panel_x = 4.05
        panel_w = 4.6
        if digital:
            centers = [1.95, -0.35, -2.65]      # hist, Var, kernel
        else:
            centers = [1.55, -1.35, None]

        right = VGroup()

        # --- opinion histogram
        bins = np.linspace(-1, 1, 21)
        hist_h = 1.35
        hist_axes = Axes(
            x_range=[-1, 1, 0.5], y_range=[0, 0.6, 0.2],
            x_length=panel_w, y_length=hist_h,
            axis_config={"stroke_width": 1.5, "include_ticks": False},
            tips=False,
        ).move_to([panel_x, centers[0], 0])
        hist_title = tex("opinion distribution", size=13)
        hist_title.next_to(hist_axes, UP, buff=0.10)
        hist_xticks = VGroup(*[
            tex(f"${v:g}$", size=11, color="#b8c2cc").next_to(
                hist_axes.c2p(v, 0), DOWN, buff=0.10)
            for v in (-1, 0, 1)])
        hist_xlabel = tex("$x$", size=13).next_to(
            hist_axes.c2p(1, 0), RIGHT, buff=0.16)
        right.add(hist_axes, hist_title, hist_xticks, hist_xlabel)

        bar_colors = color_gradient(
            [opinion_color(-1), opinion_color(0), opinion_color(1)],
            len(bins) - 1)

        def make_hist():
            k = fidx()
            counts, _ = np.histogram(traj.x[k], bins=bins)
            frac = counts / max(counts.sum(), 1)
            group = VGroup()
            for b in range(len(bins) - 1):
                h = min(float(frac[b]), 0.6)
                p0 = hist_axes.c2p(bins[b], 0)
                p1 = hist_axes.c2p(bins[b + 1], h)
                bar = Rectangle(width=abs(p1[0] - p0[0]) * 0.9,
                                height=max(abs(p1[1] - p0[1]), 1e-3),
                                fill_color=bar_colors[b], fill_opacity=0.9,
                                stroke_width=0)
                bar.move_to([(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, 0])
                group.add(bar)
            return group
        hist = always_redraw(make_hist)

        # --- polarization time series on a fixed axis (scene-comparable)
        T = float(traj.times[-1])
        var_h = 1.35
        var_axes = Axes(
            x_range=[0, T, T / 4], y_range=[0, 1.05, 0.5],
            x_length=panel_w, y_length=var_h,
            axis_config={"stroke_width": 1.5, "include_ticks": False},
            tips=False,
        ).move_to([panel_x, centers[1], 0])
        var_title = tex("polarization $\\mathrm{Var}(x)$", size=13)
        var_title.next_to(var_axes, UP, buff=0.10)
        var_yticks = VGroup(*[
            tex(f"${v:g}$", size=11, color="#b8c2cc").next_to(
                var_axes.c2p(0, v), LEFT, buff=0.10)
            for v in (0, 0.5, 1)])
        var_xticks = VGroup(*[
            tex(f"${v:g}$", size=11, color="#b8c2cc").next_to(
                var_axes.c2p(v, 0), DOWN, buff=0.10)
            for v in (0, T / 2, T)])
        var_xlabel = tex("$t$", size=13).next_to(
            var_axes.c2p(T, 0), RIGHT, buff=0.16)
        var_top = DashedLine(var_axes.c2p(0, 1.0), var_axes.c2p(T, 1.0),
                             color=GRAY, stroke_width=0.8,
                             dash_length=0.06, stroke_opacity=0.5)
        right.add(var_axes, var_title, var_yticks, var_xticks, var_xlabel,
                  var_top)

        def make_var_curve():
            k = fidx()
            pts = [var_axes.c2p(traj.times[m], min(variance[m], 1.05))
                   for m in range(0, k + 1, 2)]
            curve = VGroup()
            if len(pts) > 1:
                seg = Line(pts[0], pts[0])
                seg.set_points_as_corners(pts)
                seg.set_stroke(C_VAR, width=2.5)
                curve.add(seg)
            return curve
        var_curve = always_redraw(make_var_curve)

        # --- engagement kernel of the platform (Level-4 scenes only)
        kernel_group = VGroup()
        if digital:
            kern_h = 1.15
            kern_axes = Axes(
                x_range=[0, 2, 0.5], y_range=[0, 1.1, 0.5],
                x_length=panel_w, y_length=kern_h,
                axis_config={"stroke_width": 1.5, "include_ticks": False},
                tips=False,
            ).move_to([panel_x, centers[2], 0])
            kern_title = tex("engagement kernel $E(\\Delta)$", size=13)
            kern_title.next_to(kern_axes, UP, buff=0.10)

            d_grid = np.linspace(0.0, 2.0, 200)
            e_vals = engagement_kernel(d_grid, p)
            e_vals = e_vals / e_vals.max()
            graph = kern_axes.plot_line_graph(
                x_values=d_grid, y_values=e_vals, add_vertex_dots=False,
                line_color=WHITE, stroke_width=2.0)

            # repulsive zone Delta >= eps2: the exposure that radicalizes
            zone_pts = ([kern_axes.c2p(p.eps2, 0)]
                        + [kern_axes.c2p(d, e) for d, e in
                           zip(d_grid, e_vals) if d >= p.eps2]
                        + [kern_axes.c2p(2, 0)])
            zone = Polygon(*zone_pts, stroke_width=0,
                           fill_color=C_REPEL, fill_opacity=0.30)

            marks = VGroup()
            for v, name in [(p.eps1, "$\\epsilon_1$"),
                            (p.eps2, "$\\epsilon_2$")]:
                marks.add(DashedLine(kern_axes.c2p(v, 0),
                                     kern_axes.c2p(v, 1.05),
                                     color=GRAY, stroke_width=0.9,
                                     dash_length=0.05))
                marks.add(tex(name, size=11, color="#b8c2cc").next_to(
                    kern_axes.c2p(v, 0), DOWN, buff=0.08))
            kern_xlabel = tex("$\\Delta$", size=13).next_to(
                kern_axes.c2p(2, 0), RIGHT, buff=0.16)
            kernel_group.add(kern_axes, kern_title, zone, graph, marks,
                             kern_xlabel)

        # ---------------- assemble and animate
        self.add(box, edges, dots, hist, var_curve, clock)
        self.add(right, kernel_group, captions)
        self.play(FadeIn(title), FadeIn(params), run_time=0.8)
        self.play(frame.animate.set_value(n_frames - 1),
                  run_time=self.DURATION, rate_func=lambda t: t)
        dots.clear_updaters()
        self.wait(1.5)


# ---------------------------------------------------------------------- #
# Scenes -- parameters are the paper's reference values
# ---------------------------------------------------------------------- #

def _level1_params(**kw):
    """Level 1: Brownian motion + bounded confidence (exp1 reference)."""
    defaults = dict(n_agents=200, D=1e-3, dt=0.02, n_steps=3000,
                    epsilon=0.3, ell=0.02, attention_digital=0.0,
                    record_every=10, seed=3)
    defaults.update(kw)
    return ModelParams(**defaults)


def _level4_params(engagement, **kw):
    """Level 4 reference point: lambda = 0.6, repulsion, heavy tails (exp4)."""
    defaults = dict(n_agents=200, D=1e-3, dt=0.02, n_steps=4000,
                    epsilon=0.3, ell=0.02, sigma_x=0.02,
                    digital=True, k_digital=10, rewire_prob=5.0,
                    engagement=engagement, gamma=4.0, delta=0.8, s_width=0.2,
                    repulsion=True, eps1=0.3, eps2=0.9, eta=0.4,
                    heterogeneous_influence=True, kappa=2.5,
                    attention_digital=0.6, record_every=10, seed=3)
    defaults.update(kw)
    return ModelParams(**defaults)


L4_PARAM_LINE = ("$\\lambda = 0.6$,  $\\epsilon_1 = 0.3$,  "
                 "$\\epsilon_2 = 0.9$,  $\\eta = 0.4$,  $\\kappa = 2.5$,  "
                 "$k = 10$,  $\\rho = 5$,  $N = 200$")


class LowMobilityScene(EvolutionScene):
    """Level 1, D = 1e-5: opinions freeze into many local clusters."""
    PARAMS = _level1_params(D=1e-5)
    TITLE = ("Level 1, frozen agents: locality-induced fragmentation "
             "into many opinion clusters")
    PARAM_LINE = ("$D = 10^{-5}$,  $\\epsilon = 0.3$,  $\\ell = 0.02$,  "
                  "$\\lambda = 0$,  $N = 200$")


class HighMobilityScene(EvolutionScene):
    """Level 1, D = 1e-1: mixing recovers the mean-field BC outcome."""
    PARAMS = _level1_params(D=1e-1)
    TITLE = ("Level 1, fast mixing: Brownian motion recovers the "
             "mean-field bounded-confidence clusters")
    PARAM_LINE = ("$D = 10^{-1}$,  $\\epsilon = 0.3$,  $\\ell = 0.02$,  "
                  "$\\lambda = 0$,  $N = 200$")


class SimilarityPlatformScene(EvolutionScene):
    """Level 4: algorithmic homophily withholds repulsive-zone content."""
    PARAMS = _level4_params("similarity")
    TITLE = ("Level 4, similarity-driven platform: algorithmic homophily "
             "buffers extremism")
    PARAM_LINE = L4_PARAM_LINE + ",  $E(\\Delta) = e^{-\\gamma\\Delta}$"


class NeutralPlatformScene(EvolutionScene):
    """Level 4 headline: uncurated exposure maximizes radicalization."""
    PARAMS = _level4_params("neutral")
    TITLE = ("Level 4, neutral platform: uncurated cross-cutting exposure "
             "radicalizes fastest")
    PARAM_LINE = L4_PARAM_LINE + ",  $E(\\Delta) = 1$"


class ControversyPlatformScene(EvolutionScene):
    """Level 4: engagement kernel tuned near the repulsion threshold."""
    PARAMS = _level4_params("controversy")
    TITLE = ("Level 4, controversy-driven platform: engagement-tuned "
             "exposure, nearly as radicalizing")
    PARAM_LINE = (L4_PARAM_LINE
                  + ",  $\\delta = 0.8$,  $w = 0.2$")


class HomophilicMobilityScene(EvolutionScene):
    """Level 5: Schelling-type drift creates geographic opinion structure.

    Same parameters as the paper's Fig. 11 snapshot (chi = 0.1, D = 1e-3,
    seed 1): Pe = 2, just above the threshold Pe ~ 1.
    """
    PARAMS = _level1_params(chi=0.1, D=1e-3, seed=1)
    # cmr10 has no precomposed "é"; accent it in math mode as in the figures
    TITLE = ("Level 5, homophilic mobility: geographic opinion structure "
             "above the P$\\mathrm{\\acute{e}}$clet threshold")
    PARAM_LINE = ("$\\chi = 0.1$,  $D = 10^{-3}$,  "
                  "$\\mathrm{Pe} = \\chi\\ell/D = 2$,  $\\epsilon = 0.3$,  "
                  "$N = 200$")
