"""Shared helpers for experiment scripts: paths, plotting style, persistence."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIG_DIR = ROOT / "paper" / "figures"
RESULTS_DIR = ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # match the paper's (REVTeX / Computer Modern) typography
    "font.family": "serif",
    "font.serif": ["cmr10", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

OPINION_CMAP = "coolwarm"


def run_parallel(worker, jobs, chunksize=2):
    """Run `worker` over `jobs` with a process pool; preserves job order."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    workers = max(1, (os.cpu_count() or 2) - 1)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, row in enumerate(pool.map(worker, jobs, chunksize=chunksize)):
            rows.append(row)
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(jobs)} runs done", flush=True)
    return rows


def save_fig(fig, name: str):
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(path)
    fig.savefig(FIG_DIR / f"{name}.png")
    print(f"saved {path.relative_to(ROOT)}")


def save_json(obj, name: str):
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=default))
    print(f"saved {path.relative_to(ROOT)}")


def plot_trajectories(ax, traj, max_agents=None, lw=0.6, alpha=0.5):
    """Opinion trajectories coloured by final opinion."""
    x = traj.x
    n = x.shape[1] if max_agents is None else min(max_agents, x.shape[1])
    cmap = plt.get_cmap(OPINION_CMAP)
    for i in range(n):
        ax.plot(traj.times, x[:, i], lw=lw, alpha=alpha,
                color=cmap((x[-1, i] + 1) / 2))
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("$t$")


def plot_spatial(ax, traj, snapshot=-1, size=14):
    """Spatial snapshot with agents coloured by opinion."""
    r, x = traj.r[snapshot], traj.x[snapshot]
    L = traj.params.box_size
    sc = ax.scatter(r[:, 0], r[:, 1], c=x, cmap=OPINION_CMAP, vmin=-1, vmax=1,
                    s=size, edgecolors="k", linewidths=0.2)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return sc
