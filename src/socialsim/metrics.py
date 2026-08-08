"""Observables for the multiplex opinion-dynamics model."""

from __future__ import annotations

import numpy as np
import networkx as nx


def polarization(x: np.ndarray) -> float:
    """Crude polarization measure P = Var(x)."""
    return float(np.var(x))


def opinion_clusters(x: np.ndarray, gap: float = 0.1, min_size: int = 2):
    """Cluster opinions in 1D by splitting the sorted profile at gaps > `gap`.

    Returns (n_clusters, labels, centers). Agents in groups smaller than
    `min_size` are treated as isolates and not counted as clusters.
    """
    order = np.argsort(x)
    xs = x[order]
    breaks = np.flatnonzero(np.diff(xs) > gap)
    groups = np.split(order, breaks + 1)

    labels = -np.ones(len(x), dtype=int)
    centers = []
    k = 0
    for g in groups:
        if len(g) >= min_size:
            labels[g] = k
            centers.append(float(x[g].mean()))
            k += 1
    return k, labels, np.asarray(centers)


def classify_state(x: np.ndarray, gap: float = 0.1, consensus_var: float = 0.01):
    """Classify the final opinion profile as consensus / polarization / fragmentation.

    - consensus: a single cluster (or negligible variance);
    - polarization: exactly two well-separated clusters;
    - fragmentation: three or more clusters.
    """
    n, _, centers = opinion_clusters(x, gap=gap)
    if n <= 1 or np.var(x) < consensus_var:
        return "consensus"
    if n == 2:
        return "polarization"
    return "fragmentation"


def opinion_assortativity(x: np.ndarray, edges: np.ndarray) -> float:
    """Pearson correlation of opinions across directed digital edges."""
    if edges is None or len(edges) == 0:
        return np.nan
    xi, xj = x[edges[:, 0]], x[edges[:, 1]]
    if np.std(xi) < 1e-12 or np.std(xj) < 1e-12:
        return np.nan
    return float(np.corrcoef(xi, xj)[0, 1])


def digital_modularity(x: np.ndarray, edges: np.ndarray, gap: float = 0.1) -> float:
    """Modularity of the (symmetrised) digital graph with respect to the
    partition induced by the opinion clusters."""
    if edges is None or len(edges) == 0:
        return np.nan
    _, labels, _ = opinion_clusters(x, gap=gap, min_size=1)
    G = nx.Graph()
    G.add_nodes_from(range(len(x)))
    G.add_edges_from(map(tuple, edges))
    communities = [np.flatnonzero(labels == k) for k in np.unique(labels)]
    communities = [set(c.tolist()) for c in communities if len(c) > 0]
    if len(communities) < 2:
        return 0.0
    return float(nx.algorithms.community.modularity(G, communities))


def spatial_opinion_correlation(x: np.ndarray, r: np.ndarray, box_size: float,
                                ell: float) -> float:
    """Moran's I of opinions with Gaussian spatial weights: does geography
    predict opinion? 1 = perfectly clustered in space, ~0 = no structure."""
    n = len(x)
    d = r[:, None, :] - r[None, :, :]
    d -= box_size * np.round(d / box_size)
    dist2 = (d**2).sum(axis=-1)
    W = np.exp(-dist2 / (2.0 * ell**2))
    np.fill_diagonal(W, 0.0)
    z = x - x.mean()
    denom = (z**2).sum()
    if denom < 1e-12 or W.sum() < 1e-12:
        return np.nan
    return float(n / W.sum() * (W * np.outer(z, z)).sum() / denom)


def local_agreement_gap(x: np.ndarray, r: np.ndarray, box_size: float,
                        radius: float, epsilon: float = 0.3) -> float:
    """P(|x_i - x_j| < epsilon | d_ij < radius) - P(|x_i - x_j| < epsilon).

    Positive values mean physically close agents agree more often than a
    random pair: opinions carry spatial structure. Raw opinion-position
    correlations (Moran's I) miss this because Brownian positions are
    opinion-independent, so opposite clusters interleave instead of
    segregating; conditioning on distance exposes the structure.
    """
    n = len(x)
    d = r[:, None, :] - r[None, :, :]
    d -= box_size * np.round(d / box_size)
    dist = np.sqrt((d**2).sum(axis=-1))
    iu = np.triu_indices(n, k=1)
    agree = (np.abs(x[:, None] - x[None, :]) < epsilon)[iu]
    near = (dist < radius)[iu]
    if near.sum() == 0:
        return np.nan
    return float(agree[near].mean() - agree.mean())


def local_disagreement(x: np.ndarray, edges: np.ndarray) -> float:
    """Mean |x_i - mean of i's digital in-neighbourhood|.

    Small values with large global variance indicate echo chambers: everyone
    agrees with what they see while the population disagrees globally.
    """
    if edges is None or len(edges) == 0:
        return np.nan
    n = len(x)
    sums = np.zeros(n)
    counts = np.zeros(n)
    np.add.at(sums, edges[:, 0], x[edges[:, 1]])
    np.add.at(counts, edges[:, 0], 1.0)
    has = counts > 0
    return float(np.mean(np.abs(x[has] - sums[has] / counts[has])))


def summarize(x: np.ndarray, r: np.ndarray, box_size: float, ell: float,
              edges: np.ndarray = None, gap: float = 0.05,
              epsilon: float = 0.3) -> dict:
    n_clusters, _, _ = opinion_clusters(x, gap=gap)
    return {
        "polarization": polarization(x),
        "n_clusters": n_clusters,
        "state": classify_state(x, gap=gap),
        "morans_I": spatial_opinion_correlation(x, r, box_size, ell),
        "local_agreement_gap": local_agreement_gap(x, r, box_size,
                                                   3.0 * ell, epsilon),
        "assortativity": opinion_assortativity(x, edges),
        "modularity": digital_modularity(x, edges, gap=gap),
        "local_disagreement": local_disagreement(x, edges),
    }
