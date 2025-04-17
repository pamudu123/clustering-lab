"""Unified API for classical, density‑based, model‑based, and deep‑learning
clustering — **with built‑in logging and per‑algorithm plots**.

Supported algorithms:
- kmeans
- minibatch
- kmedoids
- dbscan
- optics
- hdbscan
- agglomerative (ward linkage)
- agglomerative_average (average linkage)
- birch
- spectral
- mean_shift
- gmm
- bgmm
- autoencoder

Usage (inside your main script)
-------------------------------
>>> from clustering_algorithms import run_clustering
>>> labels, model, metrics = run_clustering(X, algorithm="kmeans", n_clusters=4,
                                            vis=True, pca=True)

Every adapter returns `(labels, model, metrics)` so downstream code stays
identical.  Set `vis=True` to see a quick‑look plot saved as PNG (or pop‑up if
running in an interactive backend).
"""
from __future__ import annotations
import logging, time, inspect, pathlib, warnings
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

# Import algorithm adapters from submodules
from .partitioning import _kmeans, _minibatch, _kmedoids
from .density_based import _dbscan, _optics, _hdbscan, _mean_shift
from .model_based import _gmm, _bgmm, _agglomerative, _agglomerative_average, _birch, _spectral
from .deep_learning import _autoencoder

# ────────────────────────────── logging setup ───────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s] %(message)s")
log = logging.getLogger("clust")

# ───────────────────────────── helper utils ─────────────────────────────
def _ensure_2d(X):
    """Convert to 2‑D via PCA if dims>2; leave 2‑D data untouched."""
    if X.shape[1] <= 2:
        return X
    return PCA(n_components=2, random_state=42).fit_transform(X)

def _scatter(X2, lbl, centers=None, title="", fname=None):
    """Generic 2‑D scatter with optional centers."""
    plt.figure(figsize=(6, 5))
    uniq = sorted(set(lbl))
    colours = plt.cm.get_cmap("tab10", len(uniq))
    for k in uniq:
        mask = lbl == k
        plt.scatter(X2[mask, 0], X2[mask, 1], s=12,
                    label=f"cluster {k}" if k != -1 else "noise",
                    c=[colours(k)])
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c="red", s=100,
                    marker="x", label="centers")
    plt.title(title)
    plt.legend(fontsize="small")
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=200)
    else:
        plt.show(); plt.close()

def _ellipse(ax, mean, cov, colour):
    """Draw 1‑σ ellipse for a 2‑D Gaussian component."""
    from matplotlib.patches import Ellipse
    import scipy.linalg as LA
    vals, vecs = LA.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals)
    ell = Ellipse(xy=mean, width=width, height=height, angle=theta,
                  edgecolor=colour, fc="none", lw=1)
    ax.add_patch(ell)

# ───────────────────────── visualisation switchboard ────────────────────
def _visualise(method, X, labels, model, outdir=None):
    """Route to an algo‑specific quick plot."""
    X2 = _ensure_2d(X)
    p = (Path(outdir) if outdir else None)
    if method in {"kmeans", "minibatch", "agglomerative", "birch",
                  "spectral", "mean_shift", "dbscan", "optics", "hdbscan"}:
        ctrs = getattr(model, "cluster_centers_", None)
        _scatter(X2, labels, ctrs,
                 title=f"{method.capitalize()} result",
                 fname=p/ f"{method}.png" if p else None)
    elif method in {"gmm", "bgmm"} and X2.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(6,5))
        uniq = sorted(set(labels))
        cmap = plt.cm.get_cmap("tab10", len(uniq))
        for k in uniq:
            m = labels == k
            ax.scatter(X2[m,0], X2[m,1], s=12, color=cmap(k))
        for m,c in zip(model.means_, model.covariances_):
            cov = c if model.covariance_type == "full" else np.diag(c) if model.covariance_type=="diag" else np.eye(2)*c
            _ellipse(ax, m[:2], cov[:2,:2], "black")
        ax.set_title("GMM components (1σ ellipses)")
        fig.tight_layout()
        if p: fig.savefig(p/"gmm.png", dpi=200)
        else: plt.show(); plt.close()
    elif method == "autoencoder":
        _scatter(X2, labels, None, "Autoencoder latent clustering",
                 fname=p/"autoencoder.png" if p else None)
    else:
        log.info("No visualiser for %s", method)

# ───────────────────────── registry + façade ───────────────────────────
_REGISTRY = {
    "kmeans": _kmeans,
    "minibatch": _minibatch,
    "kmedoids": _kmedoids,
    "gmm": _gmm,
    "bgmm": _bgmm,
    "dbscan": _dbscan,
    "optics": _optics,
    "hdbscan": _hdbscan,
    "agglomerative": _agglomerative,
    "agglomerative_average": _agglomerative_average,
    "birch": _birch,
    "spectral": _spectral,
    "mean_shift": _mean_shift,
    "autoencoder": _autoencoder,
}

def run_clustering(X: np.ndarray, algorithm: str, *, vis=False, outdir=None, **params):
    """Fit chosen algorithm and (optionally) drop a figure.

    Parameters
    ----------
    X : ndarray            – feature matrix
    algorithm : str        – key from _REGISTRY
    vis : bool             – draw plot if True
    outdir : str | Path    – save images to this folder instead of showing
    params : **kwargs      – forwarded to underlying adapter
    """
    algo = _REGISTRY.get(algorithm.lower())
    if algo is None:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {list(_REGISTRY)}")
    t0=time.time()
    labels, model, metrics = algo(X, **params)
    log.info("%s done in %.2fs – metrics: %s", algorithm, time.time()-t0, metrics)
    if vis:
        _visualise(algorithm.lower(), X, np.array(labels), model, outdir)
    return labels, model, metrics
