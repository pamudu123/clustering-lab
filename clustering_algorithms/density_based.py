import numpy as np
from sklearn.metrics import silhouette_score

def _safe_metric(X, lbl):
    return (None if len(set(lbl)) in (0, 1, len(X))
            else silhouette_score(X, lbl))

def _dbscan(X, eps=0.5, min_samples=5, **kw):
    from sklearn.cluster import DBSCAN
    mdl = DBSCAN(eps=eps, min_samples=min_samples,
                 **{k:v for k,v in kw.items() if k not in {"vis","outdir"}}).fit(X)
    lbl = mdl.labels_
    return lbl, mdl, {"silhouette": _safe_metric(X,lbl)}

def _optics(X, min_samples=10, **kw):
    from sklearn.cluster import OPTICS
    mdl = OPTICS(min_samples=min_samples,
                 **{k:v for k,v in kw.items() if k not in {"vis","outdir"}}).fit(X)
    return mdl.labels_, mdl, {"silhouette": _safe_metric(X, mdl.labels_)}

def _hdbscan(X, min_cluster_size=10, **kw):
    try:
        import hdbscan
    except ImportError:
        raise ImportError("pip install hdbscan to use this adapter")
    mdl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                          **{k:v for k,v in kw.items() if k not in {"vis","outdir"}}).fit(X)
    lbl = mdl.labels_
    return lbl, mdl, {"silhouette": _safe_metric(X,lbl)}

def _mean_shift(X, bandwidth=None, **kw):
    from sklearn.cluster import MeanShift, estimate_bandwidth
    if bandwidth is None:
        bandwidth = estimate_bandwidth(X, quantile=kw.pop("quantile",0.2))
    mdl = MeanShift(bandwidth=bandwidth,
        **{k:v for k,v in kw.items() if k not in {"vis","outdir"}}).fit(X)
    lbl = mdl.labels_
    return lbl, mdl, {"silhouette": _safe_metric(X,lbl)}
