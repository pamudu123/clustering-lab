import numpy as np
from sklearn.metrics import silhouette_score

def _safe_metric(X, lbl):
    return (None if len(set(lbl)) in (0, 1, len(X))
            else silhouette_score(X, lbl))

def _kmeans(X, n_clusters=3, **kw):
    from sklearn.cluster import KMeans
    mdl = KMeans(n_clusters=n_clusters, random_state=42,
                 **{k:v for k,v in kw.items() if k not in {"vis","outdir"}}).fit(X)
    lbl = mdl.labels_
    return lbl, mdl, {"inertia": mdl.inertia_, "silhouette": _safe_metric(X,lbl)}

def _minibatch(X, n_clusters=3, **kw):
    from sklearn.cluster import MiniBatchKMeans
    mdl = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                          **{k:v for k,v in kw.items() if k not in {"vis","outdir"}}).fit(X)
    return mdl.labels_, mdl, {"silhouette": _safe_metric(X, mdl.labels_)}

def _kmedoids(X, n_clusters=3, **kw):
    try:
        from sklearn_extra.cluster import KMedoids
    except ImportError:
        raise ImportError("pip install scikit-learn-extra to use K-Medoids")
    mdl = KMedoids(n_clusters=n_clusters, random_state=42,
                   **{k:v for k,v in kw.items() if k not in {"vis","outdir"}}).fit(X)
    lbl = mdl.labels_
    return lbl, mdl, {"silhouette": _safe_metric(X, lbl)}
