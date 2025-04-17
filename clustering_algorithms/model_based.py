import numpy as np
from sklearn.metrics import silhouette_score

def _safe_metric(X, lbl):
    return (None if len(set(lbl)) in (0, 1, len(X))
            else silhouette_score(X, lbl))

def _gmm(X, n_clusters=3, **kw):
    from sklearn.mixture import GaussianMixture
    mdl = GaussianMixture(n_components=n_clusters, random_state=42,
                          **{k:v for k,v in kw.items() if k not in {"vis","outdir"}}).fit(X)
    lbl = mdl.predict(X)
    return lbl, mdl, {"aic": mdl.aic(X), "bic": mdl.bic(X), "silhouette": _safe_metric(X,lbl)}

def _bgmm(X, n_clusters=5, **kw):
    from sklearn.mixture import BayesianGaussianMixture
    mdl = BayesianGaussianMixture(n_components=n_clusters, random_state=42,
        **{k:v for k,v in kw.items() if k not in {"vis","outdir"}}).fit(X)
    lbl = mdl.predict(X)
    return lbl, mdl, {"aic": mdl.aic(X), "bic": mdl.bic(X), "silhouette": _safe_metric(X,lbl)}

def _agglomerative(X, n_clusters=3, linkage="ward", **kw):
    from sklearn.cluster import AgglomerativeClustering
    mdl = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage,
        **{k:v for k,v in kw.items() if k not in {"vis","outdir"}}).fit(X)
    lbl = mdl.labels_
    return lbl, mdl, {"silhouette": _safe_metric(X,lbl)}

def _agglomerative_average(X, n_clusters=3, **kw):
    return _agglomerative(X, n_clusters=n_clusters, linkage="average", **kw)

def _birch(X, n_clusters=3, threshold=0.5, **kw):
    from sklearn.cluster import Birch
    mdl = Birch(n_clusters=n_clusters, threshold=threshold,
        **{k:v for k,v in kw.items() if k not in {"vis","outdir"}}).fit(X)
    lbl = mdl.labels_
    return lbl, mdl, {"silhouette": _safe_metric(X,lbl)}

def _spectral(X, n_clusters=3, affinity="rbf", **kw):
    from sklearn.cluster import SpectralClustering
    mdl = SpectralClustering(n_clusters=n_clusters, affinity=affinity,
        random_state=42, **{k:v for k,v in kw.items() if k not in {"vis","outdir"}})
    lbl = mdl.fit_predict(X)
    return lbl, mdl, {"silhouette": _safe_metric(X,lbl)}
