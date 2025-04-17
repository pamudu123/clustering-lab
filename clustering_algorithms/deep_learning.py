import numpy as np
from sklearn.metrics import silhouette_score

def _safe_metric(X, lbl):
    return (None if len(set(lbl)) in (0, 1, len(X))
            else silhouette_score(X, lbl))

def _autoencoder(X, n_clusters=3, latent_dim=8, epochs=20, batch_size=64, **kw):
    import torch, torch.nn as nn
    import time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_t = torch.from_numpy(X.astype("float32")).to(device)
    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(X.shape[1],64), nn.ReLU(),
                                     nn.Linear(64,latent_dim))
            self.dec = nn.Sequential(nn.Linear(latent_dim,64), nn.ReLU(),
                                     nn.Linear(64,X.shape[1]))
        def forward(self,x):
            z = self.enc(x); return self.dec(z), z
    ae = AE().to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(X_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    t0=time.time()
    for _ in range(epochs):
        for xb, in dl:
            opt.zero_grad(); recon,_ = ae(xb); loss=crit(recon,xb); loss.backward(); opt.step()
    with torch.no_grad():
        _, z = ae(X_t)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(z.cpu())
    lbl = km.labels_
    metrics = {"silhouette": _safe_metric(z.cpu().numpy(), lbl)}
    ae.cpu();
    return lbl, (ae, km), metrics
