# Terminal Usage Guide for Clustering Project

This guide describes how to run clustering algorithms and preprocess your data using the unified `main.py` script. The script supports a variety of clustering methods and allows you to specify algorithm parameters directly from the command line.

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Usage

```bash
python main.py <csv_file> [algorithm] [param1=value1 param2=value2 ...]
```
- `<csv_file>`: Path to your data CSV (required)
- `[algorithm]`: Clustering algorithm key (optional, defaults to `agglomerative_average`)
- `[param1=value1 ...]`: Optional parameters for the chosen algorithm (see below)

**Defaults:**
- If you omit the algorithm, hierarchical clustering with average linkage (`agglomerative_average`) will be used.
- By default, cleaning of unrealistic values is skipped. To enable cleaning, add the `--clean` flag.

## 3. Examples

### K-Means
```bash
python main.py data.csv kmeans n_clusters=4
```

### Mini-Batch K-Means
```bash
python main.py data.csv minibatch n_clusters=4 batch_size=100
```

### K-Medoids (PAM)
```bash
python main.py data.csv kmedoids n_clusters=3
```

### DBSCAN
```bash
python main.py data.csv dbscan eps=0.5 min_samples=5
```

### OPTICS
```bash
python main.py data.csv optics min_samples=10
```

### HDBSCAN
```bash
python main.py data.csv hdbscan min_cluster_size=10
```

### Mean Shift
```bash
python main.py data.csv mean_shift bandwidth=0.6
```

### Hierarchical (Ward linkage)
```bash
python main.py data.csv agglomerative n_clusters=5
```

### Hierarchical (Average linkage)
```bash
python main.py data.csv agglomerative_average n_clusters=5
```

### Spectral (RBF affinity)
```bash
python main.py data.csv spectral n_clusters=4 affinity=rbf
```

### GMM (Full covariance, k=6)
```bash
python main.py data.csv gmm n_clusters=6 covariance_type=full
```

### Autoencoder + K-Means
```bash
python main.py data.csv autoencoder latent_dim=8 epochs=20 batch_size=64
```

---

**Tip:**
- You can add or omit algorithm parameters as needed. If a parameter is not specified, the algorithm's default will be used.
- The output CSV will be named `<algorithm>_clusters.csv` and will contain `patient_id` (if present) and the assigned `cluster_id` for each row.
