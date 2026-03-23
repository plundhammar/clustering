# clustering

A Python library implementing common clustering algorithms from scratch using NumPy.

## Algorithms

| Algorithm | Class | Description |
|-----------|-------|-------------|
| K-Means | `KMeans` | Partitions data into *k* clusters by iteratively updating centroids |
| DBSCAN | `DBSCAN` | Density-based clustering; discovers clusters of arbitrary shape and marks outliers as noise |

## Installation

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from clustering import KMeans, DBSCAN

X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]], dtype=float)

# K-Means
km = KMeans(n_clusters=2, random_state=0).fit(X)
print(km.labels_)      # e.g. [0 0 0 1 1 1]
print(km.centroids_)   # centroid coordinates

# DBSCAN
db = DBSCAN(eps=3, min_samples=2).fit(X)
print(db.labels_)      # -1 marks noise points
```

## Running tests

```bash
pytest tests/
```