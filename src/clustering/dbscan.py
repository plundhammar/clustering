"""DBSCAN clustering algorithm."""

import numpy as np


class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Parameters
    ----------
    eps : float
        Maximum distance between two samples to be considered neighbours
        (default: 0.5).
    min_samples : int
        Minimum number of samples in a neighbourhood for a point to be a
        core point (default: 5).
    """

    NOISE = -1

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """Perform DBSCAN clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")

        n_samples = X.shape[0]
        labels = np.full(n_samples, self.NOISE, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        neighbours = self._region_query(X)

        for idx in range(n_samples):
            if visited[idx]:
                continue
            visited[idx] = True
            if len(neighbours[idx]) < self.min_samples:
                continue  # noise (for now)

            # Start a new cluster
            labels[idx] = cluster_id
            seeds = list(neighbours[idx])
            i = 0
            while i < len(seeds):
                pt = seeds[i]
                if not visited[pt]:
                    visited[pt] = True
                    if len(neighbours[pt]) >= self.min_samples:
                        seeds.extend(neighbours[pt])
                if labels[pt] == self.NOISE:
                    labels[pt] = cluster_id
                i += 1

            cluster_id += 1

        self.labels_ = labels
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _region_query(self, X: np.ndarray) -> list[list[int]]:
        """Return indices of all points within *eps* of each point."""
        # Vectorised pairwise distances via broadcasting
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]   # (n, n, d)
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))      # (n, n)
        return [list(np.where(dist_matrix[i] <= self.eps)[0]) for i in range(X.shape[0])]
