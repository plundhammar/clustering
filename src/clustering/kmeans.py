"""K-Means clustering algorithm."""

import numpy as np


class KMeans:
    """K-Means clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form.
    max_iter : int, optional
        Maximum number of iterations (default: 300).
    tol : float, optional
        Convergence tolerance based on centroid movement (default: 1e-4).
    random_state : int or None, optional
        Seed for reproducible centroid initialisation (default: None).
    """

    def __init__(self, n_clusters: int, max_iter: int = 300, tol: float = 1e-4,
                 random_state=None):
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray) -> "KMeans":
        """Compute K-Means clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        n_samples, _ = X.shape
        if n_samples < self.n_clusters:
            raise ValueError("n_samples must be >= n_clusters")

        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = X[indices].copy()

        for i in range(self.max_iter):
            labels = self._assign(X, centroids)
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
                for k in range(self.n_clusters)
            ])
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift <= self.tol:
                self.n_iter_ = i + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.centroids_ = centroids
        self.labels_ = labels
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign cluster labels to new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        if self.centroids_ is None:
            raise RuntimeError("Call fit() before predict()")
        X = np.asarray(X, dtype=float)
        return self._assign(X, self.centroids_)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Return the index of the nearest centroid for each sample."""
        dists = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        return np.argmin(dists, axis=1)
