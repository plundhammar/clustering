"""Shared test fixtures and helpers."""

import numpy as np
import pytest


def make_blobs(n: int = 50, sep: float = 5.0, scale: float = 0.3,
               random_state: int = 0) -> np.ndarray:
    """Create two well-separated 2-D Gaussian clusters.

    Parameters
    ----------
    n : int
        Number of points per cluster.
    sep : float
        Distance between the two cluster centres along both axes.
    scale : float
        Standard deviation of each cluster.
    random_state : int
        Seed for reproducibility.
    """
    rng = np.random.default_rng(random_state)
    c1 = rng.normal(loc=[0.0, 0.0], scale=scale, size=(n, 2))
    c2 = rng.normal(loc=[sep, sep], scale=scale, size=(n, 2))
    return np.vstack([c1, c2])


@pytest.fixture
def two_blobs():
    """Pytest fixture returning a two-cluster dataset (100 × 2)."""
    return make_blobs()
