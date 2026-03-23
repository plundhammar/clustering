"""Tests for the DBSCAN clustering algorithm."""

import numpy as np
import pytest

from clustering import DBSCAN
from conftest import make_blobs


class TestDBSCANBasic:
    def test_fit_returns_self(self):
        X = make_blobs()
        model = DBSCAN(eps=0.5, min_samples=3)
        assert model.fit(X) is model

    def test_labels_shape(self):
        X = make_blobs()
        model = DBSCAN(eps=0.5, min_samples=3).fit(X)
        assert model.labels_.shape == (len(X),)

    def test_two_clusters_found(self):
        X = make_blobs()
        model = DBSCAN(eps=0.5, min_samples=3).fit(X)
        cluster_ids = set(model.labels_[model.labels_ != DBSCAN.NOISE])
        assert len(cluster_ids) == 2

    def test_no_noise_on_clean_blobs(self):
        X = make_blobs()
        model = DBSCAN(eps=0.5, min_samples=3).fit(X)
        assert np.all(model.labels_ != DBSCAN.NOISE)

    def test_noise_detection(self):
        """A lone outlier should be labelled as noise."""
        X = np.array([
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
            [0.05, 0.05], [0.02, 0.08],
            [100.0, 100.0],  # outlier
        ])
        model = DBSCAN(eps=0.5, min_samples=3).fit(X)
        assert model.labels_[-1] == DBSCAN.NOISE


class TestDBSCANValidation:
    def test_invalid_eps(self):
        with pytest.raises(ValueError):
            DBSCAN(eps=0)

    def test_invalid_min_samples(self):
        with pytest.raises(ValueError):
            DBSCAN(min_samples=0)

    def test_1d_input_raises(self):
        with pytest.raises(ValueError):
            DBSCAN().fit(np.array([1.0, 2.0, 3.0]))
