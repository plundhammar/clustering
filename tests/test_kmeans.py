"""Tests for the KMeans clustering algorithm."""

import numpy as np
import pytest

from clustering import KMeans
from conftest import make_blobs


class TestKMeansBasic:
    def test_fit_returns_self(self):
        X = make_blobs()
        model = KMeans(n_clusters=2, random_state=42)
        assert model.fit(X) is model

    def test_labels_shape(self):
        X = make_blobs()
        model = KMeans(n_clusters=2, random_state=42).fit(X)
        assert model.labels_.shape == (len(X),)

    def test_centroids_shape(self):
        X = make_blobs()
        model = KMeans(n_clusters=2, random_state=42).fit(X)
        assert model.centroids_.shape == (2, 2)

    def test_two_clusters_separated(self):
        """Points from the two blobs should receive different labels."""
        X = make_blobs()
        model = KMeans(n_clusters=2, random_state=42).fit(X)
        # first 50 points should share one label, last 50 another
        assert len(set(model.labels_[:50])) == 1
        assert len(set(model.labels_[50:])) == 1
        assert model.labels_[0] != model.labels_[50]

    def test_predict_consistent_with_fit(self):
        X = make_blobs()
        model = KMeans(n_clusters=2, random_state=42).fit(X)
        predicted = model.predict(X)
        np.testing.assert_array_equal(model.labels_, predicted)


class TestKMeansValidation:
    def test_invalid_n_clusters(self):
        with pytest.raises(ValueError):
            KMeans(n_clusters=0)

    def test_more_clusters_than_samples(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError):
            KMeans(n_clusters=5).fit(X)

    def test_1d_input_raises(self):
        with pytest.raises(ValueError):
            KMeans(n_clusters=2).fit(np.array([1.0, 2.0, 3.0]))

    def test_predict_before_fit_raises(self):
        model = KMeans(n_clusters=2)
        with pytest.raises(RuntimeError):
            model.predict(np.array([[1.0, 2.0]]))
