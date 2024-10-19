import numpy as np
import pytest

from custom import k_means_clustering


@pytest.fixture
def k_means_setup():
    points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)]
    k = 2
    initial_centroids = [(1, 1), (10, 1)]
    max_iterations = 10
    return points, k, initial_centroids, max_iterations


class TestKMeansClustering:
    def test_basic_function(self, k_means_setup):
        points, k, initial_centroids, max_iterations = k_means_setup
        expected_centroids = np.round(np.array([(1, 2), (10, 2)]), 4)

        actual_centroids = k_means_clustering(
            points, k, initial_centroids, max_iterations
        )

        assert np.allclose(expected_centroids, actual_centroids, rtol=1e-5)

    def test_empty_points(self, k_means_setup):
        _, k, initial_centroids, max_iterations = k_means_setup
        points = np.array([])

        with pytest.raises(
            ValueError,
            match="The points array is empty. Please provide a non-empty array of points.",
        ):
            k_means_clustering(points, k, initial_centroids, max_iterations)

    def test_limited_iteration(self, k_means_setup):
        points, k, initial_centroids, _ = k_means_setup
        max_iterations = 1

        centroids = k_means_clustering(points, k, initial_centroids, max_iterations)

        assert len(centroids) == k
        assert all(isinstance(centroid, tuple) for centroid in centroids)

    def test_single_point(self, k_means_setup):
        _, k, initial_centroids, max_iteration = k_means_setup
        points = np.array([(1, 2)])

        centroids = k_means_clustering(points, k, initial_centroids, max_iteration)

        assert np.allclose(centroids, np.array([(1, 2), (10, 1)]), rtol=1e-5)


if __name__ == "__main__":
    pytest.main()
