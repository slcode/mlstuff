import numpy as np
from typing import List, Tuple


def euclidean_distance(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def k_means_clustering(
    points, k, initial_centroids, max_iterations
) -> Tuple[List[List[float]], List[int]]:
    if len(points) == 0:
        raise ValueError(
            "The points array is empty. Please provide a non-empty array of points."
        )

    points = np.array(points)
    centroids = np.array(initial_centroids)

    for iteration in range(max_iterations):
        # Assign points to the nearest centroid
        distances = np.array(
            [euclidean_distance(points, centroid) for centroid in centroids]
        )
        assignments = np.argmin(distances, axis=0)
        # update centroids
        new_centroids = np.array(
            [
                (
                    points[assignments == i].mean(axis=0)
                    if len(points[assignments == i]) > 0
                    else centroids[i]
                )
                for i in range(k)
            ]
        )

        new_centroids = np.round(new_centroids, 4)
        # Check for convergence
        if np.allclose(centroids, new_centroids, rtol=1e-05):
            break

        centroids = new_centroids

    return [list(centroid) for centroid in centroids], assignments
