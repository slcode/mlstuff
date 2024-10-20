import torch
from typing import List, Tuple, Optional


def k_means_clustering(
    points: List[List[float]],
    k: int,
    initial_centroids: Optional[List[List[float]]] = None,
    max_iterations: int = 10,
) -> Tuple[List[List[float]], List[int]]:
    if len(points) == 0:
        raise ValueError(
            "The points array is empty. Please provide a non-empty array of points."
        )

    points = torch.tensor(points, dtype=torch.float)

    if initial_centroids is not None:
        centroids = torch.tensor(initial_centroids, dtype=torch.float)
    else:
        centroids = points[torch.randperm(points.size(0))[:k]]

    for _ in range(max_iterations):
        distances = torch.cdist(points, centroids)
        assignments = torch.argmin(distances, dim=1)
        new_centroids = torch.stack(
            [
                (
                    points[assignments == i].mean(dim=0)
                    if len(points[assignments == i]) > 0
                    else centroids[i]
                )
                for i in range(k)
            ]
        )

        if torch.allclose(centroids, new_centroids, rtol=1e-5):
            break

        centroids = new_centroids

    return centroids.tolist(), assignments.tolist()
