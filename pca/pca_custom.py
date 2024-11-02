import numpy as np


def pca(data: np.ndarray, k: int) -> np.ndarray:
    if data.size == 0:
        raise ValueError("Input data is empty")

    if data.shape[1] < k:
        raise ValueError(
            f"Number of components ({k}) cannot be larger than number of features ({data.shape[1]})"
        )
    # standardize data, get covarance matrix the eigenv
    data_standardized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    covariance_matrix = np.cov(data_standardized, rowvar=False)
    U, S, Vt = np.linalg.svd(covariance_matrix)

    principal_components = Vt.T[:, :k]

    return principal_components
