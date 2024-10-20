from sklearn.model_selection import KFold
from typing import List, Tuple


def k_fold_split(
    data: List[List[float]], k: int, seed: int = 42
) -> List[Tuple[List[List[float]], List[List[float]]]]:
    if k <= 1:
        raise ValueError("k should be an integer greater than 1.")
    if len(data) < k:
        raise ValueError("k cannot be greater than the number of data points.")

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    splits = []

    for train_index, test_index in kf.split(data):
        train_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]
        splits.append((train_data, test_data))

    return splits
