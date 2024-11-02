import pytest
from kfold.k_fold_split import k_fold_split


@pytest.fixture
def data():
    return [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]


class TestKFoldSplit:
    def test_k_fold_split_k_2(self, data):
        seed = 42
        expected_splits = [
            ([[1.0, 2.0], [7.0, 8.0]], [[3.0, 4.0], [5.0, 6.0], [9.0, 10.0]]),
            ([[3.0, 4.0], [5.0, 6.0], [9.0, 10.0]], [[1.0, 2.0], [7.0, 8.0]]),
        ]

        splits = k_fold_split(data, k=2, seed=seed)
        print(splits)
        assert (
            splits == expected_splits
        ), f"Expected {expected_splits}, but got {splits}"
        assert len(splits) == 2, "There should be 2 splits."
        assert all(
            len(train) + len(test) == len(data) for train, test in splits
        ), "Each split should contain all data points."

    def test_k_fold_split_k_5(self, data):
        seed = 42
        expected_splits = [
            ([[1.0, 2.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], [[3.0, 4.0]]),
            ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0]]),
            ([[1.0, 2.0], [3.0, 4.0], [7.0, 8.0], [9.0, 10.0]], [[5.0, 6.0]]),
            ([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], [[1.0, 2.0]]),
            ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [9.0, 10.0]], [[7.0, 8.0]]),
        ]

        splits = k_fold_split(data, k=5, seed=seed)
        print(splits)

        assert (
            splits == expected_splits
        ), f"Expected {expected_splits}, but got {splits}"
        assert len(splits) == 5, "There should be 5 splits."
        assert all(
            len(train) + len(test) == len(data) for train, test in splits
        ), "Each split should contain all data points."

    def test_invalid_k(self, data):
        with pytest.raises(ValueError, match="k should be an integer greater than 1."):
            k_fold_split(data, k=1)

    def test_k_greater_than_data_length(self, data):
        with pytest.raises(
            ValueError, match="k cannot be greater than the number of data points."
        ):
            k_fold_split(data, k=6)


# Run tests
if __name__ == "__main__":
    pytest.main()
