import re
import numpy as np
import pytest

from pca.pca_custom import pca


@pytest.fixture
def pca_setup():
    return np.array(
        [
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2, 1.6],
            [1, 1.1],
            [1.5, 1.6],
            [1.1, 0.9],
        ]
    )


def test_pca_custom_normal_case(pca_setup):
    k = 2
    expected_result = np.array([[-0.70711, -0.70711], [-0.70711, 0.70711]])

    principal_components = pca(pca_setup, k)

    np.testing.assert_almost_equal(
        principal_components,
        expected_result,
        decimal=5,
        err_msg="Principal components do not match expected result",
    )


def test_pca_single_component(pca_setup):
    k = 1
    expected_result = np.array([[-0.70711], [-0.70711]])

    principal_components = pca(pca_setup, k)

    np.testing.assert_almost_equal(
        principal_components,
        expected_result,
        decimal=5,
        err_msg="Principal components do not match expected result",
    )


def test_pca_more_components_than_features(pca_setup):
    k = 3
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Number of components (3) cannot be larger than number of features (2)"
        ),
    ):
        pca(pca_setup, k)


def test_perform_pca_empty_input():
    X = np.array([])
    k = 2
    with pytest.raises(ValueError, match="Input data is empty"):
        pca(X, k)
