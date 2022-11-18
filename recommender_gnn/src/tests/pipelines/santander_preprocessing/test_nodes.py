import pytest

from recommender_gnn.pipelines.santander_preprocessing.nodes import _stratify


@pytest.mark.parametrize(
    "customers_limit, expected_result",
    [
        (0.2, 1),
        (1, 5),
    ],
)
class TestSampling:
    def test_stratify(santander_dummy_df, customers_limit, expected_result):
        assert len(_stratify(santander_dummy_df, customers_limit)) == expected_result

    def test_sample():
        pass
