import numpy as np
import pandas as pd
import pytest

from recommender_gnn.pipelines.otto_preprocessing.nodes import (
    _stratify,
    sample,
)


class TestStratify:
    @pytest.mark.parametrize(
        "sessions_frac, expected_result",
        [(0.1, 687), (0.3, 2060), (0.4, 2746)],
    )
    def test_stratification_shape_given_sessions_frac(
        self, otto_dummy_df, sessions_frac, expected_result
    ):
        np.random.seed(0)
        assert len(_stratify(otto_dummy_df, sessions_frac)) == expected_result

    @pytest.mark.parametrize(
        "sessions_frac, expected_result",
        [
            (
                0.001,
                [13111813, 13717906, 14010015, 14332907, 13093193, 13611696, 13363021],
            ),
            (
                0.0011,
                [
                    13111813,
                    13717906,
                    14010015,
                    14306966,
                    13093193,
                    13611696,
                    13363021,
                    14332907,
                ],
            ),
            (
                0.0012,
                [
                    13075074,
                    13717906,
                    14166868,
                    14306966,
                    14332907,
                    13038803,
                    14010015,
                    13611696,
                    13363021,
                ],
            ),
        ],
    )
    def test_stratification_output_given_sessions_frac(
        self, otto_dummy_df, sessions_frac, expected_result
    ):
        np.random.seed(0)
        assert all(_stratify(otto_dummy_df, sessions_frac) == expected_result)

    def test_stratification_given_empty_df(self):
        empty_df = pd.DataFrame()
        with pytest.raises(KeyError):
            _stratify(empty_df, 0.1)


class TestSample:
    @pytest.mark.parametrize(
        "sessions_frac, expected_result",
        [(0, 0), (0.1, 697), (1, 6928)],
    )
    def test_sample_shape_given_sessions_frac(
        self, otto_dummy_df, sessions_frac, expected_result
    ):
        np.random.seed(0)
        sample_df = sample(otto_dummy_df, sessions_frac)
        assert sample_df.shape[0] == expected_result

    @pytest.mark.parametrize(
        "sessions_frac, expected_result",
        [
            (0.001, [14060852, 13220050, 13832718, 13147422, 14448265, 13711093]),
            (
                0.0011,
                [14060852, 13220050, 12967871, 13832718, 13147422, 14448265, 13711093],
            ),
            (
                0.0012,
                [
                    14060852,
                    13220050,
                    12967871,
                    13832718,
                    13147422,
                    13600052,
                    14448265,
                    13711093,
                ],
            ),
        ],
    )
    def test_sample_output_values_given_sessions_frac_without_stratification(
        self, otto_dummy_df, sessions_frac, expected_result
    ):
        np.random.seed(0)
        sample_df = sample(otto_dummy_df, sessions_frac)
        assert list(sample_df.loc[:, "session"]) == expected_result
