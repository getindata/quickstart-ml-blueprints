import numpy as np
import pandas as pd
import pytest
from dateutil.parser._parser import ParserError

from recommender_gnn.pipelines.santander_preprocessing.nodes import (
    _stratify,
    clean_santander,
    filter_santander,
    sample_santander,
    split_santander,
)


class TestStratify:
    @pytest.mark.parametrize(
        "customers_limit, expected_result",
        [(3, 4), (30, 30), (50, 50)],
    )
    def test_sampling(self, santander_dummy_df, customers_limit, expected_result):
        assert len(_stratify(santander_dummy_df, customers_limit)) == expected_result

    @pytest.mark.parametrize(
        "customers_limit, expected_result",
        [
            (5, [90037, 173437, 259205, 965820, 1319458]),
            (4, [90037, 259205, 965820, 1319458]),
        ],
    )
    def test_stratification(self, santander_dummy_df, customers_limit, expected_result):
        np.random.seed(0)
        assert all(_stratify(santander_dummy_df, customers_limit) == expected_result)

    def test_given_not_enough_classes(
        self, santander_small_dummy_df, customers_limit=1
    ):
        with pytest.raises(Exception):
            _stratify(santander_small_dummy_df, customers_limit)


class TestSampleSantander:
    @pytest.mark.parametrize(
        "sample_customer_frac, expected_result",
        [(0, 0), (0.1, 136), (1, 1365)],
    )
    def test_sample_given_customer_frac(
        self, santander_dummy_df, sample_customer_frac, expected_result
    ):
        sample_df = sample_santander(santander_dummy_df, sample_customer_frac)
        assert sample_df.shape[0] == expected_result

    @pytest.mark.parametrize(
        "cutoff_date, expected_result",
        [
            ("2016-05-28", 136),
            ("2015-05-28", 29),
            ("2015-09-28", 60),
            ("2014-09-28", 0),
        ],
    )
    @pytest.mark.parametrize("stratify", [True, False])
    def test_sample_given_cutoff_date(
        self, santander_dummy_df, cutoff_date, expected_result, stratify
    ):
        sample_df = sample_santander(
            santander_dummy_df, cutoff_date=cutoff_date, stratify=stratify
        )
        assert sample_df.shape[0] == expected_result

    @pytest.mark.parametrize(
        "cutoff_date, expected_result",
        [
            ("2016-05-28", "2016-05-28"),
            ("2015-05-28", "2015-05-28"),
            ("2015-09-28", "2015-09-28"),
        ],
    )
    def test_valid_cutoff_date(self, santander_dummy_df, cutoff_date, expected_result):
        sample_df = sample_santander(santander_dummy_df, cutoff_date=cutoff_date)
        assert sample_df["fecha_dato"].max() <= pd.to_datetime(expected_result)

    def test_invalid_cutoff_date(self, santander_dummy_df, cutoff_date="wrong format"):
        with pytest.raises(ParserError):
            sample_santander(santander_dummy_df, cutoff_date=cutoff_date)


def test_filter_santander(santander_dummy_df):
    filtered_df = filter_santander(santander_dummy_df)
    assert filtered_df.shape == (1365, 47)
    assert not set(["tipodom", "cod_prov"]).intersection(filtered_df.columns)


def test_clean_santander(santander_dummy_df):
    cleanded_df = clean_santander(santander_dummy_df)
    assert not any(cleanded_df["nomprov"] == "CORU\xc3\x91A, A")


def test_split_santander(santander_dummy_df):
    train_df, test_df = split_santander(santander_dummy_df)
    assert train_df.shape[1] == test_df.shape[1] == santander_dummy_df.shape[1]
