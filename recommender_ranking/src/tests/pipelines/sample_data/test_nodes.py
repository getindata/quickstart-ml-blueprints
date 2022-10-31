import datetime
from math import ceil
from pathlib import Path

import pandas as pd
import pytest

from src.gid_ml_framework.pipelines.sample_data.nodes import (
    _calculate_distinct_transactions,
    _copy_images,
    filter_out_old_transactions,
    sample_customers,
)


# TRANSACTIONS
def test_valid_cutoff_date(transactions_dummy_df):
    cutoff_date = "2020-05-01"
    result_transactions = filter_out_old_transactions(
        transactions_dummy_df, cutoff_date=cutoff_date
    )
    assert result_transactions["t_dat"].min() >= pd.to_datetime("2020-05-01")


def test_future_cutoff_date(transactions_dummy_df):
    cutoff_date = datetime.datetime.today() + datetime.timedelta(days=1)
    result_transactions = filter_out_old_transactions(
        transactions_dummy_df, cutoff_date=cutoff_date
    )
    assert result_transactions.empty is True


def test_invalid_cutoff_date(transactions_dummy_df):
    cutoff_date = "not a date"
    with pytest.raises(TypeError):
        filter_out_old_transactions(transactions_dummy_df, cutoff_date=cutoff_date)


def test_calculate_distinct_transactions_cols(transactions_dummy_df):
    result_df = _calculate_distinct_transactions(transactions_dummy_df)
    assert "no_transactions" in result_df.columns


# CUSTOMERS
def test_sample_customers_cols(customers_dummy_df, transactions_dummy_df):
    result_columns = sample_customers(
        customers_dummy_df, transactions_dummy_df, sample_size=0.5
    ).columns
    expected_columns = [
        "customer_id",
        "FN",
        "Active",
        "club_member_status",
        "fashion_news_frequency",
        "age",
        "postal_code",
    ]
    assert set(result_columns) == set(expected_columns)


def test_sample_customers_test_size(customers_dummy_df, transactions_dummy_df):
    sample_size = 0.5
    initial_size = len(customers_dummy_df)
    result = sample_customers(
        customers_dummy_df, transactions_dummy_df, sample_size=sample_size
    )
    expexted_size = ceil(initial_size * sample_size)
    assert len(result) == expexted_size


def test_bigger_float_sample_size(customers_dummy_df, transactions_dummy_df):
    invalid_sample_size = 100
    with pytest.raises(ValueError) as exc_info:
        sample_customers(
            customers_dummy_df, transactions_dummy_df, sample_size=invalid_sample_size
        )
    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == "sample_size should be a float in the (0, 1) range"


class TestCopyImages:
    def test_copy_images(self):
        path = Path("src/tests/fixtures/img/")
        article_ids = ["0550827007"]
        img_src_dir = path / "from"
        img_dst_dir = path / "to"
        _copy_images(img_src_dir, img_dst_dir, article_ids)

    def test_strings_as_paths(self):
        article_ids = ["0550827007"]
        img_src_dir = "src/tests/fixtures/img/from"
        img_dst_dir = "src/tests/fixtures/img/to"
        with pytest.raises(AttributeError) as exc_info:
            _copy_images(img_src_dir, img_dst_dir, article_ids)
        assert exc_info.value.args[0] == "'str' object has no attribute 'glob'"
