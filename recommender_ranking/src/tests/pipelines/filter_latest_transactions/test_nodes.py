import pandas as pd
import pytest

from gid_ml_framework.pipelines.filter_latest_transactions.nodes import (
    filter_dataframe_by_last_n_weeks,
)


# date_column
def test_not_a_datetime_column(transactions_dummy_df):
    not_a_datetime_col = "customer_id"
    with pytest.raises(ValueError):
        filter_dataframe_by_last_n_weeks(transactions_dummy_df, not_a_datetime_col)


def test_not_existing_column(transactions_dummy_df):
    not_a_col = "does_not_exist"
    with pytest.raises(KeyError):
        filter_dataframe_by_last_n_weeks(transactions_dummy_df, not_a_col)


# no_weeks
def test_negative_weeks(transactions_dummy_df):
    no_weeks = -5
    result_df = filter_dataframe_by_last_n_weeks(
        transactions_dummy_df, "t_dat", no_weeks
    )
    assert result_df.empty is True


def test_zero_weeks(transactions_dummy_df):
    no_weeks = 0
    result_df = filter_dataframe_by_last_n_weeks(
        transactions_dummy_df, "t_dat", no_weeks
    )
    pd.testing.assert_frame_equal(result_df, transactions_dummy_df)


def test_many_weeks(transactions_dummy_df):
    no_weeks = 1_000
    result_df = filter_dataframe_by_last_n_weeks(
        transactions_dummy_df, "t_dat", no_weeks
    )
    pd.testing.assert_frame_equal(result_df, transactions_dummy_df)


def test_few_weeks(transactions_dummy_df):
    no_weeks = 6
    initial_size = len(transactions_dummy_df)
    result_df = filter_dataframe_by_last_n_weeks(
        transactions_dummy_df, "t_dat", no_weeks
    )
    assert len(result_df) <= initial_size


def test_missing_optional_no_weeks(transactions_dummy_df):
    result_df = filter_dataframe_by_last_n_weeks(transactions_dummy_df, "t_dat")
    assert result_df.empty is False


# all columns
def test_input_output_cols(transactions_dummy_df):
    expected_columns = transactions_dummy_df.columns
    result_columns = filter_dataframe_by_last_n_weeks(
        transactions_dummy_df, "t_dat"
    ).columns
    assert set(result_columns) == set(expected_columns)
