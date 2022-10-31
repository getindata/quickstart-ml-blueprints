import pandas as pd
import pytest

from src.gid_ml_framework.helpers.utils import (
    filter_dataframe_by_last_n_days,
    log_memory_usage,
    reduce_memory_usage,
)


class TestMemoryUsage:
    def test_log_memory_usage_input_output(self, transactions_dummy_df):
        @log_memory_usage
        def my_function(df):
            return df

        df_result = my_function(transactions_dummy_df)
        pd.testing.assert_frame_equal(transactions_dummy_df, df_result)

    def test_reduce_memory_usage_input_output(self, transactions_dummy_df):
        df_result = reduce_memory_usage(transactions_dummy_df)
        assert df_result.shape == transactions_dummy_df.shape
        assert set(df_result.columns) == set(transactions_dummy_df.columns)


# filter_dataframe_by_last_n_days
class TestFilterDataFrameByLastNDays:
    def test_not_a_datetime_column(self, transactions_dummy_df):
        not_a_datetime_col = "customer_id"
        with pytest.raises(ValueError):
            filter_dataframe_by_last_n_days(
                transactions_dummy_df, n_days=15, date_column=not_a_datetime_col
            )

    def test_not_existing_column(self, transactions_dummy_df):
        not_a_col = "does_not_exist"
        with pytest.raises(KeyError):
            filter_dataframe_by_last_n_days(
                transactions_dummy_df, n_days=15, date_column=not_a_col
            )

    def test_negative_days(self, transactions_dummy_df):
        n_days = -5
        result_df = filter_dataframe_by_last_n_days(
            transactions_dummy_df, n_days=n_days, date_column="t_dat"
        )
        assert result_df.empty is True

    def test_zero_days(self, transactions_dummy_df):
        n_days = 0
        result_df = filter_dataframe_by_last_n_days(
            transactions_dummy_df, n_days=n_days, date_column="t_dat"
        )
        pd.testing.assert_frame_equal(result_df, transactions_dummy_df)

    def test_none_n_days(self, transactions_dummy_df):
        n_days = None
        result_df = filter_dataframe_by_last_n_days(
            transactions_dummy_df, n_days=n_days, date_column="t_dat"
        )
        assert result_df.empty is False

    def test_many_days(self, transactions_dummy_df):
        n_days = 100_000
        result_df = filter_dataframe_by_last_n_days(
            transactions_dummy_df, n_days=n_days, date_column="t_dat"
        )
        pd.testing.assert_frame_equal(result_df, transactions_dummy_df)

    def test_few_days(self, transactions_dummy_df):
        n_days = 30
        initial_size = len(transactions_dummy_df)
        result_df = filter_dataframe_by_last_n_days(
            transactions_dummy_df, n_days=n_days, date_column="t_dat"
        )
        assert len(result_df) <= initial_size

    def test_input_output_cols(self, transactions_dummy_df):
        n_days = 30
        expected_columns = transactions_dummy_df.columns
        result_columns = filter_dataframe_by_last_n_days(
            transactions_dummy_df, n_days=n_days, date_column="t_dat"
        ).columns
        assert set(expected_columns) == set(result_columns)
