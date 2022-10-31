import pandas as pd
import pytest

from gid_ml_framework.pipelines.train_val_split.nodes import train_val_split


# date_column
def test_not_a_datetime_column(transactions_dummy_df):
    not_a_datetime_col = "customer_id"
    with pytest.raises(ValueError):
        train_val_split(transactions_dummy_df, not_a_datetime_col)


def test_not_existing_column(transactions_dummy_df):
    not_a_col = "does_not_exist"
    with pytest.raises(KeyError):
        train_val_split(transactions_dummy_df, not_a_col)


# input/output cols
def test_input_output_cols(transactions_dummy_df):
    expected_columns = transactions_dummy_df.columns
    train_df, val_df = train_val_split(transactions_dummy_df, "t_dat")
    assert set(train_df.columns) == set(val_df.columns) == set(expected_columns)


# train/val separate split
def test_train_val_separate(transactions_dummy_df):
    train_df, val_df = train_val_split(transactions_dummy_df, "t_dat", 4, 3)
    assert train_df["t_dat"].max() < val_df["t_dat"].min()


# no validation/training sets
def test_no_validation_set(transactions_dummy_df):
    train_df, val_df = train_val_split(
        transactions_dummy_df, "t_dat", no_train_weeks=4, no_val_weeks=0
    )
    assert val_df.empty is True
    assert train_df.empty is False


def test_no_training_set(transactions_dummy_df):
    train_df, val_df = train_val_split(
        transactions_dummy_df, "t_dat", no_train_weeks=0, no_val_weeks=4
    )
    assert train_df.empty is True
    assert val_df.empty is False


def test_many_validation_weeks(transactions_dummy_df):
    no_val_weeks = 1_000
    train_df, val_df = train_val_split(
        transactions_dummy_df, "t_dat", no_val_weeks=no_val_weeks
    )
    pd.testing.assert_frame_equal(val_df, transactions_dummy_df)
    assert train_df.empty is True


def test_many_train_weeks(transactions_dummy_df):
    no_train_weeks = 1_000
    initial_shape = len(transactions_dummy_df)
    train_df, val_df = train_val_split(
        transactions_dummy_df, "t_dat", no_train_weeks=no_train_weeks
    )
    assert len(train_df) + len(val_df) == initial_shape
