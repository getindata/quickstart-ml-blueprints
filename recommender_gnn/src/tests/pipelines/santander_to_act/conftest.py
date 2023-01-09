import pandas as pd
import pytest


@pytest.fixture
def preprocessed_train_df():
    path = "src/tests/fixtures/dataframes/bank_train_preprocessed_sample.csv"
    bank_train_df = pd.read_csv(path)
    return bank_train_df


@pytest.fixture
def preprocessed_val_df():
    path = "src/tests/fixtures/dataframes/bank_val_preprocessed_sample.csv"
    bank_val_df = pd.read_csv(path)
    return bank_val_df


@pytest.fixture
def preprocessed_train_bigger_df():
    path = "src/tests/fixtures/dataframes/bank_train_preprocessed_bigger_sample.csv"
    bank_train_df = pd.read_csv(path)
    return bank_train_df
