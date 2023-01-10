import pandas as pd
import pytest


@pytest.fixture
def bank_train_transactions():
    path = "src/tests/fixtures/dataframes/bank_train_transactions.csv"
    bank_train_df = pd.read_csv(path)
    return bank_train_df


@pytest.fixture
def bank_val_transactions():
    path = "src/tests/fixtures/dataframes/bank_val_transactions.csv"
    bank_val_df = pd.read_csv(path)
    return bank_val_df


@pytest.fixture
def bank_concat_transactions():
    path = "src/tests/fixtures/dataframes/bank_concat_transactions.csv"
    bank_concat_df = pd.read_csv(path)
    return bank_concat_df
