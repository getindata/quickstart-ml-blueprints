import pandas as pd
import pytest


@pytest.fixture
def santander_train_transactions():
    path = "src/tests/fixtures/csv/santander_train_transactions.csv"
    santander_train_df = pd.read_csv(path)
    return santander_train_df


@pytest.fixture
def santander_val_transactions():
    path = "src/tests/fixtures/csv/santander_val_transactions.csv"
    santander_val_df = pd.read_csv(path)
    return santander_val_df


@pytest.fixture
def santander_concat_transactions():
    path = "src/tests/fixtures/csv/santander_concat_transactions.csv"
    santander_concat_df = pd.read_csv(path)
    return santander_concat_df
