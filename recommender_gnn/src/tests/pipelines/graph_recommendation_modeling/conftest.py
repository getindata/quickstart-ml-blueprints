import pandas as pd
import pytest


@pytest.fixture
def mapped_transactions():
    path = "src/tests/fixtures/dataframes/bank_mapped_transactions.csv"
    mapped_df = pd.read_csv(path)
    return mapped_df


@pytest.fixture
def mapped_transactions_small():
    path = "src/tests/fixtures/dataframes/bank_mapped_transactions_small.csv"
    mapped_df = pd.read_csv(path)
    return mapped_df
