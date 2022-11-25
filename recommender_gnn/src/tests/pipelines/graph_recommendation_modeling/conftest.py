import pandas as pd
import pytest


@pytest.fixture
def mapped_transactions():
    path = "src/tests/fixtures/csv/santander_mapped_transactions.csv"
    mapped_df = pd.read_csv(path)
    return mapped_df


@pytest.fixture
def mapped_transactions_small():
    path = "src/tests/fixtures/csv/santander_mapped_transactions_small.csv"
    mapped_df = pd.read_csv(path)
    return mapped_df


@pytest.fixture
def mapped_transactions_custom():
    transactions_dict = {
        "user_id": {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 3,
        },
        "item_id": {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 2,
            "4": 3,
        },
        "time": {
            "0": 1453939200,
            "1": 1453039200,
            "2": 1453032200,
            "3": 1453132200,
            "4": 1453032200,
        },
    }
    transactions_df = pd.DataFrame(transactions_dict)
    return transactions_df
