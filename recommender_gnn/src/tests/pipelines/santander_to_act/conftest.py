import pandas as pd
import pytest


@pytest.fixture
def preprocessed_train_df():
    path = "src/tests/fixtures/dataframes/santander_train_preprocessed_sample.csv"
    santander_train_df = pd.read_csv(path)
    return santander_train_df


@pytest.fixture
def preprocessed_val_df():
    path = "src/tests/fixtures/dataframes/santander_val_preprocessed_sample.csv"
    santander_val_df = pd.read_csv(path)
    return santander_val_df


@pytest.fixture
def preprocessed_train_bigger_df():
    path = (
        "src/tests/fixtures/dataframes/santander_train_preprocessed_bigger_sample.csv"
    )
    santander_train_df = pd.read_csv(path)
    return santander_train_df
