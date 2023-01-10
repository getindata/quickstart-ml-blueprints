import pandas as pd
import pytest


@pytest.fixture
def otto_small_dummy_df():
    path = "src/tests/fixtures/dataframes/otto_preprocessed_small_sample.csv"
    otto_df = pd.read_csv(path)
    return otto_df
