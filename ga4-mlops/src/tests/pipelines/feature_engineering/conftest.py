import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def engineer_features_sample():
    df = pd.DataFrame(
        {
            "i_full_visitor_id": [
                3033910.355860057,
                81793309.0616803,
                4293031.296243032,
            ],
            "i_visit_start_time": [
                1611619614341157.0,
                1611820704267587.0,
                1611661585573344.0,
            ],
        }
    )

    return df


@pytest.fixture
def exclusion_sample():
    df = pd.DataFrame(
        {
            "i_id_col": [1, 2, 3, 4, 5],
            "n_col_to_use": [3.2, 2.1, 9.8, 0.3, 3.1],
            "c_col_to_use": ["cat1", "cat2", "cat3", "cat1", "cat2"],
            "n_col_to_exclude": [4, 7, 3, 1, 2],
            "c_col_to_exclude": ["cat1", "cat1", "cat1", "cat2", "cat3"],
        }
    )

    return df


@pytest.fixture
def imputation_sample():
    df = pd.DataFrame(
        {
            "i_id_col": [1, 2, 3, 4, 5, 6],
            "n_num_col_complete": [4, 2, 6, 1, 0, 1],
            "n_num_col_missing_not_listed": [4.0, np.nan, 2.1, 3.2, np.nan, np.nan],
            "n_num_col_missing_for_mean": [2, np.nan, 5, np.nan, 8, np.nan],
            "n_num_col_missing_for_zero": [np.nan, 9.3, -1.2, np.nan, np.nan, np.nan],
            "c_cat_col_for_mostfreq": ["cat1", np.nan, "cat2", "cat1", "cat1", "cat1"],
            "c_cat_col_for_unknown": ["cat1", "cat2", np.nan, "cat3", "cat2", pd.NA],
        }
    )

    return df


@pytest.fixture
def encoding_sample():
    df = pd.DataFrame(
        {
            "i_id_col": [1, 2, 3, 4, 5],
            "c_cat_col_for_binary": ["0", "0", "1", "0", "1"],
            "c_cat_col_for_onehot": ["cat1", "cat2", "cat3", "cat3", "cat2"],
            "c_cat_col_for_ordinal": ["cat3", "cat2", "cat1", "cat3", "cat2"],
        }
    )

    return df
