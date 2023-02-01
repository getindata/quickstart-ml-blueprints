import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def contaminated_column_names_sample():
    df = pd.DataFrame(
        {
            "c_first_good_colname": ["cat1", "cat2", "cat1"],
            "n_second_good_colname": [1, 3, 5],
            "c_first_bad_<colname>": ["cat4", "cat4", "cat3"],
            "n_[second_bad_colname]": [3.0, 2.2, 5.6],
        }
    )

    return df


@pytest.fixture
def wrong_column_types_sample():
    df = pd.DataFrame(
        {
            "c_first_good_colname": ["cat1", "cat2", "cat1"],
            "n_second_good_colname": [1, 3, 5],
            "c_first_bad_colname": [7, 8, 9],
            "n_second_bad_colname": ["3.0", "2.2", "5.6"],
        }
    )

    return df


@pytest.fixture
def column_names_sample():
    df = pd.DataFrame(
        {
            "i_info_col_1": [123],
            "i_info_col_2": ["abc"],
            "c_cat_col": ["cat1"],
            "n_num_col_1": [4],
            "n_num_col_2": [6.54],
            "y_target_col": [1],
            "f_unknown_prefix_col": ["value"],
        }
    )

    return df


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
def imputation_sample():
    df = pd.DataFrame(
        {
            "i_id_col": [1, 2, 3, 4, 5],
            "n_num_col_complete": [4, 2, 6, 1, 0],
            "n_num_col_missing_not_listed": [4.0, np.nan, 2.1, 3.2, np.nan],
            "n_num_col_missing_for_mean": [2, np.nan, 5, np.nan, 8],
            "n_num_col_missing_for_zero": [np.nan, 9.3, -1.2, np.nan, np.nan],
            "c_cat_col_for_mostfreq": ["cat1", np.nan, "cat2", "cat1", "cat1"],
            "c_cat_col_for_unknown": ["cat1", "cat2", np.nan, "cat3", "cat2"],
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


@pytest.fixture
def exclusion_sample():
    df = pd.DataFrame(
        {
            "i_id_col": [1, 2, 3, 4, 5],
            "n_col_to_use": [3.2, 2.1, 9.8, 0.3, 3.1],
            "c_col_to_use_0": [0, 1, 0, 0, 0],
            "c_col_to_use_1": [1, 0, 1, 1, 1],
            "c_first_col_to_exclude": [1, 3, 2, 1, 2],
            "c_second_col_to_exclude_cat1": [1, 1, 0, 0, 0],
            "c_second_col_to_exclude_cat2": [0, 0, 1, 1, 0],
            "c_second_col_to_exclude_cat3": [0, 0, 0, 0, 1],
        }
    )

    return df
