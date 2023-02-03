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
            "c_first_good_coltype": ["cat1", "cat2", "cat1"],
            "n_second_good_coltype": [1, 3, 5],
            "c_third_good_coltype": ["cat2", pd.NA, "cat3"],
            "n_fourth_good_coltype": [4.2, np.NaN, 2.9],
            "c_fifth_good_coltype": [pd.NA, pd.NA, pd.NA],
            "n_sixth_good_coltype": [np.NaN, np.NaN, np.NaN],
            "c_first_bad_coltype": [7, 8, 9],
            "c_second_bad_coltype": [4.1, 2.3, 8.9],
            "n_third_bad_coltype": ["3.0", "2.2", "5.6"],
            "n_fourth_bad_coltype": [4.2, pd.NA, 2.9],
            "n_fifth_bad_coltype": [3, pd.NA, 2],
            "c_sixth_bad_coltype": [np.NaN, np.NaN, np.NaN],
            "n_seventh_bad_coltype": [pd.NA, pd.NA, pd.NA],
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
