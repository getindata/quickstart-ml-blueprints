import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def abt_sample():
    df = pd.DataFrame(
        {
            "i_id": [1, 2, 3, 4, 5],
            "i_info_col": ["2022", "2021", "2022", "2022", "2021"],
            "n_num_col": [1.3, 5.6, 2.3, 7.6, 9.2],
            "c_cat_col": [1, 3, 3, 1, 2],
            "y_target_col": [1, 0, 0, 1, 0],
        }
    )

    return df


@pytest.fixture
def raw_scores_sample():
    arr = np.array([0.99, 0.02, 0.01, 0.98, 0.01])

    return arr


@pytest.fixture
def calibrated_scores_sample():
    arr = np.array([0.88, 0.42, 0.23, 0.84, 0.22])

    return arr
