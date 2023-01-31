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
