import pandas as pd
import pytest


@pytest.fixture
def explanation_sample():
    df = pd.DataFrame({"i_id": list(range(20)), "y_target": [1] * 5 + [0] * 15})

    return df
