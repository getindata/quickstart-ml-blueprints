from uuid import uuid4

import mlflow
import numpy as np
import pandas as pd
import pytest


def generate_time_series(n: int) -> list:
    rng = np.random.RandomState(1111)
    y = np.arange(n) + rng.rand(n) * 0.5
    return y


# data
@pytest.fixture
def raw_data_df() -> pd.DataFrame:
    n = 10
    id = np.repeat("1234", n)
    dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
    y = generate_time_series(n)
    df = pd.DataFrame(
        {
            "product_id": id,
            "sale_date": dates,
            "items_sold": y,
        }
    )
    return df


@pytest.fixture
def model_input_df() -> pd.DataFrame:
    n = 10
    id = np.repeat("1234", n)
    dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
    y = generate_time_series(n)
    df = pd.DataFrame(
        {
            "unique_id": id,
            "ds": dates,
            "y": y,
        }
    )
    return df


@pytest.fixture
def predictions_df() -> pd.DataFrame:
    n = 6
    data = 10 * np.random.random_sample((n, 2)) + 10
    df = pd.DataFrame(data, columns=["y", "model_1"])
    return df


@pytest.fixture
def cross_validation_df() -> pd.DataFrame:
    """result:
    unique_id (index)  y   model_1   model_2   model_3  ds            cutoff
    '1234'             3   4         3         2        '2023-01-01'  '2023-01-01'
    '1234'             4   5         6         4        '2023-01-02'  '2023-01-01'
    ...
    """
    n = 10
    ids = ["1234"] * 5 + ["4321"] * 5
    y_data = 10 * np.random.random_sample((n, 4)) + 10
    date_series = pd.Series(pd.date_range(start="2023-01-01", periods=n / 2, freq="D"))
    dates = pd.concat([date_series, date_series], axis=0)
    cutoff_date = np.repeat(date_series.min(), n)
    df = pd.DataFrame(y_data, index=ids, columns=["y", "model_1", "model_2", "model_3"])
    df.index.name = "unique_id"
    df.loc[:, "ds"] = dates.values
    df.loc[:, "cutoff"] = cutoff_date
    return df


# mlflow setup
@pytest.fixture(scope="function")
def reset_active_experiment(tmp_path):
    exp_name = f"random_experiment_{uuid4()}"
    exp_id = mlflow.create_experiment(exp_name, artifact_location=str(tmp_path))
    yield exp_id, tmp_path
    mlflow.delete_experiment(exp_id)
