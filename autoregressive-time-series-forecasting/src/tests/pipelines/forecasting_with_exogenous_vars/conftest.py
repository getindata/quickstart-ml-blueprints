import numpy as np
import pandas as pd
import pytest
from tests.conftest import generate_time_series


@pytest.fixture
def df_raw_input_exo(raw_data_df) -> pd.DataFrame:
    df_raw_input_exo = raw_data_df
    df_raw_input_exo["exo"] = np.array([22, 97, 38, 16, 86, 80, 37, 91, 45, 11])
    return df_raw_input_exo


@pytest.fixture
def df_raw_input_exo_long(raw_data_df) -> pd.DataFrame:
    df_raw_input_exo_long = pd.concat([raw_data_df, raw_data_df, raw_data_df], axis=0)
    df_raw_input_exo_long["exo"] = np.array(
        [
            68,
            53,
            59,
            75,
            9,
            12,
            87,
            97,
            53,
            84,
            82,
            14,
            62,
            94,
            60,
            74,
            66,
            97,
            60,
            46,
            6,
            17,
            90,
            42,
            78,
            69,
            5,
            41,
            48,
            83,
        ]
    )
    df_raw_input_exo_long["product_id"] = [str(i) for i in range(1, 31)]
    return df_raw_input_exo_long


@pytest.fixture
def df_input_exo(model_input_df) -> pd.DataFrame:
    df_input_exo = model_input_df
    df_input_exo["exo"] = np.array([14, 59, 60, 67, 83, 50, 47, 89, 85, 83])
    return df_input_exo


@pytest.fixture
def X_test_exo() -> pd.DataFrame:
    n = 10
    exo = np.array([48, 82, 99, 58, 90, 19, 90, 46, 55, 47])
    dates = pd.date_range(start="2023-01-11", periods=n, freq="D")
    id = np.repeat("1234", n)
    X_test_exo = pd.DataFrame(
        {
            "unique_id": id,
            "ds": dates,
            "exo": exo,
        }
    )
    return X_test_exo


@pytest.fixture
def Y_test_exo() -> pd.DataFrame:
    n = 10
    y = generate_time_series(n)
    dates = pd.date_range(start="2023-01-11", periods=n, freq="D")
    id = np.repeat("1234", n)
    Y_test_exo = pd.DataFrame(
        {
            "unique_id": id,
            "ds": dates,
            "y": y,
        }
    )
    return Y_test_exo


def prepare_forecast_df(exo: bool) -> pd.DataFrame:
    forecast_name = "AutoARIMA_exogenous" if exo else "AutoARIMA"
    n = 10
    ids = np.repeat("1234", n)
    forecast = 10 * np.random.random_sample((n,)) + 10
    dates = pd.date_range(start="2023-01-11", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "unique_id": ids,
            "ds": dates,
            forecast_name: forecast,
        }
    )
    df = df.set_index("unique_id")
    return df


@pytest.fixture
def forecast_exo() -> pd.DataFrame:
    forecast_exo_df = prepare_forecast_df(exo=True)
    return forecast_exo_df


@pytest.fixture
def forecast_not_exo() -> pd.DataFrame:
    forecast_df = prepare_forecast_df(exo=False)
    return forecast_df


@pytest.fixture
def merged_forecasts_exo() -> pd.DataFrame:
    n = 10
    ids = np.repeat("1234", n)
    y = 10 * np.random.random_sample((n,)) + 10
    forecast = 10 * np.random.random_sample((n,)) + 10
    forecast_exo = 10 * np.random.random_sample((n,)) + 10
    dates = pd.date_range(start="2023-01-11", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "unique_id": ids,
            "ds": dates,
            "y": y,
            "AutoARIMA": forecast,
            "AutoARIMA_exogenous": forecast_exo,
        }
    )
    return df
