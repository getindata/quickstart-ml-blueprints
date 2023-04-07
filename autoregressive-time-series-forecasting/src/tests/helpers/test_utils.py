import logging
import os

import mlflow
import numpy as np
import pandas as pd
import pytest
from autoregressive_forecasting.helpers.utils import calculate_forecasting_metrics
from autoregressive_forecasting.helpers.utils import filter_dataframe_by_date_cutoffs
from autoregressive_forecasting.helpers.utils import get_all_available_models
from autoregressive_forecasting.helpers.utils import load_model
from autoregressive_forecasting.helpers.utils import log_dataframe_as_html_artifact
from autoregressive_forecasting.helpers.utils import rename_columns
from statsforecast.models import _TS


def test_log_df_as_html(reset_active_experiment):
    df = pd.DataFrame(np.random.randint(0, 100, size=(10, 4)), columns=list("ABCD"))
    experiment_id, temp_path = reset_active_experiment
    items_before = os.listdir(temp_path)
    with mlflow.start_run(experiment_id=experiment_id):
        log_dataframe_as_html_artifact(
            df=df,
            artifact_path="a_path",
            file_name="f_name",
        )
    items_after = os.listdir(temp_path)
    assert len(items_before) == 0
    assert len(items_after) == 1


def test_get_all_available_models(monkeypatch):
    models = ["arima", "ar", "ma", "garch", "ets"]

    def mock_return():
        return models

    monkeypatch.setattr("statsforecast.models.__all__", mock_return())
    output = get_all_available_models()
    assert output == models


class TestLoadModel:
    def test_load_model(self):
        model_name = "Naive"
        model_object = load_model(model_name)
        assert issubclass(model_object, _TS)

    def test_wrong_model_name(self, caplog):
        caplog.set_level(
            logging.CRITICAL
        )  # disable error logging messages from load_model()
        model_name = "Not existing model"
        with pytest.raises(AttributeError):
            load_model(model_name)


class TestCalculateForecastingMetrics:
    def test_calculate_forecasting_metrics(self, predictions_df):
        model_name = predictions_df.columns[1]
        loss_metrics = calculate_forecasting_metrics(
            predictions_df, model_name=model_name
        )
        expected_metrics = {"mae", "rmse", "mape"}
        assert len(loss_metrics) == len(expected_metrics)
        assert expected_metrics == set(loss_metrics.keys())
        assert all(
            isinstance(loss_metric, float) for loss_metric in loss_metrics.values()
        )

    def test_wrong_model_name(self, predictions_df):
        model_name = "model_999"
        with pytest.raises(KeyError):
            calculate_forecasting_metrics(predictions_df, model_name=model_name)


class TestRenameColumns:
    # basic functionality
    def test_rename_columns(self, raw_data_df):
        correct_mapper = {
            "product_id": "unique_id",
            "sale_date": "ds",
            "items_sold": "y",
        }
        df_output = rename_columns(raw_data_df, correct_mapper)
        expected_cols_set = {"unique_id", "ds", "y"}
        assert set(df_output.columns) == expected_cols_set
        assert df_output.shape == raw_data_df.shape

    # test mapper
    def test_more_keys_in_mapper(self, raw_data_df):
        more_cols_mapper = {
            "product_id": "unique_id",
            "sale_date": "ds",
            "items_sold": "y",
            "not_existing_col": "abc",
        }
        df_output = rename_columns(raw_data_df, more_cols_mapper)
        expected_cols_set = {"unique_id", "ds", "y"}
        assert set(df_output.columns) == expected_cols_set
        assert df_output.shape == raw_data_df.shape

    def test_less_keys_in_mapper(self, raw_data_df):
        less_cols_mapper = {
            "product_id": "unique_id",
            "sale_date": "ds",
        }
        df_output = rename_columns(raw_data_df, less_cols_mapper)
        expected_cols_set = {"unique_id", "ds", "items_sold"}
        assert set(df_output.columns) == expected_cols_set
        assert df_output.shape == raw_data_df.shape

    def test_no_keys_in_mapper(self, raw_data_df):
        no_keys_mapper = {
            "not_existing_col_1": "unique_id",
            "not_existing_col_2": "ds",
            "not_existing_col_3": "y",
        }
        df_output = rename_columns(raw_data_df, no_keys_mapper)
        expected_cols_set = {"product_id", "sale_date", "items_sold"}
        assert set(df_output.columns) == expected_cols_set
        assert df_output.shape == raw_data_df.shape


class TestFilterDataframeByCutoffs:
    @pytest.mark.parametrize(
        "min_date, max_date, df_length",
        [
            pytest.param(None, None, 10, id="unchanged dataframe"),
            pytest.param(None, "2023-01-05", 5, id="filtering from the beginning"),
            pytest.param("2023-01-05", None, 6, id="filtering from the end"),
            pytest.param("2023-01-03", "2023-01-06", 4, id="filtering from the middle"),
            pytest.param("2023-01-07", "2023-01-07", 1, id="filtering to a single day"),
        ],
    )
    def test_correct_filtering(self, model_input_df, min_date, max_date, df_length):
        filtered_df = filter_dataframe_by_date_cutoffs(
            model_input_df, min_date, max_date
        )

        assert len(filtered_df) == df_length

    def test_empty_dataframe(self, model_input_df):
        min_date = "2080-01-01"
        max_date = "2080-12-01"

        with pytest.raises(ValueError):
            filter_dataframe_by_date_cutoffs(model_input_df, min_date, max_date)
