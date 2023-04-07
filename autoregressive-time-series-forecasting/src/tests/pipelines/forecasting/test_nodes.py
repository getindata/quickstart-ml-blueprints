import mlflow
import pandas as pd
import pytest
from autoregressive_forecasting.pipelines.forecasting.nodes import forecast_time_series
from autoregressive_forecasting.pipelines.forecasting.nodes import (
    log_training_artifacts,
)


@pytest.fixture
def model_dict():
    model_dict = {"name": "SeasonalNaive", "parameters": {"season_length": 3}}
    return model_dict


@pytest.fixture
def fallback_dict():
    fallback_model_dict = {"name": "HistoricAverage", "parameters": {}}
    return fallback_model_dict


@pytest.fixture
def fcst_options():
    fcst_options = {
        "frequency": "D",
        "horizon": 1,
        "n_jobs": 1,
    }
    return fcst_options


@pytest.mark.slow
class TestForecast:
    @pytest.mark.parametrize(
        "n_horizon",
        [
            pytest.param(1, id="n_horizon=1"),
            pytest.param(5, id="n_horizon=5"),
            pytest.param(10, id="n_horizon=10"),
        ],
    )
    def test_forecast_dimensions(
        self, model_input_df, model_dict, fallback_dict, fcst_options, n_horizon
    ):
        fcst_options_horizon = fcst_options
        fcst_options_horizon["horizon"] = n_horizon
        forecast, _ = forecast_time_series(
            model_input_df,
            model_dict,
            fallback_dict,
            fcst_options_horizon,
        )
        assert len(forecast) == n_horizon
        assert len(forecast.columns) == 2
        assert forecast.index.name == "unique_id"

    def test_in_sample_predictions_dimensions(
        self, model_input_df, model_dict, fallback_dict, fcst_options
    ):
        _, in_sample_predictions = forecast_time_series(
            model_input_df,
            model_dict,
            fallback_dict,
            fcst_options,
        )
        assert len(in_sample_predictions) == len(model_input_df)
        assert len(in_sample_predictions.columns) == 3
        assert in_sample_predictions.index.name == "unique_id"

    def test_forecast_output(
        self, model_input_df, model_dict, fallback_dict, fcst_options
    ):
        model_name = model_dict["name"]
        forecast, _ = forecast_time_series(
            model_input_df,
            model_dict,
            fallback_dict,
            fcst_options,
        )
        assert {"ds", model_name} == set(forecast.columns)
        assert pd.api.types.is_numeric_dtype(forecast[model_name])

    def test_in_sample_predictions_output(
        self, model_input_df, model_dict, fallback_dict, fcst_options
    ):
        model_name = model_dict["name"]
        _, in_sample_predictions = forecast_time_series(
            model_input_df,
            model_dict,
            fallback_dict,
            fcst_options,
        )
        assert {"ds", "y", model_name} == set(in_sample_predictions.columns)
        assert pd.api.types.is_numeric_dtype(in_sample_predictions[model_name])


def test_log_forecasting_metrics(predictions_df, reset_active_experiment):
    model_name = predictions_df.columns[1]
    experiment_id, _ = reset_active_experiment
    with mlflow.start_run(experiment_id=experiment_id) as active_run:
        log_training_artifacts(predictions_df, model_name=model_name)
    finished_run = mlflow.tracking.MlflowClient().get_run(active_run.info.run_id)
    output_metrics = finished_run.data.metrics
    assert "y_true_sum" in output_metrics
    assert "y_pred_sum" in output_metrics
    assert all(type(metric) == float for metric in output_metrics.values())
