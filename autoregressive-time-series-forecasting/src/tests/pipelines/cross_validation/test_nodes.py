import mlflow
import pandas as pd
import pytest
from autoregressive_forecasting.pipelines.cross_validation.nodes import (
    _aggregate_error_metrics,
)
from autoregressive_forecasting.pipelines.cross_validation.nodes import (
    _calculate_cross_validation_loss,
)
from autoregressive_forecasting.pipelines.cross_validation.nodes import (
    calculate_cross_validation_errors,
)
from autoregressive_forecasting.pipelines.cross_validation.nodes import (
    cross_validate_forecasting_models,
)
from autoregressive_forecasting.pipelines.cross_validation.nodes import (
    log_cross_validation_errors,
)
from datasetsforecast.losses import mae


@pytest.fixture
def cv_options():
    cv_options_dict = {
        "h": 1,
        "step_size": 1,
        "n_windows": 1,
        "freq": "D",
        "n_jobs": 1,
        "verbose": False,
        "test_run": True,
    }
    return cv_options_dict


@pytest.fixture
def model_params():
    model_params_dict = {"season_length": 3}
    return model_params_dict


@pytest.mark.slow
class TestCrossValidation:
    @pytest.mark.parametrize(
        "h, n_windows",
        [
            pytest.param(*(1, 1), id="h=1, n_windows=1"),
            pytest.param(*(5, 1), id="h=5, n_windows=1"),
            pytest.param(*(1, 2), id="h=1, n_windows=2"),
            pytest.param(*(5, 2), id="h=5, n_windows=2"),
        ],
    )
    def test_dimensions(self, model_input_df, cv_options, model_params, h, n_windows):
        cv_options_copy = cv_options.copy()
        cv_options_copy["h"] = h
        cv_options_copy["n_windows"] = n_windows

        cv_df = cross_validate_forecasting_models(
            model_input_df, cv_options_copy, model_params
        )

        assert len(cv_df) == n_windows * h

    def test_output(self, model_input_df, cv_options, model_params):
        cv_df = cross_validate_forecasting_models(
            model_input_df, cv_options, model_params
        )
        output_cols = cv_df.columns.to_list()

        assert "y" in output_cols
        assert "ds" in output_cols
        assert "cutoff" in output_cols
        assert cv_df.index.name == "unique_id"

    def test_output_types(self, model_input_df, cv_options, model_params):
        cv_df = cross_validate_forecasting_models(
            model_input_df, cv_options, model_params
        )
        ds_cols = ["ds", "cutoff"]
        y_cols = [col for col in cv_df.columns if col not in ["ds", "cutoff"]]

        assert cv_df.select_dtypes(include="datetime64").columns.to_list() == ds_cols
        assert cv_df.select_dtypes(include="number").columns.to_list() == y_cols
        assert pd.api.types.is_string_dtype(cv_df.index.dtype)


class TestCalculateCVLoss:
    def test_loss_output(self, cross_validation_df):
        input_cv_df = cross_validation_df.reset_index()
        no_unique_ids = input_cv_df.unique_id.nunique()

        evals = _calculate_cross_validation_loss(input_cv_df, mae)

        assert no_unique_ids == len(evals)
        assert set(evals.columns) == {"model_1", "model_2", "model_3"}
        assert pd.api.types.is_numeric_dtype(evals.values)

    def test_multiple_cutoffs(self, cross_validation_df):
        cutoff_dates = [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02")] * 5
        input_cv_df = cross_validation_df.reset_index()
        input_cv_df.loc[:, "cutoff"] = cutoff_dates

        no_unique_ids = input_cv_df.unique_id.nunique()
        evals = _calculate_cross_validation_loss(input_cv_df, mae)

        assert no_unique_ids == len(evals)
        assert set(evals.columns) == {"model_1", "model_2", "model_3"}
        assert pd.api.types.is_numeric_dtype(evals.values)

    def test_missing_y_values(self, cross_validation_df):
        input_cv_df = cross_validation_df.reset_index().drop(["y"], axis=1)

        with pytest.raises(KeyError):
            _calculate_cross_validation_loss(input_cv_df, mae)

    def test_no_forecasts(self, cross_validation_df):
        input_cv_df = cross_validation_df.reset_index()
        input_cv_df = input_cv_df[["unique_id", "ds", "y"]]

        with pytest.raises(ValueError):
            _calculate_cross_validation_loss(input_cv_df, mae)


def test_aggregate_error_metrics(cross_validation_df):
    input_cv_df = cross_validation_df.reset_index()

    evals = _aggregate_error_metrics(input_cv_df)

    assert {"mae", "mape", "rmse"} == set(evals.columns)
    assert {"model_1", "model_2", "model_3"} == set(evals.index)
    assert pd.api.types.is_numeric_dtype(evals.values)


def test_calculate_cross_validation_errors(cross_validation_df):
    errors_df = calculate_cross_validation_errors(cross_validation_df).reset_index()

    assert {"cutoff", "models", "mae", "mape", "rmse"} == set(errors_df.columns)
    assert pd.api.types.is_numeric_dtype(
        errors_df.loc[:, ["mae", "mape", "rmse"]].values
    )
    assert pd.api.types.is_string_dtype(errors_df.loc[:, "models"])


def test_log_cross_validation_errors(cross_validation_df, reset_active_experiment):
    experiment_id, _ = reset_active_experiment
    with mlflow.start_run(experiment_id=experiment_id):
        log_cross_validation_errors(cross_validation_df)
