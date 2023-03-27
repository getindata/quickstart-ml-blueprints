import mlflow
import pandas as pd
import pytest
from autoregressive_forecasting.pipelines.forecasting_with_exogenous_vars.nodes import (
    _split_train_test,
)
from autoregressive_forecasting.pipelines.forecasting_with_exogenous_vars.nodes import (
    _sum_sales,
)
from autoregressive_forecasting.pipelines.forecasting_with_exogenous_vars.nodes import (
    forecast_with_exogenous,
)
from autoregressive_forecasting.pipelines.forecasting_with_exogenous_vars.nodes import (
    log_exogenous_metrics,
)
from autoregressive_forecasting.pipelines.forecasting_with_exogenous_vars.nodes import (
    merge_dataframes,
)
from autoregressive_forecasting.pipelines.forecasting_with_exogenous_vars.nodes import (
    preprocess_data,
)


class TestSplitTrainTest:
    def test_columns(self, df_input_exo):
        df_train, X_test, Y_test = _split_train_test(df_input_exo, no_test_periods=2)

        assert set(df_train.columns) == {"ds", "y", "unique_id", "exo"}
        assert set(X_test.columns) == {"ds", "exo", "unique_id"}
        assert set(Y_test.columns) == {"ds", "y", "unique_id"}

    @pytest.mark.parametrize("no_test_periods", [0, 2, 5, 999_999])
    def test_shape(self, df_input_exo, no_test_periods):
        input_len = len(df_input_exo)
        df_train, X_test, Y_test = _split_train_test(
            df_input_exo, no_test_periods=no_test_periods
        )

        expected_test_len = min(input_len, no_test_periods)
        expected_train_len = input_len - expected_test_len

        assert len(Y_test) == expected_test_len
        assert len(X_test) == expected_test_len
        assert len(df_train) == expected_train_len


class TestPreprocessData:
    def setup_method(self):
        self.mapper = {
            "product_id": "unique_id",
            "sale_date": "ds",
            "items_sold": "y",
        }

    def test_output(self, df_raw_input_exo):
        df_train, X_test, Y_test = preprocess_data(
            df=df_raw_input_exo,
            mapper=self.mapper,
            no_test_periods=2,
            test_run=False,
            exo_columns=["exo"],
        )

        assert set(df_train.columns) == {"ds", "y", "unique_id", "exo"}
        assert set(X_test.columns) == {"ds", "exo", "unique_id"}
        assert set(Y_test.columns) == {"ds", "y"}
        assert Y_test.index.name == "unique_id"

    def test_test_run(self, df_raw_input_exo_long):
        test_run = True
        no_articles_before = df_raw_input_exo_long["product_id"].nunique()
        df_train, _, _ = preprocess_data(
            df=df_raw_input_exo_long,
            mapper=self.mapper,
            no_test_periods=0,
            test_run=test_run,
            exo_columns=["exo"],
        )
        no_articles_after = df_train["unique_id"].nunique()

        assert no_articles_before > 10
        assert no_articles_after == 10

    def test_sum_sales(self, df_input_exo):
        exo_cols = ["exo"]
        df_sum = _sum_sales(df_input_exo, exo_cols)
        init_len = len(df_input_exo)

        assert {"exo", "ds", "y", "unique_id"} == set(df_sum.columns)
        assert init_len == len(df_sum)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestExogenousForecasts:
    def setup_method(self):
        self.fcst_options = {
            "horizon": 10,
            "season_length": 3,
            "frequency": "D",
            "n_jobs": 1,
        }

    def test_exo_forecast(self, df_input_exo, X_test_exo):
        fit_exo = True
        forecast = forecast_with_exogenous(
            df_train=df_input_exo,
            X_test=X_test_exo,
            fcst_options=self.fcst_options,
            fit_exogenous=fit_exo,
        )

        assert set(forecast.columns) == {"AutoARIMA_exogenous", "ds"}
        assert forecast.index.name == "unique_id"
        assert len(forecast) == self.fcst_options["horizon"]

    def test_not_exo_forecast(self, df_input_exo):
        fit_exo = False
        forecast = forecast_with_exogenous(
            df_train=df_input_exo,
            X_test=pd.DataFrame(),
            fcst_options=self.fcst_options,
            fit_exogenous=fit_exo,
        )

        assert set(forecast.columns) == {"AutoARIMA", "ds"}
        assert forecast.index.name == "unique_id"
        assert len(forecast) == self.fcst_options["horizon"]

    def test_horizon_not_matching_X_test(self, df_input_exo, X_test_exo):
        fit_exo = True
        changed_fcst_options = self.fcst_options
        changed_fcst_options["horizon"] = 20

        with pytest.raises(ValueError):
            forecast_with_exogenous(
                df_train=df_input_exo,
                X_test=X_test_exo,
                fcst_options=changed_fcst_options,
                fit_exogenous=fit_exo,
            )

    def test_missing_test_exo_variables(self, df_input_exo, X_test_exo):
        fit_exo = True
        changed_X_test = X_test_exo.drop(["exo"], axis=1)

        with pytest.raises(ValueError):
            forecast_with_exogenous(
                df_train=df_input_exo,
                X_test=changed_X_test,
                fcst_options=self.fcst_options,
                fit_exogenous=fit_exo,
            )


def test_merge_dataframes(forecast_exo, forecast_not_exo, Y_test_exo):
    expected_len = len(Y_test_exo)
    merged_forecasts = merge_dataframes(forecast_exo, forecast_not_exo, Y_test_exo)

    assert len(merged_forecasts) == expected_len
    assert set(merged_forecasts.columns) == {
        "AutoARIMA",
        "AutoARIMA_exogenous",
        "ds",
        "y",
        "unique_id",
    }


def test_log_exo_metrics(merged_forecasts_exo, reset_active_experiment):
    experiment_id, _ = reset_active_experiment
    with mlflow.start_run(experiment_id=experiment_id) as active_run:
        log_exogenous_metrics(merged_forecasts_exo)
    # cleaning up NESTED RUNS
    query = "tags.mlflow.parentRunId = '{}'".format(active_run.info.run_id)
    child_runs = mlflow.search_runs(experiment_ids=[], filter_string=query)
    for run in child_runs["run_id"].to_list():
        mlflow.delete_run(run)
