"""
This is a 'forecasting_with_exogenous_vars' pipeline, which demonstrates how to use the AutoARIMA model
    from the StatsForecasts library to forecast with exogenous variables.

The pipeline:
- takes in model input,
- splits it into train and test dataframes,
- fits two models on the training set (AutoARIMA with exogenous variables; AutoARIMA without exegenous variables),
- forecasts test dataframe using these 2 models,
- saves metrics to MLflow.

The output dataframe (forecast_exogenous) contains 5 columns:
- 'unique_id': A unique identifier for the time series,
- 'ds': A time index column that can be either an integer index or a datestamp,
- 'AutoARIMA_exogenous': A column representing the point predictions with exogenous variables for time series,
- 'AutoARIMA': A column representing the point predictions without exogenous variables for time series,
- 'y': A column representing the true values for time series.

It is important to note that:
- to forecast values in the future, you will need to have future exogenous variable values available,
- exogenous variables must have numeric types,
- AutoARIMA model is the only model in the StatsForecasts library that supports forecasting with exogenous variables.
"""
from kedro.pipeline import node
from kedro.pipeline import Pipeline
from kedro.pipeline import pipeline

from .nodes import forecast_with_exogenous
from .nodes import log_exogenous_metrics
from .nodes import merge_dataframes
from .nodes import preprocess_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs=[
                    "input_exogenous",
                    "params:data_processing.column_mapper",
                    "params:data_processing.no_test_periods",
                    "params:data_processing.test_run",
                    "params:data_processing.exo_columns",
                    "params:data_processing.date_cutoffs.min_date",
                    "params:data_processing.date_cutoffs.max_date",
                ],
                outputs=["train_exogenous", "test_exogenous", "Y_test"],
                name="preprocess_exogenous_data_node",
            ),
            node(
                func=forecast_with_exogenous,
                inputs=[
                    "train_exogenous",
                    "test_exogenous",
                    "params:forecast_options",
                    "params:exo_vars.true_value",
                ],
                outputs="forecast_with_exogenous",
                name="forecast_with_exogenous_node",
            ),
            node(
                func=forecast_with_exogenous,
                inputs=[
                    "train_exogenous",
                    "test_exogenous",
                    "params:forecast_options",
                    "params:exo_vars.false_value",
                ],
                outputs="forecast_without_exogenous",
                name="forecast_without_exogenous_node",
            ),
            node(
                func=merge_dataframes,
                inputs=[
                    "forecast_with_exogenous",
                    "forecast_without_exogenous",
                    "Y_test",
                ],
                outputs="forecast_exogenous",
                name="merge_dataframes_node",
            ),
            node(
                func=log_exogenous_metrics,
                inputs=[
                    "forecast_exogenous",
                ],
                outputs=None,
                name="log_exogenous_metrics_node",
            ),
        ],
        namespace="forecasting_with_exogenous_vars",
        inputs=["input_exogenous"],
        outputs=["forecast_exogenous"],
    )
