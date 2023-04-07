"""
This is a 'forecasting' pipeline, which forecasts time series input, and logs in-sample metrics to MLflow.

The pipeline takes in model input, loads a model and fallback model from the `statsforecast` library,
forecasts `params:forecast_options.horizon` steps ahead, and logs in-sample metrics to MLflow.

The output dataframe (forecast) contains three columns:
- 'unique_id': A unique identifier for the time series,
- 'ds': A time index column that can be either an integer index or a datestamp,
- `params:model.name`: A column representing the point predictions for time series.
"""
from kedro.pipeline import node
from kedro.pipeline import Pipeline
from kedro.pipeline import pipeline

from .nodes import forecast_time_series
from .nodes import log_training_artifacts


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=forecast_time_series,
                inputs=[
                    "model_input",
                    "params:model",
                    "params:fallback_model",
                    "params:forecast_options",
                ],
                outputs=["forecast", "in_sample_predictions"],
                name="forecast_time_series_node",
            ),
            node(
                func=log_training_artifacts,
                inputs=[
                    "in_sample_predictions",
                    "params:model.name",
                ],
                outputs=None,
                name="log_training_artifacts_node",
            ),
        ],
        namespace="forecasting",
        inputs=["model_input"],
        outputs=["forecast"],
    )
