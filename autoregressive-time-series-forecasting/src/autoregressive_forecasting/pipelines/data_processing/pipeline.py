"""
This is a 'data_processing' pipeline, which preprocesses data and prepares a dataframe for model forecasting.

The pipeline takes in raw data and performs data cleaning and transformation steps to create a prepared dataframe.
The output dataframe contains three columns:
- 'unique_id': A unique identifier for the time series,
- 'ds': A time index column that can be either an integer index or a datestamp,
- 'y': A column representing the target variable or measurement we wish to forecast.

The 'data_processing' pipeline ensures that the resulting dataframe is ready to be used as input to a forecasting model.
"""
from kedro.pipeline import node
from kedro.pipeline import Pipeline
from kedro.pipeline import pipeline

from .nodes import preprocess_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs=[
                    "input",
                    "params:column_mapper",
                    "params:date_cutoffs.min_date",
                    "params:date_cutoffs.max_date",
                ],
                outputs="model_input",
                name="preprocess_data_node",
            ),
        ],
        namespace="data_processing",
        inputs=["input"],
        outputs=["model_input"],
    )
