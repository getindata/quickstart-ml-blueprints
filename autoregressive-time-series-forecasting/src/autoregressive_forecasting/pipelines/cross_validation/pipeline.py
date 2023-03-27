"""
This is a 'cross_validation' pipeline, which performs temporal cross-validation for multiple models.

The pipeline takes in model input, efficiently fits multiple models, and calculates errors.
The error metrics are saved to MLflow for easy tracking and analysis.
"""
from kedro.pipeline import node
from kedro.pipeline import Pipeline
from kedro.pipeline import pipeline

from .nodes import calculate_cross_validation_errors
from .nodes import cross_validate_forecasting_models
from .nodes import log_cross_validation_errors


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=cross_validate_forecasting_models,
                inputs=[
                    "model_input",
                    "params:cv_options",
                    "params:model_params",
                ],
                outputs="cross_validation_df",
                name="cross_validate_forecasting_models_node",
            ),
            node(
                func=calculate_cross_validation_errors,
                inputs=[
                    "cross_validation_df",
                ],
                outputs="errors_df",
                name="calculate_cross_validation_errors_node",
            ),
            node(
                func=log_cross_validation_errors,
                inputs=[
                    "errors_df",
                ],
                outputs=None,
                name="log_cross_validation_errors_node",
            ),
        ],
        namespace="cross_validation",
        inputs=["model_input"],
        outputs=None,
    )
