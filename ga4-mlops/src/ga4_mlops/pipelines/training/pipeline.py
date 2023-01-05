"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    optimize_hyperparameters,
    test_model,
    train_and_validate_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                name="optimize_hyperparameters_node",
                func=optimize_hyperparameters,
                inputs=[
                    "train.abt",
                    "valid.abt",
                    "params:seed",
                    "params:optim_time",
                    "params:objective",
                    "params:eval_metric",
                    "params:direction",
                ],
                outputs="best_params",
            ),
            node(
                name="training_and_validation_node",
                func=train_and_validate_model,
                inputs=[
                    "train.abt",
                    "valid.abt",
                    "best_params",
                    "params:eval_metric",
                ],
                outputs=["fitted.model", "model_config"],
            ),
            node(
                name="test_node",
                func=test_model,
                inputs=["test.abt", "fitted.model", "params:eval_metric"],
                outputs=None,
            ),
        ]
    )
