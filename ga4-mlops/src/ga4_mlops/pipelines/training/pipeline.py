"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import test_model, train_and_validate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                name="training_and_validation_node",
                func=train_and_validate_model,
                inputs=[
                    "abt_train",
                    "abt_valid",
                    "params:seed",
                    "params:optim_time",
                    "params:objective",
                    "params:eval_metric",
                    "params:direction",
                ],
                outputs=["model_train", "model_config"],
            ),
            node(
                name="test_node",
                func=test_model,
                inputs=["abt_test", "model_train", "params:eval_metric"],
                outputs=None,
            ),
        ]
    )
