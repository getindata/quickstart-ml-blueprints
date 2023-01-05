"""
This is a boilerplate pipeline 'prediction'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                name="predict_node",
                func=predict,
                inputs=["predict.abt", "stored.model"],
                outputs="predictions",
            ),
        ]
    )
