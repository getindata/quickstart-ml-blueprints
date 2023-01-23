"""
This is a boilerplate pipeline 'prediction'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from ..modeling_utils import score_abt
from .nodes import create_predictions


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                name="calculate_raw_scores_node",
                func=score_abt,
                inputs=["predict.abt", "stored.model"],
                outputs="raw_scores",
            ),
            node(
                name="calculate_calibrated_scores_node",
                func=score_abt,
                inputs=["predict.abt", "stored.calibrator"],
                outputs="calibrated_scores",
            ),
            node(
                name="create_predictions_node",
                func=create_predictions,
                inputs=[
                    "predict.abt",
                    "raw_scores",
                    "calibrated_scores",
                    "params:threshold",
                    "params:classify_on_calibrated",
                ],
                outputs="predictions",
            ),
        ]
    )
