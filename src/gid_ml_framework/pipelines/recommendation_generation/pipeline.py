from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import generate_predictions


def create_pipeline(**kwargs) -> Pipeline:
    generate_predictions_pipeline = pipeline(
        [
            node(
                func=generate_predictions,
                name="generate_predictions_node",
                inputs=[
                    "final_candidates",
                    "params:recommendation_generation.models",
                    "params:recommendation_generation.k",
                    ],
                outputs="recommendations",
            ),
        ],
        namespace="generate_predictions",
        inputs="final_candidates",
        outputs="recommendations",
    )
    
    return generate_predictions_pipeline
