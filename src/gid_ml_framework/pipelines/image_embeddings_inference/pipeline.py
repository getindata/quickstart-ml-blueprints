from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import calculate_image_embeddings


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=calculate_image_embeddings,
                name="calculate_image_embeddings",
                inputs=[
                    "params:run_id",
                    "params:image_path",
                    "params:platform",
                    "params:batch_size",
                    ],
                outputs="image_embeddings",
            ),
        ],
        namespace="image_embeddings_inference",
        inputs=None,
        outputs=["image_embeddings"],
    )
