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
                    "params:image_embeddings_inference.run_id",
                    "params:image_embeddings_inference.image_path",
                    "params:image_embeddings_inference.batch_size",
                    ],
                outputs="image_embeddings",
            ),
        ],
        namespace="calculate_image_embeddings",
        inputs=None,
        outputs=["image_embeddings"],
    )