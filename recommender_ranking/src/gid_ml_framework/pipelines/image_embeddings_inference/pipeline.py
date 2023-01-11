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
                    "params:model_uri",
                    "params:image_path",
                    "params:platform",
                    "params:batch_size",
                    "image_autoencoder_training_metadata",
                ],
                outputs="image_embeddings",
            ),
        ],
        namespace="image_embeddings_inference",
        inputs=["image_autoencoder_training_metadata"],
        outputs=["image_embeddings"],
    )
