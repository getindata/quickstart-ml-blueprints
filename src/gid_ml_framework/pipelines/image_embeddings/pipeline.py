from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import train_image_embeddings


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_image_embeddings,
                name="train_image_embeddings",
                inputs=[
                    "params:image_path",
                    "params:encoder",
                    "params:decoder",
                    "params:batch_size",
                    "params:image_size",
                    "params:embedding_size",
                    "params:num_epochs",
                    "params:shuffle_reconstructions",
                    "params:save_model",
                    "params:model_name",
                    "params:seed"
                    ],
                outputs=None,
            ),
        ],
        namespace="image_embeddings",
        inputs=None,
        outputs=None,
    )
