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
                    "params:image_embeddings.image_path",
                    "params:image_embeddings.encoder",
                    "params:image_embeddings.decoder",
                    "params:image_embeddings.batch_size",
                    "params:image_embeddings.image_size",
                    "params:image_embeddings.embedding_size",
                    "params:image_embeddings.num_epochs",
                    "params:image_embeddings.shuffle_reconstructions",
                    "params:image_embeddings.save_model",
                    "params:image_embeddings.model_name",
                    "params:image_embeddings.seed"
                    ],
                outputs=None,
            ),
        ],
        namespace="image_embeddings",
        inputs=None,
        outputs=None,
    )
