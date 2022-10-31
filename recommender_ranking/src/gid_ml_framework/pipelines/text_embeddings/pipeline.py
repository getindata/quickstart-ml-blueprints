from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import generate_embeddings, prepare_desciptions_and_labels


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_desciptions_and_labels,
                name="prepare_desciptions_and_labels_node",
                inputs=[
                    "articles",
                ],
                outputs=[
                    "article_descriptions",
                    "article_labels",
                ],
            ),
            node(
                func=generate_embeddings,
                name="generate_embeddings_node",
                inputs=[
                    "article_descriptions",
                    "article_labels",
                    "params:transformer_model",
                ],
                outputs="text_embeddings",
            ),
        ],
        namespace="text_embeddings",
        inputs=["articles"],
        outputs=["text_embeddings"],
    )
