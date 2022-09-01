from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import process_images


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=process_images,
                name="process_images",
                inputs=[
                    "input_images",
                    "params:output_size",
                    "params:method",
                    ],
                outputs="resized_images",
            ),
        ],
        namespace="process_images",
        inputs="input_images",
        outputs=["resized_images"],
    )