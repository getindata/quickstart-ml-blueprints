"""
This is a boilerplate pipeline 'test_gpu'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import test_gpu


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=test_gpu,
                inputs=None,
                outputs="test_gpu_pickle",
                name="test_gpu_node",
            ),
        ]
    )
