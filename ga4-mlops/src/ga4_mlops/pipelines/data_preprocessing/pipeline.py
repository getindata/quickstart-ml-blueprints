"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import aggregate_data


def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [
            node(
                name="aggregate_data_node",
                func=aggregate_data,
                inputs=["ga4_raw_data"],
                outputs=None,
            )
        ]
    )
