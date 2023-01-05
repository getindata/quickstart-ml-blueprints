"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import get_and_aggregate_data


def create_pipeline(subset: str, **kwargs) -> Pipeline:
    """Creates a parametrized pipeline for data extraction and aggregation

    Args:
        subset (str): data subset. Possible values: ["train", "valid", "test", "predict"].
    """
    possible_subsets = ["train", "valid", "test", "predict"]
    assert subset in possible_subsets, f"Subset should be one of: {possible_subsets}"

    namespace = f"{subset}"

    main_pipeline = pipeline(
        [
            node(
                name="get_and_aggregate_data_node",
                func=get_and_aggregate_data,
                inputs=["ga4_data"],
                outputs="df",
            ),
        ],
        namespace=namespace,
    )

    return main_pipeline
