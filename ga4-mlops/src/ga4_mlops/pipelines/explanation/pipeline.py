"""
This is a boilerplate pipeline 'explanation'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import explain_model, sample_data


def create_pipeline(subset: str, **kwargs) -> Pipeline:
    """_summary_

    Args:
        subset (str): _description_

    Returns:
        Pipeline: _description_
    """
    possible_subsets = ["train", "valid", "test"]
    assert subset in possible_subsets, f"Subset should be one of: {possible_subsets}"

    main_pipeline = pipeline(
        [
            node(
                name=f"{subset}.sample_data_node",
                func=sample_data,
                inputs=[f"{subset}.abt", "params:n_obs", "params:sampling_seed"],
                outputs=f"{subset}.abt_sample",
            ),
            node(
                name=f"{subset}.explain_model_node",
                func=explain_model,
                inputs=[f"{subset}.abt_sample", "stored.model"],
                outputs=None,
            ),
        ]
    )

    return main_pipeline
