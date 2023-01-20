"""
This is a boilerplate pipeline 'explanation'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import calculate_shap, create_explanations, sample_data


def create_pipeline(subset: str, **kwargs) -> Pipeline:
    """Create model explanation pipeline.

    Args:
        subset (str): subset to explain model on (train, valid or test). It has to have ground truth (target) column.

    Returns:
        Pipeline: Kedro pipeline for model explanation
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
                name=f"{subset}.calculate_shap_node",
                func=calculate_shap,
                inputs=[f"{subset}.abt_sample", "stored.model"],
                outputs="shap_values",
            ),
            node(
                name=f"{subset}.create_explanations_node",
                func=create_explanations,
                inputs=[
                    "shap_values",
                    f"{subset}.abt_sample",
                    "stored.model",
                    "params:pdp_top_n",
                ],
                outputs=[
                    "shap_summary_plot",
                    "feature_importance",
                    "partial_dependence_plots",
                ],
            ),
        ]
    )

    return main_pipeline
