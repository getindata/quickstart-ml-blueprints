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
        subset (str): Data subset to explain model on (train, valid, test). Has to contain ground truth column (target).

    Returns:
        Pipeline: explanation Kedro pipeline
    """
    possible_subsets = ["train", "valid", "test"]
    assert subset in possible_subsets, f"Subset should be one of: {possible_subsets}"

    namespace = subset
    models = ["model", "calibrator"]

    sub_pipelines = []

    for model in models:
        pipeline_template = pipeline(
            [
                node(
                    name="sample_data_node",
                    func=sample_data,
                    inputs=["abt", "params:n_obs", "params:sampling_seed"],
                    outputs="abt_sample",
                ),
                node(
                    name=f"{model}.calculate_shap_node",
                    func=calculate_shap,
                    inputs=["abt_sample", f"stored.{model}"],
                    outputs=f"{model}.shap_values",
                ),
                node(
                    name=f"{model}.create_explanations_node",
                    func=create_explanations,
                    inputs=[
                        f"{model}.shap_values",
                        "abt_sample",
                        f"stored.{model}",
                        "params:pdp_top_n",
                    ],
                    outputs=[
                        f"{model}.shap_summary_plot",
                        f"{model}.feature_importance",
                        f"{model}.partial_dependence_plots",
                    ],
                ),
            ]
        )

        sub_pipeline = pipeline(
            pipe=pipeline_template,
            inputs={
                f"stored.{model}": f"stored.{model}",
            },
            parameters={
                "n_obs": "n_obs",
                "sampling_seed": "sampling_seed",
                "pdp_top_n": "pdp_top_n",
            },
            outputs={
                f"{model}.shap_values": f"{model}.shap_values",
                f"{model}.shap_summary_plot": f"{model}.shap_summary_plot",
                f"{model}.feature_importance": f"{model}.feature_importance",
                f"{model}.partial_dependence_plots": f"{model}.partial_dependence_plots",
            },
            namespace=namespace,
        )

        sub_pipelines += [sub_pipeline]

    main_pipeline = sub_pipelines[0] + sub_pipelines[1]

    return main_pipeline
