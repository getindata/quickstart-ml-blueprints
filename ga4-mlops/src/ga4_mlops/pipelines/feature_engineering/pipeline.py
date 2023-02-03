"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    apply_encoders,
    apply_imputers,
    engineer_features,
    exclude_features,
    fit_encoders,
    fit_imputers,
)


def create_pipeline(subset: str, **kwargs) -> Pipeline:
    """Creates a parametrized pipeline feature engineering

    Args:
        subset (str): data subset. Possible values: ["train", "valid", "test", "predict"].
    """
    possible_subsets = ["train", "valid", "test", "predict"]
    assert subset in possible_subsets, f"Subset should be one of: {possible_subsets}"

    trasnformations_fitting_and_application_pipeline = pipeline(
        [
            node(
                name="train.feature_engineering_node",
                func=engineer_features,
                inputs=["train.df"],
                outputs="train.df_fe_temp",
            ),
            node(
                name="train.exclude_features_node",
                func=exclude_features,
                inputs=["train.df_fe_temp", "params:features_to_exclude"],
                outputs="train.df_excl_temp",
            ),
            node(
                name="train.imputers_fitting_node",
                func=fit_imputers,
                inputs=["train.df_excl_temp", "params:imputation_strategies"],
                outputs="fitted.imputers",
            ),
            node(
                name="train.imputers_application_node",
                func=apply_imputers,
                inputs=["train.df_excl_temp", "fitted.imputers"],
                outputs="train.df_imp_temp",
            ),
            node(
                name="train.encoders_fitting_node",
                func=fit_encoders,
                inputs=["train.df_imp_temp", "params:encoder_types"],
                outputs="fitted.feature_encoders",
            ),
            node(
                name="train.encoders_application_node",
                func=apply_encoders,
                inputs=["train.df_imp_temp", "fitted.feature_encoders"],
                outputs="train.abt",
            ),
        ]
    )

    transformations_source = "stored" if subset == "predict" else "fitted"

    trasnformations_application_pipeline = pipeline(
        [
            node(
                name=f"{subset}.feature_engineering_node",
                func=engineer_features,
                inputs=[f"{subset}.df"],
                outputs=f"{subset}.df_fe_temp",
            ),
            node(
                name=f"{subset}.exclude_features_node",
                func=exclude_features,
                inputs=[f"{subset}.df_fe_temp", "params:features_to_exclude"],
                outputs=f"{subset}.df_excl_temp",
            ),
            node(
                name=f"{subset}.imputers_application_node",
                func=apply_imputers,
                inputs=[f"{subset}.df_excl_temp", f"{transformations_source}.imputers"],
                outputs=f"{subset}.df_imp_temp",
            ),
            node(
                name=f"{subset}.encoders_application_node",
                func=apply_encoders,
                inputs=[
                    f"{subset}.df_imp_temp",
                    f"{transformations_source}.feature_encoders",
                ],
                outputs=f"{subset}.abt",
            ),
        ]
    )

    if subset == "train":
        return trasnformations_fitting_and_application_pipeline
    else:
        return trasnformations_application_pipeline
