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
                name="feature_engineering_train_node",
                func=engineer_features,
                inputs=["df_train"],
                outputs="df_train_fe_temp",
            ),
            node(
                name="imputers_fitting_node",
                func=fit_imputers,
                inputs=["df_train_fe_temp", "params:imputation_strategies"],
                outputs="imputers_fitted",
            ),
            node(
                name="imputers_application_train_node",
                func=apply_imputers,
                inputs=["df_train_fe_temp", "imputers_fitted"],
                outputs="df_train_imp_temp",
            ),
            node(
                name="encoders_fitting_node",
                func=fit_encoders,
                inputs=["df_train_imp_temp", "params:encoder_types"],
                outputs="feature_encoders_fitted",
            ),
            node(
                name="encoders_application_train_node",
                func=apply_encoders,
                inputs=["df_train_imp_temp", "feature_encoders_fitted"],
                outputs="df_train_fe",
            ),
            node(
                name="exclude_features_train_node",
                func=exclude_features,
                inputs=["df_train_fe", "params:features_to_exclude"],
                outputs="abt_train",
            ),
        ]
    )

    transformations_source = "stored" if subset == "predict" else "fitted"

    trasnformations_application_pipeline = pipeline(
        [
            node(
                name=f"feature_engineering_{subset}_node",
                func=engineer_features,
                inputs=[f"df_{subset}"],
                outputs=f"df_{subset}_fe_temp",
            ),
            node(
                name=f"imputers_application_{subset}_node",
                func=apply_imputers,
                inputs=[f"df_{subset}_fe_temp", f"imputers_{transformations_source}"],
                outputs=f"df_{subset}_imp_temp",
            ),
            node(
                name=f"encoders_application_{subset}_node",
                func=apply_encoders,
                inputs=[
                    f"df_{subset}_imp_temp",
                    f"feature_encoders_{transformations_source}",
                ],
                outputs=f"df_{subset}_fe",
            ),
            node(
                name=f"exclude_features_{subset}_node",
                func=exclude_features,
                inputs=[f"df_{subset}_fe", "params:features_to_exclude"],
                outputs=f"abt_{subset}",
            ),
        ]
    )

    if subset == "train":
        return trasnformations_fitting_and_application_pipeline
    else:
        return trasnformations_application_pipeline
