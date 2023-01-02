"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    apply_encoders,
    apply_imputers,
    engineer_features,
    exclude_features,
    fit_encoders,
    fit_imputers,
)


def create_pipeline(train: bool = True, **kwargs) -> Pipeline:

    feature_engineering_train_pipeline = pipeline(
        [
            node(
                name="feature_engineering_train_node",
                func=engineer_features,
                inputs=["df_train"],
                outputs="df_train_fe_temp",
            ),
            node(
                name="feature_engineering_valid_node",
                func=engineer_features,
                inputs=["df_valid"],
                outputs="df_valid_fe_temp",
            ),
            node(
                name="feature_engineering_test_node",
                func=engineer_features,
                inputs=["df_test"],
                outputs="df_test_fe_temp",
            ),
            node(
                name="imputers_fitting_node",
                func=fit_imputers,
                inputs=["df_train_fe_temp", "params:imputation_strategies"],
                outputs="imputers_train",
            ),
            node(
                name="imputers_application_train_node",
                func=apply_imputers,
                inputs=["df_train_fe_temp", "imputers_train"],
                outputs="df_train_imp_temp",
            ),
            node(
                name="imputers_application_valid_node",
                func=apply_imputers,
                inputs=["df_valid_fe_temp", "imputers_train"],
                outputs="df_valid_imp_temp",
            ),
            node(
                name="imputers_application_test_node",
                func=apply_imputers,
                inputs=["df_test_fe_temp", "imputers_train"],
                outputs="df_test_imp_temp",
            ),
            node(
                name="encoders_fitting_node",
                func=fit_encoders,
                inputs=["df_train_imp_temp", "params:encoder_types"],
                outputs="feature_encoders_train",
            ),
            node(
                name="encoders_application_train_node",
                func=apply_encoders,
                inputs=["df_train_imp_temp", "feature_encoders_train"],
                outputs="df_train_fe",
            ),
            node(
                name="encoders_application_valid_node",
                func=apply_encoders,
                inputs=["df_valid_imp_temp", "feature_encoders_train"],
                outputs="df_valid_fe",
            ),
            node(
                name="encoders_application_test_node",
                func=apply_encoders,
                inputs=["df_test_imp_temp", "feature_encoders_train"],
                outputs="df_test_fe",
            ),
            node(
                name="exclude_features_train_node",
                func=exclude_features,
                inputs=["df_train_fe", "params:features_to_exclude"],
                outputs="abt_train",
            ),
            node(
                name="exclude_features_valid_node",
                func=exclude_features,
                inputs=["df_valid_fe", "params:features_to_exclude"],
                outputs="abt_valid",
            ),
            node(
                name="exclude_features_test_node",
                func=exclude_features,
                inputs=["df_test_fe", "params:features_to_exclude"],
                outputs="abt_test",
            ),
        ]
    )

    feature_engineering_predict_pipeline = pipeline(
        [
            node(
                name="feature_engineering_predict_node",
                func=engineer_features,
                inputs=["df_predict"],
                outputs="df_predict_fe_temp",
            ),
            node(
                name="imputers_application_predict_node",
                func=apply_imputers,
                inputs=["df_predict_fe_temp", "imputers_predict"],
                outputs="df_predict_imp_temp",
            ),
            node(
                name="encoders_application_predict_node",
                func=apply_encoders,
                inputs=["df_predict_imp_temp", "feature_encoders_predict"],
                outputs="df_predict_fe",
            ),
            node(
                name="exclude_features_predict_node",
                func=exclude_features,
                inputs=["df_predict_fe", "params:features_to_exclude"],
                outputs="abt_predict",
            ),
        ]
    )

    if train:
        return feature_engineering_train_pipeline
    else:
        return feature_engineering_predict_pipeline
