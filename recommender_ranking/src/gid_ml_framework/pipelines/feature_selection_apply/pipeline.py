from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import apply_feature_selection


def create_pipeline(**kwargs) -> Pipeline:
    apply_feature_selection_pipeline = pipeline(
        [
            node(
                func=apply_feature_selection,
                name="apply_feature_selection_automated_articles_features_node",
                inputs=[
                    "automated_articles_features_temp",
                    "automated_articles_columns",
                ],
                outputs="automated_articles_features",
            ),
            node(
                func=apply_feature_selection,
                name="apply_feature_selection_automated_customers_features_node",
                inputs=[
                    "automated_customers_features_temp",
                    "automated_customers_columns",
                ],
                outputs="automated_customers_features",
            ),
            node(
                func=apply_feature_selection,
                name="apply_feature_selection_manual_articles_features_node",
                inputs=[
                    "manual_articles_features_temp",
                    "manual_articles_columns",
                ],
                outputs="manual_articles_features",
            ),
            node(
                func=apply_feature_selection,
                name="apply_feature_selection_manual_customers_features_node",
                inputs=[
                    "manual_customers_features_temp",
                    "manual_customers_columns",
                ],
                outputs="manual_customers_features",
            ),
        ],
        namespace="apply_feature_selection",
        inputs=[
            "automated_articles_features_temp",
            "automated_articles_columns",
            "automated_customers_features_temp",
            "automated_customers_columns",
            "manual_articles_features_temp",
            "manual_articles_columns",
            "manual_customers_features_temp",
            "manual_customers_columns",
        ],
        outputs=[
            "automated_articles_features",
            "automated_customers_features",
            "manual_articles_features",
            "manual_customers_features",
        ],
    )

    return apply_feature_selection_pipeline
