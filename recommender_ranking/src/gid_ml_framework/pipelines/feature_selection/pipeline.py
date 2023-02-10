from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import feature_selection


def create_pipeline(**kwargs) -> Pipeline:
    feature_selection_pipeline = pipeline(
        [
            node(
                func=feature_selection,
                name="feature_selection_automated_articles_columns_node",
                inputs=[
                    "automated_articles_features_temp",
                    "params:feature_selection_params",
                    "params:feature_selection",
                ],
                outputs="automated_articles_columns",
            ),
            node(
                func=feature_selection,
                name="feature_selection_automated_customers_columns_node",
                inputs=[
                    "automated_customers_features_temp",
                    "params:feature_selection_params",
                    "params:feature_selection",
                ],
                outputs="automated_customers_columns",
            ),
            node(
                func=feature_selection,
                name="feature_selection_manual_articles_columns_node",
                inputs=[
                    "manual_articles_features_temp",
                    "params:feature_selection_params",
                    "params:feature_selection",
                ],
                outputs="manual_articles_columns",
            ),
            node(
                func=feature_selection,
                name="feature_selection_manual_customers_columns_node",
                inputs=[
                    "manual_customers_features_temp",
                    "params:feature_selection_params",
                    "params:feature_selection",
                ],
                outputs="manual_customers_columns",
            ),
        ],
        namespace="feature_selection",
        inputs=[
            "automated_articles_features_temp",
            "automated_customers_features_temp",
            "manual_articles_features_temp",
            "manual_customers_features_temp",
        ],
        outputs=[
            "automated_articles_columns",
            "automated_customers_columns",
            "manual_articles_columns",
            "manual_customers_columns",
        ],
    )

    return feature_selection_pipeline
