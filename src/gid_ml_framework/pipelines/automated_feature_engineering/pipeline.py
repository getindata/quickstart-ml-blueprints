from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import create_static_features, feature_selection


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_static_features,
                name="create_static_features_node",
                inputs=[
                    "transactions_sample",
                    "customers_sample",
                    "articles_sample",
                    "params:automated_feature_engineering.n_days",
                    ],
                outputs=["automated_articles_features_temp", "automated_customers_features_temp"],
            ),
            node(
                func=feature_selection,
                name="feature_selection_articles_node",
                inputs=[
                    "automated_articles_features_temp",
                    "params:automated_feature_engineering.articles.feature_selection",
                    "params:automated_feature_engineering.articles.feature_selection_params",
                    ],
                outputs="automated_articles_features",
            ),
            node(
                func=feature_selection,
                name="feature_selection_customers_node",
                inputs=[
                    "automated_customers_features_temp",
                    "params:automated_feature_engineering.customers.feature_selection",
                    "params:automated_feature_engineering.customers.feature_selection_params",
                    ],
                outputs="automated_customers_features",
            ),
        ],
        namespace="automated_feature_engineering",
        inputs=["transactions_sample", "customers_sample", "articles_sample"],
        outputs=["automated_articles_features", "automated_customers_features"],
    )