from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import create_entity_set, create_static_articles_features, create_static_customer_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_entity_set,
                name="create_entity_set",
                inputs=[
                    "transactions_sample",
                    "customers_sample",
                    "articles_sample",
                    ],
                outputs="entity_set",
            ),
            node(
                func=create_static_articles_features,
                name="create_static_articles_features",
                inputs=[
                    "entity_set",
                    "params:automated_feature_engineering.articles.feature_selection",
                    "params:automated_feature_engineering.articles.null_threshold",
                    ],
                outputs="automated_articles_features",
            ),
            node(
                func=create_static_customer_features,
                name="create_static_customer_features",
                inputs=[
                    "entity_set",
                    "params:automated_feature_engineering.customers.feature_selection",
                    "params:automated_feature_engineering.customers.null_threshold",
                    ],
                outputs="automated_customers_features",
            ),
        ],
        namespace="automated_feature_engineering",
        inputs=["transactions_sample", "customers_sample", "articles_sample"],
        outputs=["automated_articles_features", "automated_customers_features"],
    )
