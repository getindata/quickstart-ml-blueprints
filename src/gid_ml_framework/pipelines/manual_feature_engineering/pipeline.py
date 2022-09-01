from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import create_article_features, create_customer_features, create_customer_product_group_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_article_features,
                name="create_article_features_node",
                inputs=[
                    "transactions_sample",
                    ],
                outputs="manual_article_features",
            ),
            node(
                func=create_customer_features,
                name="create_customer_features_node",
                inputs=[
                    "transactions_sample",
                    "articles_sample",
                    "params:customers.n_days",
                    ],
                outputs="manual_customer_features",
            ),
            node(
                func=create_customer_product_group_features,
                name="create_customer_product_group_features_node",
                inputs=[
                    "transactions_sample",
                    "articles_sample",
                    ],
                outputs="manual_customer_prod_group_features",
            ),
        ],
        namespace="manual_feature_engineering",
        inputs=["transactions_sample", "articles_sample"],
        outputs=["manual_article_features", "manual_customer_features", "manual_customer_prod_group_features"],
    )
