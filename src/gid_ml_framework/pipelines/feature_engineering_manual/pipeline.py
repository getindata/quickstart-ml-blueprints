from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import create_article_features, create_customer_features #, create_customer_product_group_features


def create_pipeline(**kwargs) -> Pipeline:
    # Train vs inference flag
    TRAIN_FLAG='' # 'train_'/''

    return pipeline(
        [
            node(
                func=create_article_features,
                name="create_article_features_node",
                inputs=[
                    f"{TRAIN_FLAG}transactions",
                    ],
                outputs="manual_articles_features_temp",
            ),
            node(
                func=create_customer_features,
                name="create_customer_features_node",
                inputs=[
                    f"{TRAIN_FLAG}transactions",
                    "articles",
                    "params:customers.n_days",
                    ],
                outputs="manual_customers_features_temp",
            ),
            # node(
            #     func=create_customer_product_group_features,
            #     name="create_customer_product_group_features_node",
            #     inputs=[
            #         f"{TRAIN_FLAG}transactions",
            #         "articles",
            #         ],
            #     outputs="manual_customers_prod_group_features",
            # ),
        ],
        namespace="feature_engineering_manual",
        inputs=[f"{TRAIN_FLAG}transactions", "articles"],
        outputs=[
            "manual_articles_features_temp",
            "manual_customers_features_temp",
            # "manual_customers_prod_group_features",
        ],
    )
