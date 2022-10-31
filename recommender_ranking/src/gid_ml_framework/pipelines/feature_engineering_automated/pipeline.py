from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import create_static_features


def create_pipeline(**kwargs) -> Pipeline:
    # Train vs inference flag
    TRAIN_FLAG = ""  # 'train_'/''

    return pipeline(
        [
            node(
                func=create_static_features,
                name="create_static_features_node",
                inputs=[
                    f"{TRAIN_FLAG}transactions",
                    "customers",
                    "articles",
                    "params:n_days",
                ],
                outputs=[
                    "automated_articles_features_temp",
                    "automated_customers_features_temp",
                ],
            ),
        ],
        namespace="feature_engineering_automated",
        inputs=[f"{TRAIN_FLAG}transactions", "customers", "articles"],
        outputs=[
            "automated_articles_features_temp",
            "automated_customers_features_temp",
        ],
    )
