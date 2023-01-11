from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import create_static_features


def create_pipeline(train_flag: bool = False, **kwargs) -> Pipeline:
    "Use empty string '' if inference, or 'train_' if training flag"
    mode_prefix = "train_" if train_flag else ""

    return pipeline(
        [
            node(
                func=create_static_features,
                name="create_static_features_node",
                inputs=[
                    f"{mode_prefix}transactions",
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
        inputs=[f"{mode_prefix}transactions", "customers", "articles"],
        outputs=[
            "automated_articles_features_temp",
            "automated_customers_features_temp",
        ],
    )
