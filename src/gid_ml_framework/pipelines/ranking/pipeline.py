from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import add_article_features, add_customer_features, add_dict_features


def create_pipeline(**kwargs) -> Pipeline:
    prepare_dataset_pipeline = pipeline(
        [
            node(
                func=add_article_features,
                name="add_article_features_node",
                inputs=[
                    "candidates",
                    "automated_articles_features",
                    "manual_article_features",
                    "params:ranking.data.fill_na_pattern",
                    ],
                outputs="candidates_step_1",
            ),
            node(
                func=add_customer_features,
                name="add_customer_features_node",
                inputs=[
                    "candidates_step_1",
                    "automated_customers_features",
                    "manual_customer_features",
                    "params:ranking.data.fill_na_pattern",
                    ],
                outputs="candidates_step_2",
            ),
            node(
                func=add_dict_features,
                name="add_dict_features_node",
                inputs=[
                    "candidates_step_2",
                    "articles",
                    "customers",
                    "params:ranking.data.categorical_cols",
                    "params:ranking.data.drop_cols",
                    ],
                outputs="training_data",
            ),
        ],
        namespace="ranking",
        inputs=["candidates",
                "automated_articles_features",
                "manual_article_features",
                "automated_customers_features",
                "manual_customer_features",
                "articles",
                "customers",
                ],
        outputs="training_data",
    )

    
    return prepare_dataset_pipeline
