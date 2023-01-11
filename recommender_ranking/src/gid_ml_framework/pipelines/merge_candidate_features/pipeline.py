from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    add_article_features,
    add_customer_features,
    add_dict_features,
    add_label,
)


def create_pipeline(**kwargs) -> Pipeline:
    prepare_dataset_pipeline = pipeline(
        [
            node(
                func=add_label,
                name="add_label_node",
                inputs=[
                    "candidates_similarity_features",
                    "val_transactions",
                ],
                outputs="candidates_step_0",
            ),
            node(
                func=add_article_features,
                name="add_article_features_node",
                inputs=[
                    "candidates_step_0",
                    "automated_articles_features",
                    "manual_articles_features",
                    "params:data.fill_na_pattern",
                ],
                outputs="candidates_step_1",
            ),
            node(
                func=add_customer_features,
                name="add_customer_features_node",
                inputs=[
                    "candidates_step_1",
                    "automated_customers_features",
                    "manual_customers_features",
                    "params:data.fill_na_pattern",
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
                    "params:data.categorical_cols",
                    "params:data.drop_cols",
                ],
                outputs="final_candidates",
            ),
        ],
        namespace="merge_candidate_features",
        inputs=[
            "candidates_similarity_features",
            "val_transactions",
            "automated_articles_features",
            "manual_articles_features",
            "automated_customers_features",
            "manual_customers_features",
            "articles",
            "customers",
        ],
        outputs=[
            "final_candidates",
            "candidates_step_0",
            "candidates_step_1",
            "candidates_step_2",
        ],
    )

    return prepare_dataset_pipeline
