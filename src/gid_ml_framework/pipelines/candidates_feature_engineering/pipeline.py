from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import create_set_of_attributes, filter_last_n_rows_per_customer, apply_avg_jaccard_similarity


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_set_of_attributes,
                name="create_set_of_attributes_node",
                inputs=[
                    "articles_sample",
                    "params:candidates_feature_engineering.jaccard.attribute_cols",
                    ],
                outputs="article_attributes",
            ),
            node(
                func=filter_last_n_rows_per_customer,
                name="filter_last_n_rows_per_customer_node",
                inputs=[
                    "transactions_sample",
                    "params:candidates_feature_engineering.jaccard.last_n_rows",
                    ],
                outputs="latest_transactions",
            ),
            node(
                func=apply_avg_jaccard_similarity,
                name="apply_avg_jaccard_similarity_node",
                inputs=[
                    "candidates_sample",
                    "article_attributes",
                    "latest_transactions",
                    ],
                outputs="similarity_features",
            ),
        ],
        namespace="candidates_feature_engineering",
        inputs=["articles_sample", "transactions_sample", "candidates_sample"],
        outputs="similarity_features",
    )
