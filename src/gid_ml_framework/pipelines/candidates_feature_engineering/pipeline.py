from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    create_set_of_attributes, filter_last_n_rows_per_customer, apply_avg_jaccard_similarity,
    calculate_customer_embeddings, apply_cosine_similarity
)


def create_pipeline(**kwargs) -> Pipeline:
    filter_last_n_rows_per_customer_node = node(
                func=filter_last_n_rows_per_customer,
                name="filter_last_n_rows_per_customer_node",
                inputs=[
                    "transactions_sample",
                    "params:candidates_feature_engineering.last_n_rows",
                    ],
                outputs="latest_transactions",
    )

    jaccard_similarity_pipeline = pipeline(
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
            filter_last_n_rows_per_customer_node,
            node(
                func=apply_avg_jaccard_similarity,
                name="apply_avg_jaccard_similarity_node",
                inputs=[
                    "candidates_sample",
                    "article_attributes",
                    "latest_transactions",
                    ],
                outputs="jaccard_similarity_features",
            ),
        ],
        namespace="candidates_feature_engineering",
        inputs=["articles_sample", "transactions_sample", "candidates_sample"],
        outputs="jaccard_similarity_features",
    )

    image_cosine_similarity_pipeline = pipeline(
        [
            filter_last_n_rows_per_customer_node,
            node(
                func=calculate_customer_embeddings,
                name="calculate_customer_embeddings_image_node",
                inputs=[
                    "latest_transactions",
                    "image_embeddings",
                    ],
                outputs="image_customer_embeddings",
            ),
            node(
                func=apply_cosine_similarity,
                name="apply_cosine_similarity_image_node",
                inputs=[
                    "candidates_sample",
                    "image_customer_embeddings",
                    "image_embeddings",
                    "params:candidates_feature_engineering.cosine.image.col_name",
                    ],
                outputs="image_cosine_similarity_features",
            ),
        ],
        namespace="candidates_feature_engineering",
        inputs=["transactions_sample", "image_embeddings", "candidates_sample"],
        outputs="image_cosine_similarity_features",
    )

    text_cosine_similarity_pipeline = pipeline(
        [
            filter_last_n_rows_per_customer_node,
            node(
                func=calculate_customer_embeddings,
                name="calculate_customer_embeddings_text_node",
                inputs=[
                    "latest_transactions",
                    "text_embeddings",
                    ],
                outputs="text_customer_embeddings",
            ),
            node(
                func=apply_cosine_similarity,
                name="apply_cosine_similarity_text_node",
                inputs=[
                    "candidates_sample",
                    "text_customer_embeddings",
                    "text_embeddings",
                    "params:candidates_feature_engineering.cosine.text.col_name",
                    ],
                outputs="text_cosine_similarity_features",
            ),
        ],
        namespace="candidates_feature_engineering",
        inputs=["transactions_sample", "text_embeddings", "candidates_sample"],
        outputs="text_cosine_similarity_features",
    )
    return jaccard_similarity_pipeline + text_cosine_similarity_pipeline + image_cosine_similarity_pipeline
