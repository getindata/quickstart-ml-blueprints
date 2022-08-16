from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    unpack_candidates,
    create_set_of_attributes, create_list_of_previously_bought_articles, filter_last_n_rows_per_customer, apply_avg_jaccard_similarity,
    calculate_customer_embeddings, apply_cosine_similarity
)

# nodes for multiple use
unpack_candidates_node = node(
    func=unpack_candidates,
    name="unpack_candidates_node",
    inputs=[
        "candidates_sample",
        ],
        outputs="unpacked_candidates",
)
filter_last_n_rows_per_customer_node = node(
    func=filter_last_n_rows_per_customer,
    name="filter_last_n_rows_per_customer_node",
    inputs=[
        "transactions_sample",
        "params:candidates_feature_engineering.last_n_rows",
        ],
        outputs="latest_transactions",
)
create_list_of_previously_bought_articles_node = node(
    func=create_list_of_previously_bought_articles,
    name="create_list_of_previously_bought_articles_node",
    inputs=[
        "latest_transactions",
        ],
    outputs="customer_list_of_articles",
)

def create_pipeline(**kwargs) -> Pipeline:
    jaccard_similarity_pipeline = pipeline(
        [
            unpack_candidates_node,
            filter_last_n_rows_per_customer_node,
            create_list_of_previously_bought_articles_node,
            node(
                func=create_set_of_attributes,
                name="create_set_of_attributes_node",
                inputs=[
                    "articles",
                    "params:candidates_feature_engineering.jaccard.attribute_cols",
                    ],
                outputs="article_attributes",
            ),
            node(
                func=apply_avg_jaccard_similarity,
                name="apply_avg_jaccard_similarity_node",
                inputs=[
                    "unpacked_candidates",
                    "article_attributes",
                    "customer_list_of_articles",
                    ],
                outputs="jaccard_similarity_features",
            ),
        ],
        namespace="candidates_feature_engineering",
        inputs=["candidates_sample", "articles", "transactions_sample"],
        outputs="jaccard_similarity_features",
    )

    image_cosine_similarity_pipeline = pipeline(
        [
            unpack_candidates_node,
            filter_last_n_rows_per_customer_node,
            create_list_of_previously_bought_articles_node,
            node(
                func=calculate_customer_embeddings,
                name="calculate_customer_embeddings_image_node",
                inputs=[
                    "customer_list_of_articles",
                    "image_embeddings",
                    ],
                outputs="image_customer_embeddings",
            ),
            node(
                func=apply_cosine_similarity,
                name="apply_cosine_similarity_image_node",
                inputs=[
                    "unpacked_candidates",
                    "image_customer_embeddings",
                    "image_embeddings",
                    "params:candidates_feature_engineering.cosine.image.col_name",
                    ],
                outputs="image_cosine_similarity_features",
            ),
        ],
        namespace="candidates_feature_engineering",
        inputs=["candidates_sample", "transactions_sample", "image_embeddings"],
        outputs="image_cosine_similarity_features",
    )

    text_cosine_similarity_pipeline = pipeline(
        [
            unpack_candidates_node,
            filter_last_n_rows_per_customer_node,
            create_list_of_previously_bought_articles_node,
            node(
                func=calculate_customer_embeddings,
                name="calculate_customer_embeddings_text_node",
                inputs=[
                    "customer_list_of_articles",
                    "text_embeddings",
                    ],
                outputs="text_customer_embeddings",
            ),
            node(
                func=apply_cosine_similarity,
                name="apply_cosine_similarity_text_node",
                inputs=[
                    "unpacked_candidates",
                    "text_customer_embeddings",
                    "text_embeddings",
                    "params:candidates_feature_engineering.cosine.text.col_name",
                    ],
                outputs="text_cosine_similarity_features",
            ),
        ],
        namespace="candidates_feature_engineering",
        inputs=["candidates_sample", "transactions_sample", "text_embeddings"],
        outputs="text_cosine_similarity_features",
    )
    return jaccard_similarity_pipeline + text_cosine_similarity_pipeline + image_cosine_similarity_pipeline
