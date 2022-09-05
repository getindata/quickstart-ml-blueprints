from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    unpack_candidates, filter_last_n_rows_per_customer, create_list_of_previously_bought_articles,
    create_set_of_attributes, apply_avg_jaccard_similarity,
    calculate_customer_embeddings, apply_cosine_similarity,
    merge_similarity_features
)

# nodes for multiple use
unpack_candidates_node = node(
    func=unpack_candidates,
    name="unpack_candidates_node",
    inputs=[
        "train_candidates",
        "params:drop_random_strategies",
        ],
        outputs="unpacked_candidates",
)
filter_last_n_rows_per_customer_node = node(
    func=filter_last_n_rows_per_customer,
    name="filter_last_n_rows_per_customer_node",
    inputs=[
        "train_transactions",
        "params:last_n_rows",
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
                    "params:jaccard.attribute_cols",
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
        inputs=["train_candidates", "articles", "train_transactions"],
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
                    "params:cosine.image.col_name",
                    ],
                outputs="image_cosine_similarity_features",
            ),
        ],
        namespace="candidates_feature_engineering",
        inputs=["train_candidates", "train_transactions", "image_embeddings"],
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
                    "params:cosine.text.col_name",
                    ],
                outputs="text_cosine_similarity_features",
            ),
        ],
        namespace="candidates_feature_engineering",
        inputs=["train_candidates", "train_transactions", "text_embeddings"],
        outputs="text_cosine_similarity_features",
    )

    merge_similarity_features_pipeline = pipeline(
        [
            node(
                func=merge_similarity_features,
                name="merge_similarity_features_node",
                inputs=[
                    "jaccard_similarity_features",
                    "image_cosine_similarity_features",
                    "text_cosine_similarity_features",
                    ],
                outputs="candidates_similarity_features",
            ),
        ],
        namespace="candidates_feature_engineering",
        inputs=[
            "jaccard_similarity_features",
            "image_cosine_similarity_features",
            "text_cosine_similarity_features",
            ],
        outputs="candidates_similarity_features",
    )
    
    return (
        jaccard_similarity_pipeline + 
        text_cosine_similarity_pipeline + 
        image_cosine_similarity_pipeline + 
        merge_similarity_features_pipeline
    )
