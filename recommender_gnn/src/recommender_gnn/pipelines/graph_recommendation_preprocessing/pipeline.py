from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import map_users_and_items, preprocess_transactions


def create_pipeline(dataset: str, comments: str = None, **kwargs) -> Pipeline:
    """Creates pipeline for preprocessing transactions dataframes into format required by recommender graph neural
     network models.

    Args:
        dataset (str): dataset name
        comments (str): comments to add to the pipeline namespace
    """
    namespace = f"graph_recommendation_preprocessing_{dataset}"
    to_act_namespace = f"{dataset}_to_act"
    if comments:
        namespace += f"_{comments}"
    main_pipeline_instance = pipeline(
        [
            node(
                func=preprocess_transactions,
                inputs=[
                    "train_transactions",
                    "test_transactions",
                    "params:train_subset",
                    "params:test_subset",
                    "params:original_date_column",
                ],
                outputs="transactions_preprocessed",
                name="preprocess_transactions_node",
                tags=["preprocessing_tag"],
            ),
            node(
                func=map_users_and_items,
                inputs=[
                    "transactions_preprocessed",
                ],
                outputs=[
                    "transactions_mapped",
                    "users_mapping",
                    "items_mapping",
                ],
                name="map_users_and_items_node",
            ),
        ]
    )

    main_pipeline = pipeline(
        pipe=main_pipeline_instance,
        inputs={
            "train_transactions": f"{to_act_namespace}_train.transactions",
            "test_transactions": f"{to_act_namespace}_test.transactions",
        },
        namespace=namespace,
    )

    return main_pipeline
