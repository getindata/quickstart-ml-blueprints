from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import map_users_and_items, preprocess_transactions


def create_pipeline(dataset: str, comments: str = None, **kwargs) -> Pipeline:
    """Creates pipeline for preprocessing transactions dataframes into format required by recommender graph neural
     network models.

    Args:
        dataset (str): dataset name
        train_subset (bool): whether to include train subset
        val_subset (bool): whether to include val subset
        comments (str): comments to add to the pipeline namespace
    """
    namespace = f"graph_recommendation_preprocessing_{dataset}"
    if comments:
        namespace += f"_{comments}"
    main_pipeline_instance = pipeline(
        [
            node(
                func=preprocess_transactions,
                inputs=[
                    "transactions_train",
                    "transactions_val",
                    "params:train_subset",
                    "params:val_subset",
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
            "transactions_train": f"{dataset}_transactions_train_act",
            "transactions_val": f"{dataset}_transactions_val_act",
        },
        outputs={
            "transactions_mapped": f"{namespace}_transactions_mapped",
            "users_mapping": f"{namespace}_users_mapping",
            "items_mapping": f"{namespace}_items_mapping",
        },
        namespace=namespace,
    )

    return main_pipeline
