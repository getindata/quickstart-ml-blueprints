from typing import List

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import map_users_and_items, preprocess_transactions


def create_pipeline(
    dataset: str, transactions_subsets: List[str], comments: str = None, **kwargs
) -> Pipeline:
    """Creates pipeline for preprocessing transactions dataframes into format required by recommender graph neural
     network models. It can use both train and test subsets of the dataset as split into train/test/val subsets for
     graph recommendation is done in the graph recommendation modelling pipeline. It is useful if we want to utilize
     all available information form model training and evaluation.

    Args:
        dataset (str): dataset name
        transactions_subsets (List[str]): list of transaction subsets to use for preprocessing (e.g. ["train", "test"])
        comments (str): comments to add to the pipeline namespace
    """
    namespace = f"graph_recommendation_preprocessing_{dataset}"
    to_act_namespace = f"{dataset}_to_act"
    if comments:
        namespace += f"_{comments}"
    preprocessing_params = [
        "params:original_date_column",
        "params:original_item_column",
        "params:original_user_column",
        "params:new_date_column",
        "params:new_item_column",
        "params:new_user_column",
    ]
    preprocessing_inputs = transactions_subsets.copy()
    preprocessing_inputs[1:1] = preprocessing_params

    main_pipeline_instance = pipeline(
        [
            node(
                func=preprocess_transactions,
                inputs=preprocessing_inputs,
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
    inputs_mapping = {}
    for subset in transactions_subsets:
        inputs_mapping[subset] = f"{to_act_namespace}_{subset}.transactions"

    main_pipeline = pipeline(
        pipe=main_pipeline_instance,
        inputs=inputs_mapping,
        namespace=namespace,
    )

    return main_pipeline
