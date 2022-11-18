from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import concat_train_val, map_users_and_items


def create_pipeline(dataset_namespace: str, **kwargs) -> Pipeline:
    main_pipeline_instance = pipeline(
        [
            node(
                func=concat_train_val,
                inputs=[
                    "transactions_train",
                    "transactions_val",
                    "params:concat.date_column",
                ],
                outputs="transactions_graph",
                name="concat_train_val_node",
                tags=["preprocessing_tag"],
            ),
            node(
                func=map_users_and_items,
                inputs=[
                    "transactions_graph",
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
            "transactions_train": f"{dataset_namespace}_transactions_train",
            "transactions_val": f"{dataset_namespace}_transactions_val",
        },
        outputs={
            "transactions_mapped": f"{dataset_namespace}_transactions_mapped",
            "users_mapping": f"{dataset_namespace}_users_mapping",
            "items_mapping": f"{dataset_namespace}_items_mapping",
        },
        namespace=dataset_namespace,
    )

    return main_pipeline
