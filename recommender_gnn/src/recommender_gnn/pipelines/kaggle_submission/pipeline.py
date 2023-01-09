from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_submission


def create_pipeline(dataset: str, **kwargs) -> Pipeline:
    """Creates pipeline for generating kaggle submission file from saved predictions

    Args:
        dataset (str): dataset name [santander/hm]
    """
    namespace = f"{dataset}_ks"
    gr_namespace = "dgsr_kaggle_gr"
    pipeline_template = pipeline(
        [
            node(
                func=generate_submission,
                inputs=[
                    "predictions",
                    "all_users",
                    "users_mapping",
                    "items_mapping",
                    "test_input",
                    "params:new_item_column",
                    "params:new_user_column",
                    "params:original_user_column",
                    "params:filter_by_test_users",
                ],
                outputs="submission",
                name=f"generate_{dataset}_submission_node",
            )
        ]
    )

    main_pipeline = pipeline(
        pipe=pipeline_template,
        inputs={
            "predictions": f"{dataset}_{gr_namespace}_predictions",
            "all_users": f"{dataset}_customers",
            "users_mapping": f"{dataset}_users_mapping",
            "items_mapping": f"{dataset}_items_mapping",
            "test_input": f"{dataset}_test_input",
        },
        outputs={
            "submission": f"{dataset}_submission",
        },
        namespace=namespace,
    )

    return main_pipeline
