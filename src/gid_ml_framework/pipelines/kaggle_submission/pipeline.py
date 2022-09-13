from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_submission


def create_pipeline(dataset: str, **kwargs) -> Pipeline:
    namespace = dataset
    gr_namespace = "dgsr_kaggle_gr"
    pipeline_template = pipeline(
        [
            node(
                func=generate_submission,
                inputs=[
                    "predictions",
                    "user_mapping",
                    "item_mapping",
                    "params:new_item_column",
                    "params:new_user_column",
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
            "user_mapping": f"{dataset}_users_mapping",
            "item_mapping": f"{dataset}_items_mapping",
        },
        outputs={
            "submission": f"{dataset}_submission",
        },
        namespace=namespace,
    )

    return main_pipeline
