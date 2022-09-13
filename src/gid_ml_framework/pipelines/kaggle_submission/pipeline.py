from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_submission


def create_pipeline(dataset: str, **kwargs) -> Pipeline:
    gr_namespace = "dgsr_kaggle_gr"
    pipeline_template = pipeline(
        [
            node(
                func=generate_submission(dataset),
                inputs=[
                    "predictions",
                    "user_mapping",
                    "item_mapping",
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
        namespace=dataset,
    )

    return main_pipeline
