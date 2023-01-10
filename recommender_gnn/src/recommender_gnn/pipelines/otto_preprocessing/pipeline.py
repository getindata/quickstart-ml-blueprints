from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import sample


def create_pipeline(subset: str, **kwargs) -> Pipeline:
    """Creates pipeline for preprocessing otto dataset

    Args:
        subset (str): subset name [train/test]
    """
    namespace = f"otto_preprocessing_{subset}"
    pipeline_template = pipeline(
        [
            node(
                func=sample,
                inputs=[
                    "input_df",
                    "params:sample.sessions_frac",
                    "params:sample.stratify",
                ],
                outputs="sample_df",
                name="sample_otto_node",
            )
        ]
    )

    main_pipeline = pipeline(
        pipe=pipeline_template,
        inputs={
            "input_df": f"otto_{subset}_parquet",
        },
        namespace=namespace,
    )

    return main_pipeline
