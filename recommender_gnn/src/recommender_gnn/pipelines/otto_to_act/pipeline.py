from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import extract_articles, extract_customers, extract_transactions


def create_pipeline(subset: str, **kwargs) -> Pipeline:
    """Creates pipeline for converting otto dataset to act format

    Args:
        subset (str): subset name [train/test]
    """
    namespace = f"otto_to_act_{subset}"
    preprocessing_namespace = f"otto_preprocessing_{subset}"
    pipeline_template = pipeline(
        [
            node(
                func=extract_transactions,
                inputs=[
                    "sample_df",
                ],
                outputs="transactions",
                name="extract_transactions_node",
            ),
            node(
                func=extract_articles,
                inputs=[
                    "transactions",
                ],
                outputs="articles",
                name="extract_articles_node",
            ),
            node(
                func=extract_customers,
                inputs=[
                    "transactions",
                ],
                outputs="customers",
                name="extract_customers_node",
            ),
        ]
    )
    main_pipeline = pipeline(
        pipe=pipeline_template,
        inputs={
            "sample_df": f"{preprocessing_namespace}.sample_df",
        },
        namespace=namespace,
    )

    return main_pipeline
