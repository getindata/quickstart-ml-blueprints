from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import extract_articles, extract_customers, extract_transactions


def create_pipeline(subset: str, **kwargs) -> Pipeline:
    """Creates pipeline for converting otto dataset to act format

    Args:
        subset (str): subset name [train/test]
    """
    namespace = f"otto_to_act_{subset}"
    pipeline_template = pipeline(
        [
            node(
                func=extract_transactions,
                inputs=[
                    "preprocessed_df",
                ],
                outputs="otto_transactions_act",
                name=f"otto_{subset}_to_transactions_node",
            ),
            node(
                func=extract_articles,
                inputs=[
                    "transactions_df",
                ],
                outputs="otto_articles_act",
                name=f"otto_{subset}_to_articles_node",
            ),
            node(
                func=extract_customers,
                inputs=[
                    "transactions_df",
                ],
                outputs="otto_customers_act",
                name=f"otto_{subset}_to_customers_node",
            ),
        ]
    )
    act_subset = "val" if subset == "test" else subset
    main_pipeline = pipeline(
        pipe=pipeline_template,
        inputs={
            "preprocessed_df": f"otto_{subset}_preprocessed",
            "transactions_df": f"otto_transactions_{act_subset}_act",
        },
        outputs={
            "otto_transactions_act": f"otto_transactions_{act_subset}_act",
            "otto_articles_act": f"otto_articles_{act_subset}_act",
            "otto_customers_act": f"otto_customers_{act_subset}_act",
        },
        namespace=namespace,
    )

    return main_pipeline
