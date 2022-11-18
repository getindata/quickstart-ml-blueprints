from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    santander_to_articles,
    santander_to_customers,
    santander_to_transactions,
)


def create_pipeline(**kwargs) -> Pipeline:

    main_pipeline_instance = pipeline(
        [
            node(
                func=santander_to_articles,
                inputs="santander_train",
                outputs="santander_articles",
                name="santander_to_articles_node",
                tags=["preprocessing_tag"],
            ),
            node(
                func=santander_to_customers,
                inputs=[
                    "santander_train",
                    "santander_val",
                    "params:customers.merge_type",
                ],
                outputs="santander_customers",
                name="santander_to_customers_node",
                tags=["preprocessing_tag"],
            ),
            node(
                func=santander_to_transactions,
                inputs=["santander_train", "santander_val"],
                outputs=["santander_transactions_train", "santander_transactions_val"],
                name="santander_to_transactions_node",
                tags=["preprocessing_tag"],
            ),
        ]
    )

    main_pipeline = pipeline(
        pipe=main_pipeline_instance,
        inputs=["santander_train", "santander_val"],
        outputs=[
            "santander_articles",
            "santander_customers",
            "santander_transactions_train",
            "santander_transactions_val",
        ],
        namespace="santander_to_act_main",
        parameters={
            "params:customers.merge_type": "params:santander_to_act_main.customers.merge_type"
        },
    )

    return main_pipeline
